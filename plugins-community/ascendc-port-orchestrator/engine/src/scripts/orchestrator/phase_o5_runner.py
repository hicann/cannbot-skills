# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O5 runner — real SSH-based verifier execution (P0kk Step 2.1).

Plan: docs/design/CONTRACT_AND_MATURITY_NOTES.md#fsm-phase-gap-fix-plan Step 2.1.

Step 2 (phase_o5.py) shipped the framework with a stub runner. This module
provides the real runner that:
1. Reads .ascendc_env for A5 credentials
2. SSH+docker-exec to invoke workspace/<op>/run_pass_b.py + run_det_check.py
   on the A5 NPU container
3. Captures stdout (verifier scripts print JSON summaries)
4. Parses JSON → MeasuredResult

Architecture per user direction (2026-05-06): pure Python, no LLM. The
verifier scripts themselves are op-specific and produce structured output;
the runner just executes them and parses.

Subprocess layer is testable via monkeypatch — unit tests inject fake
subprocess.run, integration smoke can do the same.

P0abh (2026-05-08): canonical Pass A via precision_eval_two_tier.py.
The vendor 昇腾算子精度标准 2.1 §4.5.3 small-value rule is implemented
in src/scripts/precision_eval_two_tier.py:classify_output(). Cold-start
evidence (1_GELU 2026-05-07) showed that putting §4.5.3 in the kw_brief
as instruction is a compliance gap — agents read the brief but don't
invoke the script under load. Structural fix: runner pushes the canonical
script to the container, runs it as the AUTHORITATIVE Pass A measurement,
and the worker's pass_a self-claim is overridden if §4.5.3 promotes
cases. The phase_o5 MISMATCH gate then routes back to await_worker with
the runner-measured count, so the next kw spawn updates verification.json
with the §4.5.3-aware numbers.

Safeguards against moral hazard (a buggy precision_eval_two_tier.py
silently promoting real failures):
1. The §4.5.3 rule itself has 11/11 unit tests (test_p0abh_*) — the
   threshold table matches vendor wiki, the gate enforces
   ours_count ≤ 2 × max(cann_count, 1) AND ≥10% of cases in the
   small-value regime. A real precision gap fails the gate.
2. The JSON output preserves n_pass_t1, n_pass_t1_smallval,
   n_pass_t1_inclusive separately so audit can see how often the rule
   fired and on which cases.
3. T3-axis ops (no CPU truth) fall back to the worker's pass_a_runner.py
   because precision_eval_two_tier.py needs CPU truth. The canonical
   script is for T1/T2 ops only.
"""
from __future__ import annotations
import logging

import json
import os
import subprocess
from pathlib import Path
from typing import Optional

from phase_o5 import MeasuredResult
from lane_health import resolve_healthy_device


_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent.parent


def _is_port_a3_mode(workspace: Path) -> bool:
    """True for an arch22 to arch35 migration (build target = A5 host).

    task#24-item2: the build/verify sites below resolve their host/CANN from
    `{target}_*` (current-target prefix). For port_a3 on a TARGET=a3 agent that
    would route the A5 build to the A3 host with A3 CANN. We mode-gate the
    A5-explicit precedence on port_a3 ONLY so that backward / benchmark builds
    (which DO build on `{target}_HOST`) are untouched even when A5_HOST is set
    in the env (e.g. the back-agent's npu-a3-back, TARGET=a3).
    """
    from plugins import detect_plugin as _detect_plugin
    p = _detect_plugin(workspace)
    return p is not None and getattr(p, "name", "") == "port_a3_to_a5"


def _port_a3_claims_pass_a(workspace: Path) -> bool:
    """True iff this is a port_a3 op whose verification.json CLAIMS a real
    (non-N/A) precision.pass_a. Used by gap-(b) to distinguish "worker claims a
    pass_a but emitted no O5-re-measurable verifier" (a worker gap → fail loud,
    actionable) from "pass_a legitimately N/A" (→ leave measured.pass_a None,
    nothing to cross-check)."""
    if not _is_port_a3_mode(workspace):
        return False
    try:
        vj = json.loads((workspace / "verification.json").read_text())
    except Exception:
        return False
    pa = (vj.get("precision") or {}).get("pass_a")
    if not isinstance(pa, dict):
        return False
    status = str(pa.get("status", "")).upper()
    if status in ("N/A", "NA", "SKIPPED", ""):
        return False
    # A genuine claim carries a measurable count.
    return pa.get("tier1_pass") is not None or pa.get("total") is not None


def _a5_build_host(env: dict, workspace: Path, target: str) -> str:
    """A5 build/verify host. In port_a3 mode prefer the explicit A5_HOST (split
    A3-ref-host / A5-build-host); otherwise the legacy `{target}_HOST or A5_HOST`
    (no-op for TARGET=a5 agents where {target}==A5)."""
    if _is_port_a3_mode(workspace) and env.get("A5_HOST"):
        return env["A5_HOST"]
    return env.get(f"{target}_HOST") or env.get("A5_HOST", "")


def _a5_build_container(env: dict, workspace: Path, target: str) -> str:
    if _is_port_a3_mode(workspace) and env.get("A5_CONTAINER"):
        return env["A5_CONTAINER"]
    return env.get(f"{target}_CONTAINER") or env.get("A5_CONTAINER", "npu_dev3")


def _a5_build_cann_path(env: dict, workspace: Path, target: str) -> str:
    if _is_port_a3_mode(workspace) and env.get("A5_CANN_PATH"):
        return env["A5_CANN_PATH"]
    return env.get(f"{target}_CANN_PATH") or env.get("CANN_PATH", "/usr/local/Ascend/cann")


def _normalise_target(env: dict) -> tuple[str, str]:
    """Return the configured target and its environment-variable prefix."""
    target = (env.get("TARGET") or "a5").lower()
    if target.endswith("-ds"):
        target = target[:-3]
    return target, target.upper()


def _plugin_pass_b_skip(workspace: Path) -> Optional[dict]:
    """Return the canonical pass-B skip result when the active plugin permits it."""
    from plugins import detect_plugin as _detect_plugin

    active_plugin = _detect_plugin(workspace)
    if active_plugin is None or active_plugin.pass_b_required():
        return None
    vj_path = workspace / "verification.json"
    if not vj_path.is_file():
        return None
    try:
        summary = json.loads(vj_path.read_text())
        status = (summary.get("precision") or {}).get("pass_b", {}).get("status")
    except Exception as error:
        logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)
        return None
    if status != "N/A":
        return None
    return {
        "status": "SKIPPED",
        "reason": (
            f"plugin '{active_plugin.name}' pass_b_required=False; "
            "verification.json.precision.pass_b.status=N/A — pass_b script not invoked"
        ),
    }


def _pass_b_script_or_error(
    workspace: Path, op: str, plugin_skip: Optional[dict]
) -> tuple[Optional[Path], Optional[MeasuredResult]]:
    """Resolve the Pass-B script and retain the original fail-closed error."""
    if plugin_skip is not None:
        return None, None
    pass_b_script = _find_verifier(
        workspace, ["pass_b_runner.py", "run_pass_b.py", "run_edge_pass_b.py"]
    )
    if not pass_b_script:
        return None, MeasuredResult(runner_error=(
            f"workspace/{op}: no Pass B verifier found "
            f"(tried pass_b_runner.py, run_pass_b.py, run_edge_pass_b.py) "
            f"— worker did not produce verifier"
        ))
    cycle_error = _verify_runner_independence(workspace, pass_b_script)
    return pass_b_script, MeasuredResult(runner_error=cycle_error) if cycle_error else None


def _fallback_pass_a(
    workspace: Path, op: str, env: dict, pass_a: object, lane: int
) -> tuple[object, Optional[MeasuredResult]]:
    """Use the worker verifier when canonical Pass A is unavailable."""
    if not isinstance(pass_a, str) and pass_a is not None:
        return pass_a, None
    pass_a_script = _find_verifier(
        workspace, ["pass_a_runner.py", "edge_verify.py", "verify_edge.py"]
    )
    if pass_a_script:
        cycle_error = _verify_runner_independence(workspace, pass_a_script)
        if cycle_error:
            return cycle_error, None
        return _run_verifier(workspace, op, env, pass_a_script, "pass_a", lane=lane), None
    if not _port_a3_claims_pass_a(workspace):
        return None, None
    found = sorted(path.name for path in workspace.glob("*runner*")) or ["none"]
    return None, MeasuredResult(runner_error=(
        f"port_a3 pass_a re-measure: op '{op}' claims precision.pass_a in "
        f"verification.json but no canonical Python pass_a verifier was found "
        f"(searched pass_a_runner.py / edge_verify.py / verify_edge.py). Found "
        f"instead: {found}. A C++ <op>_runner.cpp is NOT an O5-re-measurable "
        f"verifier (P140: the A5 verify path must be a Python pass_a_runner.py "
        f"that runs ModelNew via pybind+ACLRT_LAUNCH_KERNEL and classifies via "
        f"precision_eval_port_a3_two_tier). Worker must emit "
        f"workspace/{op}/pass_a_runner.py importing precision_eval_port_a3_two_tier; "
        f"a C++ runner may remain for the A3 perf-baseline side only."
    ))


def _measure_ssh_verifiers(
    workspace: Path,
    op: str,
    env: dict,
    pass_b_script: Optional[Path],
    plugin_skip: Optional[dict],
    lane: int,
) -> MeasuredResult:
    """Run the remote measurements after the workspace has been synchronized."""
    pass_b = plugin_skip if pass_b_script is None else _run_verifier(
        workspace, op, env, pass_b_script, "pass_b", lane=lane
    )
    pass_a, fallback_error = _fallback_pass_a(
        workspace, op, env, _run_canonical_pass_a(workspace, op, env, lane=lane), lane
    )
    if fallback_error:
        return fallback_error
    gate_error = _gate_port_a3_two_tier(workspace, pass_a)
    if gate_error:
        return MeasuredResult(runner_error=gate_error)
    det_script = _find_verifier(
        workspace, ["run_det_check.py", "det_check.py", "determinism_check_inplace.py"]
    )
    determinism = _run_verifier(
        workspace, op, env, det_script, "determinism", lane=lane
    ) if det_script else None
    if isinstance(pass_b, str):
        return MeasuredResult(runner_error=pass_b)
    perf = _maybe_port_a3_perf_remeasure(workspace, op, env, lane=lane)
    return MeasuredResult(
        pass_a=pass_a if isinstance(pass_a, dict) else None,
        pass_b=pass_b if isinstance(pass_b, dict) else None,
        determinism=determinism if isinstance(determinism, dict) else None,
        perf=perf if isinstance(perf, dict) else None,
    )


def ssh_runner(workspace: Path, op: str, lane: int = 0) -> MeasuredResult:
    """Real runner for phase_o5.post_verify_for_finalize.

    Reads workspace/.ascendc_env (or project/workspace/.ascendc_env fallback),
    executes the verifier scripts, and returns a fail-closed MeasuredResult.
    """
    env = _read_ascendc_env(workspace)
    if not env:
        return MeasuredResult(runner_error="missing .ascendc_env")
    target, target_upper = _normalise_target(env)
    host_key = f"{target_upper}_HOST"
    if not env.get(host_key) and not env.get("A5_HOST"):
        return MeasuredResult(
            runner_error=f"missing {host_key} (and no A5_HOST fallback) in .ascendc_env"
        )
    plugin_skip = _plugin_pass_b_skip(workspace)
    pass_b_script, script_error = _pass_b_script_or_error(workspace, op, plugin_skip)
    if script_error:
        return script_error
    container = (env.get(f"{target_upper}_CONTAINER") or env.get("A5_CONTAINER", "")).strip().lower()
    if container == "local":
        return _run_verifier_local(workspace, op, env, lane=lane)
    sync_error = _resync_workspace_to_container(workspace, env, lane=lane)
    if sync_error:
        return MeasuredResult(runner_error=f"pre-O5 workspace sync failed: {sync_error}")
    return _measure_ssh_verifiers(workspace, op, env, pass_b_script, plugin_skip, lane)


def _maybe_port_a3_perf_remeasure(
    workspace: Path, op: str, env: dict, lane: int = 0
) -> Optional[dict]:
    """DEBT-192 (2026-07-04): orchestrator-side INDEPENDENT perf
    re-measure for port_a3 ops (worker-perf-report contract).

    Why: port_a3 mode splits verification into precision (pass_a/pass_b_runner.py)
    and perf (perf_runner.py — a standalone script that times the A5 ModelNew vs
    the captured A3 baseline in a3_baseline_perf.json and emits canonical JSON).
    ssh_runner re-measures precision but never perf, so measured.perf was always
    None and phase_o5's irm-writer no-op'd. The worker (kw) CORRECTLY refuses to
    self-populate independent_re_measure (author!=measurer) and instead ships
    {ran:false, status:PENDING_ORCHESTRATOR}; without this the orchestrator never
    fulfilled that promise → the finalize irm gate (_check_post_worker_audit)
    rejected the archive forever (feeding the finalize death-loop that main's
    DEBT-192 engine-half fail-fasts). This closes the POSITIVE path.

    Returns:
      - dict (perf_runner.py's canonical JSON) when the re-measure ran+parsed
        → phase_o5 writes performance.independent_re_measure {ran:true, ratio}.
      - {"status":"N/A", "reason":..., "_remeasure_na":True} when perf is
        structurally unmeasurable (no perf_runner.py / cpu-truth op whose
        perf_runner has no a3_baseline_perf.json to time against / runner
        errored) → phase_o5 writes irm {ran:false, status:N/A, reason}. This is
        the fail-loud N/A the contract demands — never a bare self-report PASS.
      - None when this is NOT a port_a3 op (leave perf to the mode's own path;
        benchmark uses phase_o5_perf_capture, backward uses backward_verify_runner).

    Note: this only supplies performance.independent_re_measure. The finalize
    perf gate LOGIC (accept/reject) is unchanged and owned elsewhere — this fix
    only feeds the field that gate already reads.
    """
    # Scope strictly to port_a3. Reuse the same signal ssh_runner already uses
    # to recognise a port_a3 op that claims pass_a (avoids a plugin-detect import
    # cycle and matches the existing gap-(b) detection).
    if not _port_a3_claims_pass_a(workspace):
        return None

    perf_script = _find_verifier(workspace, ["perf_runner.py", "run_perf.py"])
    if not perf_script:
        return {
            "status": "N/A",
            "reason": (
                "port_a3 perf independent re-measure: no perf_runner.py found in "
                "workspace (searched perf_runner.py / run_perf.py). The A3-vs-A5 "
                "perf ratio could not be independently measured by the "
                "orchestrator — DEBT-192 worker-perf-report contract: unmeasured "
                "perf is N/A-with-reason, never a self-reported PASS."
            ),
            "_remeasure_na": True,
        }
    # P94 anti-cycle: perf_runner.py must not read verification.json (a
    # self-citing perf verifier would fabricate independence).
    _cycle_err = _verify_runner_independence(workspace, perf_script)
    if _cycle_err:
        return {"status": "N/A", "reason": _cycle_err, "_remeasure_na": True}

    # raw=True: perf_runner.py emits {ratio, ratio_min, ...}; the default
    # normalization folds output into {tier1_pass, total, status} and would
    # STRIP the ratio -> a measurable op would false-N/A (caught in FA device
    # e2e 2026-07-04: perf_runner emitted ratio 1.2118 but normalization
    # returned {'status':'PASS'} with no ratio).
    parsed = _run_verifier(workspace, op, env, perf_script, "perf", lane=lane, raw=True)
    if isinstance(parsed, dict) and ("ratio" in parsed or "median_ratio" in parsed):
        return parsed
    return {
        "status": "N/A",
        "reason": (
            f"port_a3 perf independent re-measure ran {perf_script} but produced "
            f"no parseable ratio JSON: {parsed!r}. Unmeasured → N/A (DEBT-192)."
        ),
        "_remeasure_na": True,
    }


def _backward_summary_error(summary: dict) -> Optional[str]:
    """Reject empty or non-passing backward NPU measurements."""
    passed = summary.get("tier1_pass")
    total = summary.get("total")
    if summary.get("status") != "PASS":
        return f"backward verifier summary status is {summary.get('status')!r}, not PASS"
    valid_passed = type(passed) is int and passed >= 0
    valid_total = type(total) is int and total > 0
    if not valid_passed or not valid_total or passed > total:
        return (
            "backward verifier summary requires integer 0 <= tier1_pass <= "
            "total with total > 0"
        )
    return None


def _backward_result_from_summary(summary: dict) -> MeasuredResult:
    """Convert a validated backward verifier summary to a measurement."""
    summary_error = _backward_summary_error(summary)
    if summary_error:
        return MeasuredResult(runner_error=summary_error)
    perf = summary.get("performance")
    return MeasuredResult(
        pass_a=_normalize_verifier_output(summary, "pass_a"),
        pass_b=None,
        perf=perf if isinstance(perf, dict) else None,
    )


def _run_backward_local_verify(workspace: Path, verify_script: Path, lane: int) -> MeasuredResult:
    """Execute the self-contained backward verifier on a local target."""
    try:
        local_env = os.environ.copy()
        local_env["ASCEND_RT_VISIBLE_DEVICES"] = str(lane)
        result = subprocess.run(
            ["python3", str(workspace / verify_script)], capture_output=True,
            text=True, timeout=600, cwd=str(workspace), env=local_env,
        )
        parsed = _try_parse_json_tail(result.stdout)
    except subprocess.TimeoutExpired:
        return MeasuredResult(runner_error="backward re-measure (local): verify timed out after 600s")
    except Exception as error:
        return MeasuredResult(runner_error=f"backward re-measure (local): verify exception: {error!r}")
    if result.returncode != 0:
        return MeasuredResult(runner_error=f"backward re-measure (local): verify exit {result.returncode}")
    if not parsed:
        return MeasuredResult(runner_error=(
            f"backward re-measure (local): verify exit {result.returncode}, "
            f"no summary JSON object on stdout (stdout_tail={result.stdout[-200:]!r})"
        ))
    return _backward_result_from_summary(parsed)


def backward_verify_runner(workspace: Path, op: str, lane: int = 0) -> MeasuredResult:
    """Independently re-measure a backward operator with its autograd verifier."""
    env = _read_ascendc_env(workspace)
    if not env:
        return MeasuredResult(runner_error="backward re-measure: missing .ascendc_env")
    _, target_upper = _normalise_target(env)
    if not env.get(f"{target_upper}_HOST") and not env.get("A5_HOST"):
        return MeasuredResult(
            runner_error=f"backward re-measure: missing {target_upper}_HOST (no A5_HOST fallback)"
        )
    verify_script = _find_verifier(workspace, [f"verify_{op}.py", "verify.py"])
    if not verify_script:
        return MeasuredResult(runner_error=(
            f"backward re-measure: workspace/{op} has no self-contained verify "
            f"script (tried verify_{op}.py, verify.py). The backward kw_brief "
            "Phase D must emit verify_<op>.py."
        ))
    cycle_error = _verify_runner_independence(workspace, verify_script)
    if cycle_error:
        return MeasuredResult(runner_error=cycle_error)
    container = (env.get(f"{target_upper}_CONTAINER") or env.get("A5_CONTAINER", "")).strip().lower()
    if container == "local":
        return _run_backward_local_verify(workspace, verify_script, lane)
    sync_error = _resync_workspace_to_container(workspace, env, lane=lane)
    if sync_error:
        return MeasuredResult(runner_error=f"backward re-measure: pre-O5 workspace sync failed: {sync_error}")
    result = _run_verifier(workspace, op, env, verify_script, "pass_a", lane=lane, raw=True)
    if isinstance(result, str):
        return MeasuredResult(runner_error=result)
    if not isinstance(result, dict):
        return MeasuredResult(runner_error="backward verifier returned no summary object")
    return _backward_result_from_summary(result)


# NOTE (DEBT-201, 2026-07-06): _run_verifier_local + _run_canonical_pass_a_local
# moved to phase_o5_verify.py; re-imported at module bottom (patch surface stable).


def _resync_force_update_scripts() -> set[str]:
    """Files whose local copy must replace an existing container copy."""
    return {
        "pass_a_runner.py", "pass_b_runner.py", "run_pass_b.py", "run_edge_pass_b.py",
        "det_check.py", "run_det_check.py", "edge_verify.py", "verify_edge.py",
        "native_capture.pt", "a3_baseline_perf.json", "a3_capture_manifest.json",
        "run_a3_reference.py", ".opgen_state.json",
    }


def _collect_resync_payload(workspace: Path) -> tuple[list[Path], set[str]]:
    """Collect the workspace files required by the remote O5 verifier."""
    payload_names = (
        "run_pass_b.py", "pass_b_runner.py", "run_edge_pass_b.py",
        "pass_a_runner.py", "edge_verify.py", "verify_edge.py",
        "run_det_check.py", "det_check.py", "edge_dataset.pt", "edge_inputs.pt",
        "a5_capture.pt", "cpu_truth_outputs.pt", "cpu_truth_reference.py",
        "native_capture.pt", "a3_baseline_perf.json", "a3_capture_manifest.json",
        "run_a3_reference.py", ".opgen_state.json", "backward_cpu_truth.pt",
        "model.py", "model_new_ascendc.py", "verification.json", "manifest.json",
    )
    push_files = [workspace / name for name in payload_names if (workspace / name).exists()]
    present_names = {path.name for path in push_files}
    forced_json: set[str] = set()
    for json_path in sorted(workspace.glob("*.json")):
        if json_path.name in present_names or json_path.name == "verification.json":
            continue
        try:
            is_small = json_path.stat().st_size <= 5 * 1024 * 1024
        except OSError as error:
            logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)
            continue
        if is_small:
            push_files.append(json_path)
            forced_json.add(json_path.name)
    return push_files, forced_json


def _resync_runner_text(push_files: list[Path]) -> str:
    """Read local Python verifier sources for the oversized-input guard."""
    runner_text = ""
    for path in push_files:
        if path.suffix != ".py":
            continue
        try:
            runner_text += path.read_text(errors="ignore")
        except OSError as error:
            logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)
    return runner_text


def _resync_extra_scripts() -> list[Path]:
    """Return the canonical grading scripts that must accompany the payload."""
    script_names = (
        "precision_eval_two_tier.py", "precision_eval_port_a3_two_tier.py",
        "precision_tier1.py", "precision_tier2.py",
    )
    scripts_dir = _PROJECT_ROOT / "src" / "scripts"
    return [scripts_dir / name for name in script_names if (scripts_dir / name).exists()]


def _probe_resync_resident_files(
    env: dict, target: str, a5_host: str, a5_user: str, a5_password: str,
    a5_container: str, benchmark_root: str,
) -> set[str]:
    """Best-effort inventory of files already present in current_task."""
    host_mode = str(env.get("A5_HOST_MODE", "")).strip() in ("1", "true", "yes")
    current_task = f"{benchmark_root}/current_task"
    list_command = f"ls -1 {current_task} 2>/dev/null"
    if not host_mode:
        list_command = f"docker exec {a5_container} {list_command}"
        list_command = _maybe_sudo_wrap_remote(list_command, env, target)
    ssh_opts = [
        *_resolve_ssh_key_opts(env, target), "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=60",
        "-o", "LogLevel=ERROR",
    ]
    command = (["sshpass", "-p", a5_password, "ssh", *ssh_opts,
                f"{a5_user}@{a5_host}", list_command] if a5_password else
               ["ssh", *ssh_opts, f"{a5_user}@{a5_host}", list_command])
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except Exception as error:
        logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)
        return set()
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _filter_resync_payload(
    push_files: list[Path], resident: set[str], force_update_scripts: set[str], forced_json: set[str]
) -> list[Path]:
    """Keep missing files and explicitly refreshed verifier inputs."""
    if not resident:
        return push_files
    return [
        path for path in push_files
        if path.name not in resident or path.name in force_update_scripts or path.name in forced_json
    ]


def _oversized_resync_error(push_files: list[Path], runner_text: str) -> Optional[str]:
    """Remove unused oversized tensors or return the existing fail-closed error."""
    OVERSIZED_PT_THRESHOLD = 100 * 1024 * 1024  # 100 MiB
    OVERSIZED_EXEMPT = {"backward_cpu_truth.pt"}
    oversized: list[tuple[str, int]] = []
    for p in list(push_files):
        if p.suffix == ".pt" and p.name not in OVERSIZED_EXEMPT:
            try:
                size = p.stat().st_size
            except OSError as error:
                logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)
                continue
            if size > OVERSIZED_PT_THRESHOLD:
                if p.name not in runner_text:
                    push_files.remove(p)
                    continue
                oversized.append((p.name, size))
    if not oversized:
        return None
    details = ", ".join(f"{name} ({size / (1024 * 1024):.1f} MiB)" for name, size in oversized)
    return (f"phase_o5 SCP aborted: oversized payload .pt files exceed "
            f"100 MiB threshold — {details}. Regenerate edge_inputs.pt / "
            f"edge_dataset.pt with smaller LB stress cases (input_gen.py "
            f"size cap) before re-running. See task #23 (2026-05-18).")


def _stage_resync_tar(push_files: list[Path], extra_scripts: list[Path]) -> tuple[str, list[str], list[str]]:
    """Build the disk-backed archive and record fresh grader instruments."""
    import tarfile
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as temporary_file:
        host_tar = temporary_file.name
    instrument_files: list[str] = []
    instrument_dirs: list[str] = []
    with tarfile.open(host_tar, mode="w") as tar:
        for path in push_files:
            tar.add(str(path), arcname=path.name)
        for path in extra_scripts:
            tar.add(str(path), arcname=path.name)
            instrument_files.append(path.name)

        def no_pycache(member):
            return None if "__pycache__" in member.name or member.name.endswith(".pyc") else member

        for source_path, archive_path in (
            (_PROJECT_ROOT / "src" / "scripts" / "orchestrator" / "precision" / "cannbench_grader",
             "orchestrator/precision/cannbench_grader"),
            (_PROJECT_ROOT / "src" / "scripts" / "reference_provider" / "verify.py",
             "reference_provider/verify.py"),
        ):
            if source_path.exists():
                tar.add(str(source_path), arcname=archive_path, filter=no_pycache)
                (instrument_dirs if source_path.is_dir() else instrument_files).append(archive_path)
    return host_tar, instrument_files, instrument_dirs


def _resync_timeout() -> int:
    """Read the existing configurable SCP/SSH timeout with its safe default."""
    try:
        return int(os.environ.get("AOG_O5_SYNC_TIMEOUT", "300"))
    except (TypeError, ValueError):
        return 300


def _publish_resync_tar(
    host_tar: str, workspace: Path, env: dict, target: str, a5_host: str,
    a5_user: str, a5_password: str, a5_container: str, benchmark_root: str,
    push_files: list[Path], force_update_scripts: set[str], forced_json: set[str],
    instrument_files: list[str], instrument_dirs: list[str],
) -> Optional[str]:
    """Copy the archive and perform the additive, targeted remote extraction."""
    ssh_opts = [
        *_resolve_ssh_key_opts(env, target), "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=60",
        "-o", "LogLevel=ERROR",
    ]
    remote_tar = f"/tmp/o5_sync_{workspace.name}_{os.getpid()}.tar"
    timeout = _resync_timeout()
    scp_command = (["sshpass", "-p", a5_password, "scp", *ssh_opts,
                    host_tar, f"{a5_user}@{a5_host}:{remote_tar}"] if a5_password else
                   ["scp", *ssh_opts, host_tar, f"{a5_user}@{a5_host}:{remote_tar}"])
    result = subprocess.run(scp_command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        return f"scp failed: {result.stderr[:200]}"
    force_names = [
        path.name for path in push_files
        if path.name in force_update_scripts or path.name in forced_json
    ]
    remove_force = ("rm -f " + " ".join(force_names) + " && ") if force_names else ""
    remove_instruments = [
        f"rm -f {benchmark_root}/current_task/{path}" for path in instrument_files
    ] + [
        f"rm -rf {benchmark_root}/current_task/{path}" for path in instrument_dirs
    ]
    remove_instrument = (" && ".join(remove_instruments) + " && ") if remove_instruments else ""
    host_mode = str(env.get("A5_HOST_MODE", "")).strip() in ("1", "true", "yes")
    if host_mode:
        remote_command = (
            f"mkdir -p {benchmark_root}/current_task && cd {benchmark_root}/current_task && "
            f"{remove_force}{remove_instrument}tar --skip-old-files -xf {remote_tar} && rm -f {remote_tar}"
        )
    else:
        remote_command = (
            f"docker cp {remote_tar} {a5_container}:{remote_tar} && "
            f"docker exec {a5_container} bash -c 'mkdir -p {benchmark_root}/current_task && "
            f"cd {benchmark_root}/current_task && {remove_force}{remove_instrument}"
            f"tar --skip-old-files -xf {remote_tar} && rm {remote_tar}' && rm {remote_tar}"
        )
        remote_command = _maybe_sudo_wrap_remote(remote_command, env, target)
    ssh_command = (["sshpass", "-p", a5_password, "ssh", *ssh_opts,
                    f"{a5_user}@{a5_host}", remote_command] if a5_password else
                   ["ssh", *ssh_opts, f"{a5_user}@{a5_host}", remote_command])
    result = subprocess.run(ssh_command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        return f"docker cp / untar failed: {result.stderr[:200]}"
    return None


def _resync_workspace_to_container(workspace: Path, env: dict, lane: int = 0) -> Optional[str]:
    """Synchronize only O5 inputs to ``current_task`` without clobbering artifacts.

    The additive ``tar --skip-old-files`` deployment preserves the container's built
    artifacts while refreshing pass_a_runner.py, run_pass_b.py, edge_dataset.pt, and
    model_new_ascendc.py when their contract requires it.
    """
    _, target = _normalise_target(env)
    a5_host = _a5_build_host(env, workspace, target)
    if not a5_host:
        return f"missing {target}_HOST in env"
    a5_user = env.get(f"{target}_USER") or env.get("A5_USER", "root")
    a5_password = env.get(f"{target}_PASSWORD") or env.get("A5_PASSWORD", "")
    a5_container = _a5_build_container(env, workspace, target)
    benchmark_root = _lane_aware_benchmark_root(env, lane)
    force_update_scripts = _resync_force_update_scripts()
    push_files, forced_json = _collect_resync_payload(workspace)
    if not push_files:
        return "no verifier scripts or data files in workspace"
    runner_text = _resync_runner_text(push_files)
    resident = _probe_resync_resident_files(
        env, target, a5_host, a5_user, a5_password, a5_container, benchmark_root
    )
    push_files = _filter_resync_payload(push_files, resident, force_update_scripts, forced_json)
    oversized_error = _oversized_resync_error(push_files, runner_text)
    if oversized_error:
        return oversized_error
    host_tar, instrument_files, instrument_dirs = _stage_resync_tar(push_files, _resync_extra_scripts())
    try:
        return _publish_resync_tar(
            host_tar, workspace, env, target, a5_host, a5_user, a5_password,
            a5_container, benchmark_root, push_files, force_update_scripts, forced_json,
            instrument_files, instrument_dirs,
        )
    finally:
        try:
            os.unlink(host_tar)
        except OSError as error:
            logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)


# NOTE (DEBT-201, 2026-07-06): _gate_port_a3_two_tier + _verify_runner_independence
# moved to phase_o5_verify.py; re-imported at module bottom (patch surface stable).


def _read_ascendc_env(workspace: Optional[Path] = None) -> dict:
    """Parse .ascendc_env. Tries explicit env-file overrides first, then
    workspace-relative, CWD-relative, and project-root fallback.

    P0aba.O5 followup (2026-05-07, nms-kw-7): added workspace parameter so
    multi-project setups (a3-ds vs a5_ops) find the correct .ascendc_env.
    Previous 5 finalize failures all traced to this function reading
    a5_ops/workspace/.ascendc_env (no A3_HOST) instead of
    a3-ds/a3_ops/workspace/.ascendc_env (has A3_HOST).
    """
    candidates = []
    for key in ("ASCENDC_ENV_PATH", "ASCENDC_ENV_FILE"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value).expanduser())
    if workspace is not None:
        # workspace = .../a3-ds/a3_ops/workspace/30_NMS
        # workspace.parent = .../a3-ds/a3_ops/workspace/  → has .ascendc_env
        candidates.append(workspace.parent / ".ascendc_env")
    candidates.extend([
        Path.cwd() / "workspace" / ".ascendc_env",
        _PROJECT_ROOT / "workspace" / ".ascendc_env",
    ])
    seen = set()
    for path in candidates:
        path = path.resolve() if path.exists() else path
        if path in seen:
            continue
        seen.add(path)
        if not path.exists():
            continue
        out = {}
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip("'").strip('"')
        if out:
            return out
    return {}


# NOTE (DEBT-201, 2026-07-06): _run_verifier + _run_canonical_pass_a moved to
# phase_o5_verify.py (verifier-execution cluster). Re-imported at the bottom of
# this module so `phase_o5_runner._run_verifier` / `_run_canonical_pass_a` stay
# valid + monkeypatch.setattr(phase_o5_runner, ...) keeps biting parent-resident
# callers (ssh_runner / backward_verify_runner / _maybe_port_a3_perf_remeasure).


# ─────────────────────────────────────────────────────────────────────────────
# DEBT-201 decomposition (2026-07-06): pure leaf helpers were relocated verbatim
# to phase_o5_helpers.py to keep this module a focused FSM driver under the
# god-file line budget. Re-import them here so:
#   1. The remaining functions in THIS module call them by bare name unchanged
#      (the names are bound in this module's namespace by the star-import).
#   2. The public attribute surface `phase_o5_runner.<name>` is preserved for
#      external callers (orchestrator / plugins) and for tests that
#      access them via `phase_o5_runner.<name>` / `por.<name>` / `o5.<name>`.
# NONE of these relocated helpers are monkeypatched by any test, and none call a
# monkeypatched phase_o5_runner function, so relocating them is behavior-neutral.
from phase_o5_helpers import (  # noqa: E402,F401
    _find_verifier,
    _lane_aware_benchmark_root,
    _normalize_canonical_pass_a,
    _normalize_port_a3_two_tier_pass_a,
    _normalize_verifier_output,
    _resolve_extra_ld,
    _resolve_npu_python_bin,
    _resolve_ssh_key_opts,
    _shell_quote,
    _docker_sudo_enabled,
    _maybe_sudo_wrap_remote,
    _try_fetch_remote_result_json,
    _try_parse_json_tail,
)

# ─────────────────────────────────────────────────────────────────────────────
# DEBT-201 batch5 (2026-07-06): the verifier-execution cluster moved to
# phase_o5_verify.py to push this module under the 1000-line budget:
#   _run_verifier_local, _run_canonical_pass_a_local, _gate_port_a3_two_tier,
#   _verify_runner_independence, _run_verifier, _run_canonical_pass_a.
# Re-imported here (bottom shim) so:
#   1. This module's own PARENT-resident callers (ssh_runner /
#      backward_verify_runner / _maybe_port_a3_perf_remeasure) call these by BARE
#      name → they resolve to THIS module's re-imported attribute, so existing
#      `monkeypatch.setattr(phase_o5_runner, "_run_verifier", ...)` etc. STILL BITE.
#   2. `phase_o5_runner.<name>` / `por.<name>` / `o5.<name>` attribute access +
#      `from phase_o5_runner import <name>` stay valid for external callers/tests.
# The moved functions call back into this module's still-patchable port-mode,
# verifier-independence, and canonical-Pass-A helpers through qualified
# `phase_o5_runner` attributes, so those patches remain effective.
# This import MUST be last: phase_o5_verify top-imports the non-patched build
# helpers + _PROJECT_ROOT from THIS module (defined above) — placing it here
# keeps the graph acyclic (verify's qualified patched calls resolve at call time).
from phase_o5_verify import (  # noqa: E402,F401  re-export: keep patch surface + import paths stable
    _resolve_visible_device,
    _run_verifier_local, _run_canonical_pass_a_local, _gate_port_a3_two_tier,
    _verify_runner_independence, _run_verifier, _run_canonical_pass_a,
)
