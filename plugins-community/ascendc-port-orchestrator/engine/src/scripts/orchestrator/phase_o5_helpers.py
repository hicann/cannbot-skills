# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Phase O5 runner — pure leaf helpers (DEBT-201 decomposition, 2026-07-06).

Extracted verbatim from phase_o5_runner.py to keep that module a focused FSM
driver under the god-file line budget. This module holds ONLY the pure,
side-effect-light helpers that are:
  - NOT monkeypatched by any test (verified: none of these appear in the
    tests' patched-attr set — see phase_o5_runner.py bottom-shim re-import),
  - do NOT call any monkeypatched phase_o5_runner function (so relocating them
    to a separate module cannot change which function a `monkeypatch.setattr(
    phase_o5_runner, ...)` patch is seen by),
  - carry no phase_o5_runner-specific closure state.

The clusters here:
  * SSH/build resolve helpers  — _resolve_npu_python_bin / _resolve_extra_ld /
    _resolve_ssh_key_opts / _lane_aware_benchmark_root
  * verifier discovery         — _find_verifier
  * result normalizers         — _normalize_port_a3_two_tier_pass_a /
    _normalize_canonical_pass_a / _normalize_verifier_output
  * JSON/remote parse helpers  — _try_fetch_remote_result_json /
    _try_parse_json_tail / _shell_quote

phase_o5_runner re-imports every name here at import time (bottom shim), so the
public attribute surface `phase_o5_runner.<name>` is unchanged for callers and
tests. Behavior is byte-identical to the pre-split definitions.
"""
from __future__ import annotations
import logging

import json
import os
import subprocess
from pathlib import Path
from typing import Optional


def _resolve_npu_python_bin(env: dict, target: str) -> str:
    """Resolve the NPU-python bin dir, preferring the target-specific override.

    Checks `{target}_NPU_PYTHON_BIN` (e.g. A3_NPU_PYTHON_BIN / A5_NPU_PYTHON_BIN)
    first, then the generic NPU_PYTHON_BIN. Without the target-specific check an
    A3 op (TARGET=a3) used the A5 py311 path from the generic var -> "No such
    file or directory" exit 127 in phase_o5 verify (#448). Mirrors the
    {TARGET}_NPU_PYTHON_BIN resolution used by the scoped verify path.
    `target` is uppercase here (e.g. "A3" / "A5")."""
    return (str(env.get(f"{target}_NPU_PYTHON_BIN") or env.get("NPU_PYTHON_BIN") or "")).rstrip("/")


def _resolve_extra_ld(env: dict, target: str) -> str:
    """Resolve an extra LD_LIBRARY_PATH prefix, preferring the target-specific override.

    Checks `{target}_EXTRA_LD_LIBRARY_PATH` (e.g. A5_EXTRA_LD_LIBRARY_PATH) first, then the
    generic EXTRA_LD_LIBRARY_PATH. Empty (the default) when neither is set -> zero behavior
    change for single-CANN targets.

    Why this exists (2026-06-19, .171 npu_dev3 multi-CANN box): the run-env LD only adds the
    NPU-python's torch/torch_npu .so dirs + `source {cann_path}/set_env.sh`. On a box where the
    *runtime* acl/hccl libs live in a DIFFERENT CANN than the compile toolkit (here libacl_dvpp.so
    / libhccl.so live only in 8.5.0/x86_64-linux/lib64 while arch35 is compiled with the 9.1.T500
    bisheng), neither source covers them and the verifier python aborts with
    `libacl_dvpp.so: cannot open shared object file`. Setting {TARGET}_EXTRA_LD_LIBRARY_PATH lets
    such a box inject the runtime-lib dir without hand-mixing CANN versions. `target` is uppercase
    here (e.g. "A3" / "A5")."""
    return (str(env.get(f"{target}_EXTRA_LD_LIBRARY_PATH") or env.get("EXTRA_LD_LIBRARY_PATH") or "")).rstrip(":")


def _resolve_extra_pythonpath(env: dict, target: str) -> str:
    """Resolve an extra PYTHONPATH prefix, preferring the target-specific override.

    Symmetric with `_resolve_extra_ld`: checks `{target}_EXTRA_PYTHONPATH` (e.g.
    A3_EXTRA_PYTHONPATH) first, then the generic EXTRA_PYTHONPATH. Empty (the default)
    when neither is set -> zero behavior change (the emitted PYTHONPATH export is
    byte-identical to today, no stray colon).

    Why this exists (2026-07-20): the O5 SSH+docker-exec verifier builds PYTHONPATH from
    the CANN site-packages + inherited `${PYTHONPATH:-}` only (non-login shell). A verifier
    that needs an explicitly provisioned OUT-OF-TREE support package
    had no env hook to inject it — the LD side already has `{TARGET}_EXTRA_LD_LIBRARY_PATH`
    but there was no PYTHONPATH analog. This adds the symmetric passthrough. `target` is
    uppercase here (e.g. "A3" / "A5")."""
    return (str(env.get(f"{target}_EXTRA_PYTHONPATH") or env.get("EXTRA_PYTHONPATH") or "")).rstrip(":")


def _resolve_ssh_key_opts(env: dict, target: str) -> list:
    """SSH identity opts (`-i <key>`) for key-auth lanes, target-specific then generic.

    2026-06-20 (FA-grad .141 V351 lane): the .141/.171 V351 hosts use KEY auth
    (id_ca_team), not a password. The verify-runner SSH sites historically only
    handled sshpass (password) or a bare-ssh fallback that relies on a default key /
    ~/.ssh/config — neither available in a sandbox where ~/.ssh/config is read-only.
    Honor `{target}_SSH_KEY` (then generic `SSH_KEY`) → `["-i", <key>]` prepended into
    each site's ssh_opts so the key flows into BOTH the sshpass and key-auth branches.
    Empty (default) → [] → zero behavior change for password/default-key lanes.
    `target` is uppercase here (e.g. "A3" / "A5"). Mirrors deploy_to_npu.sh's SSH_KEY."""
    key = (str(env.get(f"{target}_SSH_KEY") or env.get("SSH_KEY") or "")).strip()
    return ["-i", key] if key else []


def _lane_aware_benchmark_root(env: dict, lane: int) -> str:
    """Bug B (2026-05-23): construct lane-aware BENCHMARK_ROOT.

    Symptom: lightning_indexer_grad lane 3 worker built `.so` files at
    `/home/npu_user/workspace/AscendOpGenAgent_lane3/current_task/kernel/build/`
    (per deploy_to_npu_lane.sh's lane 3 export), but phase_o5_runner read
    from `/root/AscendOpGenAgent/current_task/kernel/build/` (the lane 0
    default from .ascendc_env). O5 saw stale .so from prior lane 0 op
    (swi_glu), reported precision FAIL on what was actually LIG.

    Resolution:
      - lane 0: use env's BENCHMARK_ROOT (default /root/AscendOpGenAgent —
        lane 0 hard-codes /root in .ascendc_env to match container-internal
        scratch layout from earlier days).
      - lane N+: use /home/npu_user/workspace/AscendOpGenAgent_lane{N},
        matching exactly what deploy_to_npu_lane.sh exports. The host
        path is bind-mounted into the container at the same path
        (/home/npu_user -> /home/npu_user mount), so the docker exec
        commands resolve correctly.

    This avoids cross-lane current_task/ contention without needing a more
    invasive unique-token current_task subdir partitioning scheme.

    Host-mode (2026-06-10, .211 super-node): the legacy lane>0 path
    `/home/npu_user/workspace/AscendOpGenAgent_lane{N}` is wrong for the
    container-blocked .211 host — `/home` is 99% full + that workspace dir
    does not exist there. In host-mode, the writable artifact root is
    `A5_DEPLOY_STAGE_HOST` (= /data/npu_user) per `.ascendc_env`. Stage
    each lane under `<stage>/AscendOpGenAgent_lane{N}` so lane>0 stays
    isolated while living on the writable `/data` volume. Device binding
    stays decoupled (ASCEND_RT_VISIBLE_DEVICES={lane} is set separately by
    callers), so lane 4 still targets card 4 with its root on /data.
    """
    # DEBT-159 (2026-06-16): lane 0 must ALSO be isolated. The ONLY behavior change
    # vs the prior version is for lane 0 with a SHARED `/root/AscendOpGenAgent`
    # root: it now isolates to `AscendOpGenAgent_lane0` so two lane-0 op-gen runs
    # (different accounts / containers on the same host) don't share
    # `/root/AscendOpGenAgent/current_task`. This mirrors deploy_to_npu.sh's
    # LANE-explicit branch (LANE set → $HOME/workspace/AscendOpGenAgent_lane${LANE};
    # npu_user's $HOME = /home/npu_user).
    #
    # Lane N>0 is UNCHANGED (always the canonical `_lane{N}` path, ignoring env —
    # matches deploy_to_npu_lane.sh's lane>0 export). Instance-isolated configs
    # (kimi/ds/faregen, /data/.../AscendOpGenAgent) are NON-shared roots and are
    # still honored verbatim for lane 0 — only the shared /root default is the one
    # that lane-0 isolation overrides (guardrail: make the step lane-aware, don't
    # mutate the env file that carries the shared default).
    _SHARED_DEFAULTS = ("/root/AscendOpGenAgent", "/root/AscendOpGenAgent/")
    _explicit_root = env.get("BENCHMARK_ROOT")
    _explicit_root_norm = _explicit_root.rstrip("/") if _explicit_root else None
    host_mode = str(env.get("A5_HOST_MODE", "")).strip() in ("1", "true", "yes")
    if host_mode:
        stage = (env.get("A5_DEPLOY_STAGE_HOST")
                 or _explicit_root_norm
                 or "/data/npu_user").rstrip("/")
        if lane == 0:
            # lane 0 host-mode: honor a non-shared configured root; else isolate.
            if _explicit_root_norm and _explicit_root_norm not in _SHARED_DEFAULTS:
                return _explicit_root_norm
            return f"{stage}/AscendOpGenAgent_lane0"
        return f"{stage}/AscendOpGenAgent_lane{lane}"
    # DEBT-185 (config-driven container home; mirrors A3's _a3_container_home,
    # upstreamed from the cannbot bundle): the A5 container's real provisioned
    # home may not be the default literal. Resolve A5_CONTAINER_HOME (env >
    # .ascendc_env) with the default preserved for byte-compat when unset.
    _a5_home = (os.environ.get("A5_CONTAINER_HOME")
                or env.get("A5_CONTAINER_HOME")
                or "/home/npu_user").rstrip("/")
    if lane == 0:
        # lane 0 container-mode: honor a non-shared configured root (instance
        # isolation kimi/ds/faregen, /data layouts); else isolate to lane-0 root.
        if _explicit_root_norm and _explicit_root_norm not in _SHARED_DEFAULTS:
            return _explicit_root_norm
        return f"{_a5_home}/workspace/AscendOpGenAgent_lane0"
    return f"{_a5_home}/workspace/AscendOpGenAgent_lane{lane}"


def _find_verifier(workspace: Path, candidates: list) -> Optional[str]:
    """Return the first candidate filename that exists in workspace, or None.

    P0aaf (2026-05-06): workers produced `pass_b_runner.py` / `edge_verify.py`
    / `det_check.py` while this module hardcoded the more verbose
    `run_pass_b.py` / `run_det_check.py` names. Centralizing the alias lookup
    here keeps the runner backward-compatible with existing canonical
    workspaces while a follow-up task pins the worker brief to canonical
    names going forward.
    """
    for name in candidates:
        if (workspace / name).exists():
            return name
    return None


def _normalize_port_a3_two_tier_pass_a(parsed: dict) -> dict:
    """task#82: convert precision_eval_port_a3_two_tier.summarize() → pass_a-shape.

    The port_a3 native two-tier summary already carries tier1_pass / tier2_pass /
    tier1_pass_inclusive / total / tier2_status / status, so this is a thin
    selection (NOT a re-derive).

    MeasuredResult.tier1_pass is kept STRICT (T1-only), matching the worker's
    verification.json strict `tier1_pass` claim; the INCLUSIVE count is exported
    separately as `tier1_pass_inclusive` so the phase_o5 MISMATCH gate (which
    switches to INCLUSIVE_FIELDS when the worker status is an inclusive status)
    still compares like for like.
    """
    n_total = int(parsed.get("total", 0))
    t1 = int(parsed.get("tier1_pass", 0))
    t2 = int(parsed.get("tier2_pass", 0))
    inclusive = int(parsed.get("tier1_pass_inclusive", t1 + t2))
    status = parsed.get("status")
    if status not in ("PASS", "FAIL"):
        status = "PASS" if (parsed.get("n_fail", 0) == 0
                            and parsed.get("n_err", 0) == 0
                            and n_total > 0) else "FAIL"
    out = {
        "tier1_pass": t1,                   # STRICT T1-only — matches worker claim
        "tier1_pass_inclusive": inclusive,  # T1 + T2 (P0cc inclusive count)
        "tier2_pass": t2,
        "tier2_status": parsed.get("tier2_status"),
        "total": n_total,
        "status": status,
        "n_a3_unavailable": int(parsed.get("n_a3_unavailable", 0)),
        "synth_conflation": bool(parsed.get("synth_conflation", False)),
        "method": "canonical_precision_eval_port_a3_two_tier",
    }
    # ②(b) codex01: KEEP the native-provisioning diagnostics in the normalized verification.json —
    # do NOT drop them. Without these, a native_provision_failed FAIL is indistinguishable from a
    # kernel precision FAIL (and n_native_missing/native_kind are invisible to the pipeline/audit).
    for _k in ("native_capture_present", "native_usable", "native_kind", "native_provision_ok",
               "native_provision_failed", "native_provision_warning", "n_native_used",
               "n_native_missing", "missing_native_case_ids", "native_dependent_case_ids", "reason",
               # controllable-harness #4: carry the chosen precision standard + its provenance into
               # verification.json (THE INVARIANT #1 — provenance-recorded, never silent).
               "precision_standard", "precision_standard_source", "grader"):
        if _k in parsed:
            out[_k] = parsed[_k]
    return out


def _normalize_canonical_pass_a(parsed: dict) -> dict:
    """Convert precision_eval_two_tier.evaluate() summary → pass_a-shape.

    Source schema (subset):
        n_total, n_pass_t1, n_pass_t1_smallval, n_pass_t1_inclusive,
        n_pass_t2, n_pass_t3, n_fail, n_err, op_verdict, tier_axis

    Output schema (MeasuredResult.pass_a):
        tier1_pass: int  (n_pass_t1_inclusive + n_pass_t2 for T1/T2 axis;
                          n_pass_t3 for T3 axis — both already accept the
                          verdict per OL-109)
        total: int  (n_total)
        status: "PASS" | "FAIL"  (PASS iff every case passed at SOME tier)
        smallval_promotions: int  (n_pass_t1_smallval — kept for audit)
        op_verdict: str  (passed through verbatim)
        tier_axis: str  ("T1_T2" | "T3")
    """
    n_total = int(parsed.get("n_total", 0))
    n_pass_t3 = int(parsed.get("n_pass_t3", 0))
    n_pass_t1_inclusive = int(
        parsed.get("n_pass_t1_inclusive",
                   parsed.get("n_pass_t1", 0) + parsed.get("n_pass_t1_smallval", 0))
    )
    n_pass_t2 = int(parsed.get("n_pass_t2", 0))
    n_fail = int(parsed.get("n_fail", 0))
    n_err = int(parsed.get("n_err", 0))
    tier_axis = parsed.get("tier_axis", "T1_T2")

    if tier_axis == "T3":
        tier1_pass = n_pass_t3
    else:
        tier1_pass = n_pass_t1_inclusive + n_pass_t2

    # PASS only if every case passed at some tier AND none errored. We do
    # not collapse FAIL+ERR into PASS — both are real signal.
    status = "PASS" if (n_fail == 0 and n_err == 0 and n_total > 0) else "FAIL"

    return {
        "tier1_pass": tier1_pass,
        "total": n_total,
        "status": status,
        "smallval_promotions": int(parsed.get("n_pass_t1_smallval", 0)),
        "op_verdict": parsed.get("op_verdict"),
        "tier_axis": tier_axis,
        "method": "canonical_precision_eval_two_tier",
    }


def _try_fetch_remote_result_json(
    a5_host: str,
    a5_user: str,
    a5_password: str,
    a5_container: str,
    benchmark_root: str,
    label: str,
    ssh_opts: list,
    *,
    host_mode: bool = False,
    sudo_wrap: bool = False,
    _subprocess_run=None,
) -> Optional[dict]:
    """P0gh-Gap-B (2026-05-28): fetch `<label>_result.json` from the
    remote A5/A3 container via SSH+docker cat. Used as a fallback when
    `_try_parse_json_tail(result.stdout)` returns None — some verifier
    scripts (e.g. LIG `pass_b_runner.py`) write the result JSON to disk
    AT `<benchmark_root>/current_task/<label>_result.json` but emit
    prose-only stdout. Without this fallback, phase_o5 reports
    "verifier stdout had no parseable JSON" and rolls back even though
    a correct JSON result exists on disk.

    Returns parsed dict on success; None on ANY failure path (file
    missing, SSH error, cat failure, JSON parse error). The caller falls
    through to the existing "no parseable JSON" error string if None.

    Strict, not lenient: this fallback only succeeds when the file IS
    valid JSON. Does NOT manufacture passing verdicts (no graceful
    degrade per main 8:54Z direction — `Option C phase_o5 graceful
    degrade 拒, 违反 verification 严格性`).

    `_subprocess_run` kwarg: dependency injection for tests (defaults
    to `subprocess.run`).
    """
    if _subprocess_run is None:
        _subprocess_run = subprocess.run
    result_filename = f"{label}_result.json"
    docker_cat = f"cd {benchmark_root}/current_task && cat {result_filename}"
    # Host-mode (.211): benchmark_root is host-visible — cat host-direct.
    remote_cat = (docker_cat if host_mode
                  else f"docker exec {a5_container} bash -c {_shell_quote(docker_cat)}")
    if sudo_wrap and not host_mode:
        # OPTIONAL sudo-gated docker host (e.g. .32): run the docker cat as root.
        remote_cat = f"sudo su -c {_shell_quote(remote_cat)}"
    if a5_password:
        ssh_cmd = [
            "sshpass", "-p", a5_password,
            "ssh", *ssh_opts,
            f"{a5_user}@{a5_host}",
            remote_cat,
        ]
    else:
        ssh_cmd = [
            "ssh", *ssh_opts,
            f"{a5_user}@{a5_host}",
            remote_cat,
        ]
    try:
        result = _subprocess_run(
            ssh_cmd, capture_output=True, text=True, timeout=60,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    # Parse — strict. Trailing noise NOT tolerated here (this is the
    # on-disk file content; the runner that wrote it should produce
    # clean JSON. If it doesn't, that's a runner-side bug, not something
    # this fallback should paper over).
    try:
        parsed = json.loads(result.stdout)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _try_parse_json_tail(stdout: str) -> Optional[dict]:
    """Find the last JSON object in stdout (verifier scripts may have logs
    before AND noise after the final JSON summary).

    Order of attempts:
      1. Whole stdout (clean case).
      2. Truncate at last `}` so trailing non-JSON noise is ignored
         (BiSheng `[Warning]: tiling struct ...` lines after the JSON;
         empirically seen on 22_Nonzero pass_b verifier 2026-05-18).
      3. Use json.JSONDecoder().raw_decode() to walk forward from each
         `{`, asking the parser to consume only what's valid JSON and
         ignore trailing characters.
      4. Last-resort: backward scan from rfind('{'), original behavior.
    """
    if not stdout:
        return None
    # Try whole stdout first
    try:
        return json.loads(stdout)
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )
    # Truncate at last `}` and retry. This handles trailing-noise lines
    # like `[Warning]: tiling struct [ReduceOpTilingDataV2] is conflict
    # with one in file lp_norm_reduce.cc, line 41` that BiSheng emits
    # AFTER the verifier's JSON summary in some op classes.
    last_close = stdout.rfind("}")
    if last_close >= 0:
        truncated = stdout[: last_close + 1]
        try:
            return json.loads(truncated)
        except Exception:
            # Try forward-scan from each `{` using raw_decode (tolerates
            # trailing chars after the JSON body that whole-string
            # json.loads rejects).
            decoder = json.JSONDecoder()
            for idx, char in enumerate(truncated):
                if char != "{":
                    continue
                try:
                    obj, _end = decoder.raw_decode(truncated, idx)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    return obj
    # Last resort: backward scan from rfind('{'). Original behavior.
    last_brace = stdout.rfind("{")
    while last_brace >= 0:
        try:
            return json.loads(stdout[last_brace:])
        except Exception:
            last_brace = stdout.rfind("{", 0, last_brace)
    return None


def _normalize_verifier_output(parsed: dict, label: str) -> dict:
    """Coerce verifier-output dict to the {tier1_pass, total, status, ...}
    shape phase_o5 expects. Verifier scripts vary a bit in field names."""
    # task#82: port_a3 worker pass_a_runner emits the native two-tier summary
    # (tier2_status present) via precision_eval_port_a3_two_tier.summarize. Route it
    # through the dedicated normalizer so the MeasuredResult carries the inclusive
    # tier1_pass + the tier2 fields, single point of truth.
    if label == "pass_a" and parsed.get("tier2_status") is not None:
        return _normalize_port_a3_two_tier_pass_a(parsed)
    out = {}
    for k_canonical in ("tier1_pass", "total", "status", "n_pass", "n_total"):
        if k_canonical in parsed:
            out[k_canonical] = parsed[k_canonical]
    # Map legacy n_pass/n_total to canonical if canonical absent
    if "tier1_pass" not in out and "n_pass" in out:
        out["tier1_pass"] = out["n_pass"]
    if "total" not in out and "n_total" in out:
        out["total"] = out["n_total"]
    # determinism may use different fields
    if label == "determinism":
        for k in ("policy_satisfied", "n_identical_cases", "n_cases_checked"):
            if k in parsed:
                out[k] = parsed[k]
    return out


def _shell_quote(s: str) -> str:
    """POSIX-safe single-quote wrap. Escapes embedded single quotes."""
    return "'" + s.replace("'", "'\\''") + "'"


def _docker_sudo_enabled(env: dict, target: str = "a5") -> bool:
    """OPTIONAL, opt-in, per-target-resource. True iff `{TARGET}_DOCKER_SUDO`
    (or legacy `A5_DOCKER_SUDO`) in `.ascendc_env` is truthy. Default OFF."""
    flag = str(env.get(f"{target.upper()}_DOCKER_SUDO",
                       env.get("A5_DOCKER_SUDO", ""))).strip().lower()
    return flag in ("1", "true", "yes", "on")


def _maybe_sudo_wrap_remote(remote_cmd: str, env: dict, target: str = "a5") -> str:
    """Wrap a remote (post-SSH) command to run under `sudo su -c` WHEN the
    optional per-target `{TARGET}_DOCKER_SUDO`/`A5_DOCKER_SUDO` flag is set.

    Needed only on target hosts where docker requires root AND the SSH user's
    ONLY NOPASSWD sudo path is `sudo su` (not `sudo docker`) — e.g. some shared
    multi-user A5 boxes. `_shell_quote` makes the `su -c <one-arg>` nesting robust
    (no manual triple-quoting). **Backward compatible**: flag unset/false → returns
    `remote_cmd` UNCHANGED (byte-identical to prior behavior)."""
    if not _docker_sudo_enabled(env, target):
        return remote_cmd
    return f"sudo su -c {_shell_quote(remote_cmd)}"
