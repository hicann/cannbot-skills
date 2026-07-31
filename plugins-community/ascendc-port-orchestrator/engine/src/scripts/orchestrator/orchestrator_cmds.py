# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""orchestrator_cmds.py — CLI subcommand handlers (DEBT-201 god-function split).

Extracted from orchestrator.py to shrink the orchestrator god-file below the
<1000-line bar.  Only the two scoped start commands (`--port-a3` and
`--backward`) plus lifecycle handlers remain.
(`_parse_bump_caps`).

MONKEYPATCH CONTRACT (OL-160-class latent-coupling guard): symbols tests patch
through the orchestrator module — `run_single_op` and `WORKSPACE_ROOT` — are
reached via `_orch().<name>`, a CALL-TIME attribute lookup on the live
orchestrator module (see `_orch` below). This keeps
`monkeypatch.setattr(orchestrator, "run_single_op"/"WORKSPACE_ROOT", ...)`
biting AND respects files that load orchestrator.py under a private name +
register it via `monkeypatch.setitem(sys.modules, "orchestrator", <alias>)`.
orchestrator.py re-imports every public name below so `orchestrator.<cmd>`
external + test access is preserved. Handler BODIES are byte-identical to the
original; only the two patched globals were qualified (in code, not comments).
"""
from __future__ import annotations
import logging

import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import Optional

import resume as resume_mod
import state_executor
from logging_config import get_logger
from orchestrator_coldstart import _cold_start_reset_workspace
from source_arch import (
    detect_source_arch,
    record_port_a3_build_source,
    stage_source_tree,
    verify_source_stage,
)
from validation import _spec_has_backward_contract, _validate_a3_host_home_mount

log = get_logger(__name__)


# Sentinel attribute that identifies the orchestrator MODULE among candidate
# sys.modules entries (the package `orchestrator` __init__ does NOT define it;
# only orchestrator.py does).
_ORCH_MARKER = "run_single_op"


def _orch():
    """Live handle to the orchestrator MODULE, resolved lazily at CALL time.

    Resolved via `sys.modules` (NOT a module-scope `import orchestrator`, which
    would re-enter a half-loaded module — this file is imported FROM
    orchestrator.py's bottom-of-file re-export — and under `python -m
    orchestrator` could bind a second module identity). Because it is a call-time
    attribute LOOKUP on the real orchestrator module,
    `monkeypatch.setattr(orchestrator, "run_single_op"/"WORKSPACE_ROOT", ...)`
    still BITES, and a test that loads orchestrator.py under a private name +
    `monkeypatch.setitem(sys.modules, "orchestrator", <alias>)` is honored too.

    orchestrator.py can appear in sys.modules under several names by launch:
    `orchestrator` (conftest / direct import), `orchestrator.orchestrator` (the
    submodule when `-m orchestrator` resolves the PACKAGE), or `__main__` (when
    orchestrator.py itself is the `-m` target). Pick the first candidate that
    actually carries the orchestrator globals (`_ORCH_MARKER`) so the package
    shell is never mistaken for the module."""
    for _name in ("orchestrator", "orchestrator.orchestrator", "__main__"):
        _m = sys.modules.get(_name)
        if _m is not None and hasattr(_m, _ORCH_MARKER):
            return _m
    return sys.modules.get("orchestrator") or sys.modules["__main__"]


_VALID_BUMP_COUNTERS = (
    "worker", "probe", "optimizer", "fused_optimizer",
    "researcher", "det_analyzer",
)

_SCOPED_MODES = frozenset({"port_a3_to_a5", "backward"})


def _workspace_mode(workspace: Path) -> Optional[str]:
    """Return a supported persisted mode, or ``None`` for an unscoped workspace."""
    state_path = workspace / ".opgen_state.json"
    try:
        state = json.loads(state_path.read_text())
        mode = state.get("opgen_mode")
    except Exception:
        return None
    if mode == "port_a3_to_a5":
        valid, reason, _manifest = verify_source_stage(workspace, state)
        if not valid:
            log.error("migration workspace rejected before resume: %s", reason)
            return None
    return mode if mode in _SCOPED_MODES else None


def _parse_bump_caps(raw: list[str]) -> dict[str, int]:
    """Parse `--bump-cap counter:delta` entries → dict.

    Validates counter is in the YAML iter_counter set; rejects negative deltas
    AND rejects deltas > 5 (anything bigger should be a YAML edit, not a runtime
    override). Returns {} if no bumps; raises ValueError on malformed input.
    """
    out: dict[str, int] = {}
    for entry in raw:
        if ":" not in entry:
            raise ValueError(f"--bump-cap {entry!r}: expected COUNTER:DELTA")
        ctr, delta_s = entry.split(":", 1)
        ctr = ctr.strip()
        if ctr not in _VALID_BUMP_COUNTERS:
            raise ValueError(
                f"--bump-cap {entry!r}: unknown counter {ctr!r}; "
                f"valid: {sorted(_VALID_BUMP_COUNTERS)}"
            )
        try:
            delta = int(delta_s.strip())
        except ValueError as exc:
            raise ValueError(
                f"--bump-cap {entry!r}: delta {delta_s!r} not int"
            ) from exc
        if delta <= 0:
            raise ValueError(f"--bump-cap {entry!r}: delta must be positive")
        if delta > 5:
            raise ValueError(
                f"--bump-cap {entry!r}: delta {delta} > 5; for larger bumps "
                f"edit YAML iter_cap directly (audit trail in git)"
            )
        out[ctr] = delta
    return out




def _cmd_resume(*, op: str | None, all_mode: bool, lane: int, dry_run: bool) -> int:
    """Resume only workspaces created by one of the two supported modes."""
    if all_mode:
        statuses = resume_mod.scan_all()
        print(f"{'op':<40s} {'action':<25s} {'state':<25s} summary")
        print("-" * 120)
        for s in statuses:
            print(f"{s.op:<40s} {s.action.value:<25s} {s.current_state:<25s} {s.summary[:60]}")
        # Resume each that's safe
        worst_rc = 0
        for s in statuses:
            if _workspace_mode(s.workspace) is None:
                print(f"\n=== skip unscoped workspace {s.op} ===")
                continue
            if s.action in (resume_mod.ResumeAction.RESUMABLE,
                            resume_mod.ResumeAction.USER_DECISION_READY):
                print(f"\n=== resume {s.op} ===")
                rc = resume_mod.execute(s.op, workspace=s.workspace,
                                         lane=lane, dry_run=dry_run)
                if rc != 0:
                    worst_rc = max(worst_rc, rc)
        return worst_rc

    if not op:
        print("ERROR: --resume requires either op name or --all")
        return 2
    status = resume_mod.diagnose(op)
    if _workspace_mode(status.workspace) is None:
        print(
            f"ERROR: workspace {status.workspace} is not an arch migration or "
            "backward-generation run"
        )
        return 2
    return resume_mod.execute(op, lane=lane, dry_run=dry_run)









def _cmd_backward(
    *,
    forward_spec: Path,
    lane: int,
    plan_only: bool,
    cold_start: bool,
    timing: bool = False,
) -> int:
    """B3 (2026-05-29, BACKWARD_PLUGIN_DESIGN): backward (gradient) op-gen entry.

    Given a differentiable PyTorch forward spec, scaffold a `backward`-mode
    workspace: derive op name (<basename>_grad), seed op_classification.json
    with a GRADIENT tag (so `plugins.base.is_backward_class` fires → the C2
    OL-200 MIX_AIC-pipelining brief block activates) and .opgen_state.json with
    opgen_mode=backward + backward_forward_source, so `plugins.detect_plugin`
    resolves the BackwardPlugin.

    Without --plan, invoke run_single_op so its Phase O2.5 `backward` dispatch
    (phase_o25_backward) produces the SELF-CONTAINED autograd reference truth
    (BACKWARD_PLUGIN_DESIGN §5.5: forward_spec.py + canonical model.py +
    backward_ref.json + backward_cpu_truth.pt), then continues through worker
    generation and target-NPU verification. With --plan: print plan, exit 0.

    The generation logic input is ALWAYS a forward (per BACKWARD_PLUGIN_DESIGN
    §6.3 owner clarification 2026-05-29); non-PyTorch forwards must first be
    represented as a differentiable PyTorch specification. Existing backward ops are port/optimize, not
    this mode.
    """
    forward_spec = forward_spec.expanduser().resolve()
    if not forward_spec.exists():
        print(f"ERROR: --backward forward spec does not exist: {forward_spec}")
        return 2
    if not forward_spec.is_file() or forward_spec.suffix != ".py":
        print(
            f"ERROR: --backward expects a PyTorch forward spec (.py file defining the "
            f"differentiable forward), got: {forward_spec}"
        )
        return 2

    # F-H (2026-07-01, skill-entry blackbox): fail-fast .ascendc_env preflight —
    # MIRROR _cmd_port_a3's env check. Without this, a missing .ascendc_env printed the plan
    # and entered the pipeline (backward CPU-truth doesn't need the NPU immediately), only
    # crashing LATER at build = a wasted run. Runs BEFORE the --plan branch, so --plan with a
    # missing env also exits non-zero (was exit 0).
    try:
        from briefs import _common as _bc
        _bc.load_env()
    except Exception as e:
        print(f"ERROR: failed to load .ascendc_env: {e!r}")
        print(
            "  Remediation: run this plugin's init.sh, then fill "
            "workspace/.ascendc_env from workspace/.ascendc_env.template "
            "for the selected target."
        )
        return 2

    # Derive op name: <basename>_grad (strip .py; avoid double _grad suffix).
    _stem = forward_spec.stem
    op_name = _stem if _stem.endswith(("_grad", "_backward", "_bwd")) else f"{_stem}_grad"

    workspace_dir = _orch().WORKSPACE_ROOT / op_name
    if cold_start and workspace_dir.exists():
        _cold_start_reset_workspace(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Seed op_classification.json with a GRADIENT tag. detect_op_class joins
    # op_class_tags → is_backward_class(op_class) fires → C2 OL-200 brief block
    # (briefs/kw_brief._backward_perf_c2_block) activates for the worker. The
    # CLI flag IS the classification (same shortcut as _cmd_port_a3 W12).
    classification_path = workspace_dir / "op_classification.json"
    classification_path.write_text(json.dumps({
        "op": op_name,
        "op_class_tags": ["backward", "GRADIENT"],
        "op_complexity": None,
        "kb_recommendations": [],
        "source": "cli_flag_backward",
        "schema_version": 1,
    }, indent=2))

    # Seed .opgen_state.json so plugins.detect_plugin resolves BackwardPlugin
    # (its detect keys on opgen_mode == "backward").
    state_path = workspace_dir / ".opgen_state.json"
    _existing_state = {}
    if state_path.is_file():
        try:
            _existing_state = json.loads(state_path.read_text())
        except Exception:
            _existing_state = {}
    _existing_state.update({
        "schema_version": 1,
        "op": op_name,
        "opgen_mode": "backward",
        "backward_forward_source": str(forward_spec),
    })
    state_path.write_text(json.dumps(_existing_state, indent=2))

    print("=" * 72)
    print(f"BACKWARD (gradient) OP-GEN MODE — op: {op_name}")
    print("=" * 72)
    print(f"  forward spec      : {forward_spec}")
    print("  opgen_mode        : backward")
    print("  reference truth   : torch.autograd.grad of the forward (CPU/fp64)")
    print("                      via reference_provider.autograd_backward_reference")
    print("  op_class tags     : ['backward', 'GRADIENT'] → is_backward_class fires")
    print("                      → C2 OL-200 MIX_AIC-pipelining brief block active")
    print(f"  build lane        : NPU {lane}")
    print(f"  workspace         : {workspace_dir}")
    print("  phases (planned)  :")
    print("    O2.5 reference  : self-contained autograd backward truth (B3.2, §5.5)")
    print("    O4 kw spawn     : author backward AscendC kernel, verify vs autograd (B3.3)")
    print("=" * 72)
    print("  op_classification.json + .opgen_state.json seeded (opgen_mode=backward).")

    if plan_only:
        return 0

    # B3.2 (2026-05-30): end-to-end run. Mirror _cmd_port_a3 — invoke
    # run_single_op, whose Phase O2.5 dispatch (the `elif opgen_mode ==
    # "backward"` branch) picks up our just-written opgen_mode +
    # backward_forward_source and routes to phase_o25_backward, producing the
    # self-contained autograd reference truth. B3.2 stops at the reference
    # (returns 98); worker kernel-gen + on-hardware verify is B3.3.
    print("\n=== entering run_single_op (Phase O2.5 backward reference) ===")
    return _orch().run_single_op(
        op_name,
        workspace=workspace_dir,
        lane=lane,
        plan_only=False,
        timing=timing,
    )




def _cmd_port_a3(
    *,
    port_a3_dir: Path,
    lane: int,
    plan_only: bool,
    cold_start: bool,
    cap_bumps: dict[str, int],
    timing: bool = False,
) -> int:
    """W1 (2026-05-12, ROADMAP §1.5): arch22→arch35 port-mode entry point.

    Validates the ops-nn source directory shape, derives op name from
    dirname, configures the run with OPGEN_MODE=port_a3_to_a5, and routes
    through `run_single_op`.

    Args:
        port_a3_dir: Path to ops-nn op dir (e.g. ~/workspace/cann/ops-nn/loss/ctc_loss_v3).
                     Must contain op_host/ AND op_kernel/ subdirs.
        lane: A5 NPU lane (0/1/2) — A5-side build/verify target.
        plan_only: --plan flag; print plan, exit 0 without invoking state machine.
        cold_start: --cold-start flag; reset workspace before run.
        cap_bumps: --bump-cap dict (orchestrator-only audit).

    Returns:
        exit code (0 on success, non-zero on validation / state-machine error)

    Pre-conditions checked:
        - port_a3_dir exists and is a directory
        - Contains op_host/ and op_kernel/ subdirs (ops-nn shape)
        - .ascendc_env has A3_HOST + A3_CONTAINER populated (A3 reference run needs them)

    The live path validates and captures the arch22 reference before an
    independently-authored arch35 implementation is built and verified, then
    archives the result in the ops-nn mirror layout. ``--plan`` prints this
    path without entering the state machine.
    """
    port_a3_dir = port_a3_dir.expanduser().resolve()

    # Validation
    if not port_a3_dir.exists():
        print(f"ERROR: --port-a3 path does not exist: {port_a3_dir}")
        return 2
    if not port_a3_dir.is_dir():
        print(f"ERROR: --port-a3 path is not a directory: {port_a3_dir}")
        return 2
    op_host_dir = port_a3_dir / "op_host"
    op_kernel_dir = port_a3_dir / "op_kernel"
    if not op_host_dir.is_dir() or not op_kernel_dir.is_dir():
        print(
            f"ERROR: --port-a3 path does not look like an ops-nn op dir "
            f"(missing op_host/ or op_kernel/): {port_a3_dir}"
        )
        return 2

    op_name = port_a3_dir.name

    source_detection = detect_source_arch(port_a3_dir)
    if not source_detection.supported or source_detection.arch != "arch22":
        print(
            "ERROR: --port-a3 requires a detected arch22 source; "
            f"method={source_detection.method} arch={source_detection.arch!r} "
            f"confidence={source_detection.confidence}"
        )
        for evidence in source_detection.evidence:
            print(f"  evidence: {evidence}")
        return 2

    # P87+ (2026-05-15): cold-start reset MUST happen BEFORE workspace seeding.
    # The reset backs up .opgen_state.json and op_classification.json; if we
    # seed those first, the reset wipes them, and run_single_op then can't
    # detect migration mode (and would otherwise fail closed as unsupported).
    # Order: backup-and-reset → seed → run.
    # Compute workspace path manually (don't use _resolve_workspace which
    # scans existing dirs — workspace may not exist on first invocation).
    workspace_dir_early = _orch().WORKSPACE_ROOT / op_name
    if cold_start and workspace_dir_early.exists():
        _cold_start_reset_workspace(workspace_dir_early)

    # Env check: A3 reference run needs A3 connection info.
    # DEBT-101 (2026-05-28): `load_env()` no-arg now resolves
    # DEFAULT_ASCENDC_ENV at call time (sentinel default) + honors
    # `ASCENDC_ENV_PATH` env var for subprocess test overrides. The earlier
    # workaround `load_env(_bc.DEFAULT_ASCENDC_ENV)` was needed when the
    # function default bound the constant at import time; now it bypasses
    # both the call-time resolution AND the env-var override. Use no-arg
    # form so both fixes flow through.
    try:
        from briefs import _common as _bc
        env = _bc.load_env()
    except Exception as e:
        print(f"ERROR: failed to load .ascendc_env: {e!r}")
        return 2
    if env.target != "a5":
        print(
            f"WARN: TARGET={env.target} in .ascendc_env; port-a3 mode assumes TARGET=a5. "
            f"Continuing — A5 codegen will use the active target."
        )
    if not env.a3_host or not env.a3_container:
        print(
            "ERROR: --port-a3 requires A3_HOST + A3_CONTAINER in workspace/.ascendc_env "
            "for the A3-CANN reference run. Run this plugin's init.sh, then add the "
            "A3 fields documented in workspace/.ascendc_env.template."
        )
        return 2
    # P135.HV (2026-05-20 DS): port_a3 needs BOTH A3 and A5 hosts.
    # AscendCEnv resolves host/container from current target (a3-ds → A3_HOST),
    # so env.host may be the A3 host. A separate A5 build host is available if
    # EITHER an explicit A5_HOST is configured (task#24-item2: dedicated key, lets
    # a TARGET=a3 agent point port_a3 at a separate A5 host without flipping
    # TARGET) OR the active-target host already differs from a3_host (TARGET=a5
    # agent — the original DS path).
    a5_build_host = env.a5_host or (env.host if env.host != env.a3_host else "")
    if not a5_build_host:
        print(
            "ERROR: --port-a3 requires a separate A5 host for build/verify. "
            "Both host ({h}) and a3_host point to the same machine, and no "
            "explicit A5_HOST is configured. Add A5_HOST + A5_CONTAINER to "
            "workspace/.ascendc_env or ask the main agent to run this "
            "op.".format(h=env.host)
        )
        return 9  # distinct exit code for "mode requires unavailable host"

    # P129 (2026-05-17): runtime gate validating A3_HOST_HOME ↔ container
    # mount alignment. Catches the silent slice-vs-default-mount drift that
    # caused the 2026-05-16/17 P126→P129 incident (a3 destroy/recreate with
    # default mount left env.a3_host_home pointing at a slice path that the
    # container doesn't bind, so scp pushed to wrong host path → runner cd
    # failed mid-flight with confusing message). Hard-fail here instead.
    if os.environ.get("A3_HOST_MODE", "") == "1":
        # A3_HOST_MODE (2026-06-11): host-direct A3 reference bypasses the
        # npu-a3 container entirely (torch_npu runs on the host python, see
        # phase_o25_a3_ref._default_run_remote). The container mount is
        # irrelevant in this mode — skip the P129 container-mount gate.
        print("INFO: A3_HOST_MODE=1 — P129 container-mount gate skipped (host-direct A3 ref).")
    elif env.a3_host_home:
        # container_home CONFIG-DRIVEN (genericize ②): read A3_CONTAINER_HOME from .ascendc_env
        # so a scrubbed / non-npu_user deployment inspects the right mount Destination.
        import phase_o25_a3_ref
        rc = _validate_a3_host_home_mount(
            env.a3_host, env.a3_container, env.a3_host_home,
            container_home=getattr(phase_o25_a3_ref, "_a3_container_home")())
        if rc != 0:
            return rc
    else:
        print(
            "WARN: A3_HOST_HOME not set in workspace/.ascendc_env. P129 mount gate "
            "skipped. Run `setup_a3_isolated_container.sh` to regenerate env (it "
            "emits A3_HOST_HOME + A3_HOST_BACKUP). Continuing — phase_o25_a3_ref "
            "will fall back to legacy default path."
        )

    # Surface the plan regardless of --plan flag (cheap, informational).
    archive_root = Path(env.local_project or ".") / "output" / "a3_to_a5_port"
    print("=" * 72)
    print(f"arch22→arch35 PORT MODE — op: {op_name}")
    print("=" * 72)
    print(f"  source            : {port_a3_dir}")
    print(f"  source architecture: arch22 ({source_detection.method}, "
          f"confidence={source_detection.confidence})")
    print(f"  active target     : {env.target} (host={env.host}, container={env.container})")
    print(f"  A3 reference host : {env.a3_host} (container={env.a3_container}, SOC={env.a3_soc_version})")
    print(f"  A5 build lane     : NPU {lane}")
    print("  opgen_mode        : port_a3_to_a5")
    print(f"  archive target    : {archive_root}/{op_name}/")
    print("  phases (planned)  :")
    print("    O0 preflight    : standard")
    print("    O1 config       : opgen_mode=port_a3_to_a5 propagated via AscendCEnv (W2)")
    print("    O2 source sync  : fixed arch22 source stage prepared")
    print("    O2.5 reference  : live arch22 reference capture")
    print("    O3 PROGRESS     : standard")
    print("    O4 kw spawn     : independent arch35 implementation from arch22 semantics")
    print("    O5 verify       : truth = fresh live arch22 reference output")
    print("    O6 archive      : ops-nn mirror layout writer")
    print("=" * 72)

    # W12 (2026-05-12, ROADMAP §1.5): seed op_classification.json with the
    # a3_to_a5_port tag so kb_manifest_block auto-loads W8-W11 KB entries
    # without waiting for the (slower, LLM-driven) /aog-op-classify pass.
    # This is the "known classification" shortcut for the port-mode path —
    # the CLI flag IS the classification.
    # Use direct path (not _resolve_workspace) — workspace may not exist yet
    # and _resolve_workspace's iterdir-fallback errors on missing WORKSPACE_ROOT.
    workspace_dir = _orch().WORKSPACE_ROOT / op_name
    workspace_dir.mkdir(parents=True, exist_ok=True)
    try:
        source_stage = stage_source_tree(port_a3_dir, workspace_dir)
    except Exception as exc:
        print(f"ERROR: could not create source-only arch22 snapshot: {exc}")
        return 2
    source_detection = source_stage.detection
    print(
        f"  source snapshot   : {source_stage.root} "
        f"({source_stage.file_count} files, sha256={source_stage.digest[:12]})"
    )
    classification_path = workspace_dir / "op_classification.json"
    # P0gg (2026-05-28): seed algorithm-class tags from op_name in addition to
    # the mode tag. Without this, FA-class detection (`is_fa_class`, which needs
    # FUSED+SOFTMAX tags) could never fire for FA-class ports — only
    # `a3_to_a5_port` was seeded, so the kw FA template-assembly route (owner
    # 2026-06-07) would not be selected. When the CLI flag IS the classification
    # (per `_read_cached` short-circuit), the seed must include the
    # algorithm-class tags so the op-class gate resolves correctly.
    op_class_tags = ["a3_to_a5_port"]
    op_complexity: Optional[str] = None
    op_lower = op_name.lower()
    if "flash_attention" in op_lower or "fused_attention" in op_lower or "fusion_attention" in op_lower:
        op_class_tags.extend(["FUSED_SOFTMAX", "fa_class"])
        op_complexity = "L4"
    # P0hh (2026-06-01): tag cube-class (matmul-bearing) ops CUBE_MIX so the
    # brief AND the finalize gate have a cube signal at classification time.
    # Without this, non-FA cube ops (matmul/conv/rnn/gmm/attention/ffn) get
    # only ["a3_to_a5_port"] and fall through to pure-vec generation (the gate
    # then silently SKIPs when port_source is unresolvable at finalize). Derive
    # via the SHARED gate classifier (single source of truth — same family-prefix
    # + cube-marker grep the gate uses; do NOT re-implement here). See
    # docs/design/PORT_A3_CUBE_CLASS_MIX_ENFORCEMENT_DESIGN.md + ROADMAP task#23.
    try:
        import sys as _sys
        _checks_dir = str(Path(__file__).resolve().parent / "checks")
        if _checks_dir not in _sys.path:
            _sys.path.insert(0, _checks_dir)
        from architecture_class_check import _classify_reference_arch  # type: ignore
        _reference_arch = _classify_reference_arch(source_stage.root)
        if _reference_arch == "unknown":
            raise RuntimeError("arch22 architecture class could not be verified")
        if _reference_arch == "cube-required":
            op_class_tags.append("CUBE_MIX")
    except Exception as _cube_e:
        raise RuntimeError(
            f"arch22 architecture classification failed closed: {_cube_e}"
        ) from _cube_e
    classification_payload = {
        "op": op_name,
        "op_class_tags": op_class_tags,
        "op_complexity": op_complexity,
        "kb_recommendations": [],
        "source": "cli_flag_port_a3",
        "schema_version": 1,
    }
    classification_path.write_text(json.dumps(classification_payload, indent=2))
    print(f"  op_classification.json seeded with tags={op_class_tags} complexity={op_complexity} (W12 / P0gg)")

    # W15 follow-up (2026-05-12): seed workspace/.opgen_state.json with
    # opgen_mode=port_a3_to_a5 + port_a3_source BEFORE invoking run_single_op.
    # This is what the W15 dispatch in run_single_op reads to route Phase O2.5
    # to phase_o25_a3_ref instead of the stock check. The phase_o05.init_durable_state
    # called from run_single_op preserves the port_a3_to_a5 mode (see refined
    # 2026-05-12 guard — durable state never overwrites a scoped mode).
    state_path = workspace_dir / ".opgen_state.json"
    state_payload = {
        "schema_version": 2,
        "op": op_name,
        "target": env.target,
        "lane": lane,
        "opgen_mode": "port_a3_to_a5",
        "port_a3_source": str(source_stage.root),
        "source_stage_manifest": str(source_stage.manifest),
        "source_stage_digest": source_stage.digest,
        "source_stage_file_count": source_stage.file_count,
        "graybox_sandbox": True,
        "graybox_arch22_dir": str(source_stage.root),
        "source_arch": "arch22",
        "target_arch": "arch35",
        "source_arch_detection": source_detection.state_payload(),
        "started_ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "last_seen_ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "invocation_count": 1,
    }
    # If state file exists, preserve started_ts + invocation_count
    if state_path.exists():
        try:
            prior = json.loads(state_path.read_text())
            state_payload["started_ts"] = prior.get("started_ts", state_payload["started_ts"])
            state_payload["invocation_count"] = int(prior.get("invocation_count", 0)) + 1
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
    state_path.write_text(json.dumps(state_payload, indent=2))
    valid_stage, stage_reason, _stage_manifest = verify_source_stage(
        workspace_dir, state_payload
    )
    if not valid_stage:
        print(f"ERROR: source-only snapshot state validation failed: {stage_reason}")
        return 2
    try:
        record_port_a3_build_source(
            workspace_dir,
            port_a3_dir,
            source_stage_digest=source_stage.digest,
        )
    except (OSError, ValueError) as exc:
        print(f"ERROR: could not persist private build-source binding: {exc}")
        return 2
    print(
        "  .opgen_state.json seeded with opgen_mode=port_a3_to_a5 + "
        f"verified source-only snapshot ({stage_reason})"
    )

    if plan_only:
        return 0

    # End-to-end live run (post-W14b + W15 wiring): invoke run_single_op
    # which dispatches through phase_o25_a3_ref + state machine + finalize.
    # run_single_op's _opgen_state_path reader picks up our just-written
    # mode + source; W15 dispatch routes Phase O2.5 to the A3-CANN variant.
    print("\n=== entering run_single_op (full state-machine run) ===")
    # Keep the original checkout out of durable graybox state.  Phase O2.5
    # consumes this process-scoped value before any worker spawn and uses it
    # only when shipped-CANN dispatch proves a source package build is needed.
    build_source_env = "CANNBOT_PORT_A3_BUILD_SOURCE"
    prior_build_source = os.environ.get(build_source_env)
    os.environ[build_source_env] = str(port_a3_dir)
    try:
        return _orch().run_single_op(
            op_name,
            workspace=workspace_dir,
            lane=lane,
            plan_only=False,
            cap_bumps=cap_bumps,
            timing=timing,
        )
    finally:
        if prior_build_source is None:
            os.environ.pop(build_source_env, None)
        else:
            os.environ[build_source_env] = prior_build_source




def _cmd_status() -> int:
    """Show all op workspaces' current state."""
    if not _orch().WORKSPACE_ROOT.exists():
        print("no workspace/ directory")
        return 0
    rows = []
    for d in sorted(_orch().WORKSPACE_ROOT.iterdir()):
        if not d.is_dir() or not (d / "PROGRESS.md").exists():
            continue
        skip_current_item = False
        try:
            state = state_executor.current_state(d)
            agent = state_executor.next_agent(state) or "—"
            rows.append((d.name, state, agent))
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
    print(f"{'op':<50s} {'state':<25s} {'next_agent':<25s}")
    print("-" * 105)
    for name, state, agent in rows:
        print(f"{name:<50s} {state:<25s} {agent:<25s}")
    return 0
