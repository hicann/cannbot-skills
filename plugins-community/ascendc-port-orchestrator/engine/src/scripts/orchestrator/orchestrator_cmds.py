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
<1000-line bar.  Only the two scoped start commands (`--port-a3-ops` and
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
import stat
import sys
from pathlib import Path
from typing import Optional

import resume as resume_mod
import state_executor
from logging_config import get_logger
from npubench.npubench_inputs import (
    NpubenchInputError,
    atomic_write_state,
    bind_npubench_state,
    stage_npubench_inputs,
    validate_cli_npubench_args,
)
from reference_source import (
    A3_LIVE,
    CANNBENCH,
    NPUBENCH,
    VALID_REFERENCE_SOURCES,
    explicit_a3_live_binding,
    explicit_cannbench_binding,
    resolve_reference_source,
)
from orchestrator_coldstart import _cold_start_reset_workspace
from source_arch import (
    detect_source_arch,
    record_port_a3_build_source,
    stage_source_tree,
    verify_source_stage,
)
from validation import _spec_has_backward_contract, _validate_a3_host_home_mount
from a5_target_capability import a5_soc_version, is_limited_a5_soc, limited_a5_warning

log = get_logger(__name__)


TILELANG2ASCENDC_SOURCE_KIND = "port-aclnn-tilelang2ascendc"
TILELANG2ASCENDC_SOURCE_KIND_ALIAS = "port_aclnn_tilelang2ascendc"
TILELANG2ASCENDC_CANDIDATE_KIND = "tilelang2ascendc_custom_op"


def _tilelang2ascendc_source_api():
    """Return the independent TileLang2AscendC source adapter."""
    try:
        from tilelang2ascendc_source import (
            detect_tilelang2ascendc_source,
            logical_op_name,
            stage_tilelang2ascendc_source_tree,
            verify_tilelang2ascendc_source_stage,
        )
    except ImportError as exc:
        raise RuntimeError(
            "port-aclnn-tilelang2ascendc requires the tilelang2ascendc_source staging adapter"
        ) from exc
    return (
        detect_tilelang2ascendc_source,
        logical_op_name,
        stage_tilelang2ascendc_source_tree,
        verify_tilelang2ascendc_source_stage,
    )


_TILELANG2ASCENDC_REQUIRED_FIELDS = (
    ("port_a3_source", "stage root"),
    ("source_stage_manifest", "stage manifest"),
    ("source_stage_digest", "stage digest"),
    ("source_stage_file_count", "stage file count"),
    ("graybox_source_dir", "graybox source bind"),
)


def _is_tilelang2ascendc_source_state(state: dict) -> bool:
    """Return true when durable state declares the TileLang2AscendC route."""
    kind = state.get("source_kind")
    if kind in (TILELANG2ASCENDC_SOURCE_KIND, TILELANG2ASCENDC_SOURCE_KIND_ALIAS):
        return True
    port_source = state.get("port_source")
    return isinstance(port_source, dict) and port_source.get("kind") == TILELANG2ASCENDC_SOURCE_KIND


def _tilelang2ascendc_stage_binding_ok(port_source: dict, state: dict) -> bool:
    """Compare the staged metadata with durable state as two flat mappings.

    Expressed as one mapping equality rather than a long boolean chain: every
    pair below was ANDed before, and none of the lookups has a side effect, so
    the verdict is unchanged.
    """
    staged = {
        "stage_root": port_source.get("stage_root"),
        "manifest": port_source.get("manifest"),
        "digest": port_source.get("digest"),
        "file_count": port_source.get("file_count"),
        "source_arch": port_source.get("source_arch"),
        "target_arch": port_source.get("target_arch"),
        "root_layout": port_source.get("root_layout"),
        "graybox_bind": state.get("graybox_source_dir"),
    }
    durable = {
        "stage_root": state.get("port_a3_source"),
        "manifest": state.get("source_stage_manifest"),
        "digest": state.get("source_stage_digest"),
        "file_count": state.get("source_stage_file_count"),
        "source_arch": "arch35",
        "target_arch": "arch35",
        "root_layout": "model_and_kernel",
        "graybox_bind": state.get("port_a3_source"),
    }
    return staged == durable


def _tilelang2ascendc_identity_error(state: dict) -> str | None:
    """Return the first durable-identity defect, or None when the state binds."""
    port_source = state.get("port_source")
    if state.get("source_kind") != TILELANG2ASCENDC_SOURCE_KIND:
        return "TileLang2AscendC source_kind is missing or mismatched"
    if not isinstance(port_source, dict) or port_source.get("kind") != TILELANG2ASCENDC_SOURCE_KIND:
        return "TileLang2AscendC port_source.kind is missing or mismatched"
    if state.get("source_arch") != "arch35" or state.get("target_arch") != "arch35":
        return "TileLang2AscendC route must persist source_arch=target_arch=arch35"
    for field, label in _TILELANG2ASCENDC_REQUIRED_FIELDS:
        if state.get(field) in (None, ""):
            return f"TileLang2AscendC {label} is missing"
    if not _tilelang2ascendc_stage_binding_ok(port_source, state):
        return "TileLang2AscendC stage metadata disagrees with durable port_source"
    return None


def _verify_tilelang2ascendc_source_binding(workspace: Path, state: dict):
    """Authenticate the immutable TileLang2AscendC source project."""
    identity_error = _tilelang2ascendc_identity_error(state)
    if identity_error is not None:
        return False, identity_error, None
    try:
        _detect, _logical_op, _stage, verify = _tilelang2ascendc_source_api()
        return verify(workspace, state)
    except Exception as exc:
        return False, f"TileLang2AscendC source-stage verifier unavailable or failed: {exc}", None


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
        try:
            source = resolve_reference_source(state)
        except Exception as exc:
            log.error("migration workspace has no valid explicit reference: %s", exc)
            return None
        # The TileLang2AscendC explicit source kind carries its own arch35
        # snapshot contract.  It is implementation context, not NPUKernelBench
        # truth, but resume must still authenticate it before any agent can
        # read it.  In particular this check is driven only by durable state,
        # never by the one-shot CLI value that created the workspace.
        if _is_tilelang2ascendc_source_state(state):
            valid, reason, _manifest = _verify_tilelang2ascendc_source_binding(workspace, state)
        # The benchmark providers' source of truth is their frozen bundle, not
        # the legacy arch22 port-A3 source snapshot.  Keep the established
        # behaviour unchanged.
        elif source in {A3_LIVE}:
            valid, reason, _manifest = verify_source_stage(workspace, state)
        else:
            valid = True
            reason = "not required for frozen benchmark provider"
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




def _bind_explicit_reference(state_payload: dict, expected_reference: dict) -> bool:
    """Bind an explicit truth-source reference into the workspace state payload.

    Returns False (after printing the operator diagnostic) when the workspace
    already carries a different or partial binding: changing the truth source
    in place is refused, exactly as before this was factored out.
    """
    existing_reference = state_payload.get("reference")
    if existing_reference is None:
        state_payload["reference"] = expected_reference
        return True
    if existing_reference != expected_reference:
        print(
            "ERROR: workspace already has a different or partial reference binding; "
            "use an explicit migration/cold start instead of changing truth source"
        )
        return False
    return True


def _a3_container_mount_gate(env) -> int:
    """P129 runtime gate: A3_HOST_HOME must match the container's bind mount.

    Catches the silent slice-vs-default-mount drift behind the 2026-05-16/17
    P126->P129 incident (an a3 destroy/recreate with the default mount left
    ``env.a3_host_home`` pointing at a slice path the container does not bind,
    so scp pushed to the wrong host path and the runner's ``cd`` failed
    mid-flight with a confusing message).  Hard-fail here instead.

    Returns the CLI exit status; ``0`` means the gate passed or was skipped.
    """
    if os.environ.get("A3_HOST_MODE", "") == "1":
        # A3_HOST_MODE (2026-06-11): host-direct A3 reference bypasses the
        # npu-a3 container entirely (torch_npu runs on the host python, see
        # phase_o25_a3_ref._default_run_remote).  The container mount is
        # irrelevant in this mode - skip the P129 container-mount gate.
        print("INFO: A3_HOST_MODE=1 - P129 container-mount gate skipped (host-direct A3 ref).")
        return 0
    if not env.a3_host_home:
        print(
            "WARN: A3_HOST_HOME not set in workspace/.ascendc_env. P129 mount gate "
            "skipped. Run `setup_a3_isolated_container.sh` to regenerate env (it "
            "emits A3_HOST_HOME + A3_HOST_BACKUP). Continuing - phase_o25_a3_ref "
            "will fall back to legacy default path."
        )
        return 0
    # container_home CONFIG-DRIVEN (genericize (2)): read A3_CONTAINER_HOME from
    # .ascendc_env so a scrubbed / non-npu_user deployment inspects the right
    # mount Destination.
    import phase_o25_a3_ref

    return _validate_a3_host_home_mount(
        env.a3_host, env.a3_container, env.a3_host_home,
        container_home=getattr(phase_o25_a3_ref, "_a3_container_home")())


def _apply_prior_durable_state(
    state_path: Path,
    *,
    cold_start: bool,
    state_payload: dict,
) -> tuple[dict, int]:
    """Fold an existing durable state file into ``state_payload``.

    Returns ``(prior_state, rc)``.  A nonzero ``rc`` is the CLI exit status the
    caller must propagate; ``prior_state`` is empty unless a warm start found a
    usable prior binding.
    """
    if not state_path.exists():
        return {}, 0
    try:
        prior = json.loads(state_path.read_text())
        if not isinstance(prior, dict):
            raise ValueError("durable state is not a JSON object")
        if cold_start:
            survivor_keys = set(prior)
            if not survivor_keys.issubset({"lifetime_spawn_count"}):
                raise ValueError("cold-start reset left unexpected durable state fields")
            lifetime_spawn_count = prior["lifetime_spawn_count"]
            if (
                isinstance(lifetime_spawn_count, bool)
                or not isinstance(lifetime_spawn_count, int)
                or lifetime_spawn_count < 0
            ):
                raise ValueError("cold-start lifetime_spawn_count is malformed")
            state_payload["lifetime_spawn_count"] = lifetime_spawn_count
            return {}, 0
        if "reference" not in prior or prior.get("reference") is None:
            print(
                "ERROR: existing workspace has no explicit reference binding; "
                "run an explicit a3_live migration or use --cold-start"
            )
            return {}, 2
        if not isinstance(prior.get("reference"), dict):
            print("ERROR: existing workspace reference binding is malformed")
            return {}, 2
        state_payload["started_ts"] = prior.get("started_ts", state_payload["started_ts"])
        state_payload["invocation_count"] = int(prior.get("invocation_count", 0)) + 1
        return prior, 0
    except Exception as error:
        print(f"ERROR: existing workspace state is unreadable: {error}")
        return {}, 2


def _bind_reference_source(state_payload: dict, source: str, reference_stage) -> int:
    """Bind the resolved reference source into ``state_payload``.

    Returns the CLI exit status; ``0`` means the binding was written.
    """
    if source == NPUBENCH:
        try:
            bind_npubench_state(state_payload, reference_stage)
        except NpubenchInputError as exc:
            print(f"ERROR: could not bind NPUKernelBench state: {exc}")
            return 2
        return 0
    if source == A3_LIVE:
        return 0 if _bind_explicit_reference(state_payload, explicit_a3_live_binding()) else 2
    if source == CANNBENCH:
        return 0 if _bind_explicit_reference(state_payload, explicit_cannbench_binding()) else 2
    # Defensive: CLI source validation above is exhaustive.
    print(f"ERROR: unsupported reference source: {source!r}")
    return 2


def _cmd_port_a3(
    *,
    port_a3_dir: Path,
    lane: int,
    plan_only: bool,
    cold_start: bool,
    cap_bumps: dict[str, int],
    timing: bool = False,
    reference_source: str | None = None,
    npubench_task: Path | None = None,
    npubench_root: Path | None = None,
    extra_lanes: list[int] | None = None,
    source_kind: str | None = None,
    source_arch: str | None = None,
    candidate_kind: str | None = None,
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
        reference_source: Explicit reference provider override (npubench or
            a3_live).  Required for new invocations: bare --port-a3-ops without it
            is a hard error.
        npubench_task: Original old-format NPUKernelBench task Python file.
        npubench_root: Optional task source-root closure.
        extra_lanes: Explicit extra target lanes used only by the npubench
            evaluator for safe precision/performance parallelism.

    Returns:
        exit code (0 on success, non-zero on validation / state-machine error)

    Pre-conditions checked:
        - port_a3_dir exists and is a directory
        - Contains op_host/ and op_kernel/ subdirs (ops-nn shape)
        - .ascendc_env has an A5 build/verify configuration (except the
          deliberately unsupported CannBench reservation)
        - a3_live additionally requires A3_HOST + A3_CONTAINER

    The live path validates and captures the arch22 reference before an
    independently-authored arch35 implementation is built and verified, then
    archives the result in the ops-nn mirror layout. ``--plan`` prints this
    path without entering the state machine.
    """
    port_a3_dir = port_a3_dir.expanduser().resolve()

    # Validation
    if not port_a3_dir.exists():
        print(f"ERROR: --port-a3-ops path does not exist: {port_a3_dir}")
        return 2
    if not port_a3_dir.is_dir():
        print(f"ERROR: --port-a3-ops path is not a directory: {port_a3_dir}")
        return 2
    if source_kind == TILELANG2ASCENDC_SOURCE_KIND_ALIAS:
        source_kind = TILELANG2ASCENDC_SOURCE_KIND
    tilelang_source = source_kind == TILELANG2ASCENDC_SOURCE_KIND
    if source_kind not in (None, TILELANG2ASCENDC_SOURCE_KIND):
        print(f"ERROR: unsupported --source-kind: {source_kind!r}")
        return 2
    if not tilelang_source and (source_arch is not None or candidate_kind is not None):
        print(
            "ERROR: --source-arch and --candidate-kind require an explicit "
            "--source-kind (port-aclnn-tilelang2ascendc)"
        )
        return 2
    if tilelang_source:
        if source_arch not in (None, "arch35"):
            print("ERROR: port-aclnn-tilelang2ascendc accepts source architecture arch35 only")
            return 2
        if candidate_kind not in (None, TILELANG2ASCENDC_CANDIDATE_KIND):
            print(
                "ERROR: --candidate-kind tilelang2ascendc_custom_op is required "
                "for port-aclnn-tilelang2ascendc"
            )
            return 2
        try:
            (
                detect_tilelang_source,
                logical_tilelang_op,
                _stage_tilelang,
                _verify_tilelang,
            ) = _tilelang2ascendc_source_api()
            source_detection = detect_tilelang_source(port_a3_dir)
            op_name = logical_tilelang_op(port_a3_dir)
        except Exception as exc:
            print(f"ERROR: TILELANG2ASCENDC_SOURCE_DETECTION_FAILED: {exc}")
            return 2
    else:
        op_host_dir = port_a3_dir / "op_host"
        op_kernel_dir = port_a3_dir / "op_kernel"
        if not op_host_dir.is_dir() or not op_kernel_dir.is_dir():
            print(
                f"ERROR: --port-a3-ops path does not look like an ops-nn op dir "
                f"(missing op_host/ or op_kernel/): {port_a3_dir}"
            )
            return 2
        op_name = port_a3_dir.name

    try:
        npubench_args = validate_cli_npubench_args(npubench_task, npubench_root)
    except NpubenchInputError as exc:
        print(f"ERROR: {exc}")
        return 2
    if tilelang_source:
        if not source_detection.supported or source_detection.arch != "arch35":
            print(
                "ERROR: --source-kind port-aclnn-tilelang2ascendc requires a detected "
                "arch35 TileLang2AscendC project; "
                f"method={source_detection.method} arch={source_detection.arch!r} "
                f"confidence={source_detection.confidence}"
            )
            for evidence in source_detection.evidence:
                print(f"  evidence: {evidence}")
            return 2
    else:
        source_detection = detect_source_arch(port_a3_dir)
        if not source_detection.supported or source_detection.arch != "arch22":
            print(
                "ERROR: --port-a3-ops requires a detected arch22 source; "
                f"method={source_detection.method} arch={source_detection.arch!r} "
                f"confidence={source_detection.confidence}"
            )
            for evidence in source_detection.evidence:
                print(f"  evidence: {evidence}")
            return 2

    # Environment preflight.  Both reference paths need an A5 build/verify
    # target; only the legacy a3_live path additionally needs A3 connectivity.
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
    if reference_source is None and npubench_args is not None:
        print(
            "ERROR: --npubench-task requires --reference-source npubench; "
            "the benchmark provider must be selected explicitly."
        )
        return 2
    configured_source = reference_source or env.port_a3_reference_source
    if not configured_source:
        print(
            "ERROR: --port-a3-ops requires an explicit reference provider: pass "
            "--reference-source npubench --npubench-task TASK_PY for a frozen "
            "NPUKernelBench task (preferred), or explicitly select "
            "--reference-source a3_live for a fresh A3 CANN capture."
        )
        return 2
    if configured_source not in VALID_REFERENCE_SOURCES:
        allowed = ", ".join(sorted(VALID_REFERENCE_SOURCES))
        print(
            "ERROR: PORT_A3_REFERENCE_SOURCE/--reference-source must be one of "
            f"{allowed}; got {configured_source!r}"
        )
        return 2
    if tilelang_source and configured_source != NPUBENCH:
        print(
            "ERROR: UNSUPPORTED_SOURCE_REFERENCE_COMBINATION: "
            "port-aclnn-tilelang2ascendc requires --reference-source npubench "
            "so the frozen task remains the only oracle"
        )
        return 2
    if configured_source == NPUBENCH and npubench_args is None:
        print(
            "ERROR: --reference-source npubench requires --npubench-task TASK_PY"
        )
        return 2
    if configured_source != NPUBENCH and npubench_args is not None:
        print(
            "ERROR: --npubench-task/--npubench-root require "
            "--reference-source npubench"
        )
        return 2
    resolved_reference_source = configured_source
    if extra_lanes and resolved_reference_source != NPUBENCH:
        print("ERROR: --extra-lane is supported only with --reference-source npubench")
        return 2
    if env.target != "a5":
        print(
            f"WARN: TARGET={env.target} in .ascendc_env; port-a3-ops mode assumes TARGET=a5. "
            f"Continuing — A5 codegen will use the active target."
        )
    # NPUKernelBench target validation requires an A5-capable SoC.  Ascend910
    # remains useful for preflight and codegen, but the final target gate is
    # deliberately terminal for the explicit TileLang2AscendC source kind.
    if tilelang_source:
        configured_a5_soc = a5_soc_version(
            {
                "A5_SOC_VERSION": env.a5_soc_version,
                "SOC_VERSION": env.soc_version,
            }
        )
        if is_limited_a5_soc(configured_a5_soc):
            print(limited_a5_warning(configured_a5_soc))
    # Staged external references need only the A5 endpoint.  The live-A3 path
    # retains its separate-host guard.  CannBench is intentionally accepted to
    # persist an explicit unsupported-provider result without pretending an A5
    # evaluator exists yet.
    local_external_reference_target = (
        resolved_reference_source == NPUBENCH
        and (env.a5_container or (env.container if env.target.startswith("a5") else ""))
        .strip()
        .lower()
        == "local"
    )
    if resolved_reference_source == NPUBENCH:
        a5_build_host = env.a5_host or (
            env.host if env.target.startswith("a5") else ""
        )
    else:
        a5_build_host = env.a5_host or (
            env.host if env.host != env.a3_host else ""
        )
    if (
        resolved_reference_source != CANNBENCH
        and not a5_build_host
        and not local_external_reference_target
    ):
        if resolved_reference_source == NPUBENCH:
            print(
                "ERROR: staged external reference requires an A5 host "
                "for build/verify, unless A5_CONTAINER=local selects the controller "
                "as the explicit A5 target. Add A5_HOST or configure the local target."
            )
            return 9
        print(
            "ERROR: --port-a3-ops requires a separate A5 host for build/verify. "
            "Both host ({h}) and a3_host point to the same machine, and no "
            "explicit A5_HOST is configured. Add A5_HOST + A5_CONTAINER to "
            "workspace/.ascendc_env or ask the main agent to run this "
            "op.".format(h=env.host)
        )
        return 9  # distinct exit code for "mode requires unavailable host"

    if resolved_reference_source == A3_LIVE:
        if not env.a3_host or not env.a3_container:
            print(
                "ERROR: --port-a3-ops requires A3_HOST + A3_CONTAINER in workspace/.ascendc_env "
                "for the A3-CANN reference run. Run this plugin's init.sh, then add the "
                "A3 fields documented in workspace/.ascendc_env.template."
            )
            return 2
        # P129 (2026-05-17): runtime gate validating A3_HOST_HOME <-> container
        # mount alignment; see _a3_container_mount_gate for the incident it guards.
        rc = _a3_container_mount_gate(env)
        if rc != 0:
            return rc

    # Surface the plan regardless of --plan flag (cheap, informational).
    archive_root = Path(env.local_project or ".") / "output" / "a3_to_a5_port"
    print("=" * 72)
    route_label = (
        "arch35 TileLang2AscendC"
        if tilelang_source
        else "arch22→arch35"
    )
    print(f"{route_label} PORT MODE — op: {op_name}")
    print("=" * 72)
    print(f"  source            : {port_a3_dir}")
    if tilelang_source:
        print(
            "  source architecture: arch35 (explicit port-aclnn-tilelang2ascendc; "
            f"{source_detection.method}, confidence={source_detection.confidence})"
        )
    else:
        print(f"  source architecture: arch22 ({source_detection.method}, "
              f"confidence={source_detection.confidence})")
    print(f"  active target     : {env.target} (host={env.host}, container={env.container})")
    print(f"  reference source  : {resolved_reference_source}")
    if resolved_reference_source == A3_LIVE:
        print(f"  A3 reference host : {env.a3_host} (container={env.a3_container}, SOC={env.a3_soc_version})")
    elif resolved_reference_source == NPUBENCH:
        print(f"  npubench task     : {npubench_args.task_path}")
        print(f"  npubench root     : {npubench_args.root_path}")
        print("  npubench format   : original .py + same-stem JSON/JSONL sidecar")
    else:
        print("  CannBench         : provider interface reserved; evaluator unavailable")
    print(f"  A5 build lane     : NPU {lane}")
    if extra_lanes:
        print(f"  npubench perf lane: NPU {extra_lanes[0]} (leased if safe)")
    print("  opgen_mode        : port_a3_to_a5")
    print(f"  archive target    : {archive_root}/{op_name}/")
    print("  phases (planned)  :")
    print("    O0 preflight    : standard")
    print("    O1 config       : opgen_mode=port_a3_to_a5 propagated via AscendCEnv (W2)")
    print(
        "    O2 source sync  : fixed arch35 TileLang2AscendC source stage prepared"
        if tilelang_source else "    O2 source sync  : fixed arch22 source stage prepared"
    )
    if resolved_reference_source == A3_LIVE:
        print("    O2.5 reference  : live arch22 reference capture")
        print("    O5 verify       : truth = fresh live arch22 reference output")
    elif resolved_reference_source == NPUBENCH:
        print("    O2.5 reference  : immutable original NPUKernelBench task preflight")
        print("    O5 verify       : truth = frozen NPUKernelBench task (no A3 truth)")
    else:
        print("    O2.5 reference  : persist UNSUPPORTED_REFERENCE_SOURCE")
        print("    O5 verify       : unavailable until CannBench evaluator is implemented")
    print("    O3 PROGRESS     : standard")
    print(
        "    O4 kw spawn     : independent TileLang2AscendC project candidate"
        if tilelang_source else "    O4 kw spawn     : independent arch35 implementation from arch22 semantics"
    )
    print(
        "    O6 archive      : TileLang2AscendC kernel project layout writer"
        if tilelang_source else "    O6 archive      : ops-nn mirror layout writer"
    )
    print("=" * 72)

    # A plan is validation and presentation only: no cold-start reset, no
    # workspace creation, and no source/reference staging or state mutation.
    if plan_only:
        return 0

    # P87+ (2026-05-15): cold-start reset MUST happen BEFORE workspace seeding.
    # The reset backs up .opgen_state.json and op_classification.json; if we
    # seed those first, the reset wipes them, and run_single_op then can't
    # detect migration mode (and would otherwise fail closed as unsupported).
    workspace_dir_early = _orch().WORKSPACE_ROOT / op_name
    if cold_start and workspace_dir_early.exists():
        _cold_start_reset_workspace(workspace_dir_early)

    # W12 (2026-05-12, ROADMAP §1.5): seed op_classification.json with the
    # a3_to_a5_port tag so kb_manifest_block auto-loads W8-W11 KB entries
    # without waiting for the (slower, LLM-driven) /aog-op-classify pass.
    # This is the "known classification" shortcut for the port-mode path —
    # the CLI flag IS the classification.
    # Use direct path (not _resolve_workspace) — workspace may not exist yet
    # and _resolve_workspace's iterdir-fallback errors on missing WORKSPACE_ROOT.
    workspace_dir = _orch().WORKSPACE_ROOT / op_name
    workspace_dir.mkdir(parents=True, exist_ok=True)
    state_path = workspace_dir / ".opgen_state.json"
    # Explicit source-kind bindings are immutable for the lifetime of a
    # workspace.  In particular, do this check before staging so a second CLI
    # invocation cannot replace the previously bound snapshot and only then
    # discover that it disagrees with durable state.  --resume uses durable
    # state directly and never arrives through this constructor.
    if state_path.exists() or state_path.is_symlink():
        try:
            metadata = state_path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError("durable state is not a regular file")
            existing_pre_stage = json.loads(state_path.read_text(encoding="utf-8"))
            if not isinstance(existing_pre_stage, dict):
                raise ValueError("durable state is not a JSON object")
        except Exception as exc:
            print(f"ERROR: existing workspace state is unreadable: {exc}")
            return 2
        existing_port_source = existing_pre_stage.get("port_source")
        existing_kind = (
            existing_pre_stage.get("source_kind")
            or (
                existing_port_source.get("kind")
                if isinstance(existing_port_source, dict) else None
            )
        )
        # A source kind is not a mutable option.  In particular a legacy
        # workspace cannot become TileLang2AscendC merely by repeating the
        # start CLI, and a TileLang2AscendC workspace cannot fall back to a
        # different route.  Cold-start has already archived the old durable
        # state before this point and is the sole explicit reset mechanism.
        if not cold_start and (
            tilelang_source
            or existing_kind in {
                TILELANG2ASCENDC_SOURCE_KIND,
                TILELANG2ASCENDC_SOURCE_KIND_ALIAS,
            }
        ):
            conflict_code = "TILELANG2ASCENDC_SOURCE_STATE_CONFLICT"
            print(
                f"ERROR: {conflict_code}: existing workspace source binding is immutable; "
                "use --resume or --cold-start"
            )
            return 2
    try:
        if tilelang_source:
            (
                _detect_tilelang_source,
                _logical_tilelang_op,
                stage_tilelang_source_tree,
                _verify_tilelang,
            ) = _tilelang2ascendc_source_api()
            source_stage = stage_tilelang_source_tree(port_a3_dir, workspace_dir)
        else:
            source_stage = stage_source_tree(port_a3_dir, workspace_dir)
    except Exception as exc:
        source_label = (
            "arch35 TileLang2AscendC"
            if tilelang_source
            else "arch22"
        )
        print(f"ERROR: could not create source-only {source_label} snapshot: {exc}")
        return 2
    reference_stage = None
    if resolved_reference_source == NPUBENCH:
        try:
            reference_stage = stage_npubench_inputs(
                workspace_dir,
                npubench_task=npubench_args.task_path,
                npubench_root=npubench_args.root_path,
            )
        except NpubenchInputError as exc:
            print(f"ERROR: could not stage NPUKernelBench task: {exc}")
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
        if tilelang_source:
            # TileLang2AscendC is a target-format custom-op project.  Keep its
            # classification independent from the legacy arch22 source
            # architecture gates.
            op_class_tags = ["a3_to_a5_port", TILELANG2ASCENDC_SOURCE_KIND]
            op_complexity = None
            raise StopIteration
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
    except StopIteration:
        pass
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
    state_payload = {
        "schema_version": 3,
        "op": op_name,
        "target": env.target,
        "lane": lane,
        "opgen_mode": "port_a3_to_a5",
        "graybox_sandbox": True,
        "source_arch": "arch35" if tilelang_source else "arch22",
        "target_arch": "arch35",
        "source_arch_detection": source_detection.state_payload(),
        "started_ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "last_seen_ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "invocation_count": 1,
    }
    if tilelang_source:
        from tilelang2ascendc_source import tilelang2ascendc_state_block

        state_payload.update({
            "source_kind": TILELANG2ASCENDC_SOURCE_KIND,
            "port_source": tilelang2ascendc_state_block(source_stage),
            "source_format": "tilelang2ascendc_task",
            "candidate": {
                "kind": TILELANG2ASCENDC_CANDIDATE_KIND,
                "target_arch": "arch35",
            },
            "port_a3_source": str(source_stage.root),
            "source_stage_manifest": str(source_stage.manifest),
            "source_stage_digest": source_stage.digest,
            "source_stage_file_count": source_stage.file_count,
            "graybox_source_dir": str(source_stage.root),
        })
    else:
        state_payload.update({
            "port_a3_source": str(source_stage.root),
            "source_stage_manifest": str(source_stage.manifest),
            "source_stage_digest": source_stage.digest,
            "source_stage_file_count": source_stage.file_count,
            "graybox_arch22_dir": str(source_stage.root),
        })
    # If state file exists, preserve started_ts + invocation_count.  A
    # cold-start is deliberately different: `_cold_start_reset_workspace()`
    # archives the complete durable state, then may leave a minimal survivor
    # containing only `lifetime_spawn_count`.  That survivor is accounting
    # metadata, not a previous source/reference binding; treating it as the
    # latter would reject every cold-start with a nonzero prior spawn count.
    # Never carry semantic fields from it into the fresh run.
    prior_state, rc = _apply_prior_durable_state(
        state_path, cold_start=cold_start, state_payload=state_payload
    )
    if rc != 0:
        return rc
    prior_reference = prior_state.get("reference")
    if isinstance(prior_reference, dict):
        state_payload["reference"] = prior_reference
    rc = _bind_reference_source(state_payload, resolved_reference_source, reference_stage)
    if rc != 0:
        return rc
    try:
        atomic_write_state(workspace_dir, state_payload)
    except NpubenchInputError as exc:
        print(f"ERROR: could not atomically write reference state: {exc}")
        return 2
    stage_reason = "not required for frozen benchmark provider"
    if tilelang_source:
        valid_stage, stage_reason, _stage_manifest = _verify_tilelang2ascendc_source_binding(
            workspace_dir, state_payload
        )
        if not valid_stage:
            print(f"ERROR: source-only snapshot state validation failed: {stage_reason}")
            return 2
    elif resolved_reference_source in {A3_LIVE}:
        valid_stage, stage_reason, _stage_manifest = verify_source_stage(workspace_dir, state_payload)
        if not valid_stage:
            print(f"ERROR: source-only snapshot state validation failed: {stage_reason}")
            return 2
    if resolved_reference_source == A3_LIVE:
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

    # End-to-end live run (post-W14b + W15 wiring): invoke run_single_op
    # which dispatches through phase_o25_a3_ref + state machine + finalize.
    # run_single_op's _opgen_state_path reader picks up our just-written
    # mode + source; W15 dispatch routes Phase O2.5 to the A3-CANN variant.
    print("\n=== entering run_single_op (full state-machine run) ===")
    # Keep the original checkout out of durable graybox state.  Phase O2.5
    # consumes this process-scoped value before any worker spawn and uses it
    # only when shipped-CANN dispatch proves a source package build is needed.
    if resolved_reference_source != A3_LIVE:
        print(
            "\n=== entering run_single_op "
            f"({resolved_reference_source} state-machine run) ==="
        )
        return _orch().run_single_op(
            op_name,
            workspace=workspace_dir,
            lane=lane,
            plan_only=False,
            cap_bumps=cap_bumps,
            timing=timing,
            extra_lanes=list(extra_lanes or []),
        )

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
            extra_lanes=list(extra_lanes or []),
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
