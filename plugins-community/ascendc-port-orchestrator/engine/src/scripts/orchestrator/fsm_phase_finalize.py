# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""fsm_phase_finalize.py — the `finalize` FSM state handler (DEBT-201 split).

Extracted VERBATIM from run_single_op's `if snap.current_state == "finalize":`
branch (orchestrator.py) as part of the god-function decomposition via
dependency inversion. The slice is entered once per loop iteration when the FSM
is at the `finalize` state; it runs:

  Phase O5 independent post-verify (precision re-measure) + rollback routing
  → phase_o5_perf_capture (independent perf re-measure)
  → pre-finalize audit-artifact production + GE op_host assembler + schema_norm
  → batch precheck + P0ff eligibility gate + DEBT-192 loop-break
  → finalize_pipeline.finalize_op (archive promotion)
  → P0aax KB-manager merge gate → finalize→done routing.

DEPENDENCY CONTRACT (why the monkeypatch surface still bites):
- Orchestrator-MODULE-LEVEL deps (_ensure_audit_artifacts) are reached via
  `ctx.<name>` (read-through property resolving the live orchestrator module),
  so `monkeypatch.setattr(orchestrator, X)` bites. Per-run inputs
  (op/workspace/lane/extra_lanes/runtime_kwargs) come off `ctx`.
- SIBLING modules (phase_o5, phase_o5_runner, phase_o5_perf_capture,
  finalize_pipeline, schema_norm, kb_invoke, events, state_executor)
  are imported directly here — the SAME module objects the tests patch, so
  `monkeypatch.setattr(<sibling>, ...)` bites the direct reference with no
  indirection.

The block BODY is byte-identical to the original; the only mechanical edits are:
`op`→`ctx.op`, `workspace`→`ctx.workspace`, `lane`→`ctx.lane`,
`extra_lanes`→`ctx.extra_lanes`, `runtime_kwargs`→`ctx.runtime_kwargs`,
`_ensure_audit_artifacts`→`ctx._ensure_audit_artifacts`, and each `continue` /
`return N` → `return HandlerResult.cont()` / `.ret(N)`.
"""
from __future__ import annotations
import logging
import time

import functools
import importlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Optional

import events
import finalize_pipeline
import kb_invoke
import perf_irm_provenance
import phase_o5
import phase_o5_perf_capture
import schema_norm
import state_executor
from logging_config import get_logger
from orchestrator_coldstart import _prepare_npubench_candidate_repair

from fsm_context import HandlerResult, OrchestratorContext

log = get_logger(__name__)


# DEBT-O5-INFRA-0-MISCLASSIFY (2026-07-20): an infra-class O5 RUNNER_FAILED
# (e.g. a stale/missing NPU_PYTHON_BIN → exit-127 `<bin>/python3: No such file
# or directory`) means the RUNNER broke, not the kernel — the artifact is
# intact. Rolling back to await_worker + respawning the base worker is
# pointless (nothing to re-emit) and burns a real author iteration, so instead
# we RE-ATTEMPT the O5 re-measure in place, BOUNDED by this cap so a genuinely
# stuck infra issue still falls through to the await_worker rollback (fail
# loud, never silent). The bound is an in-call loop count — it CANNOT loop
# forever within a finalize iteration; cross-iteration progress stays guarded
# by the existing await_worker iter_cap loop-guard. The O5 gate is NOT
# skipped: each re-attempt runs the full, real post_verify_for_finalize.
_O5_INFRA_MAX_RETRIES = 2

# Host-level transient windows (2026-08-22 A5 campaign: 507035 copy faults +
# wedged npu-smi + msprof stalls that come and go every ~15-30 min) make an
# immediate O5 re-attempt land in the SAME bad window.  Back off between
# infra retries so the second attempt has a real chance to catch a clean
# window.  Env-overridable; tests patch _sleep.
_O5_INFRA_RETRY_BACKOFF_ENV = "CANNBOT_O5_INFRA_RETRY_BACKOFF_SEC"
_O5_INFRA_RETRY_BACKOFF_SECONDS = 600
_sleep = time.sleep


def _o5_infra_retry_backoff_seconds() -> int:
    raw = os.environ.get(_O5_INFRA_RETRY_BACKOFF_ENV)
    if raw is None:
        return _O5_INFRA_RETRY_BACKOFF_SECONDS
    try:
        value = int(raw.strip())
    except ValueError:
        return _O5_INFRA_RETRY_BACKOFF_SECONDS
    return max(value, 0)


# --- Host copy-fault escalation guard (2026-08-27, 42_CoTAttention) -----------
# torch_npu launches ops ASYNCHRONOUSLY: a candidate device-kernel hard fault
# (e.g. aivec error 340 "VEC to access UB is not aligned" wedging the device,
# surfaced as 507035/507015 device error type 3) only reports at the NEXT host
# op — frequently an H2D/D2H copy — so the measured precision `reason` carries
# the copy-fault markers below even when the candidate's own kernel caused it.
# On 2026-08-27 the 42_CoTAttention candidate reproduced such a fault
# DETERMINISTICALLY on a healthy device (per-launch aclrtSynchronizeStream
# bisect: cot_softmax_f32), while the transient classification looped 10+
# identical O5 failures and explicitly told the worker "INFRA, do not edit the
# candidate" — the exact opposite of the needed repair.
# Guard: persist a per-candidate-binding counter; once the same binding trips
# the copy-fault classification more than the cap, stop re-attempting as infra
# and let the MISMATCH roll back to await_worker with a deterministic-fault
# debugging note.
_HOST_COPY_FAULT_MAX_TRANSIENT_ENV = "AOG_O5_HOST_COPY_FAULT_MAX_TRANSIENT"
_HOST_COPY_FAULT_MAX_TRANSIENT = 2
_HOST_COPY_FAULT_ESCALATION_FILE = ".o5_host_copy_fault_escalation.json"


def _host_copy_fault_max_transient() -> int:
    raw = os.environ.get(_HOST_COPY_FAULT_MAX_TRANSIENT_ENV)
    if raw is None:
        return _HOST_COPY_FAULT_MAX_TRANSIENT
    try:
        value = int(raw.strip())
    except ValueError:
        return _HOST_COPY_FAULT_MAX_TRANSIENT
    return max(value, 0)


def _candidate_tree_from_report(report: object) -> Optional[str]:
    """Return a candidate-tree digest from one evaluation report section."""
    if not isinstance(report, dict):
        return None
    binding = report.get("evaluation_binding")
    if not isinstance(binding, dict):
        return None
    tree = binding.get("candidate_tree_sha256")
    return str(tree) if tree else None


def _o5_candidate_tree_key(o5) -> str:
    """Identity of the candidate TREE under evaluation, stable across evals.

    2026-08-27 (PR875 3_FusionAttention oc line, equiv analysis P1-b): keying
    the copy-fault escalation on the precision ``binding_sha256`` never works —
    every O5 evaluation mints a fresh binding, so the per-binding count is
    always 1 and the same wedging candidate tree retries as "transient"
    forever.  Key on ``candidate_tree_sha256`` instead: identical trees keep
    accumulating, a worker re-author (new tree) resets the counter.  Fall back
    to the per-eval binding only when no tree digest is reachable.
    """
    measured = getattr(o5, "measured", None)
    if isinstance(measured, dict):
        for section in ("precision", "performance", "evaluate"):
            tree = _candidate_tree_from_report(measured.get(section))
            if tree:
                return tree
        precision = measured.get("precision")
        if isinstance(precision, dict):
            binding = precision.get("binding_sha256")
            if binding:
                return str(binding)
    return "unknown"


def _bump_host_copy_fault_escalation(workspace: Path, o5) -> int:
    """Increment and return the per-candidate-tree copy-fault count."""
    path = workspace / _HOST_COPY_FAULT_ESCALATION_FILE
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            data = {}
    except Exception:
        data = {}
    key = _o5_candidate_tree_key(o5)
    entry = data.get(key)
    count = int(entry.get("count", 0)) + 1 if isinstance(entry, dict) else 1
    data[key] = {"count": count, "last_ts": time.time()}
    try:
        path.write_text(json.dumps(data, indent=1, sort_keys=True))
    except OSError:
        pass
    return count


def _mark_host_copy_fault_deterministic(ctx: OrchestratorContext, o5, count: int):
    """Stop calling a recurring copy fault transient; hand the worker the truth."""
    note = (
        f"host copy-fault classification fired {count}x for the SAME candidate "
        "tree — a transient host window does not reproduce byte-identically. "
        "Treat as a CANDIDATE device-kernel hard fault surfaced asynchronously "
        "at a copy op (torch_npu launches are async; a 507035/507015 at an H2D "
        "copy means an EARLIER launch wedged the device). Debug recipe: re-run "
        "the eval child with ASCEND_LAUNCH_BLOCKING=1 for true attribution, "
        "then bisect with per-launch aclrtSynchronizeStream. 2026-08-27 "
        "42_CoTAttention precedent: wedging kernel was cot_softmax_f32 (aivec "
        "error 340, VEC UB access not aligned; PB-21 pure-PipeBarrier pattern). "
        "This is NOT infra — do NOT wait out a window; fix the candidate."
    )
    log.error("O5 host copy-fault escalation: %s", note)
    events.emit(
        ctx.workspace,
        "orchestrator.phase_o5_host_copy_fault_deterministic",
        lane=ctx.lane,
        data={"count": count, "summary": o5.summary[:200]},
    )
    return phase_o5.O5Report(
        verdict="MISMATCH",
        summary=(o5.summary + " | " + note)[:2000],
        truth_source=getattr(o5, "truth_source", "unresolved"),
        mismatches=list(getattr(o5, "mismatches", None) or []),
        claimed=getattr(o5, "claimed", None) or {},
        measured=getattr(o5, "measured", None) or {},
        harness_git_state=getattr(o5, "harness_git_state", None),
        harness_dirty=getattr(o5, "harness_dirty", False),
    )


def _load_ascendc_static_checker():
    """Load the existing standalone checker without widening ``sys.path``."""
    checker_path = Path(__file__).resolve().parent.parent / "ascendc_static_check.py"
    spec = importlib.util.spec_from_file_location(
        "_aog_finalize_ascendc_static_check", checker_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load static checker from {checker_path}")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    return checker


def _is_delivery_cpp_source(path: Path, root: Path) -> bool:
    """Exclude generated/build/backup files beneath a declared source root."""
    if path.is_symlink() or not path.is_file() or path.suffix not in (".h", ".cpp"):
        return False
    try:
        relative = path.relative_to(root)
    except ValueError:
        return False
    for part in relative.parts[:-1]:
        if _is_generated_or_backup_dir(part):
            return False
    return True


def _is_generated_or_backup_dir(part: str) -> bool:
    """Return whether ``part`` identifies an excluded generated subtree."""
    lowered = part.lower()
    return (
        part.startswith(".")
        or lowered in {"build", "build_out", "cmakefiles", "__pycache__"}
        or lowered.startswith("cmake-build")
        or lowered.endswith("_bak")
    )


def _delivery_cpp_roots(workspace: Path) -> list[Path]:
    """Return only this plugin's declared, present AscendC delivery roots.

    Never fall back to walking the whole workspace: it can contain staged
    upstream sources, build trees, or other prior-art that is not this run's
    deliverable.  Missing roots are left to the existing completeness and
    provenance eligibility gates.
    """
    plugin = getattr(finalize_pipeline, "_get_active_plugin")(workspace)
    if plugin is None:
        return []
    workspace_aware = getattr(plugin, "kernel_cpp_dirs_for_workspace", None)
    declared = (
        workspace_aware(workspace)
        if callable(workspace_aware)
        else plugin.kernel_cpp_dirs()
    )
    workspace_real = workspace.resolve()
    roots: list[Path] = []
    for relative in declared:
        candidate = (workspace / relative).resolve()
        try:
            candidate.relative_to(workspace_real)
        except ValueError:
            log.warning("static check: ignoring delivery root outside workspace: %s", relative)
            continue
        if not candidate.is_dir():
            continue
        # The existing checker supports .h/.cpp.  Do not call it on an empty
        # root because its all-green empty report is not useful evidence.
        if any(_is_delivery_cpp_source(path, candidate) for path in candidate.rglob("*")):
            roots.append(candidate)
    return roots


def _run_delivery_static_check(workspace: Path) -> dict:
    """Run the existing checker on current-plugin delivery roots only.

    Checker load/runtime/report errors fail closed.  A workspace with no
    declared present source root is explicitly skipped instead of scanning
    unrelated files; existing finalize eligibility owns missing-deliverable
    rejection.
    """
    try:
        roots = _delivery_cpp_roots(workspace)
        if not roots:
            return {
                "passed": True,
                "skipped": "no declared present AscendC delivery source",
                "reports": [],
            }
        checker = _load_ascendc_static_checker()
        reports = []
        for root in roots:
            # ``run_checks`` normally walks recursively.  Feed it the same
            # check registry and prestage-aware file collector, narrowed to
            # current delivery source (not build products or stale backups).
            collect_files = checker.collect_files
            delivery_files = [
                path for path in collect_files(str(root))
                if _is_delivery_cpp_source(Path(path), root)
            ]

            def collect_delivery_files(_directory, files=tuple(delivery_files)):
                """Return the already scoped files for this delivery root."""
                return list(files)

            checker.collect_files = collect_delivery_files
            try:
                report = checker.run_checks(str(root))
            finally:
                checker.collect_files = collect_files
            if not isinstance(report, dict) or not isinstance(report.get("passed"), bool):
                raise RuntimeError(f"malformed static-check report for {root}")
            reports.append({"root": str(root), "report": report})
        return {
            "passed": all(item["report"]["passed"] for item in reports),
            "reports": reports,
        }
    except Exception as exc:
        return {"passed": False, "error": f"{type(exc).__name__}: {exc}", "reports": []}


def handle_finalize(ctx: OrchestratorContext, snap) -> HandlerResult:
    """Run the `finalize` state slice for one loop iteration.

    Thin orchestrator over the finalize sub-phases (each <200 lines per the
    architecture-lint god-function bar). Returns a HandlerResult telling the
    driver loop to `continue` (re-snapshot) or `return <exit_code>`.
    Any sub-phase may short-circuit by returning a HandlerResult; a None means
    "proceed to the next sub-phase".
    """
    # Phase O5 independent post-verify + rollback routing.
    _r = _o5_post_verify(ctx, snap)
    if _r is not None:
        return _r
    # Independent perf re-measure (best-effort; never short-circuits).
    _run_perf_capture(ctx)
    # Pre-finalize artifact production (audit + GE op_host + schema_norm) and
    # batch precheck; returns the aggregated multi-failure reason (or None).
    precheck_all_reasons = _run_finalize_prep(ctx)
    # P0ff eligibility gate + DEBT-192 loop-break rollback routing.
    _r = _check_eligibility_and_rollback(ctx, snap, precheck_all_reasons)
    if _r is not None:
        return _r
    # Archive promotion + P0aax KB-merge gate + finalize→done routing.
    return _promote_and_route(ctx)


def _o5_runner_for_workspace(workspace: Path, *, extra_lanes: Optional[list[int]] = None):
    """Reload the O5 runner and select its truth-source-specific entrypoint."""
    truth_source = phase_o5.expected_truth_source(workspace)
    if truth_source == "npubench":
        # Import no legacy O5 transport on this route: the NPUKernelBench
        # provider owns both its immutable task oracle and target evaluation.
        from npubench import npubench_o5_runner

        return functools.partial(
            npubench_o5_runner.npubench_verify_runner,
            extra_lanes=tuple(extra_lanes or ()),
        )
    import phase_o5_runner

    importlib.reload(phase_o5_runner)
    if truth_source == "backward_autograd":
        return phase_o5_runner.backward_verify_runner
    return phase_o5_runner.ssh_runner


def _run_o5_verification(ctx: OrchestratorContext, runner):
    """Run the independent O5 verifier through the selected runner."""
    return phase_o5.post_verify_for_finalize(
        ctx.workspace, ctx.op, lane=ctx.lane, runner=runner,
    )


def _retry_infra_o5_verification(ctx: OrchestratorContext, o5, runner, backoff_seconds: Optional[int] = None):
    """Retry only bounded infrastructure failures without consuming an iteration."""
    infra_retries = 0
    while (
        o5.verdict == "RUNNER_FAILED"
        and getattr(o5, "rollback_kind", None) == "infra"
        and infra_retries < _O5_INFRA_MAX_RETRIES
    ):
        infra_retries += 1
        log.warning(
            f"O5 RUNNER_FAILED (infra): {o5.summary[:160]} — re-attempting O5 "
            f"in place (retry {infra_retries}/{_O5_INFRA_MAX_RETRIES}); "
            "artifact intact, no worker respawn."
        )
        events.emit(
            ctx.workspace,
            "orchestrator.phase_o5_infra_retry",
            lane=ctx.lane,
            data={
                "attempt": infra_retries,
                "max": _O5_INFRA_MAX_RETRIES,
                "summary": o5.summary[:200],
            },
        )
        # P0-1 device-class split (2026-08-28): callers that already know the
        # failure is a device-class error pass backoff_seconds=0 — device
        # errors do not self-heal, so the 600s transient-window wait is pure
        # budget burn (3FA 507015: 4x600s over 95 minutes, zero recovery).
        backoff = _o5_infra_retry_backoff_seconds() if backoff_seconds is None else max(backoff_seconds, 0)
        if backoff > 0:
            log.info(
                "O5 infra retry backoff: sleeping %ss before re-attempt "
                "(transient host window avoidance)", backoff
            )
            _sleep(backoff)
        o5 = _run_o5_verification(ctx, runner)
        # The retry may be the first attempt that reaches the authored
        # candidate validator (for example after a transient target-build
        # error).  Preserve that structured repair signal immediately instead
        # of allowing another infra retry to erase it.
        if (
            o5.verdict == "RUNNER_FAILED"
            and _is_direct_npubench_candidate_failure(ctx.workspace)
            and _is_repairable_npubench_candidate_failure(o5)
        ):
            break
    return o5


def _mismatch_at_worker_cap(
    ctx: OrchestratorContext, snap, o5,
) -> Optional[HandlerResult]:
    """Stop an O5 mismatch before a capped worker would create a loop."""
    workspace = ctx.workspace
    if not state_executor.at_iter_cap(workspace, "await_worker"):
        return None
    count = snap.iter_counts.get("worker", 0)
    cap = state_executor.iter_cap("await_worker", workspace=workspace)
    log.critical(
        f"O5 MISMATCH while await_worker already at iter_cap ({count}/{cap}). "
        f"Cannot progress — routing back would infinite-loop. Manual intervention "
        f"required: investigate why measured ({o5.measured}) differs from claim "
        f"({o5.claimed}); possibilities: (a) stale on-A5 binary not rebuilt, "
        f"(b) re-runner uses different inputs than worker, (c) genuine "
        f"non-determinism worker didn't see, (d) worker fabricated case_verdicts. "
        f"Diagnostic: ls workspace/{ctx.op}/kernel/build/ + diff "
        f"workspace/{ctx.op}/pass_a_results.json vs verification.json."
    )
    events.emit(
        workspace,
        "orchestrator.fsm_loop_guard_o5_mismatch",
        lane=ctx.lane,
        data={
            "worker_count": count,
            "worker_cap": cap,
            "claimed": o5.claimed,
            "measured": o5.measured,
        },
    )
    return HandlerResult.ret(2)


def _clear_harness_build_artifacts(workspace: Path) -> list[str]:
    """Remove harness-owned build/evaluation artifacts before an O5 rollback
    to ``await_worker`` (2026-08-21, graybox respawn fix).

    The O5 controlled build compiles the candidate into kernel/build/ and the
    npubench evaluation stages a built copy under .npubench_exec/ (and a
    candidate stage under .npubench_candidate/).  Those trees contain .o/.so
    binaries that the graybox construction scan classifies as assembled-answer
    artifacts, so a worker respawn with them present is rejected before it can
    run ("graybox construction rejected target/answer-bearing curated input").
    The next worker iteration regenerates the candidate from its own authored
    sources; these built outputs are harness-owned, not worker state, so
    removing them is the same clean-slate contract as cold-start.
    npubench_evidence/ (the durable reports the next iteration's brief
    references, including preflight_target_receipt.json) is preserved here and
    by the candidate-repair reset in
    orchestrator_coldstart._prepare_npubench_candidate_repair.
    """
    removed: list[str] = []
    for rel in ("kernel/build", ".npubench_exec", ".npubench_candidate"):
        p = workspace / rel
        try:
            if p.is_symlink() or p.is_file():
                p.unlink()
                removed.append(rel)
            elif p.is_dir():
                shutil.rmtree(p)
                removed.append(rel)
        except OSError as exc:
            log.warning(f"O5 rollback: failed to remove {rel}: {exc}")
    for so in sorted(workspace.glob("*.so")):
        if not (so.is_file() or so.is_symlink()):
            continue
        # 2026-08-25: mirror the cold-start pattern — move to a backup dir
        # (recoverable) instead of unlink (permanent).  The backup root follows
        # the same convention as orchestrator_coldstart (env override, else
        # ~/.opgen_backups/<workspace>) and is deliberately OUTSIDE the
        # worker-visible workspace: an in-workspace copy would be a restore
        # vector and would trip the graybox construction scan on respawn.
        backup_root_env = (
            os.environ.get("NPUBENCH_REPAIR_BACKUP_ROOT")
            or os.environ.get("COLD_START_BACKUP_ROOT")
        )
        backup_root = (
            Path(backup_root_env) / workspace.name
            if backup_root_env
            else Path.home() / ".opgen_backups" / workspace.name
        )
        backup_dir = backup_root / f"o5-rollback-{int(time.time())}"
        try:
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(so), str(backup_dir / so.name))
            removed.append(so.name)
        except OSError as exc:
            log.warning(f"O5 rollback: failed to back up {so.name}: {exc}")
    if removed:
        log.info(f"O5 rollback: cleared harness build artifacts: {', '.join(removed)}")
    return removed


def _rollback_o5_mismatch(workspace: Path, o5) -> HandlerResult:
    """Record the existing mismatch rollback to ``await_worker``."""
    _clear_harness_build_artifacts(workspace)
    rollback_state = "await_worker"
    log.info(f"O5 MISMATCH rollback target: {rollback_state}")
    state_executor.record_transition(
        workspace,
        state_executor.TransitionDecision(
            next_state=rollback_state,
            matched_transition_index=-1,
            rationale=(
                f"P0kk O5 MISMATCH: {len(o5.mismatches)} count discrepancies — "
                f"rollback to {rollback_state} to re-iterate emission"
            ),
            from_state="finalize",
            handoff="",
        ),
    )
    finalize_pipeline.record_rollback(
        workspace,
        rollback_state=rollback_state,
        reason=f"P0kk O5 MISMATCH: {o5.summary}",
        gate="phase_o5_mismatch",
    )
    return HandlerResult.cont()


# ---------------------------------------------------------------------------
# Near-miss fingerprint admission (DSH ruling §4.c layer 3, 2026-08-28)
# ---------------------------------------------------------------------------
# npubench_runner's precision report carries a per-case `repeat_fingerprint`
# (class ∈ deterministic-fail / bimodal / stable-pass / reference-unstable).
# A case whose verdict flips across repeats (bimodal) or whose reference is
# itself unstable on device (reference-unstable) must NEVER be accepted via
# the near-miss rule — the evidence needed to judge "ULP-level near miss" is
# not trustworthy for those cases.  Legacy reports without the field keep the
# existing behavior (admission undecided here, no blocking).
_NEAR_MISS_BLOCKED_FINGERPRINT_CLASSES = frozenset({"bimodal", "reference-unstable"})
_PRECISION_REPORT_REL = Path("npubench_evidence") / "precision_report.json"


def _repeat_fingerprint_classes(workspace: Path) -> dict[int, str]:
    """Per-case repeat-fingerprint classes from the precision report.

    Returns {} when the report or the per-case field is absent (legacy
    reports) or unreadable — callers treat {} as "no fingerprint evidence",
    never as "all admissible" or "all blocked".
    """
    try:
        report = json.loads((workspace / _PRECISION_REPORT_REL).read_text())
    except Exception:
        return {}
    if not isinstance(report, dict):
        return {}
    cases = report.get("cases")
    if not isinstance(cases, list):
        return {}
    classes: dict[int, str] = {}
    for position, case in enumerate(cases):
        if not isinstance(case, dict):
            continue
        fingerprint = case.get("repeat_fingerprint")
        if not isinstance(fingerprint, dict):
            continue
        klass = fingerprint.get("class")
        if not isinstance(klass, str) or not klass:
            continue
        idx = case.get("case_idx", case.get("index", position))
        try:
            classes[int(idx)] = klass
        except (TypeError, ValueError):
            continue
    return classes


def _near_miss_inadmissible_cases(workspace: Path) -> dict[int, str]:
    """Cases that must not be accepted via near-miss (bimodal/ref-unstable)."""
    return {
        idx: klass
        for idx, klass in _repeat_fingerprint_classes(workspace).items()
        if klass in _NEAR_MISS_BLOCKED_FINGERPRINT_CLASSES
    }


def near_miss_case_admitted(workspace: Path, case_idx: int) -> bool:
    """Fingerprint admission predicate for the near-miss acceptance rule.

    False only when fingerprint evidence positively marks the case bimodal or
    reference-unstable; a missing fingerprint field (legacy report) preserves
    the pre-existing behavior (True).
    """
    klass = _repeat_fingerprint_classes(workspace).get(case_idx)
    return klass not in _NEAR_MISS_BLOCKED_FINGERPRINT_CLASSES


def _handle_o5_mismatch(
    ctx: OrchestratorContext, snap, o5,
) -> HandlerResult:
    """Emit an O5 mismatch and either halt at cap or roll back the worker."""
    print(phase_o5.format_block_message(ctx.op, o5))
    # Near-miss fingerprint admission (P0-3 / DSH §4.c): cases classified
    # bimodal or reference-unstable by the evaluation repeat fingerprint are
    # recorded in the durable event trail so a later near-miss acceptance
    # decision (user adjudication at await_user_decision) cannot treat them as
    # admissible.  Missing fingerprint field (legacy report) → empty, behavior
    # unchanged.
    inadmissible = _near_miss_inadmissible_cases(ctx.workspace)
    if inadmissible:
        log.warning(
            "repeat_fingerprint near-miss admission gate: cases %s are "
            "bimodal/reference-unstable — NOT admissible for near-miss "
            "acceptance regardless of MERE margin",
            inadmissible,
        )
    events.emit(
        ctx.workspace,
        "orchestrator.phase_o5_block",
        lane=ctx.lane,
        data={
            "verdict": o5.verdict,
            "claimed": o5.claimed,
            "measured": o5.measured,
            "mismatches": o5.mismatches,
            "near_miss_inadmissible_cases": {str(k): v for k, v in inadmissible.items()},
        },
    )
    cap_result = _mismatch_at_worker_cap(ctx, snap, o5)
    if cap_result is not None:
        return cap_result
    return _rollback_o5_mismatch(ctx.workspace, o5)


def _record_o5_provisional(ctx: OrchestratorContext, o5) -> None:
    """Persist the dirty-harness O5 finding without rolling back."""
    log.warning(f"O5 post-verify: {o5.summary}")
    events.emit(
        ctx.workspace,
        "orchestrator.phase_o5_provisional",
        lane=ctx.lane,
        data={
            "verdict": o5.verdict,
            "harness_git_state": o5.harness_git_state,
            "harness_dirty": o5.harness_dirty,
            "claimed": o5.claimed,
            "measured": o5.measured,
        },
    )
    phase_o5.record_harness_state(ctx.workspace, o5)


_HOST_TRANSIENT_RECLASSIFY_PREFIX = "host-transient copy fault in measured precision: "


def _retry_host_transient_mismatch(ctx: OrchestratorContext, o5, runner):
    """Re-run a host-transient MISMATCH through the infra retry loop."""
    # The measured precision failed in the host H2D copy path, not in the
    # candidate.  Roll the transient failure into the infra retry loop
    # (with backoff) instead of routing a host fault to await_worker.
    log.warning(
        "O5 MISMATCH is host-transient (copy fault in measured precision); "
        "re-attempting in place as infra: %s",
        o5.summary[:160],
    )
    reclassified = phase_o5.O5Report(
        verdict="RUNNER_FAILED",
        summary=_HOST_TRANSIENT_RECLASSIFY_PREFIX + o5.summary,
        truth_source=getattr(o5, "truth_source", "npubench"),
        rollback_kind="infra",
    )
    return _retry_infra_o5_verification(ctx, reclassified, runner)


# Device-class errors split OUT of the host-transient text classification
# (DSH ruling P0-1 §4.b, 2026-08-28): 5070xx device errors / aivec errcodes
# are not transient host windows.  They do not self-heal (2026-08-27 3FA:
# 507015 wedged device 3 for 95 minutes while the engine slept 4x600s), so
# the retry backoff is 0 and the same-signature parking threshold is 2
# (state_executor.DEVICE_SIGNATURE_PARK_THRESHOLD) instead of 3.
_DEVICE_ERROR_CODE_RE = re.compile(r"\b5070\d{2}\b")
_DEVICE_ERROR_MARKERS = ("aivec error", "device error type")

# Summary prefix used when a device-class MISMATCH is rolled into the infra
# retry loop; lets the settled report be classified back as device-class even
# after the measured payload is out of scope.
_DEVICE_RECLASSIFY_PREFIX = "device-class copy fault in measured precision: "


def _mismatch_is_device_error(o5) -> bool:
    """Whether a MISMATCH carries a device-error signature (5070xx / aivec).

    Checked BEFORE the host-transient copy markers: an async device fault
    surfaces at the next host copy op, so the same reason text carries both
    the copy marker and the device error code — the device code wins.
    """
    measured = getattr(o5, "measured", None)
    if not isinstance(measured, dict):
        return False
    precision = measured.get("precision")
    if not isinstance(precision, dict):
        return False
    reason = str(precision.get("reason") or "")
    if _DEVICE_ERROR_CODE_RE.search(reason):
        return True
    return any(marker in reason for marker in _DEVICE_ERROR_MARKERS)


def _retry_device_error_mismatch(ctx: OrchestratorContext, o5, runner):
    """Re-run a device-class MISMATCH through the infra retry loop, backoff 0."""
    log.warning(
        "O5 MISMATCH is device-class (5070xx/aivec signature in measured "
        "precision); re-attempting in place with NO backoff (device errors do "
        "not self-heal): %s",
        o5.summary[:160],
    )
    reclassified = phase_o5.O5Report(
        verdict="RUNNER_FAILED",
        summary=_DEVICE_RECLASSIFY_PREFIX + o5.summary,
        truth_source=getattr(o5, "truth_source", "npubench"),
        rollback_kind="infra",
    )
    return _retry_infra_o5_verification(ctx, reclassified, runner, backoff_seconds=0)


def _resolve_o5_report(ctx: OrchestratorContext, runner):
    """Run O5 and settle every infra-retryable verdict before dispatch."""
    first_o5 = _run_o5_verification(ctx, runner)
    # A controlled NPUBench build has already classified a pre-build candidate
    # rejection in its authenticated receipt.  Dispatch it before the generic
    # infra retry loop: this is a worker-owned defect, not a flaky evaluator
    # transport, and retrying can erase the only repairable observation.
    if (
        first_o5.verdict == "RUNNER_FAILED"
        and _is_direct_npubench_candidate_failure(ctx.workspace)
        and _is_repairable_npubench_candidate_failure(first_o5)
    ):
        o5 = first_o5
    else:
        o5 = _retry_infra_o5_verification(ctx, first_o5, runner)
    if o5.verdict == "MISMATCH" and _mismatch_is_device_error(o5):
        # Device-class errors are split OUT of the host-transient
        # classification (P0-1): zero backoff, and the same-signature parking
        # counter (threshold 2, enforced in _o5_post_verify) routes a repeat
        # to await_user_decision with device probe/reset/lane advice.
        o5 = _retry_device_error_mismatch(ctx, o5, runner)
    elif o5.verdict == "MISMATCH" and _mismatch_is_host_transient(o5):
        count = _bump_host_copy_fault_escalation(ctx.workspace, o5)
        if count > _host_copy_fault_max_transient():
            # Same candidate binding keeps tripping the copy-fault marker: a
            # genuinely transient host window does not reproduce
            # byte-identically — this is a candidate device-kernel fault
            # surfaced asynchronously at a copy op.  Dispatch the MISMATCH to
            # the worker with the deterministic-fault note instead of another
            # infra retry round.
            o5 = _mark_host_copy_fault_deterministic(ctx, o5, count)
        else:
            o5 = _retry_host_transient_mismatch(ctx, o5, runner)
    return o5


def _dispatch_o5_runner_failed(
    ctx: OrchestratorContext, snap, o5,
) -> HandlerResult:
    """Route a settled RUNNER_FAILED verdict to its owning repair path."""
    # Provider-owned taxonomy has precedence over operator-facing text.
    # A candidate contract diagnostic may mention capability or toolchain
    # words; that must still re-enter the worker repair path.
    if (
        _is_direct_npubench_candidate_failure(ctx.workspace)
        and _is_repairable_npubench_candidate_failure(o5)
    ):
        log.warning(
            "O5 rejected the NPUBench candidate before build; "
            "persisting await_worker repair re-entry: %s",
            o5.summary[:500],
        )
        return _handle_npubench_candidate_failure(ctx, snap, o5)
    if "A5_SOC_UNSUPPORTED_FOR_VALIDATION" in o5.summary:
        log.error("O5 stopped: %s", o5.summary)
        events.emit(
            ctx.workspace,
            "orchestrator.phase_o5_target_capability_block",
            lane=ctx.lane,
            data={"summary": o5.summary},
        )
        return HandlerResult.ret(2)
    if _is_direct_npubench_candidate_failure(ctx.workspace):
        if _is_worker_fixable_candidate_rejection(o5):
            log.info(
                "O5 RUNNER_FAILED is a candidate-authoring rejection "
                "(independence gate — worker can re-author independently); "
                "routing await_worker for a fix round"
            )
            return _handle_o5_runner_failed(ctx, snap, o5)
        return _stop_direct_npubench_candidate_failure(ctx, o5)
    return _handle_o5_runner_failed(ctx, snap, o5)


def _dispatch_o5_non_blocking(
    ctx: OrchestratorContext, snap, o5,
) -> Optional[HandlerResult]:
    """Record a non-blocking verdict and refuse anything outside the taxonomy."""
    if o5.verdict == "PROVISIONAL":
        _record_o5_provisional(ctx, o5)
    elif o5.verdict == "VERIFIED":
        log.info(f"O5 post-verify: {o5.summary}")
        phase_o5.record_harness_state(ctx.workspace, o5)
    elif o5.verdict != "SKIPPED":
        failed = phase_o5.O5Report(
            verdict="RUNNER_FAILED",
            summary=f"unknown O5 verdict {o5.verdict!r}; refusing finalize",
            truth_source=getattr(o5, "truth_source", "unresolved"),
        )
        return _handle_o5_runner_failed(ctx, snap, failed)
    return None


def _o5_post_verify(ctx: OrchestratorContext, snap) -> Optional[HandlerResult]:
    """Run O5 and dispatch its result without allowing unknown verdicts through."""
    runner = _o5_runner_for_workspace(ctx.workspace, extra_lanes=ctx.extra_lanes)
    # P0-2 (PR875 equiv review): never evaluate a candidate caught mid
    # repair-reset (2026-08-27 FAS 14:52 empty-window race — the repair reset
    # had just moved model_new_ascendc.py out when the evaluation started,
    # producing a bogus infra ERROR round).
    guard_result = _await_o5_candidate_stable(ctx, snap)
    if guard_result is not None:
        return guard_result
    o5 = _resolve_o5_report(ctx, runner)
    # P0-1 (DSH ruling §4.b): per-op PERSISTENT same-signature failure parking.
    # engine/infra/device classes accumulate in a workspace dotfile (survives
    # exit 77 / resume / cold-start); consecutive identical signatures past the
    # class threshold route to await_user_decision instead of another retry.
    park_result = _same_signature_park_route(ctx, o5)
    if park_result is not None:
        return park_result
    if o5.verdict == "MISMATCH":
        return _handle_o5_mismatch(ctx, snap, o5)
    if o5.verdict == "RUNNER_FAILED":
        return _dispatch_o5_runner_failed(ctx, snap, o5)
    return _dispatch_o5_non_blocking(ctx, snap, o5)


# ---------------------------------------------------------------------------
# P0-2 repair-reset race guard (2026-08-28, PR875 equiv review)
# ---------------------------------------------------------------------------
# Before an O5 evaluation, the candidate must be stable: entry file present as
# a regular non-symlink file, tree digest not changing under our feet, and no
# candidate-repair reset archived within the last 60s.  A violation used to
# surface as a bogus infra ERROR evaluation (the 14:52 "model_new_ascendc.py
# must be a regular non-symlink file" round); now it suspends with bounded
# retries and only then fails loud as an infra runner failure.
_O5_CANDIDATE_GUARD_MAX_RETRIES_ENV = "AOG_O5_CANDIDATE_GUARD_MAX_RETRIES"
_O5_CANDIDATE_GUARD_MAX_RETRIES = 3
_O5_CANDIDATE_GUARD_INTERVAL_ENV = "AOG_O5_CANDIDATE_GUARD_INTERVAL_SEC"
_O5_CANDIDATE_GUARD_INTERVAL_SECONDS = 5
_REPAIR_RESET_WINDOW_SECONDS = 60
_REPAIR_RECORD_FILENAME = ".npubench_candidate_repair.json"


def _o5_candidate_guard_max_retries() -> int:
    raw = os.environ.get(_O5_CANDIDATE_GUARD_MAX_RETRIES_ENV)
    if raw is None:
        return _O5_CANDIDATE_GUARD_MAX_RETRIES
    try:
        return max(int(raw.strip()), 0)
    except ValueError:
        return _O5_CANDIDATE_GUARD_MAX_RETRIES


def _o5_candidate_guard_interval_seconds() -> int:
    raw = os.environ.get(_O5_CANDIDATE_GUARD_INTERVAL_ENV)
    if raw is None:
        return _O5_CANDIDATE_GUARD_INTERVAL_SECONDS
    try:
        return max(int(raw.strip()), 0)
    except ValueError:
        return _O5_CANDIDATE_GUARD_INTERVAL_SECONDS


def _repair_reset_age_seconds(workspace: Path) -> Optional[float]:
    """Age of the last candidate repair-reset archival event, None if none."""
    try:
        record = json.loads((workspace / _REPAIR_RECORD_FILENAME).read_text())
    except Exception:
        return None
    created = record.get("created_at_utc") if isinstance(record, dict) else None
    if not created:
        return None
    try:
        import datetime as _dt

        stamp = _dt.datetime.fromisoformat(str(created).replace("Z", "+00:00"))
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=_dt.timezone.utc)
        return _dt.datetime.now(_dt.timezone.utc).timestamp() - stamp.timestamp()
    except (TypeError, ValueError):
        return None


def _candidate_tree_changed(
    entry_ok: bool,
    previous_tree_sha: Optional[str],
    tree_sha: Optional[str],
) -> bool:
    """Return whether two valid guard probes observed different trees."""
    if not entry_ok or not previous_tree_sha or not tree_sha:
        return False
    return tree_sha != previous_tree_sha


def _o5_candidate_stability_probe(workspace: Path, previous_tree_sha: Optional[str]):
    """One guard probe; returns (issues, tree_sha256_or_None).

    "tree sha agrees with the binding" is enforced as SELF-consistency inside
    the guard window: the digest computed here must match the previous probe's.
    Comparing against a HISTORICAL evaluation binding would mis-fire on every
    legitimately re-authored candidate, so no historical binding is consulted.
    """
    issues: list[str] = []
    age = _repair_reset_age_seconds(workspace)
    repair_recent = age is not None and age < _REPAIR_RESET_WINDOW_SECONDS
    if repair_recent:
        issues.append(
            f"candidate repair-reset archived {age:.1f}s ago "
            f"(< {_REPAIR_RESET_WINDOW_SECONDS}s window)"
        )
    tree_sha = None
    entry_ok = False
    try:
        from npubench.npubench_core import (
            _candidate_entry,
            _candidate_root,
            _candidate_tree_sha256,
        )

        candidate_root = _candidate_root(workspace)
        _candidate_entry(candidate_root)
        entry_ok = True
        tree_sha = _candidate_tree_sha256(candidate_root)
    except Exception as exc:
        # A missing/unstable entry only concerns the guard while a repair
        # reset is in flight; without one the runner's own candidate-contract
        # path reports the defect (unchanged legacy behavior).
        if repair_recent:
            issues.append(f"candidate entry not a regular file mid repair-reset: {exc}")
    if _candidate_tree_changed(entry_ok, previous_tree_sha, tree_sha):
        issues.append("candidate tree sha256 changed between guard probes")
    return issues, tree_sha


def _candidate_stability_guard_failure(
    ctx: OrchestratorContext,
    snap,
    max_retries: int,
    issues: list[str],
) -> HandlerResult:
    """Create the existing infra result after stability probes are exhausted."""
    log.error(
        "O5 candidate stability guard exhausted (%d attempts): %s — failing loud as infra",
        max_retries + 1, "; ".join(issues) or "tree sha never confirmed stable",
    )
    events.emit(
        ctx.workspace,
        "orchestrator.phase_o5_candidate_guard_failed",
        lane=ctx.lane,
        data={"issues": issues, "attempts": max_retries + 1},
    )
    failed = phase_o5.O5Report(
        verdict="RUNNER_FAILED",
        summary=(
            "O5 candidate stability guard: candidate unstable across "
            f"{max_retries + 1} probes ({'; '.join(issues)}) — a repair-reset "
            "race must not be evaluated (2026-08-27 FAS 14:52 empty-window "
            "incident). Re-invoke the orchestrator once the candidate settles."
        ),
        truth_source="npubench",
        rollback_kind="infra",
    )
    return _handle_o5_runner_failed(ctx, snap, failed)


def _await_o5_candidate_stable(ctx: OrchestratorContext, snap) -> Optional[HandlerResult]:
    """Bounded suspend-retry until the candidate is stable for O5 evaluation.

    Stability requires TWO consecutive clean probes with an identical tree
    sha256 (a single probe cannot observe a tree being rewritten between the
    guard and the evaluation).  Returns None when evaluation may proceed
    (including non-npubench routes, where no candidate snapshot race exists);
    on exhausted retries returns the infra RUNNER_FAILED dispatch (fail loud,
    free re-entry, no worker respawn semantics change).
    """
    try:
        if phase_o5.expected_truth_source(ctx.workspace) != "npubench":
            return None
    except Exception as error:
        log.warning("candidate stability guard: truth-source probe failed (%s); skipping", error)
        return None
    max_retries = _o5_candidate_guard_max_retries()
    interval = _o5_candidate_guard_interval_seconds()
    previous_tree_sha: Optional[str] = None
    confirmed = False  # one clean probe already seen; the next clean probe confirms
    issues: list[str] = []
    for attempt in range(max_retries + 1):
        issues, tree_sha = _o5_candidate_stability_probe(ctx.workspace, previous_tree_sha)
        previous_tree_sha = tree_sha
        if not issues and confirmed:
            if attempt > 1:
                log.info(
                    "O5 candidate stability guard: candidate settled after %d wait(s)", attempt
                )
            return None
        if not issues:
            confirmed = True
        else:
            confirmed = False
            log.warning(
                "O5 candidate stability guard (attempt %d/%d): %s",
                attempt + 1, max_retries + 1, "; ".join(issues),
            )
        if attempt < max_retries and interval > 0:
            _sleep(interval)
    return _candidate_stability_guard_failure(ctx, snap, max_retries, issues)


# ---------------------------------------------------------------------------
# P0-1 same-signature failure parking (DSH ruling §4.b, 2026-08-28)
# ---------------------------------------------------------------------------
def _o5_failure_signature_input(o5) -> Optional[tuple[str, str]]:
    """Classify a settled O5 failure; returns (failure_class, reason) or None.

    failure_class ∈ {engine, infra, device, candidate} per DSH §4.b:
    engine = build/route/contract-schema class (incl. candidate_contract
    repair re-entries and ModuleNotFoundError evaluation rounds); infra =
    host/network/lease transients (incl. host copy faults without a device
    error code); device = 5070xx/aivec device errors (split out of the
    host-transient text class); candidate = genuine precision MISMATCH.
    """
    verdict = getattr(o5, "verdict", None)
    summary = str(getattr(o5, "summary", "") or "")
    if verdict == "MISMATCH":
        measured = getattr(o5, "measured", None)
        reason = ""
        if isinstance(measured, dict) and isinstance(measured.get("precision"), dict):
            reason = str(measured["precision"].get("reason") or "")
        reason = reason or summary
        if _mismatch_is_device_error(o5):
            return "device", reason
        if _mismatch_is_host_transient(o5):
            return "infra", reason
        return "candidate", reason
    if verdict == "RUNNER_FAILED":
        if summary.startswith(_DEVICE_RECLASSIFY_PREFIX):
            return "device", summary
        if summary.startswith(_HOST_TRANSIENT_RECLASSIFY_PREFIX):
            return "infra", summary
        # Candidate-contract re-entries are the engine class per DSH §4.b
        # (build/route/contract-schema — the FAS 27-round ModuleNotFoundError
        # loop was exactly this shape), even when they inherit an infra
        # rollback_kind from the runner transport.
        if getattr(o5, "failure_kind", None) == "candidate_contract":
            return "engine", summary
        if getattr(o5, "rollback_kind", None) == "infra":
            return "infra", summary
        return "engine", summary
    return None


def _candidate_precision_cases(o5) -> list[dict]:
    """Return dictionary-shaped precision cases from one O5 report."""
    measured = getattr(o5, "measured", None)
    if not isinstance(measured, dict):
        return []
    precision = measured.get("precision")
    if not isinstance(precision, dict):
        return []
    cases = precision.get("cases")
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, dict)]


def _candidate_case_failure_signature(case: dict) -> Optional[str]:
    """Render one non-passing precision case for the parking signature."""
    status = str(case.get("status") or case.get("verdict") or "")
    if status.upper() in ("PASS", "SKIPPED"):
        return None
    idx = case.get("case_idx", case.get("index", "?"))
    reason = state_executor.normalize_failure_reason(
        case.get("reason") or case.get("MERE") or ""
    )
    return f"{idx}:{status}:{reason}"


def _candidate_case_signature(o5) -> str:
    """Stable per-case failure signature for the candidate-class escalation."""
    parts: list[str] = []
    for case in _candidate_precision_cases(o5):
        signature = _candidate_case_failure_signature(case)
        if signature is not None:
            parts.append(signature)
    if parts:
        return "|".join(parts)
    return str(getattr(o5, "summary", "") or "")


def _route_await_user_decision_park(
    ctx: OrchestratorContext, o5, *, failure_class: str, count: int, detail: str,
) -> HandlerResult:
    """Park at await_user_decision (never hard-stop) with the class advisory."""
    advisory = ""
    if failure_class == "device":
        advisory = (
            " Device-class errors do not self-heal — DO NOT just resume into "
            "the same lane. Suggested diagnostics (engine performs no reset "
            "itself): 1) npu-smi info — check the leased device's Health/Util; "
            "2) re-run the eval child with ASCEND_LAUNCH_BLOCKING=1 for true "
            "fault attribution (an async 507035/507015 at an H2D copy can be "
            "an EARLIER candidate launch wedging the device — 2026-08-27 "
            "42_CoTAttention precedent), then bisect per-launch with "
            "aclrtSynchronizeStream; 3) ask the host admin for "
            "npu-smi/aclrtResetDevice on the wedged device; 4) re-lease on a "
            "different lane."
        )
    rationale = (
        f"P0-1 same-signature parking: {count} consecutive identical "
        f"{failure_class}-class O5 failures (persistent counter, survives "
        f"exit 77/restart). {detail[:600]}{advisory} Halting at "
        "await_user_decision — write user_decision.md to cap-bump, switch "
        "lane, or repair the engine; the orchestrator does not hard-stop."
    )
    log.error(rationale)
    state_executor.record_transition(
        ctx.workspace,
        state_executor.TransitionDecision(
            next_state="await_user_decision",
            matched_transition_index=-1,
            rationale=rationale,
            from_state="finalize",
            handoff="",
        ),
    )
    events.emit(
        ctx.workspace,
        "orchestrator.same_signature_park",
        lane=ctx.lane,
        data={
            "failure_class": failure_class,
            "count": count,
            "summary": str(getattr(o5, "summary", "") or "")[:300],
        },
    )
    return HandlerResult.cont()


def _same_signature_park_route(ctx: OrchestratorContext, o5) -> Optional[HandlerResult]:
    """Update the persistent same-signature counters and park when stuck.

    Returns a HandlerResult routing to await_user_decision, or None to let the
    normal verdict dispatch proceed.  Success verdicts (VERIFIED/PROVISIONAL)
    break the consecutive chain; SKIPPED is not a measurement and leaves it.
    """
    workspace = ctx.workspace
    verdict = getattr(o5, "verdict", None)
    if verdict in ("VERIFIED", "PROVISIONAL"):
        state_executor.clear_same_signature_state(workspace)
        state_executor.clear_candidate_case_state(workspace)
        return None
    classified = _o5_failure_signature_input(o5)
    if classified is None:
        return None
    failure_class, reason = classified
    tree_key = _o5_candidate_tree_key(o5)
    if failure_class == "candidate":
        # A candidate MISMATCH is a different-signature insertion for the
        # engine/infra/device chain, and feeds its own escalation counter.
        state_executor.clear_same_signature_state(workspace)
        entry = state_executor.record_candidate_case_failure(
            workspace, tree_key, _candidate_case_signature(o5)
        )
        if entry["count"] >= state_executor.CANDIDATE_CASE_PARK_THRESHOLD:
            return _route_await_user_decision_park(
                ctx,
                o5,
                failure_class="candidate",
                count=entry["count"],
                detail=(
                    "Same candidate tree + same per-case failure signature for "
                    f"{entry['count']} O5 rounds (worker cannot move the "
                    "failure — the '改不动=卡死' shape, FAS kw-17..21 "
                    "precedent). Mismatch: " + str(getattr(o5, "summary", "") or "")[:200]
                ),
            )
        return None
    entry = state_executor.record_same_signature_failure(
        workspace, failure_class, reason, tree_key
    )
    threshold = state_executor.same_signature_park_threshold(failure_class)
    if entry["count"] >= threshold:
        return _route_await_user_decision_park(
            ctx,
            o5,
            failure_class=failure_class,
            count=entry["count"],
            detail=f"Signature {entry['signature'][:12]}… reason: {entry['reason_norm'][:200]}",
        )
    return None


# Host-level H2D copy faults (2026-08-22 A5 campaign): the measured
# precision report fails inside torch_npu's copy path, not in the candidate
# kernel.  A MISMATCH verdict carrying one of these markers must re-attempt
# in place as infra — the worker cannot repair the host (2026-08-23 CoT).
# WARNING (2026-08-27, 42_CoTAttention): these markers are NOT proof of a
# host fault.  torch_npu launches asynchronously, so a candidate device-kernel
# hard fault (507035/507015, e.g. aivec error 340) surfaces at the NEXT host
# op — often an H2D copy — making the reason byte-identical to a genuine
# transient host window.  The escalation guard in `_resolve_o5_report` bounds
# how many times the same candidate binding may hide behind this
# classification before it is treated as a deterministic candidate fault.
_HOST_COPY_FAULT_MARKERS = (
    "copy_between_host_and_device",
    "CopyKernelOpApi",
    "copy_with_slice",
)


def _mismatch_is_host_transient(o5) -> bool:
    measured = getattr(o5, "measured", None)
    if not isinstance(measured, dict):
        return False
    precision = measured.get("precision")
    if not isinstance(precision, dict):
        return False
    reason = str(precision.get("reason") or "")
    if not any(marker in reason for marker in _HOST_COPY_FAULT_MARKERS):
        return False
    # P0abb (2026-08-25): a candidate kernel that wedges the device surfaces as
    # a copy fault on a LATER case's H2D move (45_CrossformerAttention: case-0
    # launch 507035 → case-1 input copy fault, pass_a=None).  If a completed
    # case already failed with a device error inside candidate execution, the
    # copy fault is a symptom, not the cause — classify as candidate MISMATCH
    # (worker-fixable) instead of host-transient infra so the worker sees the
    # kernel-wedge signature instead of a phantom H2D fault.
    for case in precision.get("cases") or []:
        case_reason = str(case.get("reason") or "")
        if "case execution failed" in case_reason and any(
            code in case_reason for code in ("507035", "507014", "507033", "507034")
        ):
            return False
    return True


_WORKER_FIXABLE_CANDIDATE_REJECTION_MARKERS = (
    # The candidate independence gate rejects the WORKER's authored candidate
    # (byte-copy / comments-only edit of the staged kernel / forbidden ACLNN
    # text / source-stage references).  These are authoring failures the
    # source-only worker CAN repair by re-authoring independently — unlike
    # target-side build/toolchain failures, which it cannot.
    "TileLang2AscendC candidate",
    "ACLNN candidate",
)


def _is_worker_fixable_candidate_rejection(o5) -> bool:
    """Return whether a direct-npubench O5 failure is worker-fixable.

    The terminal direct-npubench stop was designed for TARGET-side failures
    (build toolchain/device errors) that the isolated source-only worker
    cannot repair.  The candidate independence gate rejects the worker's own
    authored candidate, and compile errors in the worker's own kernel sources
    are equally authoring failures — the worker can fix them (re-author
    independently / correct the code using the durable build receipt's
    diagnostics), so route back to ``await_worker`` like any other authoring
    failure instead of stopping with "manual target diagnostics".
    """
    if getattr(o5, "rollback_kind", None) == "candidate":
        return True
    summary = getattr(o5, "summary", "") or ""
    return any(marker in summary for marker in _WORKER_FIXABLE_CANDIDATE_REJECTION_MARKERS)


def _is_direct_npubench_candidate_failure(workspace: Path) -> bool:
    """Return whether an O5 failure belongs to a direct NPUBench candidate route.

    TileLang2AscendC workers are intentionally isolated from their generated
    candidate and target toolchain.  Target-side build or evaluation failures
    must not be sent back to the source-only worker: routing them to
    ``await_worker`` cannot fix the target-side problem and risks making the
    candidate visible as source in the next graybox — the durable build
    receipt is the repair/resume handoff.  A pre-build candidate-contract
    rejection is different: the worker owns that artifact and needs a durable
    repair re-entry.

    The same taxonomy applies to the generic authored-kernel route
    (``opgen_mode=port_a3_to_a5``): there is no ``port_source`` binding, but
    the worker owns the same ``kernel/`` + ``model_new_ascendc.py`` project
    under the same graybox isolation, so an authenticated
    ``candidate_contract`` build failure must reach the repair worker instead
    of burning infra retries (2026-08-27 flash_attention_score: a genuine
    candidate compile error was retried twice with 600s backoffs as "infra").
    """
    try:
        from reference_source import load_durable_state

        state = load_durable_state(workspace)
    except Exception:
        return False
    source = state.get("port_source")
    reference = state.get("reference")
    source_kind = source.get("kind") if isinstance(source, dict) else None
    if not (isinstance(reference, dict) and reference.get("source") == "npubench"):
        return False
    # Do not require the candidate entry here.  A missing entry is itself a
    # worker-owned candidate-contract defect; the structured O5 failure kind
    # below must still be able to route it to the repair worker.  Target-side
    # failures remain terminal because they carry a different failure_kind.
    if source_kind == "port-aclnn-tilelang2ascendc":
        return True
    if source_kind is None:
        try:
            from npubench.npubench_o5_runner import _generic_kernel_project_present

            return _generic_kernel_project_present(workspace)
        except Exception:
            return False
    return False


def _is_repairable_npubench_candidate_failure(o5) -> bool:
    """Use the provider receipt's taxonomy to identify a repairable defect.

    This intentionally does not inspect ``summary``.  The summary is an
    operator-facing diagnostic and must never decide whether a target-side
    failure is sent back to the source worker.
    """
    return getattr(o5, "failure_kind", None) == "candidate_contract"


def _stop_direct_npubench_candidate_failure(
    ctx: OrchestratorContext, o5,
) -> HandlerResult:
    """Stop after a target failure without respawning the source-only worker."""
    summary = o5.summary[:500]
    log.error(
        "direct NPUKernelBench candidate requires manual target diagnostics; "
        "not routing to await_worker: %s",
        summary,
    )
    events.emit(
        ctx.workspace,
        "orchestrator.phase_o5_direct_candidate_block",
        lane=ctx.lane,
        data={
            "summary": summary,
            "rollback_kind": getattr(o5, "rollback_kind", None),
            "next_action": (
                "inspect the route-specific build receipt under "
                "npubench_evidence/ and repair the candidate or target "
                "environment before an explicit resume"
            ),
        },
    )
    return HandlerResult.ret(2)


def _runner_failure_at_worker_cap(
    ctx: OrchestratorContext, snap, o5,
) -> Optional[HandlerResult]:
    """Return the terminal loop-guard result for a capped O5 runner failure."""
    workspace = ctx.workspace
    if not state_executor.at_iter_cap(workspace, "await_worker"):
        return None
    count = snap.iter_counts.get("worker", 0)
    cap = state_executor.iter_cap("await_worker", workspace=workspace)
    log.critical(
        f"O5 RUNNER_FAILED while await_worker already at iter_cap ({count}/{cap}). "
        "Cannot progress — routing back would infinite-loop. Fix the runner "
        "discoverability (run_pass_b.py / pass_a_runner.py name) manually OR "
        "mark precision.pass_b N/A explicitly, then resume."
    )
    events.emit(
        workspace,
        "orchestrator.fsm_loop_guard_o5_runner_failed",
        lane=ctx.lane,
        data={"worker_count": count, "worker_cap": cap, "summary": o5.summary[:200]},
    )
    return HandlerResult.ret(2)


def _npubench_candidate_failure_at_worker_cap(
    ctx: OrchestratorContext, snap, o5,
) -> Optional[HandlerResult]:
    """Stop a repeated candidate-contract defect with NPUBench-owned wording."""
    workspace = ctx.workspace
    if not state_executor.at_iter_cap(workspace, "await_worker"):
        return None
    count = snap.iter_counts.get("worker", 0)
    cap = state_executor.iter_cap("await_worker", workspace=workspace)
    log.critical(
        "NPUKernelBench candidate contract still rejected while await_worker "
        "is at iter_cap (%s/%s). Cannot progress — repair the candidate "
        "source/build contract before explicitly resuming.",
        count,
        cap,
    )
    events.emit(
        workspace,
        "orchestrator.fsm_loop_guard_o5_candidate_contract",
        lane=ctx.lane,
        data={
            "worker_count": count,
            "worker_cap": cap,
            "failure_kind": getattr(o5, "failure_kind", None),
            "summary": o5.summary[:200],
        },
    )
    return HandlerResult.ret(2)


def _rollback_npubench_candidate_failure(workspace: Path, o5) -> HandlerResult:
    """Persist an NPUBench candidate repair handoff without legacy contracts."""
    rollback_state = "await_worker"
    reason = (
        "P0aba.O5 NPUKernelBench candidate contract rejected before build: "
        f"{o5.summary}"
    )
    receipt = workspace / "npubench_evidence" / "tilelang2ascendc_build_receipt.json"
    try:
        payload = json.loads(receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        payload = None
    if isinstance(payload, dict):
        stderr_tail = payload.get("stderr_tail")
        if isinstance(stderr_tail, str) and stderr_tail.strip():
            reason += "\nAuthenticated target-build stderr tail:\n" + stderr_tail[-4096:]
    # The retry is an authoring attempt.  Remove stale answer-bearing output
    # before agent_dispatch constructs the next graybox, while preserving the
    # source stage/task/state and keeping a recoverable archive outside the
    # worker-visible workspace.
    _prepare_npubench_candidate_repair(
        workspace,
        failure_reason=reason,
        failure_kind=getattr(o5, "failure_kind", None),
    )
    state_executor.record_transition(
        workspace,
        state_executor.TransitionDecision(
            next_state=rollback_state,
            matched_transition_index=-1,
            rationale=(
                "P0aba.O5 NPUKernelBench candidate contract rejected before "
                f"build: {o5.summary}. Rollback to {rollback_state}. "
                "Worker must repair only the candidate source/build contract; "
                "the orchestrator owns controlled build, precision, and "
                "performance evidence. Do not create a legacy pass_b verifier "
                "or worker-owned NPUBench evidence."
            ),
            from_state="finalize",
            handoff="",
            # A candidate defect requires authoring work and consumes the
            # algorithm iteration budget. It is not an infra-free retry.
            rollback_kind="algorithm",
        ),
    )
    finalize_pipeline.record_rollback(
        workspace,
        rollback_state=rollback_state,
        reason=reason,
        gate="phase_o5_npubench_candidate_contract",
        reason_limit=None,
    )
    return HandlerResult.cont()


def _handle_npubench_candidate_failure(
    ctx: OrchestratorContext, snap, o5,
) -> HandlerResult:
    """Route a repairable NPUBench candidate defect to the next worker."""
    log.info(
        "O5 post-verify: NPUBench candidate contract failure — "
        "routing await_worker for candidate repair"
    )
    events.emit(
        ctx.workspace,
        "orchestrator.phase_o5_npubench_candidate_contract",
        lane=ctx.lane,
        data={
            "summary": o5.summary,
            "failure_kind": getattr(o5, "failure_kind", None),
            "next_state": "await_worker",
        },
    )
    cap_result = _npubench_candidate_failure_at_worker_cap(ctx, snap, o5)
    if cap_result is not None:
        return cap_result
    return _rollback_npubench_candidate_failure(ctx.workspace, o5)


def _rollback_o5_runner_failure(workspace: Path, o5) -> HandlerResult:
    """Record the existing runner-failure rollback and retry accounting."""
    _clear_harness_build_artifacts(workspace)
    rollback_state = "await_worker"
    rollback_kind = o5.rollback_kind
    retry_label = (
        "infra re-entry FREE (does not consume iter)"
        if rollback_kind == "infra"
        else "algorithm re-entry counts"
    )
    log.info(f"O5 RUNNER_FAILED rollback target: {rollback_state}")
    state_executor.record_transition(
        workspace,
        state_executor.TransitionDecision(
            next_state=rollback_state,
            matched_transition_index=-1,
            rationale=(
                f"P0aba.O5 RUNNER_FAILED: {o5.summary[:200]}. "
                f"Rollback to {rollback_state}. Worker must produce a canonical "
                "pass_b verifier filename at workspace root, OR set "
                "verification.json precision.pass_b to {status: 'N/A', reason: "
                "'<...>'} explicitly."
                + (
                    f" [NODE-5: rollback_kind={rollback_kind!r} — {retry_label}]"
                    if rollback_kind
                    else ""
                )
            ),
            from_state="finalize",
            handoff="",
            rollback_kind=rollback_kind,
        ),
    )
    finalize_pipeline.record_rollback(
        workspace,
        rollback_state=rollback_state,
        reason=f"P0aba.O5 RUNNER_FAILED: {o5.summary[:200]}",
        gate="phase_o5_runner_failed",
    )
    return HandlerResult.cont()


def _handle_o5_runner_failed(ctx: OrchestratorContext, snap, o5) -> HandlerResult:
    """Fail closed after O5 runner failure, bounded retries already exhausted."""
    log.info(f"O5 post-verify: RUNNER_FAILED — {o5.summary}")
    log.info("P0aba.O5 gate: routing await_worker for runner reconciliation")
    events.emit(
        ctx.workspace,
        "orchestrator.phase_o5_runner_failed",
        lane=ctx.lane,
        data={"summary": o5.summary},
    )
    cap_result = _runner_failure_at_worker_cap(ctx, snap, o5)
    if cap_result is not None:
        return cap_result
    return _rollback_o5_runner_failure(ctx.workspace, o5)


def _perf_capture_plugin(workspace: Path):
    """Return the active plugin and whether it permits independent capture."""
    plugin = getattr(finalize_pipeline, "_get_active_plugin")(workspace)
    should_capture = plugin is not None and plugin.should_run_phase_o5_perf_capture()
    return plugin, should_capture


def _capture_perf_if_requested(
    ctx: OrchestratorContext,
    plugin,
    should_capture: bool,
    verification_path: Path,
    verification: dict,
    existing_perf: dict,
) -> bool:
    """Capture, merge, and stamp performance when capture is requested."""
    override = os.environ.get("AOG_PERF_CAPTURE_OVERRIDE_WORKER", "").strip()
    requested = should_capture and (
        existing_perf.get("ratio") in (None, 0, 0.0)
        or override in ("1", "true", "TRUE", "yes")
    )
    if not requested:
        return False
    perf_result = phase_o5_perf_capture.measure_op_perf(
        ctx.op, ctx.workspace, plugin=plugin,
    )
    merged_perf = perf_irm_provenance.merge_perf_preserving_irm(
        existing_perf, perf_result
    )
    merged_perf["independent_re_measure"] = (
        perf_irm_provenance.orchestrator_irm_from_perf_result(
            perf_result, worker_ratio=existing_perf.get("ratio"),
        )
    )
    verification["performance"] = merged_perf
    if existing_perf.get("ratio"):
        verification["performance"]["worker_authored_aux"] = existing_perf
    verification_path.write_text(json.dumps(verification, indent=2))
    events.emit(
        ctx.workspace,
        "orchestrator.phase_o5_perf_capture",
        lane=ctx.lane,
        data={
            "status": perf_result.get("status"),
            "ratio": perf_result.get("ratio"),
            "plugin": getattr(plugin, "name", "unknown"),
        },
    )
    log.info(
        f"phase_o5_perf_capture: status={perf_result.get('status')} "
        f"ratio={perf_result.get('ratio')}"
    )
    return True


def _relabel_worker_authored_perf(
    plugin,
    should_capture: bool,
    verification_path: Path,
    verification: dict,
    existing_perf: dict,
) -> None:
    """Make a non-orchestrator performance report explicitly self-reported."""
    existing_irm = existing_perf.get("independent_re_measure")
    if not existing_perf or perf_irm_provenance.is_orchestrator_measured(existing_irm):
        return
    capture_reason = (
        "did not run: worker self-reported a ratio"
        if should_capture
        else f"disabled for plugin {getattr(plugin, 'name', 'unknown')!r}"
    )
    existing_perf["independent_re_measure"] = (
        perf_irm_provenance.worker_authored_irm(
            existing_irm,
            reason=(
                "no orchestrator-side perf re-measure ran for this op "
                f"(phase_o5_perf_capture {capture_reason}). The recorded "
                "performance.ratio is a WORKER SELF-REPORT, not an independent "
                "measure. Force an orchestrator capture with "
                "AOG_PERF_CAPTURE_OVERRIDE_WORKER=1."
            ),
        )
    )
    verification["performance"] = existing_perf
    verification_path.write_text(json.dumps(verification, indent=2))


def _run_perf_capture(ctx: OrchestratorContext) -> None:
    """Best-effort independent performance re-measure before eligibility."""
    # NPUKernelBench has a fixed runner-owned W3/R5 profiler contract.  Do
    # not overwrite it with the historical port performance capture, which
    # assumes a different reference callable and methodology.
    try:
        if phase_o5.expected_truth_source(ctx.workspace) == "npubench":
            return
    except Exception as error:
        log.warning("could not resolve provider before perf capture: %s", error)
        return
    try:
        plugin, should_capture = _perf_capture_plugin(ctx.workspace)
        verification_path = ctx.workspace / "verification.json"
        if not verification_path.exists():
            return
        verification = json.loads(verification_path.read_text())
        existing_perf = verification.get("performance", {}) or {}
        captured = _capture_perf_if_requested(
            ctx,
            plugin,
            should_capture,
            verification_path,
            verification,
            existing_perf,
        )
        if not captured:
            _relabel_worker_authored_perf(
                plugin,
                should_capture,
                verification_path,
                verification,
                existing_perf,
            )
    except Exception as error:
        log.warning(
            f"phase_o5_perf_capture failed ({type(error).__name__}: {error}); "
            "worker-authored perf kept as-is"
        )


def _prepare_finalize_audit(ctx: OrchestratorContext) -> None:
    """Best-effort creation of audit artifacts needed by the final gate."""
    try:
        ctx.ensure_audit_artifacts(ctx.workspace, lane=ctx.lane)
    except Exception as error:
        log.warning(
            f"audit artifact production failed ({error}); continuing — "
            "finalize gate will block if artifacts missing"
        )


def _assemble_finalize_ge_ophost(workspace: Path) -> None:
    """Best-effort deterministic GE host assembly before eligibility."""
    try:
        ge_report = finalize_pipeline.assemble_ge_ophost(workspace)
        if ge_report.get("ran"):
            log.info(
                "assemble_ge_ophost: assembled=%s preserved=%s errors=%s",
                ge_report.get("assembled"),
                ge_report.get("preserved"),
                ge_report.get("errors"),
            )
        else:
            log.info("assemble_ge_ophost: skipped (%s)", ge_report.get("skipped_reason"))
    except Exception as error:
        log.warning(
            f"GE op_host assembler failed ({error}); continuing — GE_OPHOST "
            "gate will block if GE host missing"
        )


def _normalize_finalize_workspace(workspace: Path) -> None:
    """Best-effort schema normalization for resumed or legacy workspaces."""
    try:
        norm_report = schema_norm.normalize_workspace(workspace, fail_strict=False)
        if norm_report.events:
            log.info("finalize-branch schema_norm: %s normalizations", len(norm_report.events))
    except Exception as error:
        log.warning(
            f"finalize-branch schema_norm failed ({error}); continuing — "
            "eligibility gate will catch any resulting issue"
        )


def _format_batch_precheck_failures(failures: list[dict]) -> Optional[str]:
    """Return the existing multi-failure worker handoff only when needed."""
    if len(failures) < 2:
        return None
    reasons = " || ".join(
        f"[{failure['gate']}] {failure['reason']}" for failure in failures
    )
    return (
        f"{len(failures)} finalize gates failed (fix ALL in one respawn, do "
        f"NOT fix one at a time): {reasons}"
    )


def _run_batch_finalize_precheck(workspace: Path) -> Optional[str]:
    """Run and record the collect-all precheck without weakening the backstop."""
    try:
        precheck = finalize_pipeline.batch_precheck(workspace)
        if not precheck.get("applicable"):
            return None
        report_md = finalize_pipeline.format_batch_precheck_report(precheck)
        try:
            (workspace / "finalize_precheck.md").write_text(report_md)
        except Exception as error:
            logging.getLogger(__name__).debug("Recoverable operation failed.", exc_info=error)
        failures = precheck.get("failures", [])
        if precheck.get("ok"):
            log.info("batch_precheck: all PASS-branch finalize gates clear — finalize promotes in one pass")
        elif precheck.get("precondition_block"):
            log.info(
                "batch_precheck: blocked by precondition gate %s",
                precheck["precondition_block"]["gate"],
            )
        else:
            log.info(
                "batch_precheck: %s finalize-gate failure(s) at once → worker fixes "
                "ALL in one respawn: %s",
                len(failures),
                ", ".join(failure["gate"] for failure in failures),
            )
        return _format_batch_precheck_failures(failures)
    except Exception as error:
        log.warning(
            f"batch_precheck failed ({error}); continuing — per-gate eligibility "
            "check is the backstop"
        )
        return None


def _run_finalize_prep(ctx: OrchestratorContext) -> Optional[str]:
    """Run best-effort artifact preparation before the authoritative gate."""
    _prepare_finalize_audit(ctx)
    _assemble_finalize_ge_ophost(ctx.workspace)
    _normalize_finalize_workspace(ctx.workspace)
    return _run_batch_finalize_precheck(ctx.workspace)


def _log_perf_na_recommendation(action: dict) -> None:
    """Log the pre-existing recommendation for a stuck performance gate."""
    if action["action"] == "coerce_perf_na":
        log.info(
            "DEBT-192: op is precision-PASS port_a3 on a perf-methodology loop "
            "→ recommend coercing performance.status=N/A (retract the unmeasured "
            "perf claim). Deferring the coercion to the port_a3 perf-N/A "
            "contract; loop halted."
        )


def _record_nonconvergent_loop(
    workspace: Path, lane, eligibility: dict, gate_tag: str, loop_break: dict, action: dict,
) -> HandlerResult:
    """Persist the DEBT-192 pause state after the existing rollback record."""
    _log_perf_na_recommendation(action)
    # The loop-break halts WITHOUT the normal rollback cleanup — but the
    # workspace still holds the O5 build tree (kernel/build/*.o|.so), which
    # the graybox construction scan classifies as assembled-answer input and
    # rejects on the next agent respawn (2026-08-23 Outlook: loop-break
    # pause → decision → await_worker spawn rejected, 16 answer-classified
    # files).  Clear the harness-owned build artifacts here so the respawn
    # after the operator decision succeeds.
    _clear_harness_build_artifacts(workspace)
    (workspace / ".finalize_loop_nonconvergent").write_text(
        json.dumps(
            {
                "pattern": loop_break.get("pattern"),
                "gate": gate_tag,
                "count": loop_break["loop_detected_at_count"],
                "recommended_action": action["action"],
                "reason": eligibility["reason"][:1000],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    state_executor.record_transition(
        workspace,
        state_executor.TransitionDecision(
            next_state="await_user_decision",
            matched_transition_index=-1,
            rationale=(
                f"DEBT-192 loop-break (pattern={loop_break.get('pattern')}): "
                f"perf/methodology gate fired {loop_break['loop_detected_at_count']}× "
                f"with no progress; recommended={action['action']}; halting at "
                f"await_user_decision (not another respawn). Last reason: "
                f"{eligibility['reason']}"
            ),
            from_state="finalize",
            handoff="",
        ),
    )
    events.emit(
        workspace,
        "orchestrator.finalize_loop_break",
        lane=lane,
        data={
            "gate": gate_tag,
            "pattern": loop_break.get("pattern"),
            "rollback_state": eligibility["rollback_state"],
            "loop_count": loop_break["loop_detected_at_count"],
            "recommended_action": action["action"],
            "reason": eligibility["reason"],
        },
    )
    return HandlerResult.cont()


def _nonconvergent_loop_result(
    workspace: Path, lane, eligibility: dict, gate_tag: str,
) -> Optional[HandlerResult]:
    """Detect and halt a non-convergent finalize rollback loop."""
    loop_break = finalize_pipeline.detect_nonconvergent_loop(workspace)
    if not loop_break:
        return None
    action = finalize_pipeline.classify_loop_break_action(workspace, loop_break)
    log.info(
        f"finalize LOOP-BREAK (DEBT-192, pattern={loop_break.get('pattern')}): "
        f"gate {gate_tag} / perf-methodology family fired "
        f"{loop_break['loop_detected_at_count']}× with no progress — "
        f"recommended action {action['action']!r}; halting at "
        f"await_user_decision instead of respawning "
        f"{eligibility['rollback_state']!r} to avoid infinite loop"
    )
    return _record_nonconvergent_loop(
        workspace, lane, eligibility, gate_tag, loop_break, action,
    )


def _rollback_rationale(eligibility: dict, precheck_all_reasons: Optional[str]) -> str:
    """Build the existing worker-facing rollback rationale."""
    if precheck_all_reasons is not None and eligibility["rollback_state"] == "await_worker":
        return f"P0ff rollback (batch precheck — fix ALL at once): {precheck_all_reasons}"
    return f"P0ff rollback: {eligibility['reason']}"


def _record_finalize_rollback(
    workspace: Path, lane, eligibility: dict, precheck_all_reasons: Optional[str],
) -> HandlerResult:
    """Write the existing synthetic finalize-to-worker rollback transition."""
    if eligibility.get("rollback_state") in {
        "await_worker",
        "await_probe",
        "await_optimizer",
        "await_fused_optimizer",
        "await_researcher",
        "await_det_analyzer",
    }:
        # Same graybox-respawn contract as the O5 rollback paths: the O5
        # candidate stage (.npubench_candidate) and build tree are answer-
        # classified and would reject the respawn.  2026-08-22 Crossformer:
        # an await_probe rollback kept them, and the precision-probe spawn
        # was rejected by the graybox construction scan.
        _clear_harness_build_artifacts(workspace)
    log.info(
        f"finalize ROLLBACK: not eligible ({eligibility['reason']}); rolling back "
        f"to {eligibility['rollback_state']!r}"
    )
    from briefs._common import g7_slug  # noqa: F401

    state_executor.record_transition(
        workspace,
        state_executor.TransitionDecision(
            next_state=eligibility["rollback_state"],
            matched_transition_index=-1,
            rationale=_rollback_rationale(eligibility, precheck_all_reasons),
            from_state="finalize",
            handoff="",
        ),
    )
    events.emit(
        workspace,
        "orchestrator.finalize_rollback",
        lane=lane,
        data={
            "rollback_state": eligibility["rollback_state"],
            "reason": eligibility["reason"],
        },
    )
    return HandlerResult.cont()


_DIGEST_SCHEME_DRIFT_MARKER = "candidate digest scheme drift"
_SNAPSHOT_MISSING_MARKER = "immutable candidate snapshot is missing"


def _eligibility_is_harness_refreezable(eligibility: dict) -> bool:
    """Digest-scheme drift / missing frozen snapshot are harness-state issues.

    Both mean the O5 evidence was produced under a different harness state
    (older exclusion list, or the snapshot was cleared by an operator
    recovery).  The worker cannot repair either — the harness can, by
    re-freezing the snapshot with the current code and re-running O5 (audit
    H1, Option B).
    """
    reason = str(eligibility.get("reason") or "")
    return _DIGEST_SCHEME_DRIFT_MARKER in reason or _SNAPSHOT_MISSING_MARKER in reason


def _self_heal_digest_scheme_drift(
    ctx: OrchestratorContext, snap,
) -> bool:
    """Clear stale evaluation artifacts and re-run O5 in place, once.

    Removes only the derived snapshot/exec trees (kernel/build stays for
    receipt reuse); the re-run freezes a fresh snapshot under the current
    digest scheme and rebinds the evidence reports.
    """
    from pathlib import Path as _Path

    workspace = _Path(ctx.workspace)
    cleared = []
    for rel in (".npubench_candidate", ".npubench_exec"):
        target = workspace / rel
        if target.is_symlink() or target.is_file():
            try:
                target.unlink()
                cleared.append(rel)
            except OSError:
                pass
        elif target.is_dir():
            import shutil as _shutil

            try:
                _shutil.rmtree(target)
                cleared.append(rel)
            except OSError:
                pass
    if cleared:
        log.info(
            "digest-scheme self-heal: cleared %s; re-running O5 with current "
            "exclusion list (bounded, no worker respawn)",
            ", ".join(cleared),
        )
    runner = _o5_runner_for_workspace(ctx.workspace, extra_lanes=ctx.extra_lanes)
    o5 = _run_o5_verification(ctx, runner)
    return o5.verdict in ("VERIFIED", "PROVISIONAL")


def _check_eligibility_and_rollback(
    ctx: OrchestratorContext, snap, precheck_all_reasons: Optional[str],
) -> Optional[HandlerResult]:
    """Gate finalization and route ineligible work to its existing rollback."""
    eligibility = finalize_pipeline.check_finalize_eligibility(ctx.workspace)
    if eligibility["eligible"]:
        return None
    if _eligibility_is_harness_refreezable(eligibility) and _self_heal_digest_scheme_drift(
        ctx, snap
    ):
        # The re-run froze a fresh snapshot + evidence under the current
        # scheme.  Re-run artifact prep and the authoritative gate once more;
        # on failure fall through to the normal rollback path.
        precheck_all_reasons = _run_finalize_prep(ctx)
        eligibility = finalize_pipeline.check_finalize_eligibility(ctx.workspace)
        if eligibility["eligible"]:
            return None
    gate_tag = eligibility.get("gate") or "finalize_eligibility"
    finalize_pipeline.record_rollback(
        ctx.workspace,
        rollback_state=eligibility["rollback_state"],
        reason=eligibility["reason"],
        gate=gate_tag,
    )
    loop_result = _nonconvergent_loop_result(
        ctx.workspace, ctx.lane, eligibility, gate_tag,
    )
    if loop_result is not None:
        return loop_result
    return _record_finalize_rollback(
        ctx.workspace, ctx.lane, eligibility, precheck_all_reasons,
    )


def _static_failure_reason(static_report: dict) -> str:
    """Summarize failing scoped static-check reports without changing the gate."""
    summaries = [
        f"{Path(item.get('root', '')).name}: {item.get('report', {}).get('summary', 'failed')}"
        for item in static_report.get("reports", [])
        if not item.get("report", {}).get("passed")
    ]
    return static_report.get("error") or "; ".join(summaries) or "unknown failure"


# 2026-08-30 (PR13 WP-A / A.4.2): finalize hard-fails must land on an FSM
# transition, never on a bare HandlerResult.ret(7) — a bare exit leaves no
# transition, so resume re-ran the same failing promotion with no recovery
# path.  Delivery-contract violations the worker can repair (unrecognized
# delivery files) route to await_worker with the filenames and repair
# guidance; harness-side gates (static safety) and KB-merge failures route to
# await_user_decision.
_DELIVERY_WORKER_FIXABLE_MARKERS = (
    "unrecognized file",
    "unrecognized hidden artifact",
)


def _finalize_hard_fail_transition(
    workspace: Path, *, target: str, rationale: str
) -> HandlerResult:
    """Record a finalize -> target FSM transition for a hard-fail, then continue.

    Returning ``HandlerResult.cont()`` after recording the transition mirrors
    the same-signature park route: the main loop re-reads the state and
    dispatches await_worker / parks at await_user_decision instead of the
    process dying on a bare exit code.
    """
    log.error(rationale)
    state_executor.record_transition(
        workspace,
        state_executor.TransitionDecision(
            next_state=target,
            matched_transition_index=-1,
            rationale=rationale,
            from_state="finalize",
            handoff="",
        ),
    )
    return HandlerResult.cont()


def _check_delivery_static_safety(workspace: Path, lane) -> Optional[HandlerResult]:
    """Run the scoped static safety gate before promotion and knowledge merge."""
    static_report = _run_delivery_static_check(workspace)
    if not static_report.get("passed"):
        reason = _static_failure_reason(static_report)
        log.error(
            "finalize static safety GATE failed; blocking promotion and KB merge: %s",
            reason,
        )
        events.emit(
            workspace,
            "orchestrator.finalize_static_check_failed",
            lane=lane,
            data={"reason": reason[:500]},
        )
        return _finalize_hard_fail_transition(
            workspace,
            target="await_user_decision",
            rationale=(
                "finalize static safety GATE failed; promotion and KB merge "
                f"blocked. Reason: {reason[:500]} — this is a harness-side "
                "safety net the worker cannot repair; routing to "
                "await_user_decision (write user_decision.md to adjudicate) "
                "instead of a bare exit 7."
            ),
        )
    if static_report.get("skipped"):
        log.info("finalize static safety GATE: skipped (%s)", static_report["skipped"])
    else:
        log.info(
            "finalize static safety GATE: PASS (%d delivery root(s))",
            len(static_report.get("reports", [])),
        )
    return None


def _log_finalize_promotion(finalize_report) -> None:
    """Log the existing archive-promotion outcome."""
    if finalize_report.skipped:
        log.info(f"finalize: skipped ({finalize_report.skip_reason})")
        return
    log.info(
        f"finalize: promoted {len(finalize_report.files_promoted)} files + "
        f"{len(finalize_report.dirs_promoted)} dirs → {finalize_report.archive_dir}"
    )
    for error in finalize_report.errors:
        log.info(f"finalize WARN: {error}")


def _finalize_kb_merge_error_route(workspace: Path, lane, error: Exception) -> HandlerResult:
    """Park a raised KB-merge failure at await_user_decision."""
    log.info(
        f"finalize KB merge ERROR: {error}; blocking transition to done — "
        "staying at finalize"
    )
    events.emit(
        workspace,
        "orchestrator.finalize_kb_merge_error",
        lane=lane,
        data={"error": str(error)[:500]},
    )
    return _finalize_hard_fail_transition(
        workspace,
        target="await_user_decision",
        rationale=(
            f"finalize KB merge raised {type(error).__name__}: "
            f"{str(error)[:500]} — KB-manager processing is harness-side; "
            "routing to await_user_decision (inspect .kb_merge_log.jsonl, "
            "then write user_decision.md) instead of a bare exit 7."
        ),
    )


def _finalize_kb_merge_failed_route(workspace: Path, lane, result: dict) -> HandlerResult:
    """Park an unsuccessful KB-merge result at await_user_decision."""
    log.info(
        "finalize KB merge FAILED (exit_code=%s); blocking transition to done "
        "— staying at finalize. Inspect .kb_merge_log.jsonl + retry orchestrator.",
        result.get("log_entry", {}).get("exit_code"),
    )
    events.emit(workspace, "orchestrator.finalize_kb_merge_failed", lane=lane, data=result)
    return _finalize_hard_fail_transition(
        workspace,
        target="await_user_decision",
        rationale=(
            "finalize KB merge FAILED (exit_code=%s); blocking transition "
            "to done. KB-manager processing is harness-side; routing to "
            "await_user_decision (inspect .kb_merge_log.jsonl, then write "
            "user_decision.md) instead of a bare exit 7."
            % result.get("log_entry", {}).get("exit_code")
        ),
    )


def _merge_finalize_knowledge(workspace: Path, lane) -> Optional[HandlerResult]:
    """Require successful KB-manager processing for a substantive update."""
    knowledge_update = workspace / "knowledge_update.md"
    if not knowledge_update.exists() or knowledge_update.stat().st_size < 100:
        return None
    if getattr(kb_invoke, "_prepare_existing_marker")(workspace):
        log.info("finalize: knowledge_update.md already merged (.kb_merged marker present)")
        return None
    log.info(
        "finalize KB merge GATE: invoking aog-knowledge-maintain "
        "(knowledge_update.md = %s bytes)...",
        knowledge_update.stat().st_size,
    )
    try:
        result = kb_invoke.merge_one(workspace)
    except Exception as error:
        return _finalize_kb_merge_error_route(workspace, lane, error)
    if not result.get("success"):
        return _finalize_kb_merge_failed_route(workspace, lane, result)
    log.info("finalize KB merge OK — proceeding to done")
    events.emit(
        workspace,
        "orchestrator.finalize_kb_merge_ok",
        lane=lane,
        data={"workspace": str(workspace)},
    )
    return None


def _unrecognized_delivery_files(worker_fixable: list[str]) -> list[str]:
    """Pull the offending delivery paths out of worker-fixable promote errors."""
    return [
        text.split(":", 1)[0][len("promote "):]
        if text.startswith("promote ") and ":" in text
        else text[:200]
        for text in worker_fixable
    ]


def _finalize_worker_fixable_route(workspace: Path, worker_fixable: list[str]) -> HandlerResult:
    """Hand unrecognized delivery files back to the worker for repair.

    Every blocking error is an unrecognized delivery file — the worker can
    repair it (delete the file or move it out of the delivery territory), so
    route back to await_worker with the filenames and the repair guidance
    instead of a bare exit 7.
    """
    files = _unrecognized_delivery_files(worker_fixable)
    return _finalize_hard_fail_transition(
        workspace,
        target="await_worker",
        rationale=(
            "finalize promotion blocked on %d unrecognized delivery "
            "file(s): %s. Repair guidance for the next worker round: "
            "delete each listed file from the workspace, or move it "
            "outside the delivery territory (kernel/, op_host/) when "
            "it is still needed at runtime, then hand off as usual — "
            "the delivery contract only ships recognized source files."
            % (len(worker_fixable), ", ".join(files))
        ),
    )


def _finalize_promotion_blocked_route(workspace: Path, texts: list[str]) -> HandlerResult:
    """Park promotion errors the worker cannot repair at await_user_decision."""
    return _finalize_hard_fail_transition(
        workspace,
        target="await_user_decision",
        rationale=(
            "finalize promotion reported %d blocking error(s) the worker "
            "cannot be expected to repair alone: %s — routing to "
            "await_user_decision (write user_decision.md to adjudicate) "
            "instead of a bare exit 7."
            % (
                len(texts),
                "; ".join(text[:300] for text in texts[:3]),
            )
        ),
    )


def _finalize_promotion_error_route(
    workspace: Path, lane, finalize_report, blocking_errors: list,
) -> HandlerResult:
    """Report blocking promotion errors and pick the matching repair route."""
    log.error(
        "finalize promotion reported %d error(s); refusing transition to done",
        len(finalize_report.errors),
    )
    events.emit(
        workspace,
        "orchestrator.finalize_promotion_failed",
        lane=lane,
        data={"errors": [str(error)[:500] for error in finalize_report.errors]},
    )
    texts = [str(error) for error in blocking_errors]
    worker_fixable = [
        text
        for text in texts
        if any(marker in text for marker in _DELIVERY_WORKER_FIXABLE_MARKERS)
    ]
    if len(worker_fixable) == len(texts):
        return _finalize_worker_fixable_route(workspace, worker_fixable)
    return _finalize_promotion_blocked_route(workspace, texts)


def _advance_finalize_state_to_done(
    workspace: Path, runtime_kwargs: dict,
) -> Optional[HandlerResult]:
    """Drive the state machine to done; return a result only when it errors."""
    try:
        decision = state_executor.next_state(
            workspace,
            "→ orchestrator: pipeline_done",
            dry_run=False,
            runtime_kwargs=runtime_kwargs,
        )
        log.info(
            f"route: {decision.from_state} → {decision.next_state} "
            f"({decision.rationale})"
        )
    except state_executor.StateMachineError as error:
        log.info(f"finalize→done routing error: {error}")
        return HandlerResult.ret(5)
    return None


def _route_finalize_to_done(
    workspace: Path, lane, runtime_kwargs: dict, finalize_report,
) -> HandlerResult:
    """Advance the FSM after successful promotion and knowledge processing."""
    # DEBT-100 git-add auto-stage is explicitly non-blocking (its WARN stays
    # in the report for audit) — a gitignore'd output/ tree makes git return
    # rc=1 and that must not block the done transition (2026-08-23 BAM: the
    # promotion itself archived 22 files successfully, then this single
    # non-blocking entry refused the transition).
    blocking_errors = [
        error
        for error in finalize_report.errors
        if "git-add auto-stage (non-blocking" not in str(error)
    ]
    if blocking_errors:
        return _finalize_promotion_error_route(
            workspace, lane, finalize_report, blocking_errors,
        )
    routing_result = _advance_finalize_state_to_done(workspace, runtime_kwargs)
    if routing_result is not None:
        return routing_result
    events.emit(
        workspace,
        "orchestrator.finalize_pipeline",
        lane=lane,
        data={
            "skipped": finalize_report.skipped,
            "files_promoted": len(finalize_report.files_promoted),
            "dirs_promoted": len(finalize_report.dirs_promoted),
            "errors": len(finalize_report.errors),
        },
    )
    return HandlerResult.cont()


def _promote_and_route(ctx: OrchestratorContext) -> HandlerResult:
    """Promote eligible output, process knowledge, then move the FSM to done."""
    static_result = _check_delivery_static_safety(ctx.workspace, ctx.lane)
    if static_result is not None:
        return static_result
    finalize_report = finalize_pipeline.finalize_op(ctx.op, ctx.workspace)
    _log_finalize_promotion(finalize_report)
    if finalize_report.errors:
        return _route_finalize_to_done(
            ctx.workspace, ctx.lane, ctx.runtime_kwargs, finalize_report,
        )
    merge_result = _merge_finalize_knowledge(ctx.workspace, ctx.lane)
    if merge_result is not None:
        return merge_result
    return _route_finalize_to_done(
        ctx.workspace, ctx.lane, ctx.runtime_kwargs, finalize_report,
    )
