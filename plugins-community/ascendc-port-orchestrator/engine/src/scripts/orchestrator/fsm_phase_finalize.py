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

import importlib
import json
import os
from pathlib import Path
from typing import Optional

import events
import finalize_pipeline
import kb_invoke
import perf_irm_provenance
import phase_o5
import phase_o5_perf_capture
import phase_o5_runner
import schema_norm
import state_executor
from logging_config import get_logger

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
    declared = plugin.kernel_cpp_dirs()
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
    _r = _check_eligibility_and_rollback(ctx, precheck_all_reasons)
    if _r is not None:
        return _r
    # Archive promotion + P0aax KB-merge gate + finalize→done routing.
    return _promote_and_route(ctx)


def _o5_runner_for_workspace(workspace: Path):
    """Reload the O5 runner and select its truth-source-specific entrypoint."""
    importlib.reload(phase_o5_runner)
    if phase_o5.expected_truth_source(workspace) == "backward_autograd":
        return phase_o5_runner.backward_verify_runner
    return phase_o5_runner.ssh_runner


def _run_o5_verification(ctx: OrchestratorContext, runner):
    """Run the independent O5 verifier through the selected runner."""
    return phase_o5.post_verify_for_finalize(
        ctx.workspace, ctx.op, lane=ctx.lane, runner=runner,
    )


def _retry_infra_o5_verification(ctx: OrchestratorContext, o5, runner):
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
        o5 = _run_o5_verification(ctx, runner)
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


def _rollback_o5_mismatch(workspace: Path, o5) -> HandlerResult:
    """Record the existing mismatch rollback to ``await_worker``."""
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


def _handle_o5_mismatch(
    ctx: OrchestratorContext, snap, o5,
) -> HandlerResult:
    """Emit an O5 mismatch and either halt at cap or roll back the worker."""
    print(phase_o5.format_block_message(ctx.op, o5))
    events.emit(
        ctx.workspace,
        "orchestrator.phase_o5_block",
        lane=ctx.lane,
        data={
            "verdict": o5.verdict,
            "claimed": o5.claimed,
            "measured": o5.measured,
            "mismatches": o5.mismatches,
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


def _o5_post_verify(ctx: OrchestratorContext, snap) -> Optional[HandlerResult]:
    """Run O5 and dispatch its result without allowing unknown verdicts through."""
    runner = _o5_runner_for_workspace(ctx.workspace)
    o5 = _retry_infra_o5_verification(ctx, _run_o5_verification(ctx, runner), runner)
    if o5.verdict == "MISMATCH":
        return _handle_o5_mismatch(ctx, snap, o5)
    if o5.verdict == "RUNNER_FAILED":
        return _handle_o5_runner_failed(ctx, snap, o5)
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


def _rollback_o5_runner_failure(workspace: Path, o5) -> HandlerResult:
    """Record the existing runner-failure rollback and retry accounting."""
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


def _check_eligibility_and_rollback(
    ctx: OrchestratorContext, precheck_all_reasons: Optional[str],
) -> Optional[HandlerResult]:
    """Gate finalization and route ineligible work to its existing rollback."""
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
        return HandlerResult.ret(7)
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
        return HandlerResult.ret(7)
    if not result.get("success"):
        log.info(
            "finalize KB merge FAILED (exit_code=%s); blocking transition to done "
            "— staying at finalize. Inspect .kb_merge_log.jsonl + retry orchestrator.",
            result.get("log_entry", {}).get("exit_code"),
        )
        events.emit(workspace, "orchestrator.finalize_kb_merge_failed", lane=lane, data=result)
        return HandlerResult.ret(7)
    log.info("finalize KB merge OK — proceeding to done")
    events.emit(
        workspace,
        "orchestrator.finalize_kb_merge_ok",
        lane=lane,
        data={"workspace": str(workspace)},
    )
    return None


def _route_finalize_to_done(
    workspace: Path, lane, runtime_kwargs: dict, finalize_report,
) -> HandlerResult:
    """Advance the FSM after successful promotion and knowledge processing."""
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
    merge_result = _merge_finalize_knowledge(ctx.workspace, ctx.lane)
    if merge_result is not None:
        return merge_result
    return _route_finalize_to_done(
        ctx.workspace, ctx.lane, ctx.runtime_kwargs, finalize_report,
    )
