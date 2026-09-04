# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""fsm_phase_spawn.py — the per-iteration spawn + post-spawn FSM cluster
(DEBT-201 god-FUNCTION split).

Extracted VERBATIM from `run_single_op`'s while-loop body (orchestrator.py) as
part of the god-function decomposition via dependency inversion. This is the
slice that, for one non-terminal / non-pause / non-finalize iteration, runs:

  next_agent lookup → iter-cap check (P0y legitimate-exhaustion → finalize
  PARTIAL_PERSIST; else exit 2) → spawn_index compute + stale-output archive
  → plan_only short-circuit → first-spawn self-critic (lifetime-gated)
  → agent_dispatch.spawn_for_state (StreamSilenceTimeout retry budget /
    generic-exception → exit 3) → lifetime_spawn_count persist
  → --kw-1-only terminal → schema_norm (SchemaNormalizationError → exit 4)
  → O3.kernel_regression gate (block → exit 5) → canonical-handoff extraction
  → task#56 perf_checkpoint → state_executor.next_state (StateMachineError →
    exit 5) + transition record.  KB merge is deliberately deferred to the
    finalize handler, after independent verification and safety gates pass.

`handle_spawn` is a THIN orchestrator (like fsm_phase_finalize.handle_finalize)
over three sub-phases — each <200 lines per the architecture-lint god-function
bar:
  _route_agent_and_itercap → agent lookup + iter-cap routing (+ P0y).
  _prepare_and_spawn        → spawn_index/stale-archive/plan/self-critic +
                              the spawn call with retry/exception handling.
  _post_spawn_transition    → kw-1-only / schema / O3 / handoff / perf-checkpoint
                              / next_state + record.

DEPENDENCY CONTRACT (why the monkeypatch surface still bites):
- Orchestrator-MODULE-LEVEL deps (`_is_legitimate_pipeline_exhaustion`,
  `_record_partial_persist_finalize`, `_archive_stale_outputs_before_spawn`,
  `_mark_agent_died`, `_load_silence_retry_count`, `_bump_silence_retry_count`,
  `extract_canonical_handoff`, `_CANONICAL_HANDOFF_PREFIXES`, `_resolve_env`,
  `_agent_timeout_for_target`) are reached via `ctx.<name>` read-through
  properties resolving the live orchestrator module, so
  `monkeypatch.setattr(orchestrator, X)` still bites. Per-run inputs
  (op/workspace/lane/plan_only/kw_1_only/runtime_kwargs) come off `ctx`.
- SIBLING modules (state_executor, events, critic_invoke, agent_dispatch,
  schema_norm, perf_checkpoint, kb_invoke) are imported
  directly here — the SAME module objects the tests patch, so
  `monkeypatch.setattr(<sibling>, ...)` bites the direct reference with no
  indirection needed.

MUTABLE LOOP STATE — threaded EXPLICITLY through `ctx`, never globals:
- `ctx.spawn_count`         — advanced (+= 1) after a successful spawn.
- `ctx.last_handoff`        — set from the extracted canonical handoff (and by
                              the P0y legitimate-exhaustion branch).
- `ctx.lifetime_spawn_count`— read for the first-spawn self-critic gate.
The driver loop re-reads `ctx.spawn_count` / `ctx.last_handoff` after this
handler returns, so the observable loop behavior is identical to the inline body.

The block BODY is byte-identical to the original; the only mechanical edits are
the `ctx.` rebinds above and each `continue` / `return N` →
`return HandlerResult.cont()` / `.ret(N)`.
"""
from __future__ import annotations
import logging

import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import agent_dispatch
import critic_invoke
import events
import perf_checkpoint
import schema_norm
import state_executor
from logging_config import get_logger
from backends import base as _backend_base
from backends.base import Envelope, StreamSilenceTimeout
from backends.opencode_backend import CandidateTreeStallTimeout

from fsm_context import HandlerResult, OrchestratorContext

log = get_logger(__name__)

# Budget for in-place respawns when a spawn fails at the backend/transport
# level without producing any canonical handoff (see _post_spawn_transition).
NOOP_FAILURE_RETRY_MAX = 4

# A.5 (2026-08-30, 2_FFN_evo opencode line): provider/API-error spawn retry.
# The retry decision trusts ONLY the structured Envelope.api_error_status
# field — the one control-flow channel the Envelope contract
# (backends/base.py) reserves for retry/quota decisions.  Error TEXT
# ("Unexpected server error" etc.) is vendor-version-dependent and forgeable,
# so it must never be a retry criterion.  Bounded: capped respawns, linear
# backoff, an independent per-(state, status) counter, and budget exhaustion
# escalates to the agent_died abort path.
API_ERROR_RETRY_MAX_ENV = "AOG_API_ERROR_RETRY_MAX"
API_ERROR_RETRY_MAX = 2
API_ERROR_BACKOFF_SEC_ENV = "AOG_API_ERROR_BACKOFF_SEC"
API_ERROR_BACKOFF_SEC = 30


def _result_metrics(result) -> tuple[Optional[float], Optional[float]]:
    """Return duration seconds and cost, tolerating non-Claude backends."""
    duration_ms = getattr(result, "duration_ms", None)
    duration_s = (duration_ms / 1000) if duration_ms is not None else None
    cost_usd = getattr(result, "cost_usd", None)
    return duration_s, cost_usd


def _file_stamp(path) -> tuple[int, int] | None:
    try:
        st = path.stat()
        return (int(st.st_mtime_ns), int(st.st_size))
    except Exception:
        return None


def _maybe_seed_branch_base(ctx: OrchestratorContext, workspace, snap) -> None:
    """Optionally seed the first migration worker from a proven target archive.

    Seeding is default-off (``AOG_DEBT203_SEED_ENABLED``), provenance-labeled,
    and fail-open. The seed lands in ``branched_from_kernel/``; the worker still
    writes the candidate output and must pass the normal precision and
    determinism gates for the current operator.
    """
    if not (
        ctx.spawn_count == 0
        and ctx.lifetime_spawn_count == 0
        and snap.current_state == "await_worker"
    ):
        return
    try:
        import provenance_seed as seed

        if not seed.seed_enabled():
            return
        from finalize_dispatch import _get_active_plugin
        from finalize_pipeline import _PROJECT_ROOT

        plugin = _get_active_plugin(workspace)
        if plugin is None or plugin.name != "port_a3_to_a5":
            return

        archive_project = None
        try:
            env = ctx.resolve_env()
            archive_project = getattr(env, "archive_project", None) if env else None
        except Exception:
            archive_project = None
        if not archive_project:
            archive_project = (
                plugin.archive_project_subdir()
            ) or "a3_to_a5_port"
        archive_root = _PROJECT_ROOT / "output" / archive_project / "src" / "kernels"
        reference = seed.maybe_seed_from_similar_op(
            ctx.op, workspace, archive_root=archive_root
        )
        if reference is not None:
            log.info(
                "seeded provenance-labeled branch base from %s "
                "(similarity=%.3f fitness=%.3f)",
                reference.op,
                reference.similarity,
                reference.fitness,
            )
    except Exception as exc:
        log.warning("target-archive seed hook (non-fatal): %s", exc)


def handle_spawn(ctx: OrchestratorContext, snap) -> HandlerResult:
    """Run the spawn + post-spawn slice for one loop iteration.

    Thin orchestrator over the sub-phases. Returns a HandlerResult telling the
    driver loop to `continue` (re-snapshot) or `return <exit_code>`.
    Reads/advances loop state on `ctx` (spawn_count / last_handoff /
    lifetime_spawn_count).
    """
    agent_type = state_executor.next_agent(snap.current_state)
    if agent_type is None:
        log.error(f"no agent for state {snap.current_state}")
        return HandlerResult.ret(2)

    # Iter-cap routing (+ P0y legitimate-exhaustion → finalize PARTIAL_PERSIST).
    _r = _route_itercap(ctx, snap)
    if _r is not None:
        return _r

    # spawn_index compute + stale-archive + plan_only + self-critic + spawn.
    spawn_index, result, _r = _prepare_and_spawn(ctx, snap, agent_type)
    if _r is not None:
        return _r

    # post-spawn: kw-1-only / schema / O3 / handoff / perf-checkpoint /
    # next_state + record.  Knowledge remains staged until finalize gates pass.
    return _post_spawn_transition(ctx, snap, agent_type, result, spawn_index)


def _route_itercap(ctx: OrchestratorContext, snap) -> Optional[HandlerResult]:
    """Iter-cap check. Returns a HandlerResult to short-circuit (P0y
    legitimate-exhaustion → continue; else exit 2), or None to proceed."""
    workspace = ctx.workspace
    if not state_executor.at_iter_cap(workspace, snap.current_state):
        return None

    # P0y (2026-05-05): "never let PARTIAL pass" terminal-route.
    # When iter_cap hits on await_researcher AND researcher actually
    # ran (cann_strategy_inference.md exists) AND probe verdict was
    # requirement (probe_result.json says so), the full pipeline IS
    # exhausted — this is the LEGITIMATE PARTIAL_PERSIST terminal
    # state per V3.8.8. Routing to finalize with persist_verdict
    # in verification.json is correct, NOT error code 2.
    #
    # Origin: op#28 multimodal_rope 2026-05-05. After probe →
    # researcher → worker → probe → researcher (full V3.8.8 cycle),
    # researcher cap hit. Orchestrator returned error 2 instead of
    # routing to finalize PARTIAL.
    counter_name = snap.current_state[len('await_'):]
    count = snap.iter_counts.get(counter_name, 0)
    cap = state_executor.iter_cap(snap.current_state, workspace=workspace)

    if ctx.is_legitimate_pipeline_exhaustion(workspace, snap.current_state):
        log.info(
            f"iter_cap hit for {snap.current_state} "
            f"(count={count}, cap={cap}); pipeline exhausted with "
            f"researcher evidence → routing to finalize PARTIAL_PERSIST"
        )
        ctx.record_partial_persist_finalize(workspace, snap.current_state, count, cap)
        # P0ww (2026-05-06): do NOT return 0 here. The state-log entry
        # transitions current_state to 'finalize', but exit-after-tag
        # leaves the workspace WITHOUT calling finalize_pipeline.finalize_op
        # → no archive promotion, no .finalized marker, no done state.
        # Caught 2026-05-06 via 9_topktopp run that exited with
        # persist_verdict tagged but no canonical archive promoted.
        # Continue the loop so next iter sees state=finalize and the
        # finalize branch (line ~320) runs finalize_pipeline.finalize_op
        # → archive promotion + .finalized-* marker + done transition.
        ctx.last_handoff = "[P0y orchestrator] iter_cap exhausted with full-pipeline evidence"
        return HandlerResult.cont()

    log.error(
        f"iter_cap hit for {snap.current_state} "
        f"(count={count}, cap={cap})"
    )
    return HandlerResult.ret(2)


def _prepare_and_spawn(
    ctx: OrchestratorContext, snap, agent_type: str,
) -> Tuple[int, Optional["Envelope"], Optional[HandlerResult]]:
    """Compute spawn_index, archive stale outputs, plan_only short-circuit,
    first-spawn self-critic, then spawn the agent.

    Returns (spawn_index, result, None) on a successful spawn (result advances
    ctx.spawn_count + lifetime persist here), or (spawn_index, None,
    HandlerResult) to short-circuit the loop (plan_only exit 0; silence-timeout
    continue/exit 3; spawn exception exit 3).
    """
    workspace = ctx.workspace

    # P0yy (2026-05-06): retired `post_iter_cap_warning` and `pre_finalize`
    # critics. Both are pre-Python-era checks that have been superseded by
    # mechanical gates with explicit replacements:
    #   - post_iter_cap_warning → P0y / P0aa / P0ww (orchestrator routes
    #     iter_cap-exhausted ops to finalize PARTIAL_PERSIST mechanically)
    #   - pre_finalize → P0ee (pass-count consistency) + P0ff (rollback
    #     gate at finalize for non-PASS without PARTIAL_PERSIST) + P0kk
    #     (Phase O5 independent post-verify) + P0qq (in-context
    #     introspection)
    # Both critics fired on every op as it approached terminal and produced
    # ~0% actionable BLOCK/WARN output (they narrated what mechanical gates
    # already enforced). Net cost: ~8min subprocess wallclock + token spend
    # per op for no catches. Retired per agreement with DS agent + user
    # direction (limited quota). The remaining `pre_phase_o4_first_spawn`
    # critic (below) catches a different class — cross-artifact reasoning
    # the YAML can't see (e.g. 9_topktopp BLOCK that prevented kw-2 spawn
    # when researcher had emitted PARTIAL_PERSIST exhaustion).

    # Compute spawn_index (1-based) for G7 slug
    counter_key = snap.current_state[len("await_"):] if snap.current_state.startswith("await_") else snap.current_state
    spawn_index = snap.iter_counts.get(counter_key, 0) + 1

    # P0v (2026-05-05): archive stale "optional" outputs before agent spawn,
    # so post-spawn state machine path_exists checks only see fresh files
    # written by THIS spawn. Without this, op#9 had a stale
    # optimization_directive.md from a prior session causing
    # await_researcher → await_worker false-match.
    ctx.archive_stale_outputs_before_spawn(workspace, snap.current_state, spawn_index)

    if ctx.plan_only:
        log.info(f"PLAN: would spawn {agent_type} as {counter_key}-{spawn_index}")
        return spawn_index, None, HandlerResult.ret(0)

    # First-spawn-of-op self-critic — gate on LIFETIME spawn count, not
    # per-process spawn_count. Otherwise every --resume restarts a fresh
    # orchestrator process with spawn_count=0 and re-fires self-critic
    # (~20-30s + $0.50-1.50 per resume), even though the op already had
    # N prior spawns where self-critic already ran. P135.SC (2026-05-18).
    if ctx.spawn_count == 0 and ctx.lifetime_spawn_count == 0:
        # Infra escape hatch (2026-07-02, upstreamed from cannbot bundle):
        # when the orchestrator runs inside an agent's background-task context
        # with the plugin installed in the config-dir, the fire_critic
        # harness skill child (bypassPermissions) can
        # re-enter and re-launch the orchestrator (fork-bomb -> resource
        # exhaustion -> SIGKILL of this parent). The pre-spawn self-critic is
        # non-fatal-by-design and orthogonal to port correctness, so allow
        # gating it off. Default OFF — byte-identical unless the env is set.
        if os.environ.get("AOG_DISABLE_PRESPAWN_CRITIC") == "1":
            log.info("skipping self-critic (AOG_DISABLE_PRESPAWN_CRITIC=1 — fork-bomb escape hatch)")
        else:
            log.info("firing self-critic (pre_phase_o4_first_spawn)")
            try:
                critic_invoke.fire_critic(workspace, "pre_phase_o4_first_spawn")
            except Exception as e:
                log.warning(f"critic pre_phase_o4_first_spawn failed ({e}); continuing")
    elif ctx.spawn_count == 0 and ctx.lifetime_spawn_count > 0:
        log.info(
            f"skipping self-critic (lifetime_spawn_count={ctx.lifetime_spawn_count}, "
            f"already fired on a prior orchestrator process for this op)"
        )

    _maybe_seed_branch_base(ctx, workspace, snap)

    # Build brief + spawn (foreground)
    log.info(f"spawning {agent_type} (G7 slug index {spawn_index})...")
    events.emit(workspace, "orchestrator.spawn.start", lane=ctx.lane,
                data={"agent_type": agent_type, "spawn_index": spawn_index})
    try:
        # P0aau: DS/V4 backend gets longer timeout (10800s vs 5400s)
        _env = ctx.resolve_env()
        _timeout = ctx.agent_timeout_for_target(_env.target if _env else "")
        progress_pre_spawn = _file_stamp(workspace / "PROGRESS.md")
        result = agent_dispatch.spawn_for_state(
            ctx.op, workspace, snap.current_state,
            lane=ctx.lane, spawn_index=spawn_index,
            timeout_sec=_timeout,
            handoff_from_prior=ctx.last_handoff,
        )
        try:
            result.progress_pre_spawn_stamp = progress_pre_spawn
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
    except CandidateTreeStallTimeout as e:
        # A.7 (2026-08-30, 2_FFN_evo close-out): the stream stayed alive but
        # the candidate tree digest did not move for the whole stall window
        # (default 45 min, AOG_TREE_STALL_TIMEOUT_SEC) — the worker is
        # reading/planning in a loop without writing.  The backend already
        # SIGTERMed the subprocess.  Complementary to the stream-silence
        # watchdog.  The tree diff can be gamed, so this watchdog has NO
        # independent parking surface: every firing feeds the P0-1
        # same-signature family counter and escalation rides its threshold.
        tree_sha = e.tree_sha256 or "unknown"
        entry = state_executor.record_same_signature_failure(
            workspace, "engine",
            f"candidate-tree stall watchdog: digest unchanged for {e.stall_seconds:.0f}s",
            tree_sha,
        )
        count = int(entry.get("count", 0))
        threshold = state_executor.same_signature_park_threshold("engine")
        _append_tree_stall_progress_note(workspace, agent_type, e, count, threshold)
        events.emit(workspace, "orchestrator.spawn.tree_stall_killed", lane=ctx.lane,
                    data={"agent_type": agent_type,
                          "stall_seconds": e.stall_seconds,
                          "tree_sha256": tree_sha,
                          "same_signature_count": count,
                          "park_threshold": threshold})
        if count >= threshold:
            log.info(
                f"candidate-tree stall watchdog escalated: {count}/{threshold} "
                f"consecutive stalls on the same tree; giving up. {e}"
            )
            ctx.mark_agent_died(
                workspace, snap.current_state,
                f"candidate-tree stall watchdog: {count} consecutive stalls "
                f"(same-signature family threshold {threshold}); worker makes "
                f"no candidate progress — needs a human",
            )
            events.emit(workspace, "orchestrator.spawn.failed", lane=ctx.lane,
                        data={"agent_type": agent_type,
                              "reason": "candidate-tree stall same-signature threshold reached"})
            return spawn_index, None, HandlerResult.ret(3)
        log.info(
            f"candidate-tree stall watchdog fired on {agent_type} "
            f"({count}/{threshold} consecutive): {e}. Respawning fresh."
        )
        return spawn_index, None, HandlerResult.cont()  # loop back — same state, fresh spawn
    except StreamSilenceTimeout as e:
        # P0aal-2 (2026-05-19): stdout silence detected mid-work.
        # Subprocess already SIGTERMed by transport. Respawn up to
        # STREAM_SILENCE_RETRY_MAX times before giving up.
        silence_retries = ctx.load_silence_retry_count(workspace, snap.current_state)
        events.emit(workspace, "orchestrator.spawn.silence_killed", lane=ctx.lane,
                    data={"agent_type": agent_type,
                          "silent_seconds": e.silent_seconds,
                          "last_event_type": e.last_event_type,
                          "retry": silence_retries + 1,
                          "max_retries": _backend_base.STREAM_SILENCE_RETRY_MAX})
        if silence_retries >= _backend_base.STREAM_SILENCE_RETRY_MAX:
            log.info(f"silence-timeout exceeded retry budget "
                  f"({silence_retries}/{_backend_base.STREAM_SILENCE_RETRY_MAX}); "
                  f"giving up. {e}")
            ctx.mark_agent_died(workspace, snap.current_state,
                             f"silence-timeout retry budget exhausted: {e}")
            events.emit(workspace, "orchestrator.spawn.failed", lane=ctx.lane,
                        data={"agent_type": agent_type,
                              "reason": f"silence-timeout retry budget exhausted: {e}"})
            return spawn_index, None, HandlerResult.ret(3)
        log.info(f"silence-timeout #{silence_retries + 1}/"
              f"{_backend_base.STREAM_SILENCE_RETRY_MAX} on "
              f"{agent_type}: {e}. Respawning fresh.")
        ctx.bump_silence_retry_count(workspace, snap.current_state)
        return spawn_index, None, HandlerResult.cont()  # loop back — same state, fresh spawn
    except Exception as e:
        log.info(f"spawn failed: {e}")
        ctx.mark_agent_died(workspace, snap.current_state, str(e))
        events.emit(workspace, "orchestrator.spawn.failed", lane=ctx.lane,
                    data={"agent_type": agent_type, "reason": str(e)[:500]})
        return spawn_index, None, HandlerResult.ret(3)

    ctx.spawn_count += 1
    # P94 INFRA-BLAME-LOOP fix: persist lifetime spawn count to
    # .opgen_state.json. Cold-start backup preserves this file
    # (per _cold_start_reset_workspace contract), so the count
    # survives across episodes. Surfaces hidden cost when an op
    # racks up many spawns across cold-starts.
    try:
        _state_fp_inc = workspace / ".opgen_state.json"
        if _state_fp_inc.is_file():
            _state_obj_inc = json.loads(_state_fp_inc.read_text())
            _state_obj_inc["lifetime_spawn_count"] = (
                int(_state_obj_inc.get("lifetime_spawn_count", 0)) + 1
            )
            _state_fp_inc.write_text(json.dumps(_state_obj_inc, indent=2))
    except Exception as exc:
        # don't block orchestrator on state-update failure
        log.warning("cannot persist lifetime_spawn_count: %s", exc)
    duration_s, cost_usd = _result_metrics(result)
    duration_label = f"{duration_s:.1f}" if duration_s is not None else "unknown"
    cost_label = f"{cost_usd:.2f}" if cost_usd is not None else "unknown"
    log.info(
        f"{agent_type} returned: success={result.success} "
        f"is_error={result.is_error} duration_s={duration_label} "
        f"cost_usd={cost_label}"
    )
    events.emit(workspace, "orchestrator.spawn.complete", lane=ctx.lane,
                data={"agent_type": agent_type, "success": result.success,
                      "duration_s": duration_s,
                      "cost_usd": cost_usd})
    return spawn_index, result, None


def _stop_gate_blocked(
    ctx: OrchestratorContext, agent_type: str, spawn_index: int,
) -> Optional[HandlerResult]:
    """HandlerResult when this spawn's stop gate refused its artifacts, else None.

    A rejected stop gate must STOP the run, not merely be logged. Claude Code enforces these
    gates through SubagentStop, which blocks the sub-agent from exiting. Harnesses without
    that event get them from the dispatcher instead (agent_dispatch._run_stop_gate), which can
    only mark the result — and marking was not enough: `result.is_error` had no consumer here,
    the canonical handoff line sits at the END of output_text so a prepended failure reason
    does not disturb extraction, and the marker file was read by nothing. The run therefore
    advanced its state on artifacts the gate had just refused.
    """
    workspace = ctx.workspace
    gate_marker = workspace / f".agent_gate_stop_failed_{agent_type}"
    if not gate_marker.is_file():
        return None
    try:
        reason = gate_marker.read_text(errors="replace").strip()[:800]
    except OSError as exc:
        # The marker exists, so the gate DID refuse this spawn; an unreadable marker must
        # fail closed here, not crash the orchestrator loop.
        reason = f"<stop-gate marker unreadable: {exc}>"
    log.error("[stop-gate] refusing to advance state after %s: %s", agent_type, reason)
    events.emit(
        workspace, "orchestrator.stop_gate_blocked", lane=ctx.lane,
        data={"agent_type": agent_type, "spawn_index": spawn_index, "reason": reason},
    )
    log.error(
        "stop gate rejected %s (spawn %s); artifacts were not accepted and the state was "
        "NOT advanced.\n%s", agent_type, spawn_index, reason,
    )
    # Not cleared here: the marker is the durable record that this spawn's artifacts were
    # never accepted, and a resume that does not re-run the agent must not step over it.
    # The next dispatch of this agent type clears it before the agent starts
    # (agent_dispatch._clear_stop_gate_marker), so a retry is judged by its own gate rather
    # than inheriting this verdict.
    return HandlerResult.ret(7)


def _capture_canonical_handoff(ctx: OrchestratorContext, result, workspace) -> None:
    """Populate ctx.last_handoff from worker stdout, with a PROGRESS.md-tail fallback.

    2026-05-13 (rms_norm_quant gap): some workers write the canonical handoff into
    PROGRESS.md but their backend result.output_text doesn't include it (the
    final response was a Bash command call, not a text emission containing the marker).
    Fall back to scanning the workspace PROGRESS.md tail before routing to abort.
    """
    ctx.last_handoff = ctx.extract_canonical_handoff(result.output_text)
    if ctx.last_handoff and ctx.last_handoff != result.output_text.strip():
        log.info(f"extracted handoff: {ctx.last_handoff[:120]}")
        return
    progress_md = workspace / "PROGRESS.md"
    if not progress_md.exists():
        return
    pre_stamp = getattr(result, "progress_pre_spawn_stamp", None)
    post_stamp = _file_stamp(progress_md)
    if pre_stamp == post_stamp:
        log.info("PROGRESS.md tail fallback skipped: file was not updated by this spawn")
        return
    progress_tail = "\n".join(progress_md.read_text().splitlines()[-50:])
    progress_handoff = ctx.extract_canonical_handoff(progress_tail)
    if progress_handoff and any(
        progress_handoff.startswith(p) for p in ctx.canonical_handoff_prefixes
    ):
        ctx.last_handoff = progress_handoff
        log.info(
            f"extracted handoff from PROGRESS.md tail (stdout had none): "
            f"{ctx.last_handoff[:120]}"
        )


def _run_perf_checkpoint(ctx: OrchestratorContext, workspace) -> None:
    """Task #56 (2026-05-29): IL perf-iter precision checkpoint + advance.

    After the probe agent updates verification.json, run the hill-climbing checkpoint:
    first clean PASS → snapshot kernel/ → .precision_baseline/; on a post-re-emit
    re-entry → 3-state advance (regress→revert, faster→update best, not-faster→revert).
    Reverts restore the best-known-good kernel + verification.json so the downstream
    next_state/finalize sees the protected state, and are tagged `perf_regression_revert`
    (consumes budget, caps the dice-roll loop). Closes #55 (known-good never lost to an
    in-place re-emit overwrite). Non-fatal: any error returns NOOP and the loop proceeds
    unchanged.
    """
    try:
        _ckpt = perf_checkpoint.checkpoint_and_advance(workspace)
        if _ckpt.action != perf_checkpoint.Action.NOOP:
            log.info(
                f"[task#56 perf_checkpoint] {_ckpt.action.value}: "
                f"{_ckpt.detail}"
            )
            events.emit(workspace, "orchestrator.perf_checkpoint", lane=ctx.lane,
                        data={"action": _ckpt.action.value,
                              "reverted": _ckpt.reverted,
                              "consumes_budget": _ckpt.consumes_budget,
                              "rollback_kind": _ckpt.rollback_kind,
                              "consecutive_no_improve": _ckpt.consecutive_no_improve,
                              "baseline_ratio": _ckpt.baseline_ratio,
                              "current_ratio": _ckpt.current_ratio})
    except Exception as _e:  # noqa: BLE001 - non-fatal by design, see docstring
        log.warning(f"[task#56 perf_checkpoint] non-fatal: {_e!r}")


def _record_optimizer_kernel_signature(workspace, spawn_index: int) -> None:
    """Record the current kernel md5 before next_state (deterministic, fail-open).

    iter_cap await_optimizer graceful-finalize fix (2026-07-24): after an
    aog-kernel-optimizer spawn, record the CURRENT kernel md5 to the byte-identical
    convergence ledger BEFORE computing next_state, so the `optimizer_kernel_converged`
    FSM primitive (read by the await_optimizer finalize transition) sees THIS spawn's
    signature. Deterministic + fail-open (the recorder never raises); non-optimizer
    states are untouched.
    """
    try:
        import ko_variant_ledger
        ko_variant_ledger.record_optimizer_kernel_signature(
            workspace, spawn_index=spawn_index
        )
    except Exception as _e:  # noqa: BLE001 - non-fatal by design, see docstring
        log.warning(f"optimizer kernel-signature record (non-fatal): {_e!r}")


def _is_noop_spawn_failure(ctx: OrchestratorContext, result) -> bool:
    """Return whether a failed spawn produced no usable handoff."""
    if result is None or (result.success and not result.is_error):
        return False
    return not ctx.extract_canonical_handoff(result.output_text or "")


# ---------------------------------------------------------------------------
# A.5 — structured api_error_status retry (see module header constants)
# ---------------------------------------------------------------------------
def api_error_status_retryable(status) -> bool:
    """Whether a structured provider status is worth an in-place respawn.

    Rate limits and server-side 5xx are transient by class; everything else
    (4xx auth/contract, missing field, non-int) is NOT — the worker cannot
    fix those by being re-run.
    """
    if isinstance(status, bool) or not isinstance(status, int):
        return False
    return status == 429 or 500 <= status <= 599


def _api_error_retry_max() -> int:
    raw = os.environ.get(API_ERROR_RETRY_MAX_ENV)
    if raw is None:
        return API_ERROR_RETRY_MAX
    try:
        return max(int(raw.strip()), 0)
    except ValueError:
        return API_ERROR_RETRY_MAX


def _api_error_backoff_sec() -> int:
    raw = os.environ.get(API_ERROR_BACKOFF_SEC_ENV)
    if raw is None:
        return API_ERROR_BACKOFF_SEC
    try:
        return max(int(raw.strip()), 0)
    except ValueError:
        return API_ERROR_BACKOFF_SEC


def _sleep_backoff(seconds: float) -> None:
    """Seam kept tiny on purpose: unit tests monkeypatch this, never time.sleep."""
    if seconds > 0:
        time.sleep(seconds)


def _append_tree_stall_progress_note(workspace, agent_type: str,
                                     exc: CandidateTreeStallTimeout,
                                     count: int, threshold: int) -> None:
    """A.7: leave the stall evidence where the respawned worker reads it.

    PROGRESS.md is the durable worker-facing channel; the note tells the next
    spawn that a no-edit turn will be killed again (and how close the
    same-signature family is to parking).
    """
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    note = (
        f"\n\n## ORCHESTRATOR NOTE — candidate-tree stall watchdog ({stamp})\n"
        f"- spawn of `{agent_type}` was SIGTERMed: candidate tree digest "
        f"unchanged for {exc.stall_seconds:.0f}s (tree={exc.tree_sha256 or 'unknown'}).\n"
        f"- same-signature family count {count}/{threshold}; at the threshold "
        "the run parks for a user decision.\n"
        "- You MUST change candidate source files this round — another "
        "read/plan-only turn without tree changes will be killed again.\n"
    )
    try:
        with open(Path(workspace) / "PROGRESS.md", "a", encoding="utf-8") as handle:
            handle.write(note)
    except OSError as error:
        log.warning("cannot append tree-stall note to PROGRESS.md: %s", error)


def _api_error_retry_decision(
    ctx: OrchestratorContext, snap, agent_type: str, result,
) -> Optional[HandlerResult]:
    """HandlerResult for a retryable structured API error, else None.

    Consumes ONLY ``result.api_error_status`` (A.5).  An errored result whose
    text screams "Unexpected server error" but carries no structured status
    falls through to the legacy no-op-failure path unchanged.
    """
    status = getattr(result, "api_error_status", None)
    if not getattr(result, "is_error", False) or not api_error_status_retryable(status):
        return None
    workspace = ctx.workspace
    retry_key = f"{snap.current_state}__api_error_{status}"
    retries = ctx.load_silence_retry_count(workspace, retry_key)
    max_retries = _api_error_retry_max()
    if retries >= max_retries:
        log.info(
            f"api-error retry budget exhausted ({retries}/{max_retries}) for "
            f"{agent_type} (status={status}); escalating to agent_died."
        )
        ctx.mark_agent_died(
            workspace, snap.current_state,
            f"provider API error status={status} persisted across "
            f"{retries} retries; needs a human, not another respawn",
        )
        events.emit(workspace, "orchestrator.spawn.failed", lane=ctx.lane,
                    data={"agent_type": agent_type,
                          "reason": f"api-error status={status} retry budget exhausted"})
        return HandlerResult.ret(3)
    ctx.bump_silence_retry_count(workspace, retry_key)
    backoff = _api_error_backoff_sec() * (retries + 1)
    log.info(
        f"{agent_type} spawn hit retryable provider API error status={status}; "
        f"respawning in place ({retries + 1}/{max_retries}) after {backoff}s backoff."
    )
    events.emit(workspace, "orchestrator.spawn.api_error_retry", lane=ctx.lane,
                data={"agent_type": agent_type, "api_error_status": status,
                      "retry": retries + 1, "max_retries": max_retries,
                      "backoff_sec": backoff})
    _sleep_backoff(backoff)
    return HandlerResult.cont()  # loop back — same state, fresh spawn


def _post_spawn_transition(
    ctx: OrchestratorContext, snap, agent_type: str, result, spawn_index: int,
) -> HandlerResult:
    """After a successful spawn: kw-1-only terminal, schema-norm, O3 regression
    gate, canonical-handoff extraction, task#56 perf-checkpoint, next_state
    transition + record. Returns the loop HandlerResult."""
    workspace = ctx.workspace

    blocked = _stop_gate_blocked(ctx, agent_type, spawn_index)
    if blocked is not None:
        return blocked

    # A.5: a retryable STRUCTURED provider error (api_error_status 429/5xx)
    # gets its own bounded retry with backoff BEFORE the no-op-failure
    # fallback below — the fallback has no backoff and cannot tell a
    # transient 500 from a genuine empty turn.
    api_retry = _api_error_retry_decision(ctx, snap, agent_type, result)
    if api_retry is not None:
        return api_retry

    # 2026-08-27 (3_FusionAttention opencode line): a backend-level spawn
    # failure that produced no canonical handoff in stdout (e.g. opencode
    # "Cannot connect to API" — turn died on the first request) must NOT fall
    # through to the PROGRESS.md-tail fallback below.  The post-spawn hook
    # annotates PROGRESS.md ("did not log: no signed PROGRESS entry") on every
    # such turn, which dirties the mtime-based staleness guard in
    # _capture_canonical_handoff and lets a PRIOR turn's build-ready handoff
    # route the unchanged candidate into a full O5 build+evaluation, burning
    # one NPU eval cycle per retry forever.  Treat it like a silence-timeout:
    # bounded in-place respawn, same state.  Successful turns keep the legacy
    # PROGRESS.md fallback (rms_norm_quant gap) untouched.
    if _is_noop_spawn_failure(ctx, result):
        noop_key = f"{snap.current_state}__noop_failure"
        noop_retries = ctx.load_silence_retry_count(workspace, noop_key)
        if noop_retries >= NOOP_FAILURE_RETRY_MAX:
            log.info(
                f"no-op spawn failure retry budget exhausted "
                f"({noop_retries}/{NOOP_FAILURE_RETRY_MAX}) for {agent_type}; giving up."
            )
            ctx.mark_agent_died(
                workspace, snap.current_state,
                f"backend spawn failed {noop_retries} times with no handoff "
                f"(last: success={result.success} is_error={result.is_error})",
            )
            events.emit(workspace, "orchestrator.spawn.failed", lane=ctx.lane,
                        data={"agent_type": agent_type,
                              "reason": "no-op spawn failure retry budget exhausted"})
            return HandlerResult.ret(3)
        ctx.bump_silence_retry_count(workspace, noop_key)
        log.info(
            f"{agent_type} spawn failed with no usable handoff "
            f"(success={result.success} is_error={result.is_error}); "
            f"respawning in place ({noop_retries + 1}/{NOOP_FAILURE_RETRY_MAX}) "
            f"instead of routing on a stale PROGRESS.md handoff."
        )
        events.emit(workspace, "orchestrator.spawn.noop_retry", lane=ctx.lane,
                    data={"agent_type": agent_type, "spawn_index": spawn_index,
                          "retry": noop_retries + 1, "max_retries": NOOP_FAILURE_RETRY_MAX})
        return HandlerResult.cont()  # loop back — same state, fresh spawn

    # Zheng 2026-05-21: --kw-1-only stops after kernel-worker spawn 1.
    # Skip schema_norm + state transition + all downstream phases
    # (O5 deploy, KB merge, finalize, safety net). User inspects raw
    # worker output in a suffixed workspace for quality diagnosis.
    if ctx.kw_1_only and agent_type == "aog-kernel-worker" and spawn_index == 1:
        duration_s, cost_usd = _result_metrics(result)
        log.info(
            f"--kw-1-only: kw-1 returned, stopping before "
            f"schema_norm / state transition / O5 / KB merge / finalize. "
            f"Raw kw-1 output preserved at {workspace}."
        )
        events.emit(workspace, "orchestrator.kw_1_only_terminal", lane=ctx.lane,
                    data={"agent_type": agent_type,
                          "duration_s": duration_s,
                          "cost_usd": cost_usd})
        return HandlerResult.ret(0)

    # Schema normalize worker output (DEBT-074 family)
    try:
        norm_report = schema_norm.normalize_workspace(workspace, fail_strict=True)
        if norm_report.events:
            log.info(f"schema_norm: {len(norm_report.events)} normalizations")
    except schema_norm.SchemaNormalizationError as e:
        log.info(f"SCHEMA REJECT: {e}")
        return HandlerResult.ret(4)

    # Extract canonical handoff line for state machine routing.
    # state_machine.handoff_match uses .startswith() — worker stdout starts
    # with markdown summary, canonical handoff line is at the END.
    _capture_canonical_handoff(ctx, result, workspace)

    # Task #56: IL perf-iter precision checkpoint + advance (see helper).
    if snap.current_state == "await_probe":
        _run_perf_checkpoint(ctx, workspace)

    # iter_cap await_optimizer graceful-finalize fix (2026-07-24, see helper).
    if snap.current_state == "await_optimizer":
        _record_optimizer_kernel_signature(workspace, spawn_index)

    # Compute next state + record transition.
    # P0bb (2026-05-05): pass from_state=snap.current_state explicitly.
    # The snap was taken BEFORE the agent ran. After the agent rewrites
    # PROGRESS.md (appending its handoff), an internal re-derive via
    # current_state() would re-bootstrap and return the bootstrap target,
    # not the state we actually spawned the agent at. Explicit
    # from_state preserves the canonical iter-spawn-state.
    try:
        decision = state_executor.next_state(
            workspace, ctx.last_handoff,
            from_state=snap.current_state,
            dry_run=False,
            runtime_kwargs=ctx.runtime_kwargs,
        )
    except state_executor.StateMachineError as e:
        log.info(f"state machine error: {e}")
        return HandlerResult.ret(5)

    log.info(f"route: {decision.from_state} → {decision.next_state} ({decision.rationale})")
    events.emit(workspace, "orchestrator.transition", lane=ctx.lane,
                data={"from_state": decision.from_state,
                      "to_state": decision.next_state,
                      "rationale": decision.rationale[:300]})

    # Bottom of the loop body — the driver `continue`s to re-snapshot.
    return HandlerResult.cont()
