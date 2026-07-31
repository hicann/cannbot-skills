# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Scoped AscendC workflow engine.

New customer work is admitted only through ``--port-a3`` (arch22 to arch35)
or ``--backward``. Positional invocation is reserved for lifecycle operations
on workspaces whose persisted mode was created by one of those two entries.

Architecture:
- Reads opgen_state_machine.yaml as source of truth (via state_executor)
- Per-state: build brief from templates (no LLM), spawn CC subagent via
  `claude --print --agent`, parse JSON envelope, normalize schema, route
  by YAML transition
- LLM never drives the top-level loop. Inside spawned agents, LLM does
  kernel-write / probe-analysis / KB-merge / self-critic.
"""
from __future__ import annotations

import datetime as _dt
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

# V2 #5 (Day 4): force stdout/stderr line-buffering so users tailing the log
# (or `tee`-redirecting to a file) see progress live, not all-at-once at
# process exit. Without this, op#10 run3 ate 17 min of "0 bytes in log" before
# any visible output. reconfigure() is no-op if already line-buffered.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception as error:
    logging.getLogger(__name__).debug(
        "Recoverable operation failed.", exc_info=error
    )

# Module-relative imports
_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

# cv-agent-style centralized logger (owner direction 2026-05-27 00:12Z).
# Module-level child of `a5_orchestrator` — handlers configured in main()
# via setup_run_logger. Before main() runs, log calls discard silently
# (stdlib default) — matches existing behavior where print() was sole output.
from logging_config import get_logger
log = get_logger(__name__)

import agent_dispatch
import agent_transport
import critic_invoke
import events
import finalize_pipeline
import phase_o0
import phase_o05
import phase_o15
import phase_o3
import perf_checkpoint
import phase_o5
import phase_o5_perf_capture
import phase_o5_runner
import resume as resume_mod
import schema_norm
# Decomposed sibling modules (god-file split 2026-06-30) — names re-imported into this namespace so
# existing call-sites + `orchestrator.<name>` external access are preserved. Mechanical, no behavior change.
from silence_retry import (
    _silence_retry_path,
    _load_silence_retry_count,
    _bump_silence_retry_count,
)
# Decomposed sibling module (god-file split 2026-06-30, group 2/7) — names re-imported so existing
# call-sites + `orchestrator.<name>` external access are preserved. Mechanical, no behavior change.
from resolution import (
    _agent_timeout_for_target,
    _resolve_env,
    _detect_max_lane,
    _DEFAULT_AGENT_TIMEOUT_SEC_A5,
    _DEFAULT_AGENT_TIMEOUT_SEC_DS,
)
# Decomposed sibling module (god-file split 2026-06-30, group 3/7) — names re-imported so existing
# call-sites + `orchestrator.<name>` external access are preserved. Mechanical, no behavior change.
from workspace_lifecycle import (
    _mark_agent_died,
    _archive_stale_outputs_before_spawn,
    _optimize_reentry_workspace,
    _record_partial_persist_finalize,
    _STALE_OUTPUTS_BY_STATE,
)
# Decomposed sibling module (god-file split 2026-06-30, group 4/7) — names re-imported so existing
# call-sites + `orchestrator.<name>` external access are preserved. Mechanical, no behavior change.
from handoff_audit import (
    extract_canonical_handoff,
    deleg_marker_needs_refresh,
    audit_doc_needs_refresh,
    _ensure_audit_artifacts,
    _consume_applied_user_decision,
    _extract_kb_draft_from_user_decision,
    _CANONICAL_HANDOFF_PREFIXES,
    _ARROW_TO_AT_FORM,
    _VALID_ARROW_KEYWORDS,
    SELF_CRITIC_POST_WORKER_TIMEOUT_SEC,
)
# Decomposed sibling module (god-file split 2026-06-30, group 5/7) — names re-imported so existing
# call-sites + `orchestrator.<name>` external access are preserved. Mechanical, no behavior change.
from validation import (
    _validate_a3_host_home_mount,
    _refuse_if_detached,
    _spec_has_backward_contract,
)
# Decomposed sibling modules (god-FUNCTION split, DEBT-201, 2026-07-06) — helper
# clusters pulled OUT of the 1792-line run_single_op / main god-functions into
# focused siblings. Names re-imported so run_single_op / main call-sites +
# `orchestrator.<name>` external+test access are preserved. Byte-identical
# relocation, no behavior change. Neither function is monkeypatched; both are
# self-contained (call only stdlib + schema_norm module ref).
from pipeline_exhaustion import _is_legitimate_pipeline_exhaustion
from orchestrator_coldstart import _cold_start_reset_workspace
from fsm_context import OrchestratorContext
import fsm_phase_finalize
import fsm_phase_o25_dispatch
import fsm_phase_spawn
import state_executor
# Decomposed sibling module (god-FILE split, DEBT-201, 2026-07-06) — the
# run_single_op prologue + CLI-audit + terminal-timing + workspace-resolution
# helper clusters were pulled OUT of this file to bring it below the <1000-line
# bar. Names re-imported so run_single_op's bare call-sites + `orchestrator.<name>`
# external+test access are preserved. The helpers resolve PROJECT_ROOT /
# WORKSPACE_ROOT lazily via `_orch()` read-through, so monkeypatch.setattr(
# orchestrator, "PROJECT_ROOT"/"WORKSPACE_ROOT", …) still bites.
from orchestrator_prologue import (
    _resolve_workspace,
    _audit_bump_caps,
    _generate_timing_report,
)
from source_arch import verify_source_stage


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ROOT = _HERE.parent.parent.parent.parent  # repo root
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"

_SCOPED_OPGEN_MODES = frozenset({"port_a3_to_a5", "backward"})


def _read_scoped_opgen_mode(workspace: Path) -> Optional[str]:
    """Read and validate the persisted customer mode before any run side effect."""
    state_path = workspace / ".opgen_state.json"
    try:
        mode = json.loads(state_path.read_text()).get("opgen_mode")
    except Exception:
        return None
    return mode if mode in _SCOPED_OPGEN_MODES else None


# Hard wall-clock per-agent timeout. Workers can take 30-60min for cold-start
# kernel build + verify; probes are quicker. We use a generous default and
# trust agent-internal iter_cap to terminate before this kicks in.
#
# P0aau (2026-05-09 DS investigation): DS/V4 backend has 2-3× tool-call latency
# vs the A5 backend. 525 tool events × ~10-30s each reaches 5400s cap on complex ops like
# Cumsum. Bump DS timeout to 10800s (3h) to match the backend's ground truth.
# A5/Opus keeps 5400s (default) — the backend-aware dispatch is in _agent_timeout().
# P-fix 2026-06-11 (backward input_gen): the edge-aware input_gen expands cases
# ~6→90/op, so the verify(90)+determinism(double-run)+perf(profiler warmup/active)
# +rebuild cycle exceeds the 90-min cap on heavier backward ops — gns + bmm both
# got SIGTERM'd (143) at the late det/perf stage despite reaching 90/90 PASS.
# Bumped to 9000s (2.5h) for the 90-case backward sweep; revert to 5400 once
# input_gen gets a per-op case-budget cap.
DEFAULT_AGENT_TIMEOUT_SEC = _DEFAULT_AGENT_TIMEOUT_SEC_A5  # safe default


# Total cap on agents per single-op run (hard safety fuse).
TOTAL_SPAWN_CAP_PER_OP = 12  # P0abe (2026-05-07): tightened from 20.
# Per-state caps sum to ~26 (kw=9 + probe=4 + ko=5 + fo=3 + ar=2 + da=2 + finalize=1)
# but a healthy pipeline rarely exceeds 6-8 spawns. 12 is the soft ceiling: if
# we cross it, something is wrong (loop, brief drift). Beyond cap, transition
# to await_user_decision (NOT exit code 6) so user can inspect rather than
# silently fail. Companion gate: detect_loop_break catches same-signature
# rollback before this cap fires (typical loop = 2-3 spawns).


# ---------------------------------------------------------------------------
# DEBT-PORT-A3-TARGET-ENFORCE (2026-06-05)
# ---------------------------------------------------------------------------
def enforce_port_a3_target(opgen_mode: str, target: str):
    """The arch22 to arch35 migration structurally builds on the A5 target.

    If `.ascendc_env` says a non-a5 target in port_a3 mode, that is a
    contradictory config that SILENTLY splits the worker build-host (which
    follows `{target}_HOST`) from the O5 verify-host
    (`phase_o5_runner._a5_build_host` returns `A5_HOST` in port_a3 mode
    REGARDLESS of target). The kernel builds on one host, O5 verifies on
    another where the `.so` does not exist → every case ImportErrors → 0/N
    FAIL, mis-read by O5 as a precision MISMATCH. This cost the FA-A5 gate-a
    sprint hours (2026-06-05). Force `a5` + return a LOUD warning so the run
    proceeds correctly (build+verify on the same A5 host) and the user is told
    to fix their `.ascendc_env target=` line.

    Pure function (no side effects) for unit-testability. Returns
    `(effective_target, warning_or_None)`.
    """
    if opgen_mode == "port_a3_to_a5" and target != "a5":
        return "a5", (
            f"DEBT-PORT-A3-TARGET-ENFORCE: port_a3_to_a5 mode implies TARGET=a5, "
            f"but .ascendc_env target={target!r} — overriding to 'a5'. "
            f"(target!=a5 in port_a3 silently splits the build-host from the O5 "
            f"verify-host → 0/N ImportError mis-read as precision FAIL. Fix your "
            f".ascendc_env target= line to silence this.)"
        )
    return target, None


# ---------------------------------------------------------------------------
# Single-op runner
# ---------------------------------------------------------------------------
def run_single_op(
    op: str,
    *,
    workspace: Path | None = None,
    lane: int = 0,
    plan_only: bool = False,
    cap_bumps: dict[str, int] | None = None,
    timing: bool = False,
    backend: str | None = None,
    perf_threshold: float | None = None,
    roofline_mode: bool = False,
    defer_perf_opt: bool = False,
    kw_1_only: bool = False,
    extra_lanes: list[int] | None = None,
) -> int:
    """Run the full state machine for one op until it reaches a terminal state.

    Args:
        timing: if True, generate TIMING_REPORT.md in workspace after terminal state
            (via scripts/gen_timing_report.py). Off by default; enable with --timing.
        backend: retained for internal callers; only ``ascendc`` is accepted.
        perf_threshold: Zheng 2026-05-20 / PERF_GATE_PROFILE_DESIGN. Maps to a
            PerfGateProfile via perf_gate.write_profile_marker. Persisted as
            workspace/.perf_gate_profile.json. None = no override (DEFAULT
            profile); marker auto-removed if previously set.
    Returns:
        process exit code (0 on finalize, non-zero on abort/error)
    """
    if backend not in (None, "ascendc"):
        print("ERROR: this orchestrator supports only the AscendC kernel backend")
        return 2
    if extra_lanes:
        print("ERROR: multi-rank generation is outside this orchestrator's scope")
        return 2
    if workspace is None:
        workspace = _resolve_workspace(op, backend="ascendc")

    scoped_mode = _read_scoped_opgen_mode(workspace)
    if scoped_mode is None:
        print(
            f"ERROR: workspace {workspace} has no supported persisted mode; "
            "start work with --port-a3 or --backward"
        )
        return 2
    if scoped_mode == "port_a3_to_a5":
        valid_stage, stage_reason, _stage_manifest = verify_source_stage(workspace)
        if not valid_stage:
            print(
                "ERROR: migration source-only snapshot validation failed before "
                f"run: {stage_reason}"
            )
            return 2

    # Per-invocation runtime_kwargs bag — passed through every
    # state_executor.next_state call so the generic `plugin_method` YAML
    # primitive can forward declared kwargs (see `forward_kwargs` in
    # opgen_state_machine.yaml) to plugin methods. NOT persisted to workspace
    # state. Currently empty (no per-invocation force-switches); a future
    # force-X switch adds a key here + the YAML primitive's forward_kwargs list,
    # with no new evaluator branches.
    runtime_kwargs: dict = {}

    # cv-agent-style logger: re-setup once workspace is known so per-run
    # FileHandler at <workspace>/.opgen.log captures the full op run.
    # Idempotent — clears prior stdout-only handler from main() and re-adds
    # both FileHandler + StreamHandler. Level inherited from prior setup
    # via root logger (the logging.getLogger("a5_orchestrator").level set
    # in main()).
    try:
        from logging_config import setup_run_logger
        import logging as _lg
        _existing_level = _lg.getLogger("a5_orchestrator").level or _lg.INFO
        if workspace.exists():
            setup_run_logger(workspace=workspace, level=_existing_level)
        # If workspace doesn't exist yet, defer to post-mkdir below.
    except Exception:
        # Logger setup is non-fatal — orchestrator's existing print() path
        # still works; falling back to stdout-only is graceful.
        pass

    log.info(f"op={op} workspace={workspace} lane={lane}")
    if cap_bumps:
        _audit_bump_caps(workspace, cap_bumps)
        log.info(f"applied --bump-cap (audited): {cap_bumps}")

    # Zheng 2026-05-20 / PERF_GATE_PROFILE_DESIGN: persist perf-gate profile
    # marker. With --perf-threshold flag: write marker (overwriting any stale
    # value, per §7 Q1 sticky-resume rule "flag wins, marker regenerated").
    # Without flag + marker present: marker wins (no warning). Without flag +
    # no marker: DEFAULT applies (no-op).
    if perf_threshold is not None or defer_perf_opt:
        from perf_gate import write_profile_marker
        active_profile = write_profile_marker(
            workspace, perf_threshold, roofline_mode=roofline_mode,
            defer_perf_opt=defer_perf_opt,
        )
        log.info(f"--perf-threshold={perf_threshold} defer_perf_opt={defer_perf_opt}: active "
              f"perf-gate profile = {active_profile.name} "
              f"(finalize_threshold={active_profile.finalize_threshold}, "
              f"ko_escalation={'on' if active_profile.allow_ko_escalation else 'off'})")

    events.emit(workspace, "orchestrator.start", lane=lane,
                data={"cap_bumps": cap_bumps or {},
                      "perf_threshold": perf_threshold})

    # P0pp (2026-05-06): Phase O0 hook integrity gate. Verify required
    # KB + hook + deploy infrastructure is present BEFORE spawning agents.
    # BLOCKED → refuse to start (critical KB/hook missing; would silently
    # allow forbidden operations); DEGRADED → warn (deploy missing; workers
    # will fail at build).
    o0 = phase_o0.check_hook_integrity(workspace)
    if o0.verdict == "BLOCKED":
        print(phase_o0.format_block_message(o0))
        events.emit(workspace, "orchestrator.phase_o0_block", lane=lane,
                    data={"verdict": o0.verdict,
                          "missing_files": o0.missing_files})
        return 8  # distinct exit code for "phase O0 critical infra missing"
    elif o0.verdict == "DEGRADED":
        print(phase_o0.format_block_message(o0))
    else:
        log.info(f"phase O0: {o0.summary}")

    # Phase O0.5 persists the mode seeded by the scoped entry command.  The
    # environment selects the target host only; it cannot switch the customer mode
    # or opt into another kernel-authoring backend.
    try:
        from briefs._common import load_env as _load_env
        _target = _load_env().target
    except Exception:
        _target = "a5"
    _opgen_mode = scoped_mode
    _target, _pa3_target_warn = enforce_port_a3_target(_opgen_mode, _target)
    if _pa3_target_warn:
        log.warning(_pa3_target_warn)
    _resolved_backend = "ascendc"
    o05 = phase_o05.init_durable_state(
        workspace, op, lane=lane, target=_target, opgen_mode=_opgen_mode,
        backend=_resolved_backend,
    )
    log.info(f"phase O0.5: {o05.summary}")

    # P0aak (2026-05-07): Phase O1.7 LLM-driven op classification.
    # Replaces the bench-name-keyed `OP_TAGS` dict + regex source-scan
    # (P0aai/P0aaj retired). Runs `/aog-op-classify` skill in isolated
    # subprocess, writes workspace/<op>/op_classification.json. Cached
    # via source content SHA256. Failure mode graceful: brief falls back
    # to DEFAULT_KB_SECTIONS-only if classification missing/errored.
    #
    # B3.3b (2026-05-31, live-e2e fix): backward mode SKIPS the O1.7 skill.
    # `_cmd_backward` ALREADY seeds op_classification.json (source=
    # "cli_flag_backward", tags ['backward','GRADIENT']) — the CLI flag IS the
    # classification (same shortcut as _cmd_port_a3 W12). Re-running the
    # `/aog-op-classify` claude --print subprocess is (a) redundant and (b)
    # flaky for a bare backward forward-spec: the skill, lacking a benchmark
    # <op>.py/<op>.json in scope, returns a clarifying question ("Which option
    # do you want?") instead of writing the file → "classification failed"
    # noise on every backward run (caught by the first live `orch --backward`
    # e2e 2026-05-31). The pre-seed is authoritative; skip the subprocess.
    _o17_state_path = workspace / ".opgen_state.json"
    _o17_mode = None
    try:
        if _o17_state_path.is_file():
            _o17_mode = json.loads(_o17_state_path.read_text()).get("opgen_mode")
    except Exception:
        _o17_mode = None
    _o17_cls_path = workspace / "op_classification.json"
    _o17_preseeded = False
    if _o17_mode == "backward" and _o17_cls_path.is_file():
        try:
            _o17_cls = json.loads(_o17_cls_path.read_text())
            _o17_preseeded = bool(_o17_cls.get("op_class_tags"))
        except Exception:
            _o17_preseeded = False
    # Backward classification is seeded by the explicit customer entry flag.
    if _o17_mode == "backward" and _o17_preseeded:
        _reason = (
            "op_classification.json pre-seeded by --backward CLI flag; "
            "the flag IS the classification"
        )
        log.info(
            f"phase O1.7: SKIPPED ({_o17_mode} mode — {_reason}; "
            "no /aog-op-classify subprocess needed)"
        )
        o17 = None
    else:
        o17 = "RUN"
    try:
        if o17 == "RUN":
            import phase_o17_classify
            o17 = phase_o17_classify.classify(workspace)
        if o17 is None:
            pass  # backward mode skip — already logged above; pre-seed authoritative
        elif o17.error:
            # Distinguish sentinel-cached skip ("skill unavailable") from real failures
            # so the log makes it obvious whether O1.7 cost a claude --print or not.
            if "skill unavailable" in (o17.error or ""):
                if "cached error" in (o17.error or ""):
                    log.info("phase O1.7: SKIPPED (cached error from prior timeout — 0.0s)")
                elif "cached sentinel" in (o17.error or ""):
                    log.info("phase O1.7: SKIPPED (cached: /aog-op-classify not installed in this checkout)")
                else:
                    log.info(f"phase O1.7: classification failed — {o17.error}")
            else:
                log.info(f"phase O1.7: classification failed — {o17.error}")
        else:
            log.info(f"phase O1.7: tags={o17.op_class_tags} kb_recs={len(o17.kb_recommendations)}")
            # §5.2 C1: config-gated auto-populate of tier-a routes now that op-class is
            # known. Writes workspace/.cba_required_routes.json iff a project route-config
            # ($AOG_CBA_ROUTE_CONFIG) is set AND this op's class matches an entry's
            # applies_to; else no-op (a5_ops mainline generation unaffected). Fail-open.
            try:
                from cba_route_populate import populate_cba_routes
                _routes = populate_cba_routes(o17.op, workspace, o17.op_class_tags)
                if _routes:
                    log.info(f"phase O1.7: CBA tier-a routes populated: "
                             f"{[r['skill'] for r in _routes]}")
            except Exception as _cba_e:
                log.info(f"phase O1.7: CBA route populate skipped ({_cba_e!r})")
    except Exception as e:
        log.info(f"phase O1.7: skill subprocess failed ({e!r}); brief will use default-only KB")

    # P0nn (2026-05-06): Phase O1.5 DET_POLICY classification. Reads
    # analysis.md (if worker has set it) or op_classification tags. Stored in
    # .opgen_state.json so downstream phases (briefs, O5, da_brief) act on it.
    try:
        from briefs.op_taxonomy import lookup as _op_taxonomy_lookup
        _t = _op_taxonomy_lookup(op, workspace=workspace)
        op_tags = list(_t.tags) if _t and getattr(_t, "tags", None) else []
    except Exception:
        op_tags = []
    o15 = phase_o15.classify_det_policy(workspace, op, op_tags=op_tags)
    phase_o15.store_in_durable_state(workspace, o15.policy, det_floor=o15.det_floor)
    log.info(
        f"phase O1.5: {o15.summary}"
        + (f" (det_floor={o15.det_floor})" if o15.det_floor is not None else "")
    )

    # Phase O2.5 dispatches only the two persisted customer modes. It references
    # no orchestrator-module globals — only per-run inputs + sibling modules
    # (patched directly by tests). Returns an exit code to `return` (7 = missing
    # source / ref unrecoverable, 98 = BACKWARD_E2E=0 opt-out) or None to
    # proceed to Phase O3.
    _o25_rc = fsm_phase_o25_dispatch.provision_reference(
        op, workspace, lane=lane, extra_lanes=extra_lanes, plan_only=plan_only,
    )
    if _o25_rc is not None:
        return _o25_rc

    # P0ll (2026-05-06): Phase O3 PROGRESS.md scaffold. Idempotent — workers
    # append to canonical Timeline section so handoff extraction is stable.
    o3 = phase_o3.init_progress_md(workspace, op, opgen_mode=scoped_mode)
    log.info(f"phase O3: {o3.summary}")

    spawn_count = 0
    last_handoff = ""

    # P94 attack-id INFRA-BLAME-LOOP / iter_cap-per-agent-type bypass
    # (DS Round 2 finding, 2026-05-15T09:17Z): TOTAL_SPAWN_CAP_PER_OP=12
    # is enforced PER SESSION. Cold-start episodes reset count → multi-
    # episode runs accumulate hidden cost. 10_LayerNorm shipped with 41
    # spawn events vs cap=12 because each cold-start started fresh.
    #
    # Fix: read lifetime spawn count from .opgen_state.json (survives
    # cold-start backup). Warn loudly when lifetime ≥ 30 so user/agent
    # sees the accumulated cost. Don't hard-block (legitimate retry
    # scenarios exist) but surface the hidden cost.
    _lifetime_spawn_count = 0
    _state_fp = workspace / ".opgen_state.json"
    if _state_fp.is_file():
        try:
            _state_obj = json.loads(_state_fp.read_text())
            _lifetime_spawn_count = int(_state_obj.get("lifetime_spawn_count", 0))
        except Exception:
            _lifetime_spawn_count = 0
    if _lifetime_spawn_count >= 30:
        log.info(
            f"⚠ HIGH LIFETIME SPAWN COST: this op has "
            f"accumulated {_lifetime_spawn_count} spawns across all "
            f"sessions/cold-starts. This may indicate an infra-blame-loop "
            f"or persistent reward-hacking cycle (P94 attack-id "
            f"INFRA-BLAME-LOOP). Hidden cost is visible — consider "
            f"investigating root cause before another cold-start."
        )
        events.emit(workspace, "orchestrator.high_lifetime_spawn_cost",
                    lane=lane, data={"count": _lifetime_spawn_count})

    # DEBT-201 (dependency inversion): build the per-run OrchestratorContext.
    # STEP 1 introduces the abstraction only — the loop below still reads/writes
    # its LOCALS; the ctx carries the same invariant inputs + a read-through
    # view of the monkeypatch-surface orchestrator globals so that when the loop
    # body is extracted into fsm_phase_* handlers (STEP 2) each handler reaches
    # deps via `ctx.X` and existing `monkeypatch.setattr(orchestrator, X)` still
    # bites. Behavior-neutral here.
    ctx = OrchestratorContext(
        op=op,
        workspace=workspace,
        lane=lane,
        plan_only=plan_only,
        timing=timing,
        kw_1_only=kw_1_only,
        extra_lanes=list(extra_lanes or []),
        runtime_kwargs=runtime_kwargs,
        spawn_count=spawn_count,
        last_handoff=last_handoff,
        lifetime_spawn_count=_lifetime_spawn_count,
    )

    # DEBT-201: the loop's mutable state (spawn_count / last_handoff /
    # lifetime_spawn_count) is now carried ON the ctx (seeded above from the
    # locals). The spawn + post-spawn cluster is extracted to
    # fsm_phase_spawn.handle_spawn, which reads+advances that ctx state; the
    # loop drives off ctx.spawn_count so the observable behavior is unchanged.
    while ctx.spawn_count < TOTAL_SPAWN_CAP_PER_OP:
        snap = state_executor.snapshot(workspace)
        log.info(f"iter={ctx.spawn_count} state={snap.current_state}")

        # P0bb (2026-05-05): bootstrap canonicalization is now handled inside
        # state_machine.get_current_state — when log is empty, it persists the
        # inferred state chain to state_transitions.jsonl before returning.
        # Subsequent current_state() calls read from log instead of
        # re-bootstrapping from a (potentially-changing) PROGRESS.md tail.
        # Just calling state_executor.snapshot above already triggered any
        # needed canonicalization.

        if snap.is_terminal:
            log.info(f"terminal state reached: {snap.current_state}")
            exit_code = 0 if snap.current_state == "done" else 1
            events.emit(workspace, "orchestrator.terminal", lane=lane,
                        data={"state": snap.current_state, "exit_code": exit_code})
            # --timing: auto-generate per-step timing report from event stream
            if timing:
                _generate_timing_report(workspace, op)
            return exit_code

        # P0dd (2026-05-05): finalize is a real state in the FSM, not a side
        # effect. When current_state == "finalize", run the in-process
        # finalize_pipeline (archive promotion + marker), record the
        # finalize→done transition in state log, then loop back to detect
        # done as terminal. This makes finalize behave like any other state:
        # has an "agent" (Python function), has an exit transition, leaves
        # an audit trail.
        if snap.current_state == "finalize":
            # DEBT-201: the finalize-state slice lives in fsm_phase_finalize.
            # It reaches orchestrator-module deps via `ctx.<name>` (read-through)
            # and sibling modules directly, so the monkeypatch surface is
            # unchanged. Returns a HandlerResult mapping the original
            # continue / return N control-flow.
            _fin_res = fsm_phase_finalize.handle_finalize(ctx, snap)
            if _fin_res.action == "return":
                return _fin_res.exit_code
            continue

        if state_executor.is_pause(snap.current_state):
            # V3.8.5 / DEBT-077 #59: pause states halt for user decision.
            # P0p (2026-05-05): if user_decision.md already exists, the user
            # has ALREADY made the decision (likely from a prior session) —
            # advance through state machine immediately instead of pausing
            # again. Without this check, every re-invoke after the user
            # writes user_decision.md still bails at PAUSE because the
            # orchestrator never tries to consume it.
            user_decision_path = workspace / "user_decision.md"
            if user_decision_path.exists() and user_decision_path.stat().st_size > 0:
                log.info(
                    f"state={snap.current_state} but user_decision.md "
                    f"present ({user_decision_path.stat().st_size} bytes) — advancing through state machine"
                )
                # P0ff (2026-05-23): extract strategic directive content into
                # kb_draft_from_user_decision.md so kb_manager Mode 1 can promote
                # to canonical KB (customer-side cold-clone reproducibility).
                # Per owner directive 20:48Z + independent design review.
                try:
                    _extract_kb_draft_from_user_decision(workspace, op)
                except Exception as e:
                    # Non-fatal: KB-draft extraction failure should NOT block the
                    # primary decision-consume path. Log + continue.
                    log.info(
                        f"P0ff WARN: kb_draft extraction failed "
                        f"(non-fatal, advancing state anyway): {e!r}"
                    )
                try:
                    decision = state_executor.next_state(
                        workspace, "", dry_run=False,
                        runtime_kwargs=runtime_kwargs,
                    )
                except state_executor.StateMachineError as e:
                    log.info(f"state machine error consuming user_decision.md: {e}")
                    return 5
                log.info(f"route: {decision.from_state} → {decision.next_state} ({decision.rationale})")
                events.emit(workspace, "orchestrator.transition", lane=lane,
                            data={"from_state": decision.from_state,
                                  "to_state": decision.next_state,
                                  "rationale": decision.rationale[:300]})
                # P0kk (2026-05-29): consume the decision file once applied, so a
                # STALE decision cannot re-advance every iteration (infinite-loop
                # guard — see _consume_applied_user_decision docstring; observed 99×
                # on FA-class 3_FusionAttention finalize↔await_user_decision).
                _consume_applied_user_decision(workspace)
                # Continue main loop — next iteration sees the new state from log tail
                continue

            # Genuine pause — no decision file yet
            log.info(
                f"PAUSE — state={snap.current_state}\n"
                f"  USER ACTION REQUIRED: write {workspace}/user_decision.md\n"
                f"  Format:\n"
                f"    next_state: <one of: await_worker / await_probe / await_optimizer /\n"
                f"                          await_fused_optimizer / await_researcher /\n"
                f"                          await_det_analyzer / finalize / abort>\n"
                f"    reason: <free text>\n"
                f"  Then re-invoke `python3 orchestrator.py {op} --lane {lane}` to resume."
            )
            events.emit(workspace, "orchestrator.pause", lane=lane,
                        data={"state": snap.current_state})
            # --timing: generate report even on pause — the events up to this
            # point are complete and the user may not re-invoke from here.
            if timing:
                _generate_timing_report(workspace, op)
            return 10

        # DEBT-201: the spawn + post-spawn cluster (next_agent lookup, iter-cap
        # routing, self-critic, agent spawn + retry/exception handling, schema
        # norm, O3 regression gate, handoff extraction, perf checkpoint, and
        # next_state transition) lives in fsm_phase_spawn. KB merge is deferred
        # to the finalize handler after independent gates. The spawn handler reaches
        # orchestrator-module deps via `ctx.<name>` (read-through) and sibling
        # modules directly, so the monkeypatch surface is unchanged. It advances
        # ctx.spawn_count / ctx.last_handoff and returns a HandlerResult mapping
        # the original continue / return N control-flow.
        _spawn_res = fsm_phase_spawn.handle_spawn(ctx, snap)
        if _spawn_res.action == "return":
            return _spawn_res.exit_code
        continue

    # P0abe (2026-05-07): instead of exit 6, route to await_user_decision so
    # the workspace state is preserved + user can inspect. The cap-hit
    # rationale is logged so resume.py / diagnose.py can surface it.
    log.info(
        f"total spawn cap ({TOTAL_SPAWN_CAP_PER_OP}) hit "
        f"without terminal — routing to await_user_decision (P0abe). "
        f"Inspect workspace and resume with explicit directive."
    )
    try:
        state_executor.record_transition(
            workspace,
            state_executor.TransitionDecision(
                next_state="await_user_decision",
                matched_transition_index=-1,
                rationale=(
                    f"P0abe total_spawn_cap_exhausted: {TOTAL_SPAWN_CAP_PER_OP} "
                    f"spawns without terminal; halting for user inspection"
                ),
                from_state=state_executor.snapshot(workspace).current_state,
                handoff="",
            ),
        )
        events.emit(workspace, "orchestrator.spawn_cap_hit", lane=lane,
                    data={"cap": TOTAL_SPAWN_CAP_PER_OP})
    except Exception as e:
        log.warning(f"failed to record cap-hit transition ({e})")
    return 6


# ---------------------------------------------------------------------------
# DEBT-201 god-function split (2026-07-06): main() + the CLI subcommand
# handlers moved OUT of this file into focused siblings to bring orchestrator.py
# below the <1000-line bar. Re-imported here so `orchestrator.main`,
# `orchestrator._cmd_*`, `orchestrator._parse_bump_caps`, and
# the `__main__.py` entry-point are preserved. Byte-identical relocation.
# Placed at END-OF-FILE so this module's globals (run_single_op, WORKSPACE_ROOT,
# …) are fully defined before the siblings load. The siblings do NOT
# `import orchestrator` at module scope (that would re-enter this half-loaded
# module — and re-enter it a SECOND time as `__main__` under `python -m
# orchestrator`, a circular import); they resolve it lazily via `_orch()` at
# CALL time, which returns the same, complete module object and keeps the
# monkeypatch-bite contract on `orchestrator.<name>`.
# ---------------------------------------------------------------------------
from orchestrator_cmds import (  # noqa: E402
    _VALID_BUMP_COUNTERS,
    _parse_bump_caps,
    _cmd_resume,
    _cmd_backward,
    _cmd_port_a3,
    _cmd_status,
)
from orchestrator_cli import main  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
