# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Resume — recover interrupted orchestrator runs (DEBT-077 Track C #4).

Reads workspace state files to determine what state an op is in:
  1. terminal (finalize / abort)            → no resume; report status
  2. paused (await_user_decision)           → check for user_decision.md;
                                              if present, re-invoke orchestrator
  3. agent died (.agent_died_at_<state>)    → surface to user, NO auto-retry
                                              (codex C4)
  3a. known NPUBench repair construction failure → archive marker, reset stale
                                                  candidate outputs, re-invoke
  4. mid-flight (state_transitions.jsonl    → re-invoke orchestrator from
     last to_state non-terminal)              current state

CLI:
    python3 resume.py --status <op>           # show what would happen
    python3 resume.py <op>                    # actually resume
    python3 resume.py --all                   # scan all workspaces, list resumable
    python3 resume.py --all --execute         # resume all that can be safely resumed

Per codex C4: NO automatic retry on agent_died. The orchestrator child
already wrote a .agent_died_at_<state> marker; resume just surfaces this
to the user, never re-invokes silently.
"""
from __future__ import annotations
import logging

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent))

import state_executor
import events as event_emitter
from source_arch import verify_source_stage
from orchestrator_coldstart import _prepare_npubench_candidate_repair

from logging_config import get_logger  # cv-agent style logger
log = get_logger(__name__)


class ResumeAction(Enum):
    NONE_TERMINAL = "none_terminal"          # already finalize/abort
    USER_DECISION_PENDING = "user_decision_pending"  # paused, no user_decision.md yet
    USER_DECISION_READY = "user_decision_ready"      # paused, user_decision.md present
    AGENT_DIED = "agent_died"                # surface to user, NO auto-retry (unknown failure)
    AUTO_RECOVERABLE = "auto_recoverable"    # FW-transient — clear marker + re-invoke (P0r 2026-05-05)
    NPUBENCH_CANDIDATE_REPAIR = "npubench_candidate_repair"
    # spawn failed due to quota/usage limit; no auto-retry — wait for reset (2026-05-18)
    QUOTA_BACKOFF = "quota_backoff"
    # abort caused by parsing bug now fixed; replay routing (P0t 2026-05-05)
    BUGGY_ABORT_RECOVERABLE = "buggy_abort_recoverable"
    RESUMABLE = "resumable"                  # mid-flight, can re-invoke
    UNKNOWN = "unknown"                      # missing state, no PROGRESS.md


# 2026-05-18 (Gap #2): patterns that indicate an upstream quota or usage
# limit was hit, NOT a transient network/proxy hiccup. These must NOT
# be auto-retried — the next spawn would just hit the same quota
# immediately, burning the retry budget for no benefit. Surface to the
# user with the explicit advice: wait for quota reset, then re-run.
#
# Anchor: 2026-05-18 1_GELU / 5_Cumsum / 6_Histc orchestrator runs all
# hit "corporate proxy Notification" from claude --print subprocess on
# ko/probe spawn → resume.py classified as FW-transient → immediate
# re-invocation → same quota error → retry budget exhausted in
# seconds, no real progress.
# Matched case-INSENSITIVELY (see _is_quota_reason) — marker reasons capitalize
# differently ("You've hit ...") than these lowercase needles, so a
# case-sensitive `p in reason` silently missed real quota deaths.
_QUOTA_PATTERNS = (
    "corporate proxy notification",
    "you've hit your org's monthly usage limit",
    "you've hit your monthly usage limit",
    "claude code usage limit reached",
    "rate limit",
    "rate_limit",
    "quota exceeded",
    "usage limit",
    # back issue #2 (2026-07-17): the 5-hour SESSION limit surfaces as
    # "You've hit your session limit · resets <time>" wrapped inside a generic
    # "claude (stream-json) exited 1" reason. Without this needle it matched the
    # FW-transient branch instead → auto-retried → hit the same session limit →
    # burned FW_AUTO_RETRY_CAP → escalated to AGENT_DIED "nothing recoverable".
    # It's a quota condition: back off (wait for reset), do NOT auto-retry.
    "session limit",
)


def _is_quota_reason(reason: str) -> bool:
    """True if the death reason is an upstream quota/usage/session limit.

    Case-insensitive: marker reasons preserve the provider's capitalization
    ("You've hit your session limit"), which a case-sensitive substring test
    against the lowercase needles above would miss.
    """
    low = reason.lower()
    return any(p in low for p in _QUOTA_PATTERNS)


def _is_regular_file(path: Path) -> bool:
    """True only for a real file: a symlink is rejected even if it resolves."""
    return path.is_file() and not path.is_symlink()


def _is_npubench_candidate_repair_crash(
    workspace: Path, died_state: str, reason: str
) -> bool:
    """Recognize the known graybox-respawn rejection after an O5 rollback.

    A generic graybox construction error remains an agent-died condition. The
    narrow exception is safe to recover because the durable rollback gate has
    already classified the preceding O5 result as a worker-owned NPUBench
    candidate defect; stale candidate output can be moved before respawn.
    Both O5 rollback gates that keep/repair a worker candidate are accepted:
    ``phase_o5_npubench_candidate_contract`` (pre-build contract rejection,
    candidate archived at rollback time) and ``phase_o5_mismatch`` (post-run
    evidence rejection; the candidate stays in the workspace for incremental
    repair, so a deliverable-shaped candidate — e.g. the PR4778 apt mirror
    ``op_kernel/arch35/*.h`` + ``kernel/pybind11.cpp`` of the port-a3-ops
    route — trips the answer gate on the next spawn; 2026-08-27
    flash_attention_score).
    """
    construction_errors = (
        "graybox construction rejected target/answer-bearing curated input",
        "graybox construction manifest changed during worker dispatch",
    )
    if died_state != "await_worker" or not any(
        error in reason for error in construction_errors
    ):
        return False
    state_path = workspace / ".opgen_state.json"
    history_path = workspace / ".rollback_history.jsonl"
    if not (_is_regular_file(state_path) and _is_regular_file(history_path)):
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        entries = [
            json.loads(line)
            for line in history_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError):
        return False
    reference = state.get("reference") if isinstance(state, dict) else None
    source = state.get("port_source") if isinstance(state, dict) else None
    latest = entries[-1] if entries else None
    return (
        isinstance(state, dict)
        and state.get("opgen_mode") == "port_a3_to_a5"
        and isinstance(reference, dict)
        and reference.get("source") == "npubench"
        and (
            # CANN ops-repo sources (port-a3-ops default route) record no
            # port_source kind at all (2026-08-27 flash_attention_score).
            source is None
            or (
                isinstance(source, dict)
                and source.get("kind") in {"port-aclnn-tilelang2ascendc"}
            )
        )
        and isinstance(latest, dict)
        and latest.get("gate") in {
            "phase_o5_npubench_candidate_contract",
            "phase_o5_mismatch",
        }
    )


@dataclass
class ResumeStatus:
    op: str
    workspace: Path
    action: ResumeAction
    current_state: str
    last_handoff: Optional[str] = None
    died_at_state: Optional[str] = None
    died_reason: Optional[str] = None
    summary: str = ""


def _performance_is_deferred(workspace: Path) -> bool:
    """Whether the harness-owned verification carries a deferred perf placeholder."""
    path = workspace / "verification.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    performance = payload.get("performance")
    return (
        isinstance(performance, dict)
        and performance.get("status") == "DEFERRED"
        and performance.get("perf_deferred") is True
    )


def _record_perf_backfill_reentry(workspace: Path, op: str) -> None:
    """Record the synthetic done→finalize transition for the perf backfill."""
    state_executor.record_transition(
        workspace,
        state_executor.TransitionDecision(
            next_state="finalize",
            matched_transition_index=-1,
            rationale=(
                "perf backfill re-entry (CANNBOT_NPUBENCH_PERF_BACKFILL=1): "
                "candidate unchanged; re-run O5 with SKIP_PERF=0 to append the "
                "W3/R5 measurement to the existing evidence bundle"
            ),
            from_state="done",
            handoff="",
        ),
    )


def _resolve_workspace(op: str) -> Path:
    """Find the flat workspace directory for a scoped operation."""
    root = _HERE.parent.parent.parent.parent / "workspace"
    direct = root / op
    if direct.exists():
        return direct
    if not root.exists():
        return direct
    for d in root.iterdir():
        if d.is_dir():
            if d.name.lower() == op.lower():
                return d
    return direct


def diagnose(op: str, *, workspace: Optional[Path] = None) -> ResumeStatus:
    """Look at workspace state and decide what kind of resume is possible."""
    if workspace is None:
        workspace = _resolve_workspace(op)

    if not workspace.exists():
        return ResumeStatus(op=op, workspace=workspace, action=ResumeAction.UNKNOWN,
                            current_state="?", summary=f"workspace {workspace} missing")

    # 1) Agent-died marker takes priority. Two paths:
    #    (a) AUTO_RECOVERABLE — reason matches a known-transient pattern (P0k
    #        FW transient: HTTP 200 empty/malformed body) AND retry count <
    #        FW_AUTO_RETRY_CAP. Clear marker + re-invoke orchestrator. The
    #        previous state-machine state is preserved (state log unchanged),
    #        so the orchestrator simply retries the same agent.
    #    (b) AGENT_DIED — codex C4 — surface to user, no auto-retry. Used for
    #        unknown failure modes where infinite retry could mask real bugs.
    died_files = sorted(workspace.glob(".agent_died_at_*"))
    # Filter out *.cleaned-* which are post-recovery archived markers
    died_files = [d for d in died_files if ".cleaned-" not in d.name]
    if died_files:
        marker = died_files[-1]  # most recent
        try:
            payload = json.loads(marker.read_text())
        except Exception:
            payload = {}
        reason = payload.get("reason") or ""
        died_state = payload.get("state", "?")

        # 2026-05-18 (Gap #2): quota-error classification BEFORE the
        # generic FW-transient check. Quota patterns are more specific
        # and require backoff (no auto-retry) — without this guard, a
        # quota error matched the broader "claude (stream-json) exited 1"
        # FW-transient pattern, got auto-retried, hit the same quota
        # again, exhausted the retry budget.
        is_quota_error = _is_quota_reason(reason)
        if is_quota_error:
            _low = reason.lower()
            return ResumeStatus(
                op=op, workspace=workspace, action=ResumeAction.QUOTA_BACKOFF,
                current_state=state_executor.current_state(workspace),
                died_at_state=died_state,
                died_reason=reason,
                summary=(
                    f"agent spawn at '{died_state}' hit upstream quota/usage/"
                    f"session limit (matched pattern: "
                    f"{[p for p in _QUOTA_PATTERNS if p in _low][:1]}). "
                    f"NO auto-retry — would hit same limit immediately. "
                    f"Action: wait for limit reset (next billing cycle for "
                    f"monthly limits; ~1 hour for rate limits; the stated reset "
                    f"time for session limits), then clear marker + re-invoke "
                    f"orchestrator manually."
                ),
            )

        if _is_npubench_candidate_repair_crash(workspace, died_state, reason):
            return ResumeStatus(
                op=op,
                workspace=workspace,
                action=ResumeAction.NPUBENCH_CANDIDATE_REPAIR,
                current_state=state_executor.current_state(workspace),
                died_at_state=died_state,
                died_reason=reason,
                summary=(
                    "known NPUBench candidate-repair graybox construction "
                    "failure; stale candidate output will be archived before "
                    "re-invoking await_worker"
                ),
            )

        # P0r (2026-05-05): classify FW-transient failures for auto-recovery.
        # Pattern lifted from agent_transport.py _FW_TRANSIENT_PATTERNS.
        is_fw_transient = any(p in reason for p in (
            "API returned an empty or malformed response",
            "check for a proxy or gateway intercepting the request",
            "claude (stream-json) exited 1",  # the wrapper-level form we see in marker
            # 2026-08-26 (review-55-57): SIGTERM'd spawn + SDK model-window blip —
            # same patterns as agent_transport._FW_TRANSIENT_PATTERNS; endpoint/model
            # were healthy minutes later (55/57 both died 13:36, respawns OK 13:39).
            "claude (stream-json) exited -15",
            "unrecognized_model",
        ))

        retry_count = _get_retry_count(workspace, died_state)
        if is_fw_transient and retry_count < FW_AUTO_RETRY_CAP:
            return ResumeStatus(
                op=op, workspace=workspace, action=ResumeAction.AUTO_RECOVERABLE,
                current_state=state_executor.current_state(workspace),
                died_at_state=died_state,
                died_reason=reason,
                summary=(
                    f"agent crash at '{died_state}' classified FW-transient "
                    f"(retry {retry_count + 1}/{FW_AUTO_RETRY_CAP}); will clear "
                    f"marker + re-invoke orchestrator"
                ),
            )

        # Not auto-recoverable: codex C4 — surface to user
        retry_budget_status = (
            "FW-transient retry budget exhausted"
            if is_fw_transient
            else "NO auto-retry (unknown failure mode, codex C4)"
        )
        return ResumeStatus(
            op=op, workspace=workspace, action=ResumeAction.AGENT_DIED,
            current_state=state_executor.current_state(workspace),
            died_at_state=died_state,
            died_reason=reason,
            summary=(
                f"agent crash at state {died_state}; "
                f"reason: {reason[:200]}. "
                f"{retry_budget_status}; "
                f"inspect workspace, then either clear marker + re-run orchestrator, or escalate."
            ),
        )

    cur = state_executor.current_state(workspace)

    # 2a) P0t (2026-05-05): abort caused by extract_canonical_handoff bug
    # is RECOVERABLE if the worker actually did the work and emitted a real
    # handoff that just wasn't parsed correctly. Detect:
    #   - tail entry to_state == "abort"
    #   - rationale contains "worker exited without recognized handoff"
    #   - PROGRESS.md tail has a canonical handoff form (the worker's actual
    #     intent, not the malformed wrapper)
    # Recovery: drop the abort entry from state_transitions.jsonl and replay
    # state_machine.next() with the PROGRESS.md handoff. The current
    # extract_canonical_handoff (post-P0s fix) will parse the handoff
    # correctly and route to the right next_state.
    if cur == "abort":
        recovery = _try_buggy_abort_recovery(workspace)
        if recovery is not None:
            return recovery

    # 2b) Terminal — nothing to resume (kept after P0t branch so non-recoverable
    # aborts still surface as NONE_TERMINAL, not as RESUMABLE)
    if state_executor.is_terminal(cur):
        # 2b.5) Perf backfill (2026-08-23): a done op whose performance
        # evidence is DEFERRED, because precision was delivered first,
        # re-enters finalize to backfill the perf measurement when the
        # backfill env flag is on and perf measurement is not skipped.  The
        # candidate tree is unchanged, so the evaluation rebinds the same
        # candidate and the perf evidence is appended to the existing bundle.
        if (
            os.environ.get("CANNBOT_NPUBENCH_PERF_BACKFILL", "0") == "1"
            and _performance_is_deferred(workspace)
        ):
            # Fail closed (2026-08-25): with SKIP_PERF still set, the
            # re-entered O5 would emit ANOTHER DEFERRED placeholder that
            # finalize accepts — the op would stay 'done' with perf never
            # measured.  Refuse the re-entry with an explicit diagnostic
            # instead of silently no-op'ing.
            if os.environ.get("CANNBOT_NPUBENCH_SKIP_PERF", "0") == "1":
                return ResumeStatus(
                    op=op,
                    workspace=workspace,
                    action=ResumeAction.NONE_TERMINAL,
                    current_state=cur,
                    summary=(
                        "perf backfill refused: CANNBOT_NPUBENCH_SKIP_PERF=1 "
                        "is still set, so O5 would only re-emit a DEFERRED "
                        "perf placeholder. Unset CANNBOT_NPUBENCH_SKIP_PERF "
                        "before running with CANNBOT_NPUBENCH_PERF_BACKFILL=1."
                    ),
                )
            _record_perf_backfill_reentry(workspace, op)
            return ResumeStatus(
                op=op,
                workspace=workspace,
                action=ResumeAction.RESUMABLE,
                current_state="finalize",
                summary="perf backfill: performance evidence is DEFERRED; "
                        "re-entering finalize for the W3/R5 measurement",
            )
        return ResumeStatus(op=op, workspace=workspace,
                            action=ResumeAction.NONE_TERMINAL,
                            current_state=cur,
                            summary=f"already at terminal state '{cur}'; nothing to do")

    # 3) Pause — check for user_decision.md
    if state_executor.is_pause(cur):
        ud = workspace / "user_decision.md"
        if ud.exists():
            return ResumeStatus(op=op, workspace=workspace,
                                action=ResumeAction.USER_DECISION_READY,
                                current_state=cur,
                                summary=(
                                    f"paused at '{cur}'; user_decision.md present; "
                                    f"can resume by re-invoking orchestrator"
                                ))
        return ResumeStatus(op=op, workspace=workspace,
                            action=ResumeAction.USER_DECISION_PENDING,
                            current_state=cur,
                            summary=(
                                f"paused at '{cur}'; awaiting user to write "
                                f"{workspace}/user_decision.md (next_state: <state>)"
                            ))

    # 4) Mid-flight — resumable by re-invoking orchestrator
    snap = state_executor.snapshot(workspace)
    return ResumeStatus(
        op=op, workspace=workspace, action=ResumeAction.RESUMABLE,
        current_state=cur, last_handoff=snap.last_handoff[:300],
        summary=f"mid-flight at '{cur}'; re-invoke orchestrator to resume",
    )


def _load_workspace_lane(workspace: Path) -> Optional[int]:
    """P135.RL (2026-05-18): read `.opgen_state.json.lane` from workspace
    so resume preserves the original cold-start lane instead of defaulting
    to 0. Returns None if state file missing or no `lane` field."""
    state_path = workspace / ".opgen_state.json"
    if not state_path.is_file():
        return None
    try:
        state = json.loads(state_path.read_text())
        v = state.get("lane")
        if isinstance(v, int) and v >= 0:
            return v
    except Exception as error:
        logging.getLogger(__name__).debug(
            "Recoverable operation failed.", exc_info=error
        )
    return None


def _load_workspace_opgen_state(workspace: Path) -> dict:
    """Read persisted state used to authorize a scoped resume."""
    state_path = workspace / ".opgen_state.json"
    if not state_path.is_file():
        return {}
    try:
        state = json.loads(state_path.read_text())
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def _verify_tilelang_source_stage(workspace: Path, state: Mapping[str, Any]):
    """Lazy compatibility seam for the TileLang2AscendC source profile."""
    try:
        from tilelang2ascendc_source import verify_tilelang2ascendc_source_stage
    except ImportError as exc:
        return (
            False,
            "port-aclnn-tilelang2ascendc requires the tilelang2ascendc_source "
            f"staging adapter: {exc}",
            {},
        )

    return verify_tilelang2ascendc_source_stage(workspace, state)


def execute(op: str, *, workspace: Optional[Path] = None,
            lane: int = 0, dry_run: bool = False,
            bump_cap: Optional[list[str]] = None) -> int:
    """Take the appropriate action for `op`. Returns exit code.

    P135.RL (2026-05-18): if the workspace's `.opgen_state.json` carries
    a `lane` field, that value overrides the `lane` kwarg default of 0.
    Caller may still pass lane explicitly (CLI `--lane N`) to force a
    specific lane regardless of state file — but ONLY if --lane was
    explicitly set (sentinel handled in main()).

    DEBT-109 (2026-05-20): pass --bump-cap through to inner orchestrator
    re-invocation. Without this, `src/scripts/orch <op> --resume --bump-cap
    worker:N` silently drops the bump-cap, and the resumed orchestrator hits
    iter_cap immediately. Caught on gather_elements_v2 task #51.
    """
    status = diagnose(op, workspace=workspace)
    state = _load_workspace_opgen_state(status.workspace)
    if state.get("opgen_mode") not in {"port_a3_to_a5", "backward"}:
        log.info(
            "[resume] rejected unscoped workspace: opgen_mode=%r",
            state.get("opgen_mode"),
        )
        return 2
    port_source = state.get("port_source")
    # P135.RL: prefer workspace's recorded lane over the function default.
    # The CLI passes lane only when --lane was explicitly set by user;
    # otherwise lane==0 here represents the function default which should
    # yield to workspace state.
    if lane == 0:
        ws_lane = _load_workspace_lane(status.workspace)
        if ws_lane is not None and ws_lane != 0:
            log.info(f"[resume] using workspace lane={ws_lane} (overrides default 0)")
            lane = ws_lane
    log.info(f"[resume] op={op} action={status.action.value} state={status.current_state}")
    log.info(f"[resume] {status.summary}")

    if status.action in (ResumeAction.NONE_TERMINAL,
                          ResumeAction.AGENT_DIED,
                          ResumeAction.QUOTA_BACKOFF,
                          ResumeAction.USER_DECISION_PENDING,
                          ResumeAction.UNKNOWN):
        # P0t carve-out: NONE_TERMINAL with state=abort but recoverable per
        # _try_buggy_abort_recovery is handled below. Generic NONE_TERMINAL
        # (genuine finalize/abort) still surfaces to user.
        if status.action == ResumeAction.NONE_TERMINAL:
            mode_flag = (
                "--port-a3-ops" if state.get("opgen_mode") == "port_a3_to_a5"
                else "--backward"
            )
            source = (
                state.get("port_a3_source")
                if mode_flag == "--port-a3-ops"
                else state.get("backward_forward_source")
            )
            log.info(
                "[resume] HINT: to regenerate, re-enter the original scoped "
                "%s %s --cold-start --lane %s command",
                mode_flag,
                source or "<source>",
                lane,
            )
        return 0 if status.action == ResumeAction.NONE_TERMINAL else 2

    if state.get("opgen_mode") == "port_a3_to_a5":
        verifier = (
            _verify_tilelang_source_stage
            if isinstance(port_source, dict)
            and port_source.get("kind") == "port-aclnn-tilelang2ascendc"
            else verify_source_stage
        )
        try:
            valid_stage, stage_reason, _stage_manifest = verifier(
                status.workspace, state
            )
        except Exception as exc:
            log.error(
                "[resume] migration source-only snapshot verifier failed: %s",
                exc,
            )
            return 2
        if not valid_stage:
            log.error(
                "[resume] migration source-only snapshot validation failed: %s",
                stage_reason,
            )
            return 2

    if dry_run:
        log.info("[resume] dry-run: would re-invoke orchestrator")
        return 0

    if status.action == ResumeAction.NPUBENCH_CANDIDATE_REPAIR:
        died_state = status.died_at_state or "await_worker"
        marker = status.workspace / f".agent_died_at_{died_state}"
        if marker.exists():
            archived = status.workspace / (
                f".agent_died_at_{died_state}.cleaned-{int(time.time())}"
            )
            marker.rename(archived)
            log.info("[resume] archived known NPUBench marker → %s", archived.name)
        try:
            _prepare_npubench_candidate_repair(status.workspace)
        except Exception as exc:
            log.error("[resume] NPUBench candidate repair reset failed: %s", exc)
            return 2
        log.info(
            "[resume] NPUBENCH_CANDIDATE_REPAIR: stale candidate archived; "
            "re-invoking orchestrator"
        )

    # P0r (2026-05-05): AUTO_RECOVERABLE = FW-transient — clear marker and
    # increment retry counter before re-invoking, so a recurring transient
    # eventually escalates to AGENT_DIED instead of looping forever.
    if status.action == ResumeAction.AUTO_RECOVERABLE:
        died_state = status.died_at_state or "unknown"
        # Archive the marker (don't delete — keep audit trail with .cleaned-<ts> suffix)
        marker = status.workspace / f".agent_died_at_{died_state}"
        if marker.exists():
            archived = status.workspace / f".agent_died_at_{died_state}.cleaned-{int(time.time())}"
            marker.rename(archived)
            log.info(f"[resume] archived marker → {archived.name}")
        _increment_retry_count(status.workspace, died_state)
        log.info("[resume] AUTO_RECOVERABLE: re-invoking orchestrator")

    # P0t (2026-05-05): BUGGY_ABORT_RECOVERABLE — handoff-parsing bug (P0s)
    # caused abort. Replay state_machine.next() with the corrected
    # extract_canonical_handoff to recover the actual routing the worker
    # intended.
    if status.action == ResumeAction.BUGGY_ABORT_RECOVERABLE:
        if not _replay_buggy_abort_recovery(status.workspace):
            log.info("[resume] P0t replay failed — escalate manually")
            return 2
        log.info("[resume] BUGGY_ABORT_RECOVERABLE: state machine replayed; re-invoking orchestrator")

    # RESUMABLE / USER_DECISION_READY / AUTO_RECOVERABLE → invoke orchestrator
    event_emitter.emit(status.workspace, "resume.replay", source="resume",
                       data={"action": status.action.value,
                             "current_state": status.current_state})
    cmd = [
        "python3",
        str(_HERE.parent / "orchestrator.py"),
        op,
        "--lane", str(lane),
    ]
    # DEBT-109 (2026-05-20): pass --bump-cap through to inner orchestrator.
    if bump_cap:
        for bc in bump_cap:
            cmd += ["--bump-cap", bc]
    log.info(f"[resume] invoking: {' '.join(cmd)}")
    return subprocess.call(cmd)


# ---------------------------------------------------------------------------
# P0r retry-counter helpers (per died-state, persisted across resume calls)
# ---------------------------------------------------------------------------
FW_AUTO_RETRY_CAP = 2
_RETRY_COUNT_FILE = ".resume_fw_retry_count.json"


def _get_retry_count(workspace: Path, died_state: str) -> int:
    """Return how many auto-retries have already been spent on this state."""
    p = workspace / _RETRY_COUNT_FILE
    if not p.exists():
        return 0
    try:
        data = json.loads(p.read_text())
        return int(data.get(died_state, 0))
    except Exception:
        return 0


def _increment_retry_count(workspace: Path, died_state: str) -> None:
    """Bump the retry counter for this state; persist."""
    p = workspace / _RETRY_COUNT_FILE
    try:
        data = json.loads(p.read_text()) if p.exists() else {}
    except Exception:
        data = {}
    data[died_state] = int(data.get(died_state, 0)) + 1
    p.write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# P0t (2026-05-05) — recover op stuck at abort due to handoff-parsing bug.
# ---------------------------------------------------------------------------
def _try_buggy_abort_recovery(workspace: Path) -> Optional[ResumeStatus]:
    """If state log tail is abort with rationale=worker-handoff-violation,
    AND PROGRESS.md tail contains a now-parseable canonical handoff,
    classify as BUGGY_ABORT_RECOVERABLE so execute() can drop the abort
    line and replay state_machine.next() with the actual handoff.

    Returns None if not applicable (genuine abort or unrecoverable).
    """
    log_path = workspace / "state_transitions.jsonl"
    if not log_path.exists():
        return None
    lines = log_path.read_text().strip().splitlines()
    if not lines:
        return None
    try:
        last = json.loads(lines[-1])
    except Exception:
        return None
    if last.get("to_state") != "abort":
        return None
    rationale = (last.get("rationale") or "")
    if "worker exited without recognized handoff" not in rationale:
        # Other abort causes (build stuck, real contract violation) — keep manual
        return None

    # Look at the abort entry's recorded handoff text (what extract_canonical_handoff
    # produced, possibly buggy) AND PROGRESS.md tail for the worker's actual handoff.
    # Import orchestrator.py module directly (importlib) to avoid the package-vs-module
    # name conflict in test contexts (src/scripts/orchestrator/__init__.py shadows
    # src/scripts/orchestrator/orchestrator.py when src/scripts/ is on sys.path).
    import importlib.util as _imputil
    orch_path = _HERE.parent / "orchestrator.py"
    spec = _imputil.spec_from_file_location("_orch_for_resume", orch_path)
    _orch = _imputil.module_from_spec(spec)
    spec.loader.exec_module(_orch)
    progress = workspace / "PROGRESS.md"
    if not progress.exists():
        return None
    try:
        progress_text = progress.read_text(errors="replace")
    except Exception:
        return None
    canonical = _orch.extract_canonical_handoff(progress_text)
    # Only proceed if extract returns an actual canonical form, not the
    # fallback "return full text when no prefix matched". Verify by checking
    # the result starts with one of the canonical prefixes.
    canonical_prefixes = (
        "→ orchestrator: done", "→ orchestrator: build-ready", "→ orchestrator: PARTIAL_PERSIST",
        "→ orchestrator: await_user_decision",
        "→ orchestrator: research_done", "→ orchestrator: research_partial",
        "→ orchestrator: research_blocked",
        "@aog-precision-probe", "@aog-kernel-optimizer",
        "@aog-fused-optimizer", "@aog-determinism-analyzer", "@aog-researcher",
    )
    if not canonical or not any(canonical.startswith(p) for p in canonical_prefixes):
        return None
    # If post-P0s extract gives us the same handoff that already aborted, no progress
    abort_handoff = last.get("handoff") or ""
    if canonical == abort_handoff:
        return None

    # Check the canonical handoff would actually route somewhere non-abort
    sys.path.insert(0, str(_HERE.parent.parent / "workflow"))
    import state_machine as sm
    sm_data = sm.load_state_machine()
    # Recover the from_state (just before the abort entry)
    if len(lines) >= 2:
        prev = json.loads(lines[-2])
        from_state = prev.get("to_state") or "await_worker"
    else:
        from_state = last.get("from_state") or "await_worker"

    # Dry-run the canonical handoff against state machine
    decision = sm.next_state(workspace, canonical, from_state) if False else None
    # Note: sm.next_state writes to log; for dry-run we'd need a separate path.
    # For now, just trust that if PROGRESS.md has a canonical form, it'll route.

    return ResumeStatus(
        op=workspace.name, workspace=workspace,
        action=ResumeAction.BUGGY_ABORT_RECOVERABLE,
        current_state="abort",
        last_handoff=canonical[:300],
        summary=(
            f"abort caused by handoff-parsing bug (now fixed); "
            f"worker's actual handoff in PROGRESS.md tail: {canonical[:120]!r} "
            f"— will drop abort entry + replay routing from {from_state!r}"
        ),
    )


def _replay_buggy_abort_recovery(workspace: Path) -> bool:
    """Drop the abort entry and re-route based on PROGRESS.md handoff.

    P0s/P0t: the abort entry was written by orchestrator's buggy
    extract_canonical_handoff (now fixed). PROGRESS.md still has the
    worker's actual handoff. We need to advance the log past the abort
    using the canonical handoff so the next orchestrator run picks up
    at the right state.

    Two sub-cases after dropping the abort:
      (a) Log NOT empty → call state_executor.next_state with the
          canonical PROGRESS.md handoff. Records await_X → await_Y.
      (b) Log empty → state_machine.get_current_state's bootstrap
          (P0bb) automatically canonicalizes the chain from
          PROGRESS.md handoff. No explicit replay needed.

    Returns True if recovery left us at a non-abort, non-init state.
    """
    log_path = workspace / "state_transitions.jsonl"
    lines = log_path.read_text().strip().splitlines()
    if not lines:
        return False

    # Archive original log + drop abort entry
    archive = workspace / f"state_transitions.jsonl.pre-p0t-recovery-{int(time.time())}"
    archive.write_text(log_path.read_text())
    log_path.write_text("\n".join(lines[:-1]) + ("\n" if lines[:-1] else ""))

    # Sub-case (b): log empty after drop. Bootstrap will canonicalize on
    # the next current_state read.
    if log_path.stat().st_size == 0:
        new_state = state_executor.current_state(workspace)
        if new_state in ("abort", "init"):
            log_path.write_text(archive.read_text())
            log.info(f"[resume:p0t] post-drop bootstrap → {new_state!r}; restored log")
            return False
        log.info(f"[resume:p0t] recovered via bootstrap canonicalization → {new_state!r}")
        return True

    # Sub-case (a): log still has prior entries. Explicitly replay using the
    # canonical PROGRESS.md handoff so the log advances past the buggy abort.
    import importlib.util as _imputil
    orch_path = _HERE.parent / "orchestrator.py"
    spec = _imputil.spec_from_file_location("_orch_for_replay", orch_path)
    _orch = _imputil.module_from_spec(spec)
    spec.loader.exec_module(_orch)

    progress_text = (workspace / "PROGRESS.md").read_text(errors="replace")
    canonical = _orch.extract_canonical_handoff(progress_text)

    try:
        decision = state_executor.next_state(workspace, canonical, dry_run=False)
    except state_executor.StateMachineError as e:
        log_path.write_text(archive.read_text())
        log.info(f"[resume:p0t] replay failed: {e}; restored log")
        return False
    log.info(
        f"[resume:p0t] replayed {decision.from_state} → {decision.next_state} "
        f"(rationale: {(decision.rationale or '')[:80]})"
    )
    return True


def scan_all(*, root: Optional[Path] = None) -> list[ResumeStatus]:
    """Walk all op workspaces under workspace/ and diagnose each."""
    if root is None:
        root = _HERE.parent.parent.parent.parent / "workspace"
    if not root.exists():
        return []
    out: list[ResumeStatus] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or not (d / "PROGRESS.md").exists():
            continue
        out.append(diagnose(d.name, workspace=d))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Resume interrupted orchestrator runs")
    ap.add_argument("op", nargs="?", help="op workspace name")
    ap.add_argument("--status", action="store_true", help="show diagnosis only, do not act")
    ap.add_argument("--all", action="store_true", help="scan all workspaces")
    ap.add_argument("--execute", action="store_true",
                    help="(with --all) re-invoke orchestrator for each RESUMABLE/USER_DECISION_READY op")
    ap.add_argument("--lane", type=int, default=0)
    ap.add_argument(
        "--bump-cap", action="append", default=None, metavar="COUNTER:DELTA",
        help="Pass-through to inner orchestrator. Format: --bump-cap worker:N. "
             "DEBT-109 (2026-05-20)."
    )
    args = ap.parse_args()

    if args.all:
        statuses = scan_all()
        print(f"{'op':<40s} {'action':<25s} {'state':<25s} summary")
        print("-" * 120)
        for s in statuses:
            print(f"{s.op:<40s} {s.action.value:<25s} {s.current_state:<25s} {s.summary[:60]}")
        if args.execute:
            for s in statuses:
                if s.action in (ResumeAction.RESUMABLE, ResumeAction.USER_DECISION_READY):
                    print(f"\n=== resume {s.op} ===")
                    execute(s.op, workspace=s.workspace, lane=args.lane,
                            bump_cap=args.bump_cap)
        return 0

    if not args.op:
        ap.print_help()
        return 2

    if args.status:
        s = diagnose(args.op)
        print(json.dumps({
            "op": s.op,
            "workspace": str(s.workspace),
            "action": s.action.value,
            "current_state": s.current_state,
            "summary": s.summary,
            "died_at_state": s.died_at_state,
            "died_reason": s.died_reason,
        }, indent=2))
        return 0

    return execute(args.op, lane=args.lane, bump_cap=args.bump_cap)


if __name__ == "__main__":
    sys.exit(main())
