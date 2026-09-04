# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""State executor — Python-side wrapper around opgen_state_machine.yaml.

Replaces the LLM-orchestrator pattern of "read SKILL.md prose and improvise"
with deterministic YAML-driven phase transitions.

Importantly: this module does NOT duplicate src/scripts/workflow/state_machine.py
(which has 590 lines of YAML loading, condition DSL evaluation, snapshot
parsing). Instead, it imports + wraps that module's functions to expose an
orchestrator-facing API.

The orchestrator calls state_executor.* methods on its main loop; the LLM
never makes phase decisions at the top level.
"""
from __future__ import annotations
import logging

import datetime as _dt
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Import from the existing workflow module to reuse YAML loader, condition
# evaluator, snapshot parser. We add orchestrator-facing convenience methods.
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent.parent  # src/scripts/orchestrator/state_executor.py → repo root
sys.path.insert(0, str(_PROJECT_ROOT / "src" / "scripts" / "workflow"))
import state_machine as _sm  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# P0dd: finalize is no longer terminal — it's a real state with finalize_pipeline as its agent
TERMINAL_STATES = ("done", "abort")
PAUSE_STATES = ("await_user_decision",)  # V3.8.5 / DEBT-077 #59
INITIAL_STATE = "await_worker"  # canonical first state for cold-start ops

# Map state → agent type that owns it. Used by orchestrator to decide which
# agent to spawn next.
STATE_TO_AGENT = {
    "await_worker": "aog-kernel-worker",
    "await_optimizer": "aog-kernel-optimizer",
    "await_probe": "aog-precision-probe",
    "await_fused_optimizer": "aog-fused-optimizer",
    "await_researcher": "aog-researcher",
    "await_det_analyzer": "aog-determinism-analyzer",
    "await_user_decision": None,  # pause — orchestrator exits gracefully
    # P0dd (2026-05-05): finalize is a real state with a synthetic in-process
    # "agent" — the orchestrator main loop calls finalize_pipeline.finalize_op
    # directly when current_state == finalize, then routes to `done`.
    "finalize": "aog-finalize-pipeline",
    # Bounded research recovery used only by the arch22 -> arch35 plugin opt-in.
    "await_cann_learn": "aog-cann-learner",
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------
@dataclass
class TransitionDecision:
    """Result of state_executor.next_state(ws, handoff)."""
    next_state: str
    matched_transition_index: int
    rationale: str
    from_state: str
    handoff: str
    # NODE-5 (2026-05-28): rollback classification — when this transition
    # represents an infra-rollback (e.g. O5 RUNNER_FAILED on SCP timeout /
    # JSON parse fail / oversized .pt / verifier env issue), the orchestrator
    # builds the decision with `rollback_kind="infra"` so
    # `record_transition` writes that tag into state_transitions.jsonl;
    # `state_machine.iter_counts_from_log` then skips those entries from
    # the iter_below_cap accounting (the kernel didn't change). Values:
    # None (default — no rollback, or algorithm rollback that consumes
    # iter), "infra" (free re-entry), "algorithm" (explicit — same effect
    # as None for budget accounting, but logged for audit).
    rollback_kind: Optional[str] = None


@dataclass
class StateSnapshot:
    """Workspace state at a point in time. Used for orchestrator decisions."""
    op: str
    workspace: Path
    current_state: str
    is_terminal: bool
    last_handoff: str
    iter_counts: dict[str, int]
    iter_caps: dict[str, int]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def current_state(workspace: Path) -> str:
    """Get current state from state_transitions.jsonl tail; default to INITIAL_STATE for fresh workspace."""
    sm = _sm.load_state_machine()
    return _sm.get_current_state(workspace, sm)


def next_agent(state: str) -> Optional[str]:
    """Return the agent type owning `state`, or None for terminal states."""
    if state in TERMINAL_STATES:
        return None
    return STATE_TO_AGENT.get(state)


def is_terminal(state: str) -> bool:
    return state in TERMINAL_STATES


def is_pause(state: str) -> bool:
    """V3.8.5 / DEBT-077 #59: pause states halt the orchestrator awaiting
    user decision (workspace/{op}/user_decision.md). Not terminal — resumable.
    """
    return state in PAUSE_STATES


def iter_count(workspace: Path, counter: str) -> int:
    """How many times agent type `counter` has been invoked for this op."""
    sm = _sm.load_state_machine()
    counts = _sm.iter_counts_from_log(workspace, sm)
    return counts.get(counter, 0)


def _read_cap_bumps(workspace: Path) -> dict[str, int]:
    """V3.8.5 / DEBT-077 #61: read all entries from .cap_bumps.jsonl + sum
    per-counter deltas. Returns {counter: total_bump}. Empty dict if file
    absent. Each entry is a JSON dict {ts, bumps: {counter: delta}, actor,
    rationale}."""
    log = workspace / ".cap_bumps.jsonl"
    if not log.exists():
        return {}
    out: dict[str, int] = {}
    for line in log.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        skip_current_item = False
        try:
            entry = json.loads(line)
        except Exception as error:
            logging.getLogger(__name__).debug(
                "Recoverable operation failed.", exc_info=error
            )
            skip_current_item = True
        if skip_current_item:
            continue
        bumps = entry.get("bumps") or {}
        if not isinstance(bumps, dict):
            continue
        for ctr, delta in bumps.items():
            try:
                out[ctr] = out.get(ctr, 0) + int(delta)
            except (TypeError, ValueError):
                continue
    return out


def iter_cap(state: str, *, workspace: Path | None = None, plugin=None) -> int:
    """Read iter_cap for a state from YAML plus workspace cap-bumps.

    Resolution order:
    1. YAML default (canonical, ground truth)
    2. workspace/.cap_bumps.jsonl additive bumps (V3.8.5 / DEBT-077 #61, user
       explicit only)
    """
    sm = _sm.load_state_machine()
    spec = _sm.get_state_spec(sm, state)
    if spec is None:
        return 999
    base = int(spec.get("iter_cap", 999))

    if workspace is None:
        return base
    counter = spec.get("iter_counter", state)
    bumps = _read_cap_bumps(workspace)
    return base + bumps.get(counter, 0)


def at_iter_cap(workspace: Path, state: str) -> bool:
    """True if current iter count for state's counter equals or exceeds cap.

    Cap = YAML iter_cap + cumulative .cap_bumps.jsonl bumps for the counter.
    """
    sm = _sm.load_state_machine()
    spec = _sm.get_state_spec(sm, state)
    if spec is None:
        return False
    counter = spec.get("iter_counter", state)
    cap = iter_cap(state, workspace=workspace)
    counts = _sm.iter_counts_from_log(workspace, sm)
    return counts.get(counter, 0) >= cap


def snapshot(workspace: Path) -> StateSnapshot:
    """Read full workspace state for orchestrator decisions."""
    sm = _sm.load_state_machine()
    state = _sm.get_current_state(workspace, sm)
    counts = _sm.iter_counts_from_log(workspace, sm)
    caps = {}
    for s in sm.get("phase_o4_states", []):
        counter = s.get("iter_counter", s["id"])
        caps[counter] = int(s.get("iter_cap", 999))
    last_handoff = ""
    log_file = workspace / "state_transitions.jsonl"
    if log_file.exists():
        lines = log_file.read_text().splitlines()
        for line in reversed(lines):
            if not line.strip():
                continue
            skip_current_item = False
            try:
                entry = json.loads(line)
                last_handoff = entry.get("handoff", "")
                break
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Recoverable operation failed.", exc_info=error
                )
                skip_current_item = True
            if skip_current_item:
                continue
    return StateSnapshot(
        op=workspace.name,
        workspace=workspace,
        current_state=state,
        is_terminal=is_terminal(state),
        last_handoff=last_handoff,
        iter_counts=counts,
        iter_caps=caps,
    )


def next_state(
    workspace: Path,
    handoff: str,
    *,
    from_state: Optional[str] = None,
    dry_run: bool = False,
    runtime_kwargs: Optional[dict] = None,
) -> TransitionDecision:
    """Compute next state given current state + handoff. Routes via YAML.

    If `from_state` is None, derives it from the log via current_state().
    Callers that already know the from_state (e.g. orchestrator's main loop
    has snap.current_state from BEFORE the agent ran) should pass it
    explicitly to avoid the bootstrap-re-derive race after PROGRESS.md
    changes mid-iteration.

    If `dry_run` is False, appends to state_transitions.jsonl.

    2026-05-20: resolves the active plugin from the workspace and threads it
    into _sm.next_state so the generic `plugin_method` YAML primitive (S3c)
    can dispatch to plugin methods.

    2026-05-27 (force-switch generalization, per owner direction): optional
    `runtime_kwargs` dict — per-invocation values forwarded to plugin methods
    via the YAML `forward_kwargs` list on the `plugin_method` primitive. None
    = no kwargs forwarded (back-compat).
    """
    if from_state is None:
        from_state = current_state(workspace)

    # P1-4 (PR875 equiv review, 2026-08-28): a worker that has diagnosed an
    # ENGINE-level blockage (not a candidate defect) may report it with the
    # `→ orchestrator: engine-block...` diagnostic handoff.  The YAML knows no
    # such verb; accept it here ONLY when the persistent same-signature
    # counter shows >= SAME_SIGNATURE_PARK_THRESHOLD consecutive identical
    # engine-class failures (otherwise the unknown handoff keeps its legacy
    # no-match behavior).  Routing is await_user_decision — a human clears the
    # engine gap; the worker is never asked to patch the engine.
    override = _engine_block_handoff_decision(workspace, from_state, handoff)
    if override is not None:
        if not dry_run:
            record_transition(workspace, override)
        return override

    # Plugin resolution for the `plugin_method` YAML primitive (S3c).
    # Detection errors must propagate: treating an unreadable/ambiguous
    # supported workspace as plugin=None can bypass migration research
    # recovery and other workflow-owned transitions.
    plugin = None
    if workspace is not None and workspace.exists():
        from plugins import detect_plugin
        plugin = detect_plugin(workspace)

    result = _sm.next_state(
        workspace, from_state, handoff,
        plugin=plugin, runtime_kwargs=runtime_kwargs,
    )
    if "error" in result:
        raise StateMachineError(result["error"])

    decision = TransitionDecision(
        next_state=result["next_state"],
        matched_transition_index=result.get("matched_transition_index", -1),
        rationale=result.get("rationale", ""),
        from_state=result.get("from_state", ""),
        handoff=handoff,
    )

    if not dry_run:
        record_transition(workspace, decision)

    return decision


def record_transition(workspace: Path, decision: TransitionDecision) -> None:
    """Append a canonical state_transitions.jsonl entry. Schema:
       {ts, from_state, to_state, handoff, matched_transition_index, rationale, iter_counts_snapshot, rollback_kind?}

    `rollback_kind` (NODE-5 2026-05-28) is written ONLY when non-None — keeps
    legacy entries clean and back-compat parsers happy. Values: "infra" /
    "algorithm". `iter_counts_from_log` skips entries with rollback_kind=="infra"
    so infra re-entries don't consume the iter_below_cap budget.
    """
    log_file = workspace / "state_transitions.jsonl"
    sm = _sm.load_state_machine()
    counts = _sm.iter_counts_from_log(workspace, sm)
    entry = {
        "ts": _dt.datetime.now(_dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "from_state": decision.from_state,
        "to_state": decision.next_state,
        "handoff": decision.handoff,
        "matched_transition_index": decision.matched_transition_index,
        "rationale": decision.rationale,
        "iter_counts_snapshot": counts,
    }
    if decision.rollback_kind is not None:
        entry["rollback_kind"] = decision.rollback_kind
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a") as f:
        f.write(json.dumps(entry) + "\n")


def validate_log(workspace: Path) -> tuple[bool, list[str]]:
    """Sanity-check state_transitions.jsonl: every transition declared in YAML."""
    log_file = workspace / "state_transitions.jsonl"
    if not log_file.exists():
        return True, []  # empty log is valid (cold-start)

    sm = _sm.load_state_machine()
    valid_states = {s["id"] for s in sm.get("phase_o4_states", [])} | set(TERMINAL_STATES) | {"init"}

    errors = []
    for i, line in enumerate(log_file.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception as e:
            errors.append(f"line {i}: not valid JSON ({e})")
            continue
        if "from_state" not in entry or "to_state" not in entry:
            errors.append(f"line {i}: missing from_state/to_state (canonical schema required)")
            continue
        for field in ("from_state", "to_state"):
            v = entry.get(field)
            if v and v not in valid_states:
                errors.append(f"line {i}: {field}={v!r} not in YAML phase_o4_states")
    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# P0-1 (PR875 equiv review, DSH ruling §4.b, 2026-08-28): same-signature
# failure parking — per-op PERSISTENT layer.
#
# Per-process counters cannot see the loop: spawn_count resets on every
# orchestrator restart and exit-77 engine-drift restarts were observed 7x on a
# single op (oc FAS line).  The observation window therefore lives in a
# workspace dotfile that survives exit 77 / resume / cold-start.
#
# The signature combines failure class, normalized reason, and candidate-tree
# digest. Candidate MISMATCH is tracked separately from the repair loop; run
# identifiers, timestamps, and binding hashes are normalized before hashing.
# A changed candidate-tree digest therefore resets the consecutive count.
# Consecutive-only: any different signature inserted resets the count, so
# "different causes failing in turn" is never misread as a stuck loop.
# ---------------------------------------------------------------------------
SAME_SIGNATURE_STATE_FILE = ".opgen_same_signature.json"
SAME_SIGNATURE_PARK_THRESHOLD = 3
SAME_SIGNATURE_PARK_THRESHOLD_ENV = "AOG_SAME_SIGNATURE_PARK_THRESHOLD"
# Device-class errors do not self-heal (2026-08-27 3FA: 507015 wedged device 3
# for 95 minutes while the engine slept 4x600s).  Park earlier and never
# backoff-wait on them.
DEVICE_SIGNATURE_PARK_THRESHOLD = 2
# candidate-class escalation ("改不动=卡死"): same candidate tree + same
# per-case failure signature for this many O5 rounds → await_user_decision.
CANDIDATE_CASE_PARK_THRESHOLD = 3

_SAME_SIGNATURE_CLASSES = frozenset({"engine", "infra", "device"})

_RUN_ID_RE = re.compile(r"\b(?:run|attempt|binding)[_-][A-Za-z0-9][A-Za-z0-9_.-]*\b")
_HEX_DIGEST_RE = re.compile(r"\b[0-9a-fA-F]{12,64}\b")
_ISO_TS_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_CLOCK_RE = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:\.\d+)?\b")
_WS_RE = re.compile(r"\s+")


def normalize_failure_reason(reason: object) -> str:
    """Strip per-run identifiers (run ids, timestamps, binding hashes) so two
    observations of the SAME failure produce the same signature input.
    """
    text = str(reason or "")
    text = _ISO_TS_RE.sub("<ts>", text)
    text = _CLOCK_RE.sub("<clock>", text)
    text = _HEX_DIGEST_RE.sub("<hex>", text)
    text = _RUN_ID_RE.sub("<run>", text)
    return _WS_RE.sub(" ", text).strip()[:300]


def same_failure_signature(failure_class: str, reason: object, tree_sha256: object) -> str:
    """Canonical signature per DSH §4.b: class ‖ normalized reason ‖ tree."""
    material = "\x1f".join(
        [str(failure_class), normalize_failure_reason(reason), str(tree_sha256 or "unknown")]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def load_same_signature_state(workspace: Path) -> dict:
    """Read the persistent same-signature record; {} when absent/unreadable."""
    try:
        data = json.loads((Path(workspace) / SAME_SIGNATURE_STATE_FILE).read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_same_signature_state(workspace: Path, state: dict) -> None:
    path = Path(workspace) / SAME_SIGNATURE_STATE_FILE
    try:
        path.write_text(json.dumps(state, indent=1, sort_keys=True) + "\n")
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "cannot persist same-signature state (non-fatal): %s", exc
        )


def record_same_signature_failure(
    workspace: Path, failure_class: str, reason: object, tree_sha256: object,
) -> dict:
    """Record one engine/infra/device-class failure; return the updated entry.

    Consecutive identical signatures accumulate; any different signature
    (including a candidate_tree change, which is part of the signature)
    resets the count to 1.  Caller decides the parking threshold.
    """
    if failure_class not in _SAME_SIGNATURE_CLASSES:
        raise ValueError(f"not a same-signature-parked failure class: {failure_class!r}")
    state = load_same_signature_state(workspace)
    signature = same_failure_signature(failure_class, reason, tree_sha256)
    previous = state.get("same_signature")
    count = 1
    if isinstance(previous, dict) and previous.get("signature") == signature:
        count = int(previous.get("count", 0)) + 1
    entry = {
        "signature": signature,
        "failure_class": failure_class,
        "count": count,
        "reason_norm": normalize_failure_reason(reason),
        "last_tree_sha256": str(tree_sha256 or "unknown"),
        "last_ts": time.time(),
    }
    state["schema"] = "aog.same_signature_failures/v1"
    state["same_signature"] = entry
    _save_same_signature_state(workspace, state)
    return entry


def clear_same_signature_state(workspace: Path) -> None:
    """Break the consecutive-failure chain (a successful O5 between failures,
    or a candidate-class MISMATCH, is a different-signature insertion).
    """
    state = load_same_signature_state(workspace)
    if "same_signature" in state:
        state.pop("same_signature", None)
        _save_same_signature_state(workspace, state)


def same_signature_count(workspace: Path, failure_class: str) -> int:
    """Current consecutive count for a class (0 when the chain is absent or
    belongs to another class).  Used by the lifetime-cost warning linkage
    (P2-5) and the engine-block handoff gate (P1-4).
    """
    entry = load_same_signature_state(workspace).get("same_signature")
    if isinstance(entry, dict) and entry.get("failure_class") == failure_class:
        try:
            return int(entry.get("count", 0))
        except (TypeError, ValueError):
            return 0
    return 0


def same_signature_park_threshold(failure_class: str) -> int:
    if failure_class == "device":
        return DEVICE_SIGNATURE_PARK_THRESHOLD
    raw = os.environ.get(SAME_SIGNATURE_PARK_THRESHOLD_ENV)
    if raw is not None:
        try:
            return max(int(raw.strip()), 1)
        except ValueError:
            pass
    return SAME_SIGNATURE_PARK_THRESHOLD


def record_candidate_case_failure(
    workspace: Path, tree_sha256: object, case_signature: object,
) -> dict:
    """Candidate-class escalation counter (DSH §4.b guardrail ②).

    A genuine precision MISMATCH is the repair loop itself and must NOT feed
    the same-signature parking counter — but "same candidate tree + same
    per-case failure signature for N rounds" means the worker cannot move the
    failure (FAS kw-17..21 fence whack-a-mole), which is a stuck loop of its
    own.  Keyed on (tree, case signature); any change resets.
    """
    state = load_same_signature_state(workspace)
    case_sig = hashlib.sha256(
        ("\x1f".join([str(tree_sha256 or "unknown"), normalize_failure_reason(case_signature)]))
        .encode("utf-8")
    ).hexdigest()
    previous = state.get("candidate_case")
    count = 1
    if isinstance(previous, dict) and previous.get("case_signature") == case_sig:
        count = int(previous.get("count", 0)) + 1
    entry = {
        "case_signature": case_sig,
        "count": count,
        "last_tree_sha256": str(tree_sha256 or "unknown"),
        "last_ts": time.time(),
    }
    state["schema"] = "aog.same_signature_failures/v1"
    state["candidate_case"] = entry
    _save_same_signature_state(workspace, state)
    return entry


def clear_candidate_case_state(workspace: Path) -> None:
    state = load_same_signature_state(workspace)
    if "candidate_case" in state:
        state.pop("candidate_case", None)
        _save_same_signature_state(workspace, state)


# ---------------------------------------------------------------------------
# P1-4 (PR875 equiv review, 2026-08-28): engine-block diagnostic handoff.
# ---------------------------------------------------------------------------
ENGINE_BLOCK_HANDOFF_PREFIX = "→ orchestrator: engine-block"


def _engine_block_handoff_decision(
    workspace: Path, from_state: str, handoff: str,
) -> Optional["TransitionDecision"]:
    """Accept a worker's engine-block diagnostic handoff only when the
    persistent same-signature counter proves an engine-class stuck loop
    (>= SAME_SIGNATURE_PARK_THRESHOLD consecutive identical engine failures);
    route it to await_user_decision.  Returns None to keep legacy YAML routing.
    """
    if not handoff or not handoff.lstrip().startswith(ENGINE_BLOCK_HANDOFF_PREFIX):
        return None
    count = same_signature_count(workspace, "engine")
    if count < same_signature_park_threshold("engine"):
        return None
    return TransitionDecision(
        next_state="await_user_decision",
        matched_transition_index=-1,
        rationale=(
            f"P1-4 engine-block handoff accepted: {count} consecutive identical "
            f"engine-class failures (same-signature counter); worker diagnostic: "
            f"{handoff.lstrip()[:300]}. Halting for user inspection — the engine "
            "gap needs a human, not another worker respawn."
        ),
        from_state=from_state,
        handoff=handoff,
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------
class StateMachineError(RuntimeError):
    """Raised when state machine returns an unrecoverable error."""


# ---------------------------------------------------------------------------
# CLI for smoke-testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="state_executor.py — orchestrator-side state machine wrapper")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("snapshot", help="show full state snapshot for a workspace")
    sp.add_argument("--workspace", required=True, type=Path)

    sp = sub.add_parser("next", help="dry-run state machine transition")
    sp.add_argument("--workspace", required=True, type=Path)
    sp.add_argument("--handoff", default="", help="last handoff text")
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("validate", help="validate state_transitions.jsonl is canonical")
    sp.add_argument("--workspace", required=True, type=Path)

    args = ap.parse_args()

    if args.cmd == "snapshot":
        snap = snapshot(args.workspace)
        print(json.dumps({
            "op": snap.op,
            "current_state": snap.current_state,
            "is_terminal": snap.is_terminal,
            "next_agent": next_agent(snap.current_state),
            "last_handoff": snap.last_handoff[:200],
            "iter_counts": snap.iter_counts,
            "iter_caps": snap.iter_caps,
            "at_cap": {c: snap.iter_counts.get(c, 0) >= cap for c, cap in snap.iter_caps.items()},
        }, indent=2))

    elif args.cmd == "next":
        decision = next_state(args.workspace, args.handoff, dry_run=args.dry_run)
        print(json.dumps({
            "from_state": decision.from_state,
            "to_state": decision.next_state,
            "matched_transition_index": decision.matched_transition_index,
            "rationale": decision.rationale,
            "next_agent": next_agent(decision.next_state),
            "is_terminal": is_terminal(decision.next_state),
        }, indent=2))

    elif args.cmd == "validate":
        ok, errors = validate_log(args.workspace)
        if ok:
            print(json.dumps({"valid": True}))
        else:
            print(json.dumps({"valid": False, "errors": errors}, indent=2))
            sys.exit(2)
