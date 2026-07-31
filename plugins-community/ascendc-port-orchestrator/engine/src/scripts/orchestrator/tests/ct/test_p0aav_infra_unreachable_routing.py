# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0aav (2026-05-08, DS investigation): @orchestrator: infra unreachable + BLOCKED
handoffs must route to await_user_decision, NOT abort.

Evidence: DS workers on A3 produced clear diagnostic handoffs like
"@orchestrator: infra unreachable, manual intervention needed. Host X has NPU
but missing Docker container Y." State machine had no transition matching these
patterns → fell through to abort catch-all, losing the diagnostic.
"""
import os
from pathlib import Path

import _reorg_paths  # reorg ut/it/ct: stable data-path anchors
import pytest
import yaml as _yaml

_PROJECT = _reorg_paths.REPO_ROOT


def _load_await_worker_exit_transitions():
    yaml_path = _PROJECT .parent / "workflows" / "opgen_state_machine.yaml"
    raw = _yaml.safe_load(yaml_path.read_text())
    aw = next(s for s in raw.get("phase_o4_states", [])
              if s.get("id") == "await_worker")
    return aw.get("exit_transitions", [])


def _find_transition(transitions, handoff_substring, expected_goto):
    """Scan transitions in order, return (index, condition) for first match."""
    for i, t in enumerate(transitions):
        cond = t.get("condition", {})
        all_of = cond.get("all_of", [])
        for clause in all_of:
            if isinstance(clause, dict) and "handoff_match" in clause:
                if handoff_substring in clause["handoff_match"]:
                    assert t.get("goto") == expected_goto, (
                        f"Transition {i} matched '{handoff_substring}' but "
                        f"goto={t.get('goto')} expected={expected_goto}"
                    )
                    return i
    return None


def test_infra_unreachable_routes_to_user_decision():
    """@orchestrator: infra unreachable → await_user_decision, NOT abort."""
    transitions = _load_await_worker_exit_transitions()
    idx = _find_transition(transitions, "infra unreachable", "await_user_decision")
    assert idx is not None, (
        "No transition found for 'infra unreachable' → await_user_decision. "
        "Ensure P0aav entry exists BEFORE the abort catch-all."
    )


def test_blocked_routes_to_user_decision():
    """@orchestrator: BLOCKED → await_user_decision, NOT abort."""
    transitions = _load_await_worker_exit_transitions()
    idx = _find_transition(transitions, "BLOCKED", "await_user_decision")
    assert idx is not None, (
        "No transition found for 'BLOCKED' → await_user_decision. "
        "Ensure P0aav entry exists BEFORE the abort catch-all."
    )


def test_p0aav_transitions_before_abort():
    """Both P0aav transitions must come BEFORE the abort catch-all."""
    transitions = _load_await_worker_exit_transitions()
    infra_idx = _find_transition(transitions, "infra unreachable", "await_user_decision")
    blocked_idx = _find_transition(transitions, "BLOCKED", "await_user_decision")
    abort_idx = None
    for i, t in enumerate(transitions):
        cond = t.get("condition", {})
        if cond.get("always") is True and t.get("goto") == "abort":
            abort_idx = i
            break
    assert abort_idx is not None, "abort catch-all not found"
    assert infra_idx < abort_idx, (
        f"'infra unreachable' at index {infra_idx} must be BEFORE "
        f"abort catch-all at index {abort_idx}"
    )
    assert blocked_idx < abort_idx, (
        f"'BLOCKED' at index {blocked_idx} must be BEFORE "
        f"abort catch-all at index {abort_idx}"
    )


def test_existing_orchestrator_build_stuck_still_routes_correctly():
    """P0aav must not break existing '@orchestrator: build stuck' routing."""
    transitions = _load_await_worker_exit_transitions()
    # Should route to abort (existing behavior with trajectory_sigs_stable=false)
    found_build_stuck = False
    for t in transitions:
        cond = t.get("condition", {})
        all_of = cond.get("all_of", [])
        for clause in all_of:
            if isinstance(clause, dict) and "handoff_match" in clause:
                if "build stuck" in clause["handoff_match"]:
                    found_build_stuck = True
    assert found_build_stuck, (
        "Existing '@orchestrator: build stuck' transition not found. "
        "P0aav must not have removed it."
    )


# ---------------------------------------------------------------------------
# P0abk (2026-05-09 DS-flagged): catch-all for any `@orchestrator:`-prefixed
# handoff that more-specific text patterns above didn't match. Workers use
# `@orchestrator:` as the prefix for honest diagnostics; specific text
# varies. Without this rule, "build PASS, verification blocked by infra"
# (DS 3_Add) routes to abort even though it's a valid diagnostic handoff.
# ---------------------------------------------------------------------------
def test_p0abk_catchall_routes_to_user_decision():
    """Any `@orchestrator:` prefix → await_user_decision (not abort)."""
    transitions = _load_await_worker_exit_transitions()
    # Find the catch-all `@orchestrator:` (without further text) rule
    found = False
    catchall_idx = None
    for i, t in enumerate(transitions):
        cond = t.get("condition", {})
        all_of = cond.get("all_of", [])
        for clause in all_of:
            if isinstance(clause, dict) and clause.get("handoff_match") == "@orchestrator:":
                found = True
                catchall_idx = i
                assert t.get("goto") == "await_user_decision", (
                    f"P0abk catch-all must route to await_user_decision, got "
                    f"{t.get('goto')}"
                )
                break
        if found:
            break
    assert found, "P0abk catch-all '@orchestrator:' transition not found"

    # Must be BEFORE the abort catch-all (otherwise abort wins on always:true)
    abort_idx = None
    for i, t in enumerate(transitions):
        cond = t.get("condition", {})
        if cond.get("always") is True and t.get("goto") == "abort":
            abort_idx = i
            break
    assert abort_idx is not None
    assert catchall_idx < abort_idx, (
        f"P0abk catch-all at {catchall_idx} must come before abort at {abort_idx}"
    )


def test_p0abk_specific_patterns_still_present():
    """P0abk catch-all must NOT remove the specific P0aav patterns
    (infra unreachable / BLOCKED / build stuck) — they're more specific
    and provide better forensic rationale text.
    """
    transitions = _load_await_worker_exit_transitions()
    expected_specific = ["infra unreachable", "BLOCKED", "build stuck"]
    for keyword in expected_specific:
        found = False
        for t in transitions:
            cond = t.get("condition", {})
            all_of = cond.get("all_of", [])
            for clause in all_of:
                if isinstance(clause, dict) and "handoff_match" in clause:
                    if keyword in clause["handoff_match"]:
                        found = True
                        break
            if found:
                break
        assert found, (
            f"Specific transition for '{keyword}' missing — P0abk catch-all "
            f"should ADD a fallback, not REPLACE the specific patterns."
        )
