# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Test await_user_decision pause state (DEBT-077 #59).

Covers:
- snapshot() parses user_decision.md `next_state:` value
- user_decision_target_in condition primitive
- __from_user_decision__ token resolution in next_state
- await_user_decision routing from worker handoff
- state_executor.is_pause()
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
import state_machine as sm  # noqa: E402
import state_executor as se  # noqa: E402
import orchestrator as orch  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "PROGRESS.md").write_text("# fresh\n")
    return tmp_path


def test_state_executor_is_pause():
    assert se.is_pause("await_user_decision") is True
    assert se.is_pause("await_worker") is False
    assert se.is_pause("finalize") is False
    assert se.is_pause("abort") is False


def test_state_executor_pause_state_has_no_agent():
    """Pause states must have no agent (orchestrator handles directly)."""
    assert se.next_agent("await_user_decision") is None


def test_snapshot_reads_user_decision_target(ws):
    """user_decision.md `next_state:` parsed into snapshot."""
    (ws / "user_decision.md").write_text(
        "# User decision\n\nnext_state: await_optimizer\nreason: try one more perf iter\n"
    )
    snap = sm.snapshot(ws)
    assert snap["user_decision_target"] == "await_optimizer"


def test_snapshot_user_decision_target_none_when_absent(ws):
    snap = sm.snapshot(ws)
    assert snap["user_decision_target"] is None


def test_snapshot_user_decision_with_bullet_prefix(ws):
    """Markdown bullet-prefix should also parse."""
    (ws / "user_decision.md").write_text(
        "## Decision\n\n- next_state: await_researcher\n- reason: structural rewrite needed\n"
    )
    snap = sm.snapshot(ws)
    assert snap["user_decision_target"] == "await_researcher"


def test_user_decision_target_in_condition_primitive(ws):
    """eval_condition with user_decision_target_in primitive."""
    (ws / "user_decision.md").write_text("next_state: await_worker\n")
    snap = sm.snapshot(ws)
    ctx = {"handoff": "", "snapshot": snap, "iter_counts": {}, "ws": ws, "sm": {}}
    # Allowed-list match
    assert sm.eval_condition({"user_decision_target_in": ["await_worker", "await_probe"]}, ctx) is True
    # Not in list
    assert sm.eval_condition({"user_decision_target_in": ["finalize"]}, ctx) is False


def test_user_decision_target_in_returns_false_when_absent(ws):
    """No user_decision.md → primitive returns False."""
    snap = sm.snapshot(ws)
    ctx = {"handoff": "", "snapshot": snap, "iter_counts": {}, "ws": ws, "sm": {}}
    assert sm.eval_condition({"user_decision_target_in": ["await_worker"]}, ctx) is False


def test_await_user_decision_resolves_magic_token(ws):
    """next_state() resolves __from_user_decision__ via parsed user_decision.md target."""
    (ws / "user_decision.md").write_text("next_state: await_optimizer\nreason: attempt perf bump\n")
    # Seed empty state log so current_state defaults
    (ws / "state_transitions.jsonl").write_text(
        json.dumps({
            "ts": "2026-05-04T05:00:00Z",
            "from_state": "await_worker",
            "to_state": "await_user_decision",
            "handoff": "→ orchestrator: await_user_decision — soft judgment call",
            "matched_transition_index": 0,
            "rationale": "test seed",
            "iter_counts_snapshot": {},
        }) + "\n"
    )
    result = sm.next_state(ws, "await_user_decision", "")
    assert "error" not in result, result
    # Magic token resolved to user-chosen target
    assert result["next_state"] == "await_optimizer"


def test_worker_routing_to_await_user_decision(ws):
    """Worker handoff `→ orchestrator: await_user_decision` routes there."""
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL", "pass_a": {"status": "PASS"},
                      "pass_b": {"status": "PARTIAL"}},
        "performance": {"status": "PASS", "ratio": 0.7},
        "determinism": {"policy_satisfied": True},
    }))
    handoff = "→ orchestrator: await_user_decision — should we ship PARTIAL or escalate?"
    result = sm.next_state(ws, "await_worker", handoff)
    assert "error" not in result, result
    assert result["next_state"] == "await_user_decision"


def test_extract_canonical_handoff_recognizes_await_user_decision():
    """orchestrator.extract_canonical_handoff covers await_user_decision."""
    stdout = (
        "Lots of analysis output...\n"
        "## Recommendation\n"
        "Marginal call between PARTIAL_PERSIST and researcher escalation.\n"
        "→ orchestrator: await_user_decision — please choose await_researcher or finalize\n"
    )
    extracted = orch.extract_canonical_handoff(stdout)
    assert extracted.startswith("→ orchestrator: await_user_decision")
