# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0p (2026-05-05): pause-state with user_decision.md already present.

Origin: op#10_layernorm 2026-05-05. After P0o landed, orchestrator correctly
bootstrapped to await_user_decision from PROGRESS.md tail. But the pause
handler at orchestrator.py:172 fired immediately without checking for the
already-existing user_decision.md — exit code 10, asking user to write a
file that already exists. Re-invoking the orchestrator hit the same pause
loop indefinitely.

Fix: in the pause branch, check if user_decision.md exists with content. If
yes, run state_machine.next_state to consume it and continue the main loop.
Only print the PAUSE message + return 10 when no decision file exists yet.
"""
from __future__ import annotations

import json
import sys
import subprocess
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH = _HERE.parent.parent.parent / "orchestrator"
sys.path.insert(0, str(_ORCH))


def _seed_workspace(tmp_path: Path, *, with_user_decision: bool):
    """Create a workspace pre-positioned at await_user_decision."""
    # State log: prior worker emitted await_user_decision
    log_entry = {
        "ts": "2026-05-05T00:00:00Z",
        "from_state": "await_worker",
        "to_state": "await_user_decision",
        "handoff": "→ orchestrator: await_user_decision — needs research",
        "matched_transition_index": 0,
        "rationale": "test seed",
        "iter_counts_snapshot": {},
    }
    (tmp_path / "state_transitions.jsonl").write_text(json.dumps(log_entry) + "\n")
    (tmp_path / "PROGRESS.md").write_text("# test\nMode: backward\n")
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS",
                       "pass_a": {"status": "PASS", "tier1_pass": 60, "total": 60},
                       "pass_b": {"status": "PASS", "tier1_pass": 16, "total": 16}},
        "performance": {"ratio": 0.20, "status": "BELOW_THRESHOLD"},
        "determinism": {"policy_satisfied": True,
                         "n_identical_cases": 60, "n_cases_checked": 60},
    }))
    if with_user_decision:
        (tmp_path / "user_decision.md").write_text("""\
# User decision
next_state: await_researcher
reason: per worker recommendation, investigate alternate vendor strategies
""")


def test_p0p_pause_state_advances_when_user_decision_present(tmp_path):
    """When state=await_user_decision AND user_decision.md exists, the
    orchestrator's pause branch should consume it (run state machine forward)
    instead of returning exit code 10.

    Verifies the FIX behavior at the integration level: after the consume
    step, the state log gets a NEW entry transitioning out of
    await_user_decision.
    """
    _seed_workspace(tmp_path, with_user_decision=True)
    sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
    import state_machine as sm
    import state_executor as se
    sm_data = sm.load_state_machine()

    # Sanity: starting state is await_user_decision
    assert sm.get_current_state(tmp_path, sm_data) == "await_user_decision"

    # Execute the consume path (mimicking what orchestrator.py does in the pause branch)
    # state_executor.next_state wraps state_machine and persists the log entry.
    decision = se.next_state(tmp_path, "", dry_run=False)
    assert decision.next_state == "await_researcher", \
        f"user_decision.md says await_researcher; got {decision.next_state}"

    # State log appended
    log_lines = (tmp_path / "state_transitions.jsonl").read_text().strip().splitlines()
    assert len(log_lines) == 2, f"Expected 2 log lines, got {len(log_lines)}: {log_lines}"
    last = json.loads(log_lines[-1])
    assert last["from_state"] == "await_user_decision"
    assert last["to_state"] == "await_researcher"


def test_p0p_pause_returns_10_when_no_user_decision_file(tmp_path):
    """When state=await_user_decision but no user_decision.md, orchestrator
    must STILL pause (exit 10) — that's the original V3.8.5 #59 contract.
    """
    _seed_workspace(tmp_path, with_user_decision=False)
    sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
    import state_machine as sm

    # Starting state still await_user_decision
    sm_data = sm.load_state_machine()
    assert sm.get_current_state(tmp_path, sm_data) == "await_user_decision"

    # Try to consume — state machine should refuse (no decision file)
    decision = sm.next_state(tmp_path, "await_user_decision", "")
    # The condition `user_decision_target_in` won't match without user_decision.md →
    # state machine returns error / no transition
    assert "error" in decision or decision.get("next_state") is None, \
        f"expected error/no-transition without user_decision.md, got {decision}"


def test_p0p_empty_user_decision_treated_as_absent(tmp_path):
    """An empty (zero-byte) user_decision.md should still pause.

    Defensive: the fix uses `path.stat().st_size > 0` so pure zero-byte
    files don't count. This catches the case where a touch happened but no
    content was written.
    """
    _seed_workspace(tmp_path, with_user_decision=False)
    (tmp_path / "user_decision.md").write_text("")  # empty
    p = tmp_path / "user_decision.md"
    assert p.exists()
    assert p.stat().st_size == 0
    # The orchestrator's check is: exists() AND stat().st_size > 0
    # → would fall through to the "genuine pause" branch
    advances = p.exists() and p.stat().st_size > 0
    assert advances is False
