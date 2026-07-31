# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0o (2026-05-05): empty state log + non-empty PROGRESS.md — bootstrap
current_state from PROGRESS.md tail handoff instead of always defaulting to
await_worker.

Origin: op#10_layernorm 2026-05-05. Worker emitted `→ orchestrator:
await_user_decision` in PRIOR session (lost to orchestrator hang via P0m).
Re-running orchestrator on the workspace cleared state log; bootstrap
defaulted to await_worker. New worker re-validated HARD GATE (8th time, $2.31
wasted) then read user_decision.md and emitted `@aog-researcher` — which is
NOT a valid await_worker exit transition → abort.

Fix: get_current_state, when log empty, scans PROGRESS.md tail for a
canonical handoff line and runs the state machine forward from the YAML's
initial state to derive the actual current state.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
import state_machine as sm  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    """Empty workspace — no state log."""
    (tmp_path / "state_transitions.jsonl").write_text("")
    return tmp_path


def _seed_progress_with_handoff(ws, handoff_line):
    """Write a PROGRESS.md whose tail contains the given handoff."""
    (ws / "PROGRESS.md").write_text(f"""\
# PROGRESS — test_op
Mode: backward
Started: 2026-05-04T10:00:00Z

## Targets
- Precision: 60/60 PASS
- Det: 60/60
- Perf: ≥ 0.6x

### [10:30] worker
Did stuff.

### EXIT
{handoff_line}
""")


def _seed_verification(ws, status="PASS", perf=0.20):
    """Worker baseline verification.json so condition checks have data."""
    import json
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": status,
                       "pass_a": {"status": "PASS", "tier1_pass": 60, "total": 60},
                       "pass_b": {"status": "PASS", "tier1_pass": 16, "total": 16}},
        "performance": {"ratio": perf, "status": "BELOW_THRESHOLD" if perf < 0.6 else "PASS"},
        "determinism": {"policy_satisfied": True,
                         "n_identical_cases": 60, "n_cases_checked": 60},
    }))


# ---------------------------------------------------------------------------
# P0o: bootstrap current_state from PROGRESS.md tail
# ---------------------------------------------------------------------------
def test_p0o_bootstrap_await_user_decision_from_progress(ws):
    """Worker prior session emitted `await_user_decision`; orchestrator
    cold-start (empty log) must derive current state, not default to
    await_worker.
    """
    _seed_progress_with_handoff(
        ws,
        (
            "→ orchestrator: await_user_decision — HARD_GATE_PRESERVED_DIRECTIVE_FALSIFIED. "
            "ko-1 + ko-2 exhausted; ko-2 directive falsified."
        ),
    )
    _seed_verification(ws, status="PASS", perf=0.20)
    sm_data = sm.load_state_machine()
    state = sm.get_current_state(ws, sm_data)
    assert state == "await_user_decision", \
        f"Expected await_user_decision (from PROGRESS.md handoff), got {state}"


def test_p0o_bootstrap_falls_through_to_initial_when_no_progress(tmp_path):
    """No PROGRESS.md AND no log → default to YAML initial state."""
    sm_data = sm.load_state_machine()
    state = sm.get_current_state(tmp_path, sm_data)
    expected_initial = sm_data.get("phase_o4_initial_state", "await_worker")
    assert state == expected_initial


def test_p0o_bootstrap_log_wins_over_progress(ws):
    """If state log has entries, use them. PROGRESS.md is bootstrap-only."""
    import json
    _seed_progress_with_handoff(ws, "→ orchestrator: await_user_decision — anything")
    # Pre-existing log entry says we're in await_optimizer
    log_entry = {"ts": "2026-05-05T00:00:00Z",
                 "from_state": "await_worker", "to_state": "await_optimizer",
                 "handoff": "@aog-kernel-optimizer", "matched_transition_index": 0,
                 "rationale": "test", "iter_counts_snapshot": {}}
    (ws / "state_transitions.jsonl").write_text(json.dumps(log_entry) + "\n")
    sm_data = sm.load_state_machine()
    state = sm.get_current_state(ws, sm_data)
    assert state == "await_optimizer", \
        f"Log tail should win over PROGRESS bootstrap, got {state}"


def test_p0o_extract_handoff_finds_arrow_orchestrator_form(ws):
    """Direct test of _extract_handoff_from_progress — `→ orchestrator: X` form."""
    _seed_progress_with_handoff(ws, "→ orchestrator: await_user_decision — reason here")
    handoff = getattr(sm, '_extract_handoff_from_progress')(ws)
    assert handoff is not None
    assert handoff.startswith("→ orchestrator: await_user_decision")


def test_p0o_extract_handoff_finds_at_agent_form(ws):
    """`@aog-precision-probe …` handoff form."""
    _seed_progress_with_handoff(ws, "@aog-precision-probe: signature=fp32-tanh-saturation")
    handoff = getattr(sm, '_extract_handoff_from_progress')(ws)
    assert handoff is not None
    assert handoff.startswith("@aog-precision-probe")


def test_p0o_extract_handoff_finds_markdown_wrapped(ws):
    """**Exit handoff**: `→ orchestrator: X` markdown-decorated form (worker
    sometimes wraps the handoff line in markdown).
    """
    (ws / "PROGRESS.md").write_text("""\
# PROGRESS

### EXIT
**Exit handoff**: `→ orchestrator: await_user_decision — wrapped`
""")
    handoff = getattr(sm, '_extract_handoff_from_progress')(ws)
    assert handoff is not None
    assert "await_user_decision" in handoff


def test_p0o_extract_handoff_picks_latest_when_multiple(ws):
    """If multiple handoff-shaped lines appear, the LAST one wins (most recent)."""
    (ws / "PROGRESS.md").write_text("""\
### [10:30] worker
@aog-precision-probe: stuck on iter 5

### [11:00] worker (respawn)
Recovered.

### EXIT
→ orchestrator: await_user_decision — final state
""")
    handoff = getattr(sm, '_extract_handoff_from_progress')(ws)
    assert handoff is not None
    assert "await_user_decision" in handoff
    assert "stuck on iter 5" not in handoff


def test_p0o_extract_handoff_returns_none_when_absent(ws):
    """PROGRESS.md exists but no handoff lines → None, falls through to default."""
    (ws / "PROGRESS.md").write_text("# PROGRESS\nMode: backward\nNo handoff yet.\n")
    handoff = getattr(sm, '_extract_handoff_from_progress')(ws)
    assert handoff is None
