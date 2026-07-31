# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for V3.8.8 probe_requirement → await_researcher transition (2026-05-05).

Per user direction "never let PARTIAL pass": when probe verdict=requirement
(no candidate-side fix found by probe), the state machine MUST escalate to
researcher BEFORE finalizing PARTIAL — researcher may surface alternate
vendor strategies (private aclnn dlsym, fp64 internal compute, magnitude-aware
techniques) that probe couldn't see.

Transition order in await_probe.exit_transitions (after probe_closed_loop +
actionable_fix branches):

  1. requirement + perf<threshold + optimizer cap available → await_optimizer
  2. **requirement + researcher cap available → await_researcher** [NEW V3.8.8]
  3. requirement → finalize (fallback only when researcher exhausted)
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


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "PROGRESS.md").write_text("# fresh\n")
    (tmp_path / "state_transitions.jsonl").write_text("")
    return tmp_path


def _seed_probe_requirement(ws):
    """Probe says requirement, no actionable fix."""
    (ws / "probe_result.json").write_text(json.dumps({
        "classification": "requirement",
        "confidence": "verified",
        "next_directive": None,
        "summary": "OL-83 fp32 unit-ULP floor — no candidate-side fix"
    }))
    (ws / "probe_report.md").write_text("""\
# Probe Report
## Classification
- Type: requirement
## Recommendation
- Status: NO FIX, ESCALATE — OL-class hw floor with verified evidence trail
""")


def _seed_verification(ws, perf_ratio=None):
    perf = {"ratio": perf_ratio, "status": "PASS"} if perf_ratio is not None else {}
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PARTIAL",
                       "pass_a": {"status": "PARTIAL", "tier1_pass": 22, "total": 50},
                       "pass_b": {"status": "PASS", "tier1_pass": 10, "total": 10}},
        "performance": perf,
        "determinism": {"policy_satisfied": True,
                         "n_identical_cases": 50, "n_cases_checked": 50},
    }))


def _append_log_entry(ws, from_state, to_state):
    """Append a state log entry to advance iter counters."""
    entry = {"ts": "2026-05-05T00:00:00Z",
             "from_state": from_state, "to_state": to_state,
             "handoff": "test", "matched_transition_index": 0,
             "rationale": "test", "iter_counts_snapshot": {}}
    with open(ws / "state_transitions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# V3.8.8 NEW RULE — requirement + researcher cap available → await_researcher
# ---------------------------------------------------------------------------

def test_requirement_with_researcher_budget_routes_to_researcher(ws):
    """Probe verdict=requirement, researcher iter=0 (cap=2 available),
    perf at threshold (no optimizer escalation): MUST route to await_researcher,
    NOT finalize. This is the V3.8.8 'never let PARTIAL pass' fix.
    """
    _seed_probe_requirement(ws)
    _seed_verification(ws, perf_ratio=1.5)  # perf >= 0.6 threshold

    handoff = "→ orchestrator: probe done, classification=requirement"
    result = sm.next_state(ws, "await_probe", handoff)

    assert "error" not in result, result
    assert result["next_state"] == "await_researcher", \
        f"V3.8.8 violated: requirement + researcher budget should go to researcher, got {result['next_state']}"
    assert "researcher" in result["rationale"].lower()


def test_requirement_perf_below_optimizer_cap_exhausted_routes_to_researcher(ws):
    """Optimizer cap exhausted (5 iters used, cap=5). Researcher cap available.
    Even though perf is below threshold, optimizer first-rule fails (cap),
    so researcher rule fires next. NOT finalize.
    """
    _seed_probe_requirement(ws)
    _seed_verification(ws, perf_ratio=0.1)  # perf below threshold

    # Exhaust optimizer cap (cap=5)
    for _ in range(5):
        _append_log_entry(ws, "await_probe", "await_optimizer")

    handoff = "→ orchestrator: probe done, classification=requirement"
    result = sm.next_state(ws, "await_probe", handoff)

    assert "error" not in result, result
    assert result["next_state"] == "await_researcher", \
        f"requirement + researcher budget (optimizer exhausted) should route to researcher, got {result['next_state']}"


# ---------------------------------------------------------------------------
# Existing rule preserved — optimizer-first when perf below + budget
# ---------------------------------------------------------------------------

def test_requirement_perf_below_optimizer_available_routes_to_optimizer(ws):
    """V3.3.3 rule (existing) must still win when matched first:
    requirement + perf<threshold + optimizer cap available → await_optimizer.
    Test that V3.8.8 insertion didn't reorder this incorrectly.
    """
    _seed_probe_requirement(ws)
    _seed_verification(ws, perf_ratio=0.1)  # below 0.6 threshold

    # Researcher already used (irrelevant — optimizer rule should fire first)
    handoff = "→ orchestrator: probe done, classification=requirement"
    result = sm.next_state(ws, "await_probe", handoff)

    assert "error" not in result, result
    assert result["next_state"] == "await_optimizer", (
        "V3.3.3 ordering broken: requirement + perf<thr + optimizer budget should "
        f"win first, got {result['next_state']}"
    )


# ---------------------------------------------------------------------------
# Fallback to finalize only when researcher exhausted
# ---------------------------------------------------------------------------

def test_requirement_researcher_exhausted_finalizes(ws):
    """Probe=requirement, researcher cap exhausted (2/2), perf at threshold
    (no optimizer escalation). Falls through to finalize.
    """
    _seed_probe_requirement(ws)
    _seed_verification(ws, perf_ratio=1.5)  # perf >= threshold

    # Exhaust researcher cap (cap=2)
    for _ in range(2):
        _append_log_entry(ws, "await_probe", "await_researcher")

    handoff = "→ orchestrator: probe done, classification=requirement"
    result = sm.next_state(ws, "await_probe", handoff)

    assert "error" not in result, result
    assert result["next_state"] == "finalize", \
        f"researcher exhausted + perf OK should fall to finalize, got {result['next_state']}"


def test_requirement_all_caps_exhausted_finalizes(ws):
    """Both optimizer and researcher exhausted, perf below threshold.
    Falls through to finalize PARTIAL.
    """
    _seed_probe_requirement(ws)
    _seed_verification(ws, perf_ratio=0.1)  # below threshold

    # Exhaust optimizer (cap=5) and researcher (cap=2)
    for _ in range(5):
        _append_log_entry(ws, "await_probe", "await_optimizer")
    for _ in range(2):
        _append_log_entry(ws, "await_probe", "await_researcher")

    handoff = "→ orchestrator: probe done, classification=requirement"
    result = sm.next_state(ws, "await_probe", handoff)

    assert "error" not in result, result
    assert result["next_state"] == "finalize", \
        f"all caps exhausted should finalize, got {result['next_state']}"
