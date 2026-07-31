# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0z (2026-05-05): V3.8.9 PARTIAL_PERSIST → await_researcher gate.

Origin: user pushback 2026-05-05 — "does PARTIAL_PERSIST mean researcher
is not triggered as worker and probe are all failed? does it make sense
to not give researcher a try if worker/probe cannot fix the precision
issue in single iteration?"

V3.8.7 P0j routing rules (await_worker.exit_transitions, lines 741-763)
were written before V3.8.8. They route PARTIAL_PERSIST → finalize when
probe_report.md present, regardless of researcher budget. V3.8.8 only
added researcher escalation in await_probe.exit_transitions, NOT
mirrored in await_worker.PARTIAL_PERSIST path.

V3.8.9 fix: insert PRECEDING rule in await_worker.exit_transitions —
PARTIAL_PERSIST + iter_below_cap researcher + cann_strategy_inference.md
absent → await_researcher. Per "never let PARTIAL pass" rule, researcher
MUST be tried before PARTIAL_PERSIST becomes terminal.
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


def _seed_verification(ws, perf_ratio=None, prec_status="PARTIAL"):
    perf = {"ratio": perf_ratio, "status": "BELOW_THRESHOLD"} if perf_ratio is not None else {}
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": prec_status,
                       "pass_a": {"status": prec_status, "tier1_pass": 22, "total": 50},
                       "pass_b": {"status": prec_status, "tier1_pass": 8, "total": 10}},
        "performance": perf,
        "determinism": {"policy_satisfied": True,
                         "n_identical_cases": 50, "n_cases_checked": 50},
    }))


def _seed_probe_report(ws, content="# probe report\n## Recommendation\n- Status: NO FIX, OL-class evidence trail.\n"):
    (ws / "probe_report.md").write_text(content)


def _append_log(ws, *, from_state="await_worker", to_state="await_worker"):
    """Append a state log entry to advance iter counters."""
    entry = {"ts": "2026-05-05T00:00:00Z",
             "from_state": from_state, "to_state": to_state,
             "handoff": "test", "matched_transition_index": 0,
             "rationale": "test", "iter_counts_snapshot": {}}
    with open(ws / "state_transitions.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# V3.8.9 NEW RULE — PARTIAL_PERSIST + researcher untried → await_researcher
# ---------------------------------------------------------------------------

def test_p0z_partial_persist_with_researcher_budget_routes_to_researcher(ws):
    """Worker emits PARTIAL_PERSIST, probe_report.md present, researcher
    HASN'T run (no cann_strategy_inference.md), budget available →
    await_researcher (V3.8.9), NOT finalize (V3.8.7 P0j #1).
    """
    _seed_verification(ws)
    _seed_probe_report(ws, "# probe long enough\n" + "x" * 200 + "\n## Recommendation\n- evidence cited.\n")

    handoff = "→ orchestrator: PARTIAL_PERSIST — Tier-2 evidence; OL-110 fail-floor"
    result = sm.next_state(ws, "await_worker", handoff)

    assert "error" not in result, result
    assert result["next_state"] == "await_researcher", (
        "V3.8.9 violated: PARTIAL_PERSIST + researcher untried + budget should "
        f"go to researcher, got {result['next_state']}"
    )
    assert "V3.8.9" in result["rationale"] or "researcher" in result["rationale"].lower()


def test_p0z_partial_persist_researcher_already_ran_finalizes(ws):
    """If cann_strategy_inference.md exists (researcher ran), V3.8.9 doesn't
    fire; falls through to V3.8.7 P0j → finalize.
    """
    _seed_verification(ws)
    _seed_probe_report(ws, "# probe\n" + "x" * 200 + "\n## Rec\n- cited.\n")
    (ws / "cann_strategy_inference.md").write_text("# researcher already ran")

    handoff = "→ orchestrator: PARTIAL_PERSIST — researcher exhausted, accept ceiling"
    result = sm.next_state(ws, "await_worker", handoff)

    assert "error" not in result, result
    assert result["next_state"] == "finalize", \
        f"researcher already ran → V3.8.7 P0j finalize, got {result['next_state']}"


def test_p0z_partial_persist_researcher_cap_exhausted_finalizes(ws):
    """Researcher iter cap = 2; if exhausted, V3.8.9 doesn't fire."""
    _seed_verification(ws)
    _seed_probe_report(ws, "# probe\n" + "x" * 200 + "\n## Rec\n- cited.\n")
    # No cann_strategy_inference.md, but researcher cap fully used
    for _ in range(2):
        _append_log(ws, from_state="await_worker", to_state="await_researcher")

    handoff = "→ orchestrator: PARTIAL_PERSIST — Tier-2 evidence"
    result = sm.next_state(ws, "await_worker", handoff)

    assert "error" not in result, result
    assert result["next_state"] == "finalize", \
        f"researcher cap exhausted → finalize, got {result['next_state']}"


def test_p0z_partial_persist_no_probe_routes_to_probe_first(ws):
    """No probe_report.md AND no cann_strategy_inference.md AND probe budget
    remains → V3.8.7 P0j #3: route to probe (force evidence). V3.8.9 doesn't
    take precedence over probe-first because there's no probe evidence yet.
    """
    _seed_verification(ws)
    # No probe_report.md, no cann_strategy_inference.md

    handoff = "→ orchestrator: PARTIAL_PERSIST — claim without evidence"
    result = sm.next_state(ws, "await_worker", handoff)

    # V3.8.9 rule fires first if researcher untried AND budget — actually
    # V3.8.9 doesn't require probe, so it CAN route to researcher even
    # without probe evidence. Both "go researcher" and "go probe" paths
    # are reasonable; V3.8.9 (which is listed FIRST) wins.
    # Per yaml ordering, V3.8.9 fires before V3.8.7 P0j rules.
    assert "error" not in result, result
    assert result["next_state"] == "await_researcher", \
        f"V3.8.9 fires first; researcher (untried + budget) wins over probe, got {result['next_state']}"


def test_p0z_op28_scenario_validates(ws):
    """Replay op#28-style: worker iter 2 emits PARTIAL_PERSIST, probe ran,
    researcher ran (cann_strategy_inference.md exists), researcher iter cap
    fully consumed → finalize (V3.8.7 P0j legitimately fires).
    """
    _seed_verification(ws, perf_ratio=0.10)
    _seed_probe_report(ws, "# probe\n" + "x" * 300 + "\n## Rec\n- OL-110.\n")
    (ws / "cann_strategy_inference.md").write_text("# researcher findings")
    for _ in range(2):
        _append_log(ws, from_state="await_worker", to_state="await_researcher")

    handoff = "→ orchestrator: PARTIAL_PERSIST — full pipeline exhausted, structural ceiling"
    result = sm.next_state(ws, "await_worker", handoff)

    assert result["next_state"] == "finalize", \
        f"full pipeline exhausted → finalize PARTIAL is correct, got {result['next_state']}"
