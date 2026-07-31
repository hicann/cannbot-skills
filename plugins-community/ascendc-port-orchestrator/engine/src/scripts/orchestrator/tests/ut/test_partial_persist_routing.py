# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for PARTIAL_PERSIST routing (P0j, batch4 finding).

V3.8.7 fix: kw_brief lists PARTIAL_PERSIST but YAML had no matching
transition → catch-all abort. Now ordered:

  1. PARTIAL_PERSIST + probe_report.md (>100 bytes content) → finalize
  2. PARTIAL_PERSIST + probe_report.md (any size) → finalize
  3. PARTIAL_PERSIST + probe cap available → await_probe (force evidence)
  4. PARTIAL_PERSIST + probe cap exhausted → finalize (warning)
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
    # P0xx (2026-05-06): when prec_status is PASS / PASS_WITHIN_TOLERANCE,
    # use full counts (tier1_pass == total). Otherwise pass_count_consistent
    # YAML primitive correctly flags as inconsistent (PASS-with-contradiction
    # is the bug class P0xx caught).
    if prec_status in ("PASS", "PASS_WITHIN_TOLERANCE"):
        pa_counts = {"tier1_pass": 50, "total": 50}
        pb_counts = {"tier1_pass": 10, "total": 10}
    else:
        pa_counts = {"tier1_pass": 22, "total": 50}
        pb_counts = {"tier1_pass": 8, "total": 10}
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": prec_status,
                       "pass_a": {"status": prec_status, **pa_counts},
                       "pass_b": {"status": prec_status, **pb_counts}},
        "performance": ({"ratio": perf_ratio} if perf_ratio is not None else {}),
        "determinism": {"policy_satisfied": True,
                         "n_identical_cases": 50, "n_cases_checked": 50},
    }))


# ---------------------------------------------------------------------------
# Path 1: with substantive evidence → finalize
# ---------------------------------------------------------------------------
def test_partial_persist_with_actionable_probe_report_finalizes(ws):
    """probe_report.md exists with §Recommendation > 40 chars → actionable
    flag fires → finalize.

    V3.8.9 (2026-05-05) added a preceding researcher gate: PARTIAL_PERSIST
    routes to await_researcher if researcher hasn't run. To test V3.8.7 P0j
    finalize behavior in isolation, we simulate researcher already exhausted
    (cann_strategy_inference.md present) so V3.8.9 doesn't intercept.
    """
    _seed_verification(ws)
    (ws / "probe_report.md").write_text("""\
# Probe Report — test_op

## Hypothesis
hyp 1 - dtype rounding

## Classification
- Type: requirement

## Recommendation
- Status: NO FIX, ESCALATE — OL-83 fail-floor with verified rounding evidence trail
""")
    # V3.8.9 precondition: researcher already ran
    (ws / "cann_strategy_inference.md").write_text("# researcher exhausted")
    handoff = ("→ orchestrator: PARTIAL_PERSIST — Pass A 22/50 OVERALL_T1 "
               "+ Tier-2 evidence (28/50 effective with strict NaN-match)")
    result = sm.next_state(ws, "await_worker", handoff)
    assert "error" not in result, result
    assert result["next_state"] == "finalize"


# ---------------------------------------------------------------------------
# Path 2: probe_report.md exists but small/sparse → still finalize
# ---------------------------------------------------------------------------
def test_partial_persist_with_minimal_probe_report_finalizes(ws):
    """probe_report.md exists but minimal — fall to second rule (path_exists).
    V3.8.9 precondition: researcher already ran.
    """
    _seed_verification(ws)
    (ws / "probe_report.md").write_text("# stub\nshort\n")
    (ws / "cann_strategy_inference.md").write_text("# researcher exhausted")
    handoff = "→ orchestrator: PARTIAL_PERSIST — minimal evidence"
    result = sm.next_state(ws, "await_worker", handoff)
    assert result["next_state"] == "finalize"


# ---------------------------------------------------------------------------
# Path 3: no probe_report → force probe escalation
# ---------------------------------------------------------------------------
def test_partial_persist_no_probe_report_escalates(ws):
    """Worker emits PARTIAL_PERSIST without writing probe_report.md → orchestrator
    forces probe to produce evidence (prevents short-circuit).
    V3.8.9 precondition: researcher already ran (else V3.8.9 intercepts to researcher).
    """
    _seed_verification(ws)
    (ws / "cann_strategy_inference.md").write_text("# researcher already ran")
    handoff = "→ orchestrator: PARTIAL_PERSIST — assertion only, no evidence"
    result = sm.next_state(ws, "await_worker", handoff)
    assert result["next_state"] == "await_probe"


# ---------------------------------------------------------------------------
# Path 4: probe cap exhausted, no probe_report → finalize PARTIAL anyway
# ---------------------------------------------------------------------------
def test_partial_persist_probe_cap_exhausted_finalizes(ws, tmp_path):
    """Pre-fill state log with `iter_cap` await_probe entries so probe at-cap.
    Then PARTIAL_PERSIST with no evidence → orchestrator can't escalate, must
    finalize with warning.
    V3.8.9 precondition: researcher already ran.

    NODE-4 (2026-05-28) raised await_probe iter_cap 4 → 8; the seed count is
    derived from the loaded YAML so this stays correct across future cap
    changes (previously hardcoded 4 → silently under-seeded when cap rose).
    """
    _seed_verification(ws)
    (ws / "cann_strategy_inference.md").write_text("# researcher already ran")
    _probe_cap = int(
        (sm.get_state_spec(sm.load_state_machine(), "await_probe") or {})
        .get("iter_cap", 8)
    )
    log = ws / "state_transitions.jsonl"
    entries = []
    for i in range(_probe_cap):
        entries.append(json.dumps({
            "ts": f"2026-05-04T05:{i // 60:02d}:{i % 60:02d}Z",
            "from_state": "await_worker", "to_state": "await_probe",
            "handoff": "@aog-precision-probe", "matched_transition_index": 0,
            "rationale": "test seed", "iter_counts_snapshot": {},
        }))
    log.write_text("\n".join(entries) + "\n")
    handoff = "→ orchestrator: PARTIAL_PERSIST — no evidence + probe exhausted"
    result = sm.next_state(ws, "await_worker", handoff)
    assert result["next_state"] == "finalize"


# ---------------------------------------------------------------------------
# Sanity: existing routing unchanged
# ---------------------------------------------------------------------------
def test_done_perf_pass_still_finalizes(ws):
    # ratio at/above parity default (1.0, owner-directed 2026-07-21; was 0.6)
    _seed_verification(ws, perf_ratio=1.05, prec_status="PASS")
    handoff = "→ orchestrator: done — perf 1.05x"
    result = sm.next_state(ws, "await_worker", handoff)
    assert result["next_state"] == "finalize"


def test_done_perf_low_still_escalates(ws):
    _seed_verification(ws, perf_ratio=0.2, prec_status="PASS")
    handoff = "→ orchestrator: done — perf 0.2x"
    result = sm.next_state(ws, "await_worker", handoff)
    assert result["next_state"] == "await_optimizer"


def test_partial_persist_string_appears_in_yaml_routing():
    """Sanity: confirm the fix is actually present in YAML — we're not just
    testing without the routing rule existing.
    """
    sm_dict = sm.load_state_machine()
    spec = sm.get_state_spec(sm_dict, "await_worker")
    has_partial_persist = False
    for t in spec.get("exit_transitions", []):
        cond = t.get("condition", {})
        ao = cond.get("all_of", [])
        for c in ao:
            if "handoff_match" in c and "PARTIAL_PERSIST" in str(c["handoff_match"]):
                has_partial_persist = True
    assert has_partial_persist, "PARTIAL_PERSIST routing missing from YAML"
