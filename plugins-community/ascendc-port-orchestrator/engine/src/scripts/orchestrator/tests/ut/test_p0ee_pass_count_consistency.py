# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0ee (2026-05-05): worker self-declared status="PASS" must be backed by
actual pass counts.

Origin: DS agent 30_NMS scenario. Worker wrote verification.json with
precision.status="PASS" and pass_a.tier1_pass=0/31 (everything failed).
The orchestrator's `_check_evidence_for_terminal` for the `done` alias
read the literal status field and accepted PASS without cross-checking
counts. State machine routed to finalize. Op got "done" without ever
having any case actually pass.

Fix: schema_norm._check_pass_count_consistency cross-checks pass_a/pass_b
counts against status. PASS or PASS_WITHIN_TOLERANCE with tier1_pass=0
(or tier1_pass<total and the per-pass status is "PASS") is rejected.

Allowed paths:
- pass_a.status="N/A" (Path A / OL-68 case A — Pass A genuinely didn't run)
- tier1_pass == total > 0 (real pass)
- pass missing entirely (no counts)

Rejected paths (the bug):
- status="PASS" + pass_a.tier1_pass=0 (DS agent's case)
- status="PASS" + pass_a.tier1_pass<total (incomplete pass without
  PASS_WITHIN_TOLERANCE acknowledgement)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import schema_norm  # noqa: E402


# ---------------------------------------------------------------------------
# Direct unit tests for _check_pass_count_consistency
# ---------------------------------------------------------------------------
def test_consistent_when_all_pass():
    prec = {
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
        "pass_b": {"status": "PASS", "tier1_pass": 11, "total": 11},
    }
    assert getattr(schema_norm, '_check_pass_count_consistency')(prec)["consistent"]


def test_consistent_when_pass_a_n_a():
    """OL-68 Path A: Pass A genuinely doesn't run — status N/A."""
    prec = {
        "status": "PASS",
        "pass_a": {"status": "N/A"},
        "pass_b": {"status": "PASS", "tier1_pass": 50, "total": 50},
    }
    assert getattr(schema_norm, '_check_pass_count_consistency')(prec)["consistent"]


def test_consistent_when_pass_missing():
    """No pass_a/pass_b → can't verify, allow."""
    prec = {"status": "PASS"}
    assert getattr(schema_norm, '_check_pass_count_consistency')(prec)["consistent"]


def test_rejects_zero_pass(tmp_path):
    """DS agent's exact scenario: status=PASS but tier1_pass=0/31."""
    prec = {
        "status": "PASS",
        "pass_a": {"status": "FAIL_EXPECTED", "tier1_pass": 0, "total": 31},
    }
    res = getattr(schema_norm, '_check_pass_count_consistency')(prec)
    assert not res["consistent"]
    assert "0/31" in res["reason"] or "pass_a.tier1_pass=0" in res["reason"]


def test_rejects_partial_pass_with_pass_status():
    """tier1_pass<total but pass-level status says PASS — should be
    PASS_WITHIN_TOLERANCE or PARTIAL, not PASS.
    """
    prec = {
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 47, "total": 50},
    }
    res = getattr(schema_norm, '_check_pass_count_consistency')(prec)
    assert not res["consistent"]
    assert "47/50" in res["reason"]


def test_allows_partial_pass_with_pass_within_tolerance():
    """tier1_pass<total but pass-level status is PASS_WITHIN_TOLERANCE
    explicitly — allowed.
    """
    prec = {
        "status": "PASS_WITHIN_TOLERANCE",
        "pass_a": {"status": "PASS_WITHIN_TOLERANCE", "tier1_pass": 47, "total": 50},
    }
    assert getattr(schema_norm, '_check_pass_count_consistency')(prec)["consistent"]


def test_rejects_zero_total():
    """total=0 means no cases ran. Can't claim PASS unless N/A explicitly."""
    prec = {
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 0, "total": 0},
    }
    res = getattr(schema_norm, '_check_pass_count_consistency')(prec)
    assert not res["consistent"]


def test_legacy_field_names_supported():
    """V3.7.x uses n_pass/n_total instead of tier1_pass/total."""
    prec = {
        "status": "PASS",
        "pass_a": {"status": "FAIL_EXPECTED", "n_pass": 0, "n_total": 31},
    }
    res = getattr(schema_norm, '_check_pass_count_consistency')(prec)
    assert not res["consistent"]


def test_pass_b_also_checked():
    """Cross-check fires on pass_b too, not just pass_a."""
    prec = {
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
        "pass_b": {"status": "PASS", "tier1_pass": 0, "total": 11},
    }
    res = getattr(schema_norm, '_check_pass_count_consistency')(prec)
    assert not res["consistent"]
    assert "pass_b" in res["reason"]


# ---------------------------------------------------------------------------
# End-to-end: _check_evidence_for_terminal blocks finalize routing
# ---------------------------------------------------------------------------
def _seed_workspace_with_verification(tmp_path: Path, prec: dict, perf: dict = None):
    if perf is None:
        perf = {"ratio": 1.5, "status": "PASS"}
    (tmp_path / "verification.json").write_text(json.dumps({
        "precision": prec, "performance": perf,
    }))
    # P0qq (2026-05-06): introspection gate runs first; tests targeting
    # downstream precision gates need a satisfying introspection block.
    (tmp_path / "PROGRESS.md").write_text(
        "# op log\n\n## Self-introspection (test)\n\n"
        "### Pressure modes I felt\nP1.\n\n"
        "### Decisions I almost rationalized\nnone\n\n"
        "### Verifications I might have skipped\nnone\n\n"
        "### Confidence calibration\nprecision: HIGH\nperf: HIGH\narchitectural fit: HIGH\n"
    )
    # P0aay (2026-05-11): seed knowledge_update.md so schema_norm pre-handoff
    # gate (## Findings + structure check) doesn't block done-handoff tests.
    (tmp_path / "knowledge_update.md").write_text(
        "## Context\nTest stub.\n\n"
        "## Findings\n- Stub finding\n\n"
        "## KB-promotable patterns (proposed)\nNone\n\n"
        "## Cited KB items\nNone\n\n"
        "## Anti-patterns avoided\nNone\n"
    )
    return tmp_path


def test_done_alias_rejects_zero_pass_count(tmp_path):
    """The DS agent's 30_NMS scenario: worker emitted done with status=PASS
    but pass_a 0/31. Routing to finalize must be REJECTED.
    """
    ws = _seed_workspace_with_verification(tmp_path, {
        "status": "PASS",
        "pass_a": {"status": "FAIL_EXPECTED", "tier1_pass": 0, "total": 31},
    })
    res = getattr(schema_norm, '_check_evidence_for_terminal')(ws, "done", "finalize", entry={})
    assert not res["passes"], (
        f"REGRESSION: 0/31 + PASS status routed to finalize without rejection. "
        f"reason: {res.get('reason')}"
    )
    assert "contradict" in res["reason"] or "0/31" in res["reason"]


def test_done_alias_accepts_full_pass(tmp_path):
    ws = _seed_workspace_with_verification(tmp_path, {
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
        "pass_b": {"status": "PASS", "tier1_pass": 11, "total": 11},
    })
    res = getattr(schema_norm, '_check_evidence_for_terminal')(ws, "done", "finalize", entry={})
    assert res["passes"]


def test_done_alias_accepts_path_a_n_a(tmp_path):
    """OL-68 Path A pattern — Pass A is N/A, only Pass B counts."""
    ws = _seed_workspace_with_verification(tmp_path, {
        "status": "PASS",
        "pass_a": {"status": "N/A"},
        "pass_b": {"status": "PASS", "tier1_pass": 50, "total": 50},
    })
    res = getattr(schema_norm, '_check_evidence_for_terminal')(ws, "done", "finalize", entry={})
    assert res["passes"]
