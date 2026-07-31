# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0uu (2026-05-06): perf gate must reject `done` when ratio is null
without explicit N/A justification.

Reported by DS agent: workers were skipping performance.py and writing
verification.json with `performance.ratio: null`. The previous gate
accepted that as "no perf data → OK" and routed to finalize. V3.8.4
cannot escalate to ko without a ratio number, so the worker effectively
short-circuited the perf-escalation rule.

Fix: require numeric ratio OR explicit `performance.status == "N/A"`
OR `performance.skipped + performance.skip_reason`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

import schema_norm  # noqa: E402


def _seed_progress_with_introspection(workspace: Path) -> None:
    """P0qq prereq: ## Self-introspection block must be present for the
    gate to even reach the perf check."""
    (workspace / "PROGRESS.md").write_text(
        "# op\n\n## Self-introspection (test)\n\n"
        "### Pressure modes I felt\nP1.\n\n"
        "### Decisions I almost rationalized\nnone\n\n"
        "### Verifications I might have skipped\nnone\n\n"
        "### Confidence calibration\nprecision: HIGH\nperf: HIGH\narchitectural fit: HIGH\n"
    )


def _seed_pass_precision(workspace: Path, perf: dict) -> None:
    _seed_progress_with_introspection(workspace)
    (workspace / "verification.json").write_text(json.dumps({
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "tier1_pass": 31, "total": 31},
            "pass_b": {"status": "PASS", "tier1_pass": 11, "total": 11},
        },
        "performance": perf,
    }))
    # P0aay (2026-05-11): seed knowledge_update.md for pre-handoff gate
    (workspace / "knowledge_update.md").write_text(
        "## Context\nTest stub.\n\n## Findings\n- Stub\n\n"
        "## KB-promotable patterns (proposed)\nNone\n\n"
        "## Cited KB items\nNone\n\n## Anti-patterns avoided\nNone\n"
    )


def test_done_rejected_when_ratio_null_and_no_na_flag(tmp_path):
    """ratio=null + no N/A flag → reject."""
    _seed_pass_precision(tmp_path, {"ratio": None})
    res = getattr(schema_norm, '_check_evidence_for_terminal')(tmp_path, "done", "finalize", entry={})
    assert res["passes"] is False
    assert "P0uu" in res["reason"]
    assert "performance.py" in res["reason"]


def test_done_rejected_when_perf_dict_empty(tmp_path):
    """performance: {} → reject (no ratio, no N/A flag)."""
    _seed_pass_precision(tmp_path, {})
    res = getattr(schema_norm, '_check_evidence_for_terminal')(tmp_path, "done", "finalize", entry={})
    assert res["passes"] is False
    assert "P0uu" in res["reason"]


def test_done_accepts_explicit_na_status(tmp_path):
    """`performance.status: N/A` → accept (Path A / OL-68 case A pattern)."""
    _seed_pass_precision(tmp_path, {"ratio": None, "status": "N/A"})
    res = getattr(schema_norm, '_check_evidence_for_terminal')(tmp_path, "done", "finalize", entry={})
    assert res["passes"] is True
    assert "N/A" in res["reason"]


def test_done_accepts_skipped_with_reason(tmp_path):
    """`performance.skipped` + `skip_reason` → accept (documented skip)."""
    _seed_pass_precision(tmp_path, {
        "ratio": None,
        "skipped": True,
        "skip_reason": "reference unrunnable on Ascend950PR (aclnn missing)",
    })
    res = getattr(schema_norm, '_check_evidence_for_terminal')(tmp_path, "done", "finalize", entry={})
    assert res["passes"] is True
    assert "skipped with reason" in res["reason"]


def test_done_rejects_skipped_without_reason(tmp_path):
    """`skipped: True` alone (no skip_reason) → still reject."""
    _seed_pass_precision(tmp_path, {"ratio": None, "skipped": True})
    res = getattr(schema_norm, '_check_evidence_for_terminal')(tmp_path, "done", "finalize", entry={})
    assert res["passes"] is False


def test_done_accepts_numeric_ratio_above_threshold(tmp_path):
    """ratio ≥ parity default (1.0, owner-directed 2026-07-21; was 0.6) →
    accept.
    """
    _seed_pass_precision(tmp_path, {"ratio": 1.05})
    res = getattr(schema_norm, '_check_evidence_for_terminal')(tmp_path, "done", "finalize", entry={})
    assert res["passes"] is True


def test_done_rejects_numeric_ratio_below_threshold(tmp_path):
    """ratio = 0.4 → reject (sub-parity). Reason surfaces the parity threshold
    (1.0, owner-directed 2026-07-21; was 0.6).
    """
    _seed_pass_precision(tmp_path, {"ratio": 0.4})
    res = getattr(schema_norm, '_check_evidence_for_terminal')(tmp_path, "done", "finalize", entry={})
    assert res["passes"] is False
    assert "1.0" in res["reason"]


def test_done_accepts_overall_speedup_alias(tmp_path):
    """Legacy `overall_speedup` field still recognized."""
    _seed_pass_precision(tmp_path, {"overall_speedup": 1.2})
    res = getattr(schema_norm, '_check_evidence_for_terminal')(tmp_path, "done", "finalize", entry={})
    assert res["passes"] is True


def test_done_rejects_malformed_ratio_string(tmp_path):
    """ratio = 'foo' (unparseable) → treated as missing → reject (since no N/A)."""
    _seed_pass_precision(tmp_path, {"ratio": "foo"})
    res = getattr(schema_norm, '_check_evidence_for_terminal')(tmp_path, "done", "finalize", entry={})
    assert res["passes"] is False
    assert "P0uu" in res["reason"]
