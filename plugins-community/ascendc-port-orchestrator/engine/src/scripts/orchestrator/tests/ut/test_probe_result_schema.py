# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Test probe_result.json takes precedence over probe_report.md (DEBT-076 / #58).

state_machine.read_snapshot was extended 2026-05-04 V3.8.5 to read
workspace/<op>/probe_result.json first, fall back to probe_report.md
markdown regex when JSON absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent / "workflow"))
import state_machine as sm  # noqa: E402


@pytest.fixture
def ws(tmp_path):
    """Empty workspace dir."""
    (tmp_path / "PROGRESS.md").write_text("# fresh\n")
    return tmp_path


def test_probe_result_json_classification_wins(ws):
    """JSON classification takes precedence over markdown."""
    (ws / "probe_result.json").write_text(json.dumps({
        "classification": "bug",
        "confidence": "verified",
        "next_directive": "swap kernel.h:42 from float to double",
        "summary": "real bug in line 42"
    }))
    (ws / "probe_report.md").write_text(
        "# Probe Report\n## Classification\n- Type: requirement\n## Recommendation\n"
        "- Status: NO FIX, ESCALATE\n"
    )
    snap = sm.snapshot(ws)
    assert snap["probe_classification"] == "bug"
    # next_directive non-empty → actionable fix
    assert snap["probe_report_has_actionable_fix"] is True


def test_probe_result_json_untested_cluster(ws):
    (ws / "probe_result.json").write_text(json.dumps({
        "classification": "untested-cluster",
        "confidence": "partial",
        "next_directive": None,
        "untested_clusters": [{
            "cluster_id": "C2",
            "n_cases": 5,
            "signature": "...",
            "reason_untested": "iter budget exhausted",
        }],
        "summary": "5 cases unbisected"
    }))
    snap = sm.snapshot(ws)
    assert snap["probe_classification"] == "untested-cluster"
    # next_directive null → no actionable fix
    assert snap["probe_report_has_actionable_fix"] is False


def test_probe_result_json_deferred(ws):
    (ws / "probe_result.json").write_text(json.dumps({
        "classification": "deferred",
        "confidence": "hypothesis",
        "next_directive": None,
        "summary": "NPU 0 unavailable"
    }))
    snap = sm.snapshot(ws)
    assert snap["probe_classification"] == "deferred"


def test_markdown_fallback_when_no_json(ws):
    """Legacy LLM-orchestrator path: only probe_report.md exists."""
    (ws / "probe_report.md").write_text(
        "# Probe Report\n\n## Classification\n- Type: **requirement**\n\n## Recommendation\n"
        "- Status: NO FIX, ESCALATE because of OL-83 floor with verified evidence trail.\n"
    )
    snap = sm.snapshot(ws)
    assert snap["probe_classification"] == "requirement"
    # Recommendation section non-empty
    assert snap["probe_report_has_actionable_fix"] is True


def test_malformed_json_falls_back_to_markdown(ws):
    """Defensive: malformed JSON should not break routing if markdown is fine."""
    (ws / "probe_result.json").write_text("{not valid json")
    (ws / "probe_report.md").write_text(
        "# Probe Report\n## Classification\n- Type: convention\n## Recommendation\n"
        "- Status: APPLIED IN-PLACE; precision recovered.\n"
    )
    snap = sm.snapshot(ws)
    assert snap["probe_classification"] == "convention"


def test_no_probe_artifacts(ws):
    """Fresh workspace with no probe artifacts."""
    snap = sm.snapshot(ws)
    assert snap["probe_classification"] is None
    assert snap["probe_report_has_actionable_fix"] is False
