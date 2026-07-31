# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for zero_interaction_audit — catches the 2026-05-10 over-permission incident pattern."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.scripts.workflow import zero_interaction_audit as zia


def test_clean_text_no_findings():
    text = "Spawning aog-kernel-worker (G7 slug index 1)... Phase O4 await_worker iter=1."
    result = zia.audit_text_corpus(text)
    assert len(result.findings) == 0
    assert result.is_zero_interaction()


def test_chinese_permission_ask_detected():
    """The phrase that fired in 2026-05-10 FA cold-start incident."""
    text = "Cold-start FA needs your authorization. 需你 explicit auth THIS turn per memory rule."
    result = zia.audit_text_corpus(text)
    assert len(result.findings) >= 1
    assert not result.is_zero_interaction()


def test_chinese_option_menu_detected():
    """Option menu = LLM offloading decision to user (sign of P1 user-watching drift)."""
    text = """Three paths:
A. cold-start (recommended) - run fresh
B. Kind-1 narrow fix - patch only
C. accept PARTIAL - finalize

请明示 A / B / C，或其他指引。"""
    result = zia.audit_text_corpus(text)
    # Should catch both option-menu pattern AND 请明示
    assert len(result.findings) >= 2


def test_english_permission_ask_detected():
    text = "I need your explicit authorization before proceeding."
    result = zia.audit_text_corpus(text)
    assert len(result.findings) >= 1


def test_whitelist_per_your_directive():
    """'per your auth' = citing prior, not asking new — should NOT be flagged."""
    text = "Per your explicit auth from earlier message, executing cold-start now."
    result = zia.audit_text_corpus(text)
    # whitelist suppresses this line
    findings_on_this_line = [f for f in result.findings if "per your" in f.surrounding_context.lower()]
    assert len(findings_on_this_line) == 0


def test_strict_threshold_zero():
    """Default threshold is 0 — even one permission-ask fails."""
    text = "Should I proceed with cold-start?"
    result = zia.audit_text_corpus(text)
    assert not result.is_zero_interaction(max_asks=0)


def test_threshold_allows_legitimate_asks():
    """Test threshold > 0 passes when count under threshold."""
    text = "需你 explicit auth for the destructive force-push."
    result = zia.audit_text_corpus(text)
    # Some legitimate auth-asks are unavoidable (force-push, kill live process).
    # max_asks=2 lets through 1-2 legitimate asks.
    assert result.is_zero_interaction(max_asks=2)


def test_workspace_audit_returns_structure(tmp_path):
    """Smoke test workspace audit on empty workspace."""
    result = zia.audit_workspace(tmp_path)
    assert result.findings == []
    assert result.is_zero_interaction()


def test_format_report_clean_run():
    text = "All clean output, no asks."
    result = zia.audit_text_corpus(text)
    report = zia.format_report(result)
    assert "0-interaction maintained" in report or "No permission-ask" in report


def test_format_report_failed_run():
    text = "需你 explicit auth before proceeding."
    result = zia.audit_text_corpus(text)
    report = zia.format_report(result)
    assert "FINDINGS" in report
    assert "permission-ask" in report.lower() or "permission_ask" in report.lower()


def test_2026_05_10_fa_incident_caught():
    """Regression test: the actual 2026-05-10 FA incident text MUST fail audit."""
    incident_excerpt = """
**A. `--cold-start`**（**需你 explicit auth THIS turn** per memory rule）:
- 抛弃 iter_014 byte-identical Pass B 9/9 binary
- 全新 worker spawn 从零开始

请明示 A / B / C，或其他指引。期间不动 FA。
"""
    result = zia.audit_text_corpus(incident_excerpt)
    # MUST catch at least 2 findings:
    # - "需你 explicit auth"
    # - "请明示" + option menu
    assert len(result.findings) >= 2
    assert not result.is_zero_interaction()
