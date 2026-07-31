# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P96 regression — STALE_ORCHESTRATOR gate.

Caught 2026-05-15: gather_elements_v2 finalize accepted at 20:35Z even
though P96 _check_infra_paper_over (merged to main 20:21Z) would have
rejected. The running orchestrator process was started 17:18Z, so it
had cached the pre-P96 finalize_pipeline module — new gate code on
disk was NOT in the orchestrator's runtime namespace.

This file pins:
- GateID.STALE_ORCHESTRATOR exists with expected value
- _check_stale_orchestrator returns None when module is fresh
- _check_stale_orchestrator detects post-startup mtime change
- check_finalize_eligibility short-circuits with STALE_ORCHESTRATOR before
  other gates when staleness detected
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import finalize_pipeline as fp


def test_gate_id_stable():
    """Lock GateID.STALE_ORCHESTRATOR.value."""
    assert fp.GateID.STALE_ORCHESTRATOR.value == "stale_orchestrator"


def test_check_stale_orchestrator_fresh():
    """Module just imported — startup mtime ≈ current mtime, no staleness."""
    assert getattr(fp, '_check_stale_orchestrator')() is None


def test_check_stale_orchestrator_detects_post_startup_edit(monkeypatch):
    """Simulate the gather_elements_v2 incident: cached startup mtime older
    than current on-disk mtime by >1s → gate fires.
    """
    # Force startup mtime to be 100s in the past
    real_mtime = getattr(fp, '_HERE').stat().st_mtime
    monkeypatch.setattr(fp, "_ORCH_MODULE_STARTUP_MTIME", real_mtime - 100.0)
    result = getattr(fp, '_check_stale_orchestrator')()
    assert result is not None
    assert "STALE_ORCHESTRATOR" in result
    assert "RESTART" in result.upper()


def test_check_stale_orchestrator_tolerates_subsecond_drift(monkeypatch):
    """FS quirks can cause subsecond mtime drift — don't fire on that."""
    real_mtime = getattr(fp, '_HERE').stat().st_mtime
    monkeypatch.setattr(fp, "_ORCH_MODULE_STARTUP_MTIME", real_mtime - 0.5)
    assert getattr(fp, '_check_stale_orchestrator')() is None


def test_check_stale_orchestrator_handles_unreadable_module(monkeypatch):
    """If startup mtime was unavailable at import, don't block."""
    monkeypatch.setattr(fp, "_ORCH_MODULE_STARTUP_MTIME", None)
    assert getattr(fp, '_check_stale_orchestrator')() is None


def test_check_finalize_eligibility_blocked_on_stale(monkeypatch, tmp_path):
    """Integration: check_finalize_eligibility short-circuits with
    STALE_ORCHESTRATOR gate when staleness detected, BEFORE any other
    workspace check fires.
    """
    real_mtime = getattr(fp, '_HERE').stat().st_mtime
    monkeypatch.setattr(fp, "_ORCH_MODULE_STARTUP_MTIME", real_mtime - 100.0)
    # Use a workspace with no verification.json — without stale check, that
    # would return VERIFICATION_FILE_MISSING. Stale check should fire first.
    ws = tmp_path / "empty_op"
    ws.mkdir()
    result = fp.check_finalize_eligibility(ws)
    assert not result["eligible"]
    assert result["gate"] == fp.GateID.STALE_ORCHESTRATOR.value, (
        f"Expected stale gate, got {result['gate']}"
    )
