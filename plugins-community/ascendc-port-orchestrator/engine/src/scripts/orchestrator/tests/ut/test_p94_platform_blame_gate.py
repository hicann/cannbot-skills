# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P94 attack-id PLATFORM-BLAME regression tests.

DS audit 2026-05-15 found 9 production ops shipped with platform-
attribution claims ("V220 limitation", "HBM-blocked", "c10 ABI") in
PROGRESS.md / analysis.md WITHOUT forensic backing (no probe artifact,
no msprof trace, no PB-N / hardware spec citation).

Gate enforces: any platform-blame phrase requires AT LEAST ONE of:
- workspace/probes/*.py empirical probe
- workspace/*msprof*.json hardware-counter trace
- doc-level citation of references/hardware/<chip>.md
- doc-level citation of PLATFORM_BUGS.md / PB-<N>
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import finalize_pipeline as fp


def _seed_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "test_op"
    ws.mkdir()
    return ws


def test_no_blame_phrases_gate_inactive(tmp_path):
    """No platform-blame language → gate doesn't fire (returns None)."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text("# PROGRESS\nWorker ran 8/8 PASS.\n")
    assert getattr(fp, '_check_platform_blame_backed')(ws) is None


@pytest.mark.parametrize("phrase", [
    "V220 limitation",
    "V351 limitation",
    "HBM-blocked",
    "c10 ABI",
    "fp16 not supported",
    "no scalar half",
    "AICPU fallback expected",
    "platform bug",
    "hardware limitation",
    "known limitation",
])
def test_blame_phrase_without_evidence_rejected(tmp_path, phrase):
    """Every phrase in catalog triggers reject when no evidence present."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text(
        f"# PROGRESS\n3/8 cases FAILED — root cause: {phrase}.\n"
    )
    result = getattr(fp, '_check_platform_blame_backed')(ws)
    assert result is not None
    assert "PLATFORM-BLAME" in result
    assert phrase.lower() in result.lower() or "platform-attribution" in result


def test_blame_phrase_with_probe_accepted(tmp_path):
    """Platform-blame WITH workspace/probes/*.py → gate skips (evidence present)."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text("# PROGRESS\nHBM-blocked confirmed.\n")
    (ws / "probes").mkdir()
    (ws / "probes" / "hbm_throughput_probe.py").write_text("# empirical probe\n")
    assert getattr(fp, '_check_platform_blame_backed')(ws) is None


def test_blame_phrase_with_msprof_accepted(tmp_path):
    """Platform-blame WITH msprof JSON → gate skips."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text("# PROGRESS\nV220 limitation observed.\n")
    (ws / "msprof_dump.json").write_text('{"trace": "..."}')
    assert getattr(fp, '_check_platform_blame_backed')(ws) is None


def test_blame_phrase_with_hw_citation_accepted(tmp_path):
    """Platform-blame citing references/hardware/<chip>.md → gate skips."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text(
        "# PROGRESS\nfp16 not supported per references/hardware/ascend910c.md §3.2\n"
    )
    assert getattr(fp, '_check_platform_blame_backed')(ws) is None


def test_blame_phrase_with_pb_citation_accepted(tmp_path):
    """Platform-blame citing PB-N → gate skips."""
    ws = _seed_workspace(tmp_path)
    (ws / "PROGRESS.md").write_text(
        "# PROGRESS\nKnown limitation per PB-24 (bimodal Tanh on V220).\n"
    )
    assert getattr(fp, '_check_platform_blame_backed')(ws) is None


def test_case_insensitive_detection(tmp_path):
    """Phrases should match case-insensitive (e.g., 'V220 LIMITATION')."""
    ws = _seed_workspace(tmp_path)
    (ws / "analysis.md").write_text(
        "# Analysis\nv220 LIMITATION encountered.\n"
    )
    result = getattr(fp, '_check_platform_blame_backed')(ws)
    assert result is not None


def test_blame_phrase_in_self_critic_report_scanned(tmp_path):
    """Self-critic report is in scan list."""
    ws = _seed_workspace(tmp_path)
    (ws / "self_critic_report.md").write_text(
        "C5: kernel emit hardware limitation discussed.\n"
    )
    result = getattr(fp, '_check_platform_blame_backed')(ws)
    assert result is not None


def test_blame_phrase_in_knowledge_update_scanned(tmp_path):
    """knowledge_update.md is in scan list."""
    ws = _seed_workspace(tmp_path)
    (ws / "knowledge_update.md").write_text(
        "Findings: AICPU fallback expected for fp16 transcendentals.\n"
    )
    result = getattr(fp, '_check_platform_blame_backed')(ws)
    assert result is not None


def test_gate_id_stable():
    """Lock GateID.PLATFORM_BLAME_UNBACKED.value."""
    assert fp.GateID.PLATFORM_BLAME_UNBACKED.value == "platform_blame_unbacked"
