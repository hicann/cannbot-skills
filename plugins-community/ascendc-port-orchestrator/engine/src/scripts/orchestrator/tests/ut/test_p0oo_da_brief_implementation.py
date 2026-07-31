# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0oo (2026-05-06): aog-determinism-analyzer brief builder.

Plan: docs/design/CONTRACT_AND_MATURITY_NOTES.md#fsm-phase-gap-fix-plan Step 6.

Last missing brief builder. Now O1.5 (P0nn) classifies DET_POLICY
correctly, the await_det_analyzer state can actually fire and dispatch
will succeed instead of NotImplementedError exit 3.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))


def _seed_workspace(ws: Path):
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "PROGRESS.md").write_text("# stub")
    (ws / ".ascendc_env").write_text(
        "A5_HOST=test\nA5_USER=root\nA5_PASSWORD=t\n"
        "A5_CONTAINER=t\nCANN_PATH=/test\nSOC_VERSION=Ascend950PR_9579\n"
    )


def test_da_brief_imports_cleanly():
    from briefs.da_brief import build_det_analyzer_brief
    assert callable(build_det_analyzer_brief)


def test_da_brief_builds_non_empty(tmp_path, monkeypatch):
    from briefs.da_brief import build_det_analyzer_brief
    ws = tmp_path / "test_op"
    _seed_workspace(ws)
    monkeypatch.chdir(tmp_path)
    brief = build_det_analyzer_brief(
        "test_op", ws, lane=0, spawn_index=1, iter_cap_remaining=2,
    )
    assert len(brief) > 1000
    assert "test_op" in brief


def test_da_brief_includes_analyzer_only_contract(tmp_path, monkeypatch):
    """Critical: brief must explicitly tell analyzer NOT to edit kernel."""
    from briefs.da_brief import build_det_analyzer_brief
    ws = tmp_path / "test_op"
    _seed_workspace(ws)
    monkeypatch.chdir(tmp_path)
    brief = build_det_analyzer_brief("test_op", ws, lane=0, spawn_index=1,
                                       iter_cap_remaining=2)
    assert "ANALYZER ONLY" in brief or "Do NOT edit kernel" in brief
    assert "determinism_report.md" in brief


def test_da_brief_includes_classification_options(tmp_path, monkeypatch):
    """Brief mentions all 4 classification verdicts."""
    from briefs.da_brief import build_det_analyzer_brief
    ws = tmp_path / "test_op"
    _seed_workspace(ws)
    monkeypatch.chdir(tmp_path)
    brief = build_det_analyzer_brief("test_op", ws, lane=0, spawn_index=1,
                                       iter_cap_remaining=2)
    for cls in ("kernel-side-fixable", "kernel-side-tradeoff",
                "vendor-side-OL88", "untested-cluster"):
        assert cls in brief, f"missing classification: {cls}"


def test_da_brief_cites_relevant_ols(tmp_path, monkeypatch):
    """OL-88 (vendor non-det), OL-89 (FMA), OL-83 (mantissa) referenced."""
    from briefs.da_brief import build_det_analyzer_brief
    ws = tmp_path / "test_op"
    _seed_workspace(ws)
    monkeypatch.chdir(tmp_path)
    brief = build_det_analyzer_brief("test_op", ws, lane=0, spawn_index=1,
                                       iter_cap_remaining=2)
    # OL-88 is the most important
    assert "OL-88" in brief


def test_da_brief_registered_in_dispatch():
    """The whole point: agent_dispatch.BRIEF_BUILDERS now has the entry."""
    from agent_dispatch import BRIEF_BUILDERS
    assert "aog-determinism-analyzer" in BRIEF_BUILDERS
    assert callable(BRIEF_BUILDERS["aog-determinism-analyzer"])


def test_da_brief_iter_budget_in_text(tmp_path, monkeypatch):
    from briefs.da_brief import build_det_analyzer_brief
    ws = tmp_path / "test_op"
    _seed_workspace(ws)
    monkeypatch.chdir(tmp_path)
    brief = build_det_analyzer_brief("test_op", ws, lane=0, spawn_index=1,
                                       iter_cap_remaining=2)
    assert "iter_cap_remaining = 2" in brief
