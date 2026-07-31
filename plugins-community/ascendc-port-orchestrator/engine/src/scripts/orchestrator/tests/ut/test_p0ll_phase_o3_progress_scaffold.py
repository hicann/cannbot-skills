# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""P0ll (2026-05-06): Phase O3 PROGRESS.md scaffold.

Plan: docs/design/CONTRACT_AND_MATURITY_NOTES.md#fsm-phase-gap-fix-plan Step 3.

Origin: P0bb / P0w handoff-extraction bugs traced back to PROGRESS.md
tail varying across workers. Each worker invented its own structure
(## Phase A, ### kw-1 final, etc.). State_machine.get_current_state
bootstrap parses tail to infer state — inconsistent shape made parsing
fragile.

Fix: orchestrator writes canonical PROGRESS.md skeleton at workspace
start. Workers append to a known section (## Timeline). Stable scaffold
= stable handoff extraction.

Idempotent: don't overwrite if already canonical or if worker has put
non-scaffold content there.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o3  # noqa: E402


def test_writes_skeleton_when_progress_md_missing(tmp_path):
    """Cold-start workspace → write canonical scaffold."""
    rep = phase_o3.init_progress_md(tmp_path, "test_op", opgen_mode="backward")
    assert rep.verdict == "WROTE"
    p = tmp_path / "PROGRESS.md"
    assert p.exists()
    content = p.read_text()
    assert "PROGRESS — test_op" in content
    assert "Mode: backward" in content
    assert phase_o3.SKELETON_MARKER in content
    assert "## Hard gate floors" in content
    assert "## Timeline" in content


def test_idempotent_when_marker_present(tmp_path):
    """Re-running on a scaffold-already-written workspace → ALREADY_CANONICAL."""
    phase_o3.init_progress_md(tmp_path, "test_op")
    rep2 = phase_o3.init_progress_md(tmp_path, "test_op")
    assert rep2.verdict == "ALREADY_CANONICAL"


def test_preserves_existing_non_scaffold_content(tmp_path):
    """Worker has appended real work to a non-scaffolded PROGRESS.md →
    leave it alone. Don't overwrite real work.
    """
    p = tmp_path / "PROGRESS.md"
    p.write_text("# real worker progress\n\n→ orchestrator: done — Pass A 50/50\n")
    rep = phase_o3.init_progress_md(tmp_path, "test_op")
    assert rep.verdict == "PRESERVED"
    # Content unchanged
    assert "→ orchestrator: done — Pass A 50/50" in p.read_text()


def test_writes_scaffold_when_progress_md_empty(tmp_path):
    """Empty file → still write the scaffold (treat empty as missing)."""
    p = tmp_path / "PROGRESS.md"
    p.write_text("")
    rep = phase_o3.init_progress_md(tmp_path, "test_op")
    assert rep.verdict == "WROTE"
    assert phase_o3.SKELETON_MARKER in p.read_text()


def test_hard_floors_block_renders(tmp_path):
    """Hard floors dict gets formatted into the scaffold."""
    rep = phase_o3.init_progress_md(
        tmp_path, "test_op",
        hard_floors={"pass_a": "60/60", "pass_b": "16/16", "det": "60/60", "perf": "0.19×"},
    )
    assert rep.verdict == "WROTE"
    content = (tmp_path / "PROGRESS.md").read_text()
    assert "pass_a: 60/60" in content
    assert "pass_b: 16/16" in content
    assert "perf: 0.19×" in content


def test_perf_threshold_default_documented(tmp_path):
    """Default perf threshold (1.0× parity) appears in the scaffold so workers
    see it. Owner-directed 2026-07-21: default raised 0.6 → 1.0.
    """
    phase_o3.init_progress_md(tmp_path, "test_op")
    content = (tmp_path / "PROGRESS.md").read_text()
    assert "1.0" in content


def test_det_policy_documented(tmp_path):
    """DET_POLICY appears in scaffold for worker awareness."""
    phase_o3.init_progress_md(tmp_path, "test_op", det_policy="required")
    content = (tmp_path / "PROGRESS.md").read_text()
    assert "DET_POLICY: required" in content


def test_marker_uniqueness_per_workspace(tmp_path):
    """SKELETON_MARKER is a stable string used to identify our scaffold.
    Don't accidentally match real worker content.
    """
    p = tmp_path / "PROGRESS.md"
    # Worker happens to mention the marker text accidentally
    p.write_text("# worker output\nrandom text\n")
    rep = phase_o3.init_progress_md(tmp_path, "test_op")
    assert rep.verdict == "PRESERVED"  # No marker present, preserved


def test_skeleton_has_timeline_append_marker(tmp_path):
    """Workers need a clear point to append. Scaffold has explicit marker."""
    phase_o3.init_progress_md(tmp_path, "test_op")
    content = (tmp_path / "PROGRESS.md").read_text()
    assert "WORKERS: append your section below" in content


def test_creates_workspace_dir_if_missing(tmp_path):
    """If workspace dir doesn't exist yet, init creates it. Useful for
    cold-start path where orchestrator hasn't touched workspace yet.
    """
    ws = tmp_path / "newop"
    rep = phase_o3.init_progress_md(ws, "newop")
    assert rep.verdict == "WROTE"
    assert ws.exists()
    assert (ws / "PROGRESS.md").exists()
