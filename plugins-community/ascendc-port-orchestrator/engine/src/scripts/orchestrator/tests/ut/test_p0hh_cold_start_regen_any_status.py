# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0hh (2026-05-05): --cold-start clears verification.json + .finalized markers.

User: "We should be able to re gen any op regardless its status. That could
cold start." Found: cold-start preserved verification.json (worker output —
stale claim) and .finalized-<hash> markers (would skip re-finalize as
idempotent on identical hash). After clearing those, cold-start works on
any op including ones at terminal `done`.

Also: --resume + --cold-start are mutually exclusive — refuse with clear
error instead of silently preferring resume.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import orchestrator as orch  # noqa: E402


def _seed_done_workspace(ws: Path):
    """Workspace at terminal `done` state with all the artifacts a finalized
    op carries: state log, PROGRESS, verification, .finalized marker, kernel."""
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "PROGRESS.md").write_text("# done\n→ orchestrator: pipeline_done\n")
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": "PASS", "pass_a": {"status": "PASS",
                      "tier1_pass": 50, "total": 50}},
        "performance": {"status": "PASS", "ratio": 1.5},
    }))
    (ws / "state_transitions.jsonl").write_text(
        json.dumps({"from_state": "init", "to_state": "await_worker",
                    "ts": "t1", "handoff": "", "matched_transition_index": 0,
                    "rationale": "", "iter_counts_snapshot": {}}) + "\n" +
        json.dumps({"from_state": "await_worker", "to_state": "finalize",
                    "ts": "t2", "handoff": "→ orchestrator: done",
                    "matched_transition_index": 0, "rationale": "",
                    "iter_counts_snapshot": {}}) + "\n" +
        json.dumps({"from_state": "finalize", "to_state": "done",
                    "ts": "t3", "handoff": "→ orchestrator: pipeline_done",
                    "matched_transition_index": 0, "rationale": "",
                    "iter_counts_snapshot": {}}) + "\n"
    )
    # .finalized marker
    (ws / ".finalized-abc123").write_text(json.dumps({"op": "test_op"}))
    # Phase O2.5 prep + kernel (must be PRESERVED)
    (ws / "kernel").mkdir()
    (ws / "kernel" / "k.h").write_text("// kernel")
    (ws / "model.py").write_text("# model")
    (ws / "model_new_ascendc.py").write_text("# new")
    (ws / "input_gen.py").write_text("# input")
    (ws / "edge_dataset.pt").write_bytes(b"\x80")
    (ws / "manifest.json").write_text(json.dumps({"op": "x"}))
    (ws / "analysis.md").write_text("# analysis")


def test_cold_start_clears_verification_json(tmp_path, monkeypatch):
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    """P0hh: stale verification.json must NOT survive cold-start."""
    ws = tmp_path / "test_op"
    _seed_done_workspace(ws)
    assert (ws / "verification.json").exists()

    getattr(orch, '_cold_start_reset_workspace')(ws)

    assert not (ws / "verification.json").exists(), (
        "verification.json must be cleared on cold-start (stale claim)"
    )
    # Backed up, not destroyed
    backup_dirs = list((tmp_path / ".bkp" / ws.name).glob("pre-cold-start-*")
                       if (tmp_path / ".bkp" / ws.name).exists() else [])
    assert len(backup_dirs) == 1
    assert (backup_dirs[0] / "verification.json").exists()


def test_cold_start_clears_finalized_markers(tmp_path, monkeypatch):
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    """P0hh: .finalized-<hash> must NOT survive cold-start so re-finalize fires."""
    ws = tmp_path / "test_op"
    _seed_done_workspace(ws)
    (ws / ".finalized-deadbeef").write_text("{}")
    assert len(list(ws.glob(".finalized-*"))) == 2

    getattr(orch, '_cold_start_reset_workspace')(ws)

    surviving = list(ws.glob(".finalized-*"))
    assert len(surviving) == 0, f"finalized markers must be cleared; got {surviving}"
    backup_dirs = list((tmp_path / ".bkp" / ws.name).glob("pre-cold-start-*")
                       if (tmp_path / ".bkp" / ws.name).exists() else [])
    moved = list(backup_dirs[0].glob(".finalized-*"))
    assert len(moved) == 2


def test_cold_start_preserves_phase_o25_prep(tmp_path, monkeypatch):
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    """Phase O2.5 prep + benchmark inputs (model.py, input_gen.py,
    edge_dataset.pt, manifest.json) survive cold-start. kw OUTPUTS
    (kernel/, model_new_ascendc.py, analysis.md) get wiped — they are
    contaminated by the prior iteration's KB / brief / state per the
    P0aav 2026-05-07 contract refinement."""
    ws = tmp_path / "test_op"
    _seed_done_workspace(ws)

    getattr(orch, '_cold_start_reset_workspace')(ws)

    preserved = ["model.py", "input_gen.py", "edge_dataset.pt", "manifest.json"]
    for p in preserved:
        assert (ws / p).exists(), f"{p} must be preserved as Phase O2.5 prep"

    wiped_kw_outputs = ["kernel", "model_new_ascendc.py", "analysis.md"]
    for p in wiped_kw_outputs:
        assert not (ws / p).exists(), (
            f"{p} is kw output (P0aav) — must be wiped on cold-start"
        )


def test_cold_start_clears_state_log_and_progress(tmp_path, monkeypatch):
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    """Sanity: existing state-machine artifacts cleared (P0u baseline)."""
    ws = tmp_path / "test_op"
    _seed_done_workspace(ws)
    getattr(orch, '_cold_start_reset_workspace')(ws)
    assert not (ws / "state_transitions.jsonl").exists()
    assert not (ws / "PROGRESS.md").exists()


def test_cold_start_idempotent_on_clean_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    """Cold-start on fresh workspace shouldn't crash even if no state files."""
    ws = tmp_path / "test_op"
    ws.mkdir()
    (ws / "model.py").write_text("# fresh")
    # No state files. Should not raise.
    getattr(orch, '_cold_start_reset_workspace')(ws)
    assert (ws / "model.py").exists()


def test_cold_start_after_partial_finalize_clears_finalized(tmp_path, monkeypatch):
    monkeypatch.setenv("COLD_START_BACKUP_ROOT", str(tmp_path / ".bkp"))
    """If a partial run left a .finalized marker, cold-start clears it
    (the new run should re-evaluate from scratch)."""
    ws = tmp_path / "test_op"
    _seed_done_workspace(ws)
    # Simulate a previous cold-start cleanup that left a stale marker
    (ws / ".finalized-old1").write_text("{}")
    (ws / ".finalized-old2").write_text("{}")

    getattr(orch, '_cold_start_reset_workspace')(ws)

    assert len(list(ws.glob(".finalized-*"))) == 0
