# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for agent_spawn — production wiring of aog-cann-learner.

These tests do NOT spawn a real claude --print process. They verify:
1. brief construction is well-formed (paths resolve, run_id present, sections in order)
2. _list_unverified_candidates / _extract_cann_files_read snapshot logic
3. spawn_cann_learner_agent return-dict shape conforms to Mode 5 contract
   (using a mocked backend dispatch)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[1]))

from cann_learn.agent_spawn import (  # noqa: E402
    build_cann_learner_brief,
    spawn_cann_learner_agent,
    _list_unverified_candidates,
    _extract_cann_files_read,
)


def _seed_workspace(tmp_path: Path, op: str = "TestOp") -> dict:
    """Set up a plausible workspace + module + kb tree for testing."""
    ws = tmp_path / "workspace" / op
    ws.mkdir(parents=True)
    (ws / "cann_strategy_inference.md").write_text("# strategy\nplaceholder " * 50)

    module = tmp_path / "cann_module"
    module.mkdir()
    (module / "header.h").write_text("// fake CANN header\nclass Foo {};")
    (module / "impl.cpp").write_text("// fake impl\nint bar() { return 0; }")

    kb_root = tmp_path / "kb"
    kb_root.mkdir()
    (kb_root / "patterns" / "unverified").mkdir(parents=True)
    (kb_root / "patterns" / "unverified" / "candidates.md").write_text(
        "# candidates\n\n## P-CAND-1\nexisting candidate\n"
    )

    api_catalog = tmp_path / "api_catalog.md"
    api_catalog.write_text("# api allowlist\nDataCopy\nReduceSum\n")

    sealed = ws / ".cann_learn_sealed_test123"
    sealed.mkdir()

    return {
        "op": op,
        "workspace": ws,
        "module_path": module,
        "sealed_dir": sealed,
        "run_id": "test123",
        "kb_root": kb_root,
        "api_catalog_path": api_catalog,
    }


def test_build_brief_includes_required_sections(tmp_path):
    args = _seed_workspace(tmp_path)
    brief = build_cann_learner_brief(**args)

    # Header markers
    assert "cann_learn spawn" in brief
    assert "RUN_ID: test123" in brief
    # Path resolution
    assert str(args["module_path"].resolve()) in brief
    assert str(args["workspace"].resolve()) in brief
    # Required reading
    assert "ANTI_PRESSURE_PROTOCOLS.md" in brief
    assert "cann_strategy_inference.md" in brief
    # Output paths
    assert "source_notes.md" in brief
    assert "extraction_drafts.md" in brief
    assert "cann_learn_summary.json" in brief
    assert "candidates.md" in brief
    # Phases
    for phase in ("Phase A", "Phase B", "Phase C", "Phase D", "Phase E"):
        assert phase in brief or phase.replace("Phase ", "") + "." in brief, phase
    # Anti-patterns + iter budget
    assert "Anti-patterns" in brief
    assert "ITER BUDGET" in brief
    # G7 slug
    assert "testop-cl-1" in brief


def test_list_unverified_candidates_excludes_markers_and_appendable(tmp_path):
    kb_root = tmp_path / "kb"
    p = kb_root / "patterns" / "unverified"
    p.mkdir(parents=True)
    (p / "candidates.md").write_text("appendable")
    (p / "extra_candidate.md").write_text("# extra")
    (p / ".kb_review_required-abc-extra").write_text("{}")

    result = _list_unverified_candidates(kb_root)

    # Must exclude markers (start with .) and append-only candidates.md
    assert {f.name for f in result} == {"extra_candidate.md"}


def test_extract_cann_files_read_uses_summary_json(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({
        "cann_files_read": ["/abs/path/header.h", "/abs/path/impl.cpp"],
        "self_review_verdict": "PASS",
    }))

    result = _extract_cann_files_read(summary, tmp_path / "module")
    assert result == [Path("/abs/path/header.h"), Path("/abs/path/impl.cpp")]


def test_extract_cann_files_read_falls_back_to_module_enumeration(tmp_path):
    summary = tmp_path / "missing.json"  # doesn't exist
    module = tmp_path / "module"
    module.mkdir()
    (module / "a.h").write_text("a")
    (module / "b.cpp").write_text("b")
    (module / "ignored.txt").write_text("not enumerated")

    result = _extract_cann_files_read(summary, module)
    names = {p.name for p in result}
    assert names == {"a.h", "b.cpp"}


def test_spawn_returns_mode5_contract_dict(tmp_path):
    """Verify spawn return-dict shape: sealed_files, summary_path, candidate_paths, cann_files_read."""
    args = _seed_workspace(tmp_path)

    # Simulate agent: writes summary.json + extends candidates.md + adds new sealed file.
    def _simulate(*_, **__):
        # Agent's outputs:
        (args["workspace"] / "cann_learn_summary.json").write_text(json.dumps({
            "self_review_verdict": "PASS",
            "checks": {
                "C34a": {"passed": True, "score": 0.0},
                "C34b": {"passed": True, "pass_rate": 1.0},
                "C34c": {"passed": True, "score": 0.02, "threshold": 0.05},
                "C35": {"passed": True, "matches_count": 0},
            },
            "cann_files_read": [str(args["module_path"] / "header.h")],
            "candidates_count": 2,
            "metadata_fix_proposals": 0,
        }))
        # Append to candidates.md
        candidates_md = args["kb_root"] / "patterns" / "unverified" / "candidates.md"
        candidates_md.write_text(
            candidates_md.read_text() + "\n## P-CAND-2 (new)\nfresh from cann learn\n"
        )
        # Sealed file
        (args["sealed_dir"] / "source_notes.md").write_text("# sealed notes")

        # Simulate AgentResult
        from types import SimpleNamespace
        return SimpleNamespace(
            output_text="→ orchestrator: cann_learn_done — kept 2 candidates",
            is_error=False,
            cost_usd=0.42,
            agent_type="aog-cann-learner",
            success=True,
            duration_ms=12000,
            session_id="test-session",
            terminal_reason="done",
            raw_envelope={},
        )

    with patch("cann_learn.agent_spawn.agent_transport.spawn_agent_streaming", side_effect=_simulate):
        result = spawn_cann_learner_agent(**args)

    # Mode 5's revalidate_post_agent expects these 4 keys.
    assert set(result.keys()) >= {"sealed_files", "summary_path", "candidate_paths", "cann_files_read"}

    # summary_path resolved to existing file
    assert result["summary_path"] is not None
    assert result["summary_path"].exists()

    # Candidate paths includes candidates.md (because it grew).
    cand_names = [p.name for p in result["candidate_paths"]]
    assert "candidates.md" in cand_names

    # Sealed files snapshot captures source_notes.md (taken before audit-write).
    sealed_names = [p.name for p in result["sealed_files"]]
    assert "source_notes.md" in sealed_names
    # spawn_audit.json is written into sealed dir AFTER the snapshot,
    # so it's present on disk but not in result["sealed_files"]; the
    # caller (Mode 5 archive_sealed_dir) tars the whole dir.
    assert (args["sealed_dir"] / "spawn_audit.json").exists()

    # cann_files_read pulled from summary.json
    assert any(p.name == "header.h" for p in result["cann_files_read"])


def test_spawn_audit_records_run_metadata(tmp_path):
    args = _seed_workspace(tmp_path)

    def _simulate(*_, **__):
        from types import SimpleNamespace
        return SimpleNamespace(
            output_text="→ orchestrator: cann_learn_done — 0 candidates",
            is_error=False,
            cost_usd=0.10,
            agent_type="aog-cann-learner",
            success=True,
            duration_ms=5000,
            session_id="abc",
            terminal_reason="done",
            raw_envelope={},
        )

    with patch("cann_learn.agent_spawn.agent_transport.spawn_agent_streaming", side_effect=_simulate):
        spawn_cann_learner_agent(**args)

    audit = args["sealed_dir"] / "spawn_audit.json"
    assert audit.exists()
    d = json.loads(audit.read_text())
    assert d["run_id"] == "test123"
    assert d["op"] == "TestOp"
    assert d["agent_is_error"] is False
    assert d["agent_cost_usd"] == 0.10
    assert "spawn_duration_sec" in d


def test_spawn_handles_agent_skip_no_candidates(tmp_path):
    """Agent legitimately found nothing portable — empty candidates is OK."""
    args = _seed_workspace(tmp_path)

    def _simulate(*_, **__):
        # Only write summary.json with empty candidates; don't touch candidates.md.
        (args["workspace"] / "cann_learn_summary.json").write_text(json.dumps({
            "self_review_verdict": "PASS",
            "checks": {
                "C34a": {"passed": True, "score": 0.0},
                "C34b": {"passed": True, "pass_rate": 1.0},
                "C34c": {"passed": True, "score": 0.0, "threshold": 0.05},
                "C35": {"passed": True, "matches_count": 0},
            },
            "cann_files_read": [],
            "candidates_count": 0,
            "metadata_fix_proposals": 0,
        }))
        from types import SimpleNamespace
        return SimpleNamespace(
            output_text="→ orchestrator: cann_learn_done — 0 candidates (no portable patterns)",
            is_error=False,
            cost_usd=0.05,
            agent_type="aog-cann-learner",
            success=True,
            duration_ms=3000,
            session_id="xyz",
            terminal_reason="done",
            raw_envelope={},
        )

    with patch("cann_learn.agent_spawn.agent_transport.spawn_agent_streaming", side_effect=_simulate):
        result = spawn_cann_learner_agent(**args)

    assert result["candidate_paths"] == []
    assert result["summary_path"] is not None
