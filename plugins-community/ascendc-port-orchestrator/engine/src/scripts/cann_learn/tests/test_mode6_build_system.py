# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Tests for Mode 6 (build-system extraction) extension to cann_learner.

Mode 6 extends migration research recovery to build-system evidence.

What Mode 6 changes:
- Brief generator: build_cann_learner_brief(extraction_mode="build_system")
  emits a brief whose Phase B scope is CMakeLists.txt + register_*.cpp +
  op_proto*.cpp + apt.cpp (NOT 2-5 kernel files like Mode 5).
- spawn_cann_learner_agent: accepts extraction_mode kwarg, threads through.
- Candidates dir routes to target/ascendc/build_system/candidates.md
  (NOT patterns/unverified/candidates.md).
- summary.json schema: extraction_mode optional field.

These tests pin the routing without exercising the live agent spawn.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2]))

from cann_learn import agent_spawn  # noqa: E402
from cann_learn import summary_schema  # noqa: E402


def _common_brief_args(tmp_path: Path) -> dict:
    """Minimal args for build_cann_learner_brief — just enough to render."""
    workspace = tmp_path / "ws"
    workspace.mkdir()
    module_path = tmp_path / "cann_src"
    module_path.mkdir()
    sealed_dir = workspace / ".sealed"
    sealed_dir.mkdir()
    kb_root = tmp_path / "kb"
    kb_root.mkdir()
    api_catalog = tmp_path / "api.md"
    api_catalog.write_text("")
    return dict(
        op="flash_attention_score",
        workspace=workspace,
        module_path=module_path,
        sealed_dir=sealed_dir,
        run_id="test1234",
        kb_root=kb_root,
        api_catalog_path=api_catalog,
    )


# ---------------------------------------------------------------------------
# Mode 5 (default, kernel_structural) — historical behavior preserved
# ---------------------------------------------------------------------------


def test_mode5_default_brief_mentions_kernel_files(tmp_path: Path):
    """Without --extraction-mode flag, Mode 5 historical brief is rendered.

    The kernel_structural section explicitly says 'Read 2-5 files (header +
    impl + tiling)' and routes candidates to patterns/unverified/candidates.md.
    """
    brief = agent_spawn.build_cann_learner_brief(**_common_brief_args(tmp_path))
    assert "Mode 5: kernel_structural" in brief
    assert "Read 2-5 files (header + impl + tiling)" in brief
    assert "patterns/unverified/candidates.md" in brief
    # Mode 6 specific content MUST NOT appear in Mode 5 brief
    assert "build_system" not in brief.lower() or "Mode 6" not in brief
    assert "per-source-file" not in brief.lower() or "Mode 6" not in brief


def test_mode5_explicit_kernel_structural_same_as_default(tmp_path: Path):
    """Explicit extraction_mode='kernel_structural' matches default."""
    args = _common_brief_args(tmp_path)
    default_brief = agent_spawn.build_cann_learner_brief(**args)
    explicit_brief = agent_spawn.build_cann_learner_brief(
        extraction_mode="kernel_structural",
        **args,
    )
    assert default_brief == explicit_brief


# ---------------------------------------------------------------------------
# Mode 6 (build_system) — new behavior
# ---------------------------------------------------------------------------


def test_mode6_brief_scope_is_build_system(tmp_path: Path):
    """Mode 6 brief MUST reference CMake/register/op_proto/apt files,
    NOT 'header + impl + tiling' (Mode 5 scope).
    """
    brief = agent_spawn.build_cann_learner_brief(
        extraction_mode="build_system",
        **_common_brief_args(tmp_path),
    )
    assert "Mode 6" in brief
    assert "build_system" in brief
    # Required Mode 6 file types listed
    assert "CMakeLists.txt" in brief
    assert "register_*.cpp" in brief
    assert "op_proto*" in brief
    assert "_apt.cpp" in brief
    # Mode 5's "Read 2-5 files (header + impl + tiling)" should be ABSENT
    assert "Read 2-5 files (header + impl + tiling)" not in brief


def test_mode6_brief_routes_to_build_system_kb_subdir(tmp_path: Path):
    """Mode 6 candidates land in target/ascendc/build_system/, NOT patterns/unverified/."""
    args = _common_brief_args(tmp_path)
    brief = agent_spawn.build_cann_learner_brief(
        extraction_mode="build_system",
        **args,
    )
    expected_path = args["kb_root"].resolve() / "target" / "ascendc" / "build_system" / "candidates.md"
    assert str(expected_path) in brief


def test_mode6_brief_lists_5_extraction_topics(tmp_path: Path):
    """Mode 6 Phase C focuses on 5+ specific build-system topics (per design doc)."""
    brief = agent_spawn.build_cann_learner_brief(
        extraction_mode="build_system",
        **_common_brief_args(tmp_path),
    )
    # Each represents a distinct extraction topic
    assert "Per-source-file compile flag isolation" in brief
    assert "Multi-target binary registration" in brief
    assert "Launch macro routing" in brief
    assert "CMake build dependency chains" in brief
    assert "Binary attribute metadata" in brief


def test_mode6_brief_uses_bsp_prefix(tmp_path: Path):
    """Mode 6 candidates use CAND-BSP-* prefix, promotable to BSP-N canonical."""
    brief = agent_spawn.build_cann_learner_brief(
        extraction_mode="build_system",
        **_common_brief_args(tmp_path),
    )
    assert "CAND-BSP-" in brief
    assert "BSP-N" in brief


def test_mode6_brief_requires_extraction_mode_in_summary(tmp_path: Path):
    """Mode 6 brief MUST tell the agent to write extraction_mode in summary.json."""
    brief = agent_spawn.build_cann_learner_brief(
        extraction_mode="build_system",
        **_common_brief_args(tmp_path),
    )
    assert "extraction_mode" in brief
    assert '"extraction_mode": "build_system"' in brief


# ---------------------------------------------------------------------------
# Schema validation — extraction_mode is optional
# ---------------------------------------------------------------------------


def test_schema_accepts_extraction_mode_optional(tmp_path: Path):
    """summary_schema validates a Mode-6 summary with extraction_mode field."""
    summary_path = tmp_path / "summary.json"
    summary = {
        "run_id": "test1234",
        "ts": "2026-05-21T00:00:00Z",
        "op": "test_op",
        "module_path_sha256": "abc",
        "files_read_count": 5,
        "files_read_total_bytes": 10000,
        "files_read_hashes": [],
        "candidate_count_extracted": 2,
        "candidate_count_kept": 2,
        "candidate_count_dropped_leak": 0,
        "candidate_count_dropped_compile": 0,
        "candidate_count_dropped_copy_shape": 0,
        "candidate_count_overlap_existing": 0,
        "metadata_fix_proposals_count": 0,
        "leak_score": 0.0,
        "copy_shape_score": 0.0,
        "compile_pass_rate": 1.0,
        "self_review_verdict": "PASS",
        "self_review_failures": [],
        "checks": {},
        "extraction_mode": "build_system",
    }
    import json
    summary_path.write_text(json.dumps(summary))
    result = summary_schema.validate_file(summary_path)
    assert result.valid, f"validation failed: {result.errors}"


def test_schema_still_accepts_mode5_summary_without_extraction_mode(tmp_path: Path):
    """Backward compat: Mode 5 summaries without extraction_mode still validate."""
    summary_path = tmp_path / "summary.json"
    summary = {
        "run_id": "test1234",
        "ts": "2026-05-21T00:00:00Z",
        "op": "test_op",
        "module_path_sha256": "abc",
        "files_read_count": 5,
        "files_read_total_bytes": 10000,
        "files_read_hashes": [],
        "candidate_count_extracted": 2,
        "candidate_count_kept": 2,
        "candidate_count_dropped_leak": 0,
        "candidate_count_dropped_compile": 0,
        "candidate_count_dropped_copy_shape": 0,
        "candidate_count_overlap_existing": 0,
        "metadata_fix_proposals_count": 0,
        "leak_score": 0.0,
        "copy_shape_score": 0.0,
        "compile_pass_rate": 1.0,
        "self_review_verdict": "PASS",
        "self_review_failures": [],
        "checks": {},
        # No extraction_mode field — legacy Mode 5 shape
    }
    import json
    summary_path.write_text(json.dumps(summary))
    result = summary_schema.validate_file(summary_path)
    assert result.valid, f"backward compat broke: {result.errors}"


# ---------------------------------------------------------------------------
# CLI plumbing — mode5_runner.py main() accepts --extraction-mode
# ---------------------------------------------------------------------------


def test_cli_extraction_mode_flag_accepts_valid_choices(tmp_path: Path, monkeypatch, capsys):
    """`python3 -m cann_learn.mode5_runner --extraction-mode kernel_structural ...` parses ok."""
    from cann_learn import mode5_runner
    workspace = tmp_path / "ws"
    workspace.mkdir()
    module_path = tmp_path / "mod"
    module_path.mkdir()
    kb_root = tmp_path / "kb"
    kb_root.mkdir()
    api_catalog = tmp_path / "api.md"
    api_catalog.write_text("")

    test_argv = [
        "mode5_runner.py",
        "--op", "test",
        "--workspace", str(workspace),
        "--module-path", str(module_path),
        "--kb-root", str(kb_root),
        "--api-catalog", str(api_catalog),
        "--dry-run",
        "--extraction-mode", "build_system",
        "--skip-hook-preflight",
        "--skip-compare",
    ]
    monkeypatch.setattr(sys, "argv", test_argv)
    rc = mode5_runner.main()
    # dry-run + skip preflight + skip compare → gate passes but spawn_func None
    # → failure_reason = "spawn_agent_func not provided" → rc=1 (gate passed
    # but self_review_passed False since no spawn happened)
    captured = capsys.readouterr()
    assert "extraction_mode" in captured.out or '"extraction_mode"' in captured.out or rc in (0, 1)


# ---------------------------------------------------------------------------
# Q3 main agent review 2026-05-21T16:48Z — relaxed C34c threshold for Mode 6
# ---------------------------------------------------------------------------


def test_copy_shape_threshold_dispatch_kernel_structural_uses_5pct():
    """Mode 5 (kernel_structural) keeps the historical 0.05 (5%) threshold."""
    from cann_learn import mode5_runner
    assert math.isclose(
        getattr(mode5_runner, "_COPY_SHAPE_THRESHOLD_BY_MODE")["kernel_structural"],
        0.05,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_copy_shape_threshold_dispatch_build_system_uses_30pct():
    """Mode 6 (build_system) uses relaxed 0.30 (30%) threshold per main Q3.

    Rationale: CMake boilerplate (cmake_minimum_required, project(), common
    target_link_libraries) trivially hits >5% n-gram overlap. 0.30 grants
    boilerplate amnesty while still catching actual copy-shape leaks
    (full body verbatim copies would score >>30%).
    """
    from cann_learn import mode5_runner
    assert math.isclose(
        getattr(mode5_runner, "_COPY_SHAPE_THRESHOLD_BY_MODE")["build_system"],
        0.30,
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def test_revalidate_logs_dropped_candidates_for_mode2_visibility(tmp_path: Path):
    """Per main Q3: 'log every DROPPED candidate to summary.json (don't silently
    filter — give Mode 2 reviewer the visibility)'. Implemented as side-file
    cann_learn_dropped_candidates.json (summary.json schema is pinned).

    Scenario: candidate is a near-verbatim copy of source — guaranteed to exceed
    the 0.30 threshold so we can observe the side-file logging behavior.
    """
    from cann_learn import mode5_runner
    import json
    workspace = tmp_path / "ws"
    workspace.mkdir()
    summary_path = workspace / "cann_learn_summary.json"
    summary_path.write_text(json.dumps({
        "run_id": "test", "ts": "2026-05-21T00:00:00Z", "op": "test",
        "module_path_sha256": "x" * 64,
        "files_read_count": 1, "files_read_total_bytes": 100, "files_read_hashes": [],
        "candidate_count_extracted": 1, "candidate_count_kept": 1,
        "candidate_count_dropped_leak": 0, "candidate_count_dropped_compile": 0,
        "candidate_count_dropped_copy_shape": 0, "candidate_count_overlap_existing": 0,
        "metadata_fix_proposals_count": 0,
        "leak_score": 0.0, "copy_shape_score": 0.0, "compile_pass_rate": 1.0,
        "self_review_verdict": "PASS", "self_review_failures": [], "checks": {},
    }))
    # Near-verbatim copy — guaranteed > 30% n-gram overlap
    common_text = " ".join(f"token{i}" for i in range(200))
    src_file = tmp_path / "src.txt"
    src_file.write_text(common_text)
    cand_file = tmp_path / "candidate.md"
    cand_file.write_text(common_text + " extra unique tail words")

    api_catalog = tmp_path / "api.md"
    api_catalog.write_text("")
    valid_mode6, failures_mode6 = mode5_runner.revalidate_post_agent(
        workspace, summary_path,
        cann_files_read=[src_file],
        candidate_paths=[cand_file],
        api_catalog_path=api_catalog,
        extraction_mode="build_system",
    )
    # Verbatim copy MUST trigger copy_shape failure at 0.30 threshold
    assert not valid_mode6, "verbatim copy should fail C34c even at 0.30 threshold"
    side_file = workspace / "cann_learn_dropped_candidates.json"
    assert side_file.exists(), "dropped candidates must be logged for Mode 2 visibility"
    data = json.loads(side_file.read_text())
    assert data["extraction_mode"] == "build_system"
    assert data["copy_shape_threshold"] == 0.30
    assert len(data["dropped"]) >= 1
    assert data["dropped"][0]["candidate"] == "candidate.md"
    assert data["dropped"][0]["reason"] == "C34c_copy_shape_exceeded_threshold"


def test_cli_extraction_mode_rejects_invalid_value(tmp_path: Path, monkeypatch):
    """argparse choices= rejects unknown extraction-mode values."""
    from cann_learn import mode5_runner
    workspace = tmp_path / "ws"
    workspace.mkdir()
    module_path = tmp_path / "mod"
    module_path.mkdir()
    kb_root = tmp_path / "kb"
    kb_root.mkdir()
    api_catalog = tmp_path / "api.md"
    api_catalog.write_text("")

    test_argv = [
        "mode5_runner.py",
        "--op", "test",
        "--workspace", str(workspace),
        "--module-path", str(module_path),
        "--kb-root", str(kb_root),
        "--api-catalog", str(api_catalog),
        "--extraction-mode", "nonsense_mode",
    ]
    monkeypatch.setattr(sys, "argv", test_argv)
    with pytest.raises(BaseException) as exc_info:
        mode5_runner.main()
    # argparse exits with code 2 on invalid choice
    assert type(exc_info.value).__name__ == "SystemExit"
    assert exc_info.value.code == 2
