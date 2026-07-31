# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for Mode 5 orchestration: gate + lease + scanners + sealed lifecycle."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))

from cann_learn import mode5_runner as m5  # noqa: E402


def _seed_minimal_workspace(workspace: Path, *, ratio: float = 0.19):
    """Seed workspace to satisfy gate preconditions."""
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "cann_strategy_inference.md").write_text(
        "# vendor strategy\n" + "x" * 200
    )
    (workspace / "ref_runnable.json").write_text(json.dumps({"verdict": "RUNNABLE"}))
    (workspace / "verification.json").write_text(json.dumps({
        "performance": {"ratio": ratio, "status": "BELOW_THRESHOLD"},
    }))
    log_entry = {
        "ts": "2026-05-05T00:00:00Z",
        "from_state": "await_user_decision",
        "to_state": "await_researcher",
        "handoff": "x", "matched_transition_index": 0,
        "rationale": "test", "iter_counts_snapshot": {},
    }
    (workspace / "state_transitions.jsonl").write_text(json.dumps(log_entry) + "\n")


# ---------------------------------------------------------------------------
# Gate preconditions
# ---------------------------------------------------------------------------
def test_gate_passes_with_full_preconditions(tmp_path):
    _seed_minimal_workspace(tmp_path)
    g = m5.gate_check_preconditions(tmp_path, "test_op")
    assert g.passed, f"reasons: {g.reasons}"


def test_gate_fails_without_researcher(tmp_path):
    """Missing cann_strategy_inference.md → gate FAILS."""
    _seed_minimal_workspace(tmp_path)
    (tmp_path / "cann_strategy_inference.md").unlink()
    g = m5.gate_check_preconditions(tmp_path, "test_op")
    assert not g.passed
    assert any("cann_strategy_inference" in r for r in g.reasons)


def test_gate_fails_without_state_log_researcher_iter(tmp_path):
    """state_transitions.jsonl with no await_researcher entry → gate FAILS."""
    _seed_minimal_workspace(tmp_path)
    # Overwrite state log with no researcher iter
    (tmp_path / "state_transitions.jsonl").write_text("")
    g = m5.gate_check_preconditions(tmp_path, "test_op")
    assert not g.passed
    assert any("researcher iter" in r for r in g.reasons)


def test_gate_fails_when_ref_not_runnable(tmp_path):
    _seed_minimal_workspace(tmp_path)
    (tmp_path / "ref_runnable.json").write_text(json.dumps({"verdict": "FAILED"}))
    g = m5.gate_check_preconditions(tmp_path, "test_op")
    assert not g.passed
    assert any("RUNNABLE" in r for r in g.reasons)


def test_gate_fails_when_our_kernel_already_better(tmp_path):
    """User direction: don't learn from CANN if our kernel ratio ≥ 1.0×."""
    _seed_minimal_workspace(tmp_path, ratio=1.5)  # we're 1.5× CANN — better than CANN
    g = m5.gate_check_preconditions(tmp_path, "test_op")
    assert not g.passed
    assert any("worse strategy" in r for r in g.reasons)


def test_gate_passes_when_cann_better(tmp_path):
    """Our kernel 0.19× CANN → CANN is 5× better → carve-out applicable."""
    _seed_minimal_workspace(tmp_path, ratio=0.19)
    g = m5.gate_check_preconditions(tmp_path, "test_op")
    assert g.passed


def test_gate_fails_without_verification_json(tmp_path):
    _seed_minimal_workspace(tmp_path)
    (tmp_path / "verification.json").unlink()
    g = m5.gate_check_preconditions(tmp_path, "test_op")
    assert not g.passed
    assert any("verification.json" in r for r in g.reasons)


def test_gate_skip_compare_bypasses_perf_check(tmp_path):
    """skip_compare=True allows running without verification.json (debug mode)."""
    _seed_minimal_workspace(tmp_path)
    (tmp_path / "verification.json").unlink()
    g = m5.gate_check_preconditions(tmp_path, "test_op", skip_compare=True)
    # All other preconditions met; perf check skipped → passes
    assert g.passed


# ---------------------------------------------------------------------------
# Lease management
# ---------------------------------------------------------------------------
def test_acquire_release_lease(tmp_path):
    lease = m5.acquire_lease(tmp_path, "run_id_1")
    assert lease.exists()
    payload = json.loads(lease.read_text())
    assert payload["run_id"] == "run_id_1"
    m5.release_lease(tmp_path)
    assert not lease.exists()


def test_release_lease_idempotent_when_absent(tmp_path):
    """Release when no lease exists shouldn't crash (defensive)."""
    m5.release_lease(tmp_path)  # no error


# ---------------------------------------------------------------------------
# Sealed dir lifecycle
# ---------------------------------------------------------------------------
def test_setup_sealed_dir_mode_0700(tmp_path):
    sealed = m5.setup_sealed_dir(tmp_path, "abc123")
    assert sealed.exists()
    # Mode check
    mode = oct(sealed.stat().st_mode & 0o777)
    assert mode == "0o700", f"expected 0o700, got {mode}"


# ---------------------------------------------------------------------------
# End-to-end (mocked agent)
# ---------------------------------------------------------------------------
def test_run_mode5_no_spawn_func_returns_skeleton_failure(tmp_path):
    """Without spawn_agent_func, Mode 5 reports skeleton state."""
    _seed_minimal_workspace(tmp_path)
    result = m5.run_mode5(
        op="test_op",
        workspace=tmp_path,
        module_path=tmp_path,  # dummy
        kb_root=tmp_path / "references",
        api_catalog_path=tmp_path / "catalog.md",
        skip_hook_preflight=True,
    )
    assert result.gate_passed
    assert result.failure_reason and "skeleton" in result.failure_reason


def test_run_mode5_gate_failure_short_circuits(tmp_path):
    """When gate fails, never spawns agent."""
    spawned = []

    def fake_spawn(**kw):
        spawned.append(1)
        return {}

    result = m5.run_mode5(
        op="test_op",
        workspace=tmp_path,  # empty workspace, gate will fail
        module_path=tmp_path,
        kb_root=tmp_path / "references",
        api_catalog_path=tmp_path / "catalog.md",
        skip_hook_preflight=True,
        spawn_agent_func=fake_spawn,
    )
    assert not result.gate_passed
    assert spawned == [], "agent must not spawn when gate fails"


def test_run_mode5_with_mock_agent_and_clean_output(tmp_path):
    """Full mocked run: agent produces clean candidates + valid summary →
    Mode 5 archives sealed, drops .kb_promotion_pending markers.
    """
    _seed_minimal_workspace(tmp_path)
    # API catalog
    catalog = tmp_path / "catalog.md"
    catalog.write_text("Public AscendC: DataCopy, Adds, WholeReduceSum.\n")
    kb_root = tmp_path / "references"
    kb_root.mkdir()

    def fake_spawn(*, op, workspace, module_path, sealed_dir, run_id,
                   kb_root, api_catalog_path):
        # Agent writes a candidate using only public APIs
        cand = workspace / f"candidate_{run_id}.md"
        cand.write_text(
            "# Pattern: batched reduction\n"
            "Use `DataCopy` to load N rows then `WholeReduceSum`.\n"
        )
        # Public summary (schema-valid)
        summary = workspace / "cann_learn_summary.json"
        summary_data = {
            "run_id": run_id, "ts": "2026-05-05T22:00:00Z", "op": op,
            "module_path_sha256": "a" * 64,
            "files_read_count": 1, "files_read_total_bytes": 50,
            "files_read_hashes": ["b" * 64],
            "candidate_count_extracted": 1, "candidate_count_kept": 1,
            "candidate_count_dropped_leak": 0,
            "candidate_count_dropped_compile": 0,
            "candidate_count_dropped_copy_shape": 0,
            "candidate_count_overlap_existing": 0,
            "metadata_fix_proposals_count": 0,
            "leak_score": 0.0, "copy_shape_score": 0.0, "compile_pass_rate": 1.0,
            "self_review_verdict": "PASS",
            "self_review_failures": [],
            "checks": {
                "C34a": {"passed": True}, "C34b": {"passed": True},
                "C34c": {"passed": True}, "C35": {"passed": True},
            },
        }
        summary.write_text(json.dumps(summary_data))
        return {
            "sealed_files": [],
            "summary_path": str(summary),
            "candidate_paths": [str(cand)],
            "cann_files_read": [],  # mocked agent didn't actually read
            "metadata_fix_proposals_count": 0,
        }

    result = m5.run_mode5(
        op="test_op",
        workspace=tmp_path,
        module_path=tmp_path,
        kb_root=kb_root,
        api_catalog_path=catalog,
        skip_hook_preflight=True,
        spawn_agent_func=fake_spawn,
    )
    assert result.gate_passed
    assert result.self_review_passed, f"failure: {result.failure_reason}"
    assert result.candidates_appended == 1

    # .kb_promotion_pending marker dropped (P0acl 2026-05-10 rename from .kb_review_required)
    markers = list((kb_root / "patterns" / "unverified").glob(".kb_promotion_pending-*"))
    assert len(markers) == 1


def test_run_mode5_with_mock_agent_leaky_output_rejected(tmp_path):
    """Agent's candidate contains CANN-internal identifier → re-scan catches it."""
    _seed_minimal_workspace(tmp_path)
    catalog = tmp_path / "catalog.md"
    catalog.write_text("Public: DataCopy.\n")
    kb_root = tmp_path / "references"
    kb_root.mkdir()

    # CANN file the "agent" pretends to have read
    cann_file = tmp_path / "cann_internal.h"
    cann_file.write_text("namespace c310_impl { class NormalizeVFImpl {}; }\n")

    def fake_spawn(*, sealed_dir, run_id, **kw):
        cand = kw["workspace"] / f"candidate_{run_id}.md"
        # LEAK: candidate references CANN-internal class verbatim
        cand.write_text("Use NormalizeVFImpl from c310_impl namespace.\n")
        summary = kw["workspace"] / "cann_learn_summary.json"
        summary.write_text(json.dumps({
            "run_id": run_id, "ts": "2026-05-05T22:00:00Z", "op": kw["op"],
            "module_path_sha256": "a" * 64,
            "files_read_count": 1, "files_read_total_bytes": 50,
            "files_read_hashes": ["b" * 64],
            "candidate_count_extracted": 1, "candidate_count_kept": 1,
            "candidate_count_dropped_leak": 0,
            "candidate_count_dropped_compile": 0,
            "candidate_count_dropped_copy_shape": 0,
            "candidate_count_overlap_existing": 0,
            "metadata_fix_proposals_count": 0,
            # AGENT LIES: claims clean
            "leak_score": 0.0, "copy_shape_score": 0.0, "compile_pass_rate": 1.0,
            "self_review_verdict": "PASS",
            "self_review_failures": [],
            "checks": {
                "C34a": {"passed": True}, "C34b": {"passed": True},
                "C34c": {"passed": True}, "C35": {"passed": True},
            },
        }))
        return {
            "sealed_files": [],
            "summary_path": str(summary),
            "candidate_paths": [str(cand)],
            "cann_files_read": [str(cann_file)],
            "metadata_fix_proposals_count": 0,
        }

    result = m5.run_mode5(
        op="test_op",
        workspace=tmp_path,
        module_path=tmp_path,
        kb_root=kb_root,
        api_catalog_path=catalog,
        skip_hook_preflight=True,
        spawn_agent_func=fake_spawn,
    )
    # Agent claimed PASS but independent re-scan catches the leak
    assert result.gate_passed
    assert not result.self_review_passed
    assert "C34a" in (result.failure_reason or "") or "leak" in (result.failure_reason or "").lower()
