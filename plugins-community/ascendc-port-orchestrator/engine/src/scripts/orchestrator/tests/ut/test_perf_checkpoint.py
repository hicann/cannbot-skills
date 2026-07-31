# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Task #56: regression tests for IL perf-iter precision checkpoint + advance.

Coverage focus (per main 2026-05-29 review ask):
  - 3-state advance: regression→revert, faster→UPDATE (don't drop this one),
    not-faster→revert
  - whole-dir atomic snapshot + revert (multi-file kernel)
  - reverted-iter tag (`perf_regression_revert`) + consumes_budget flag
  - N-consecutive-non-improve → should_accept_best_known_good
  - #55: checkpoint is the known-good backup (re-emit overwrite can't lose it)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

import perf_checkpoint as pc
from perf_checkpoint import Action


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

def _mk_workspace(tmp_path: Path, kernel_files: dict[str, str]) -> Path:
    ws = tmp_path / "op"
    (ws / "kernel").mkdir(parents=True)
    for name, content in kernel_files.items():
        (ws / "kernel" / name).write_text(content)
    return ws


def _write_verification(ws: Path, *, status: str, tier1: int, total: int,
                        ratio):
    vj = {
        "precision": {"status": status, "pass_a": {"tier1_pass": tier1, "total": total}},
        "performance": {"ratio": ratio},
    }
    (ws / "verification.json").write_text(json.dumps(vj))


# ---------------------------------------------------------------------------
# checkpoint creation
# ---------------------------------------------------------------------------

def test_no_baseline_precision_not_passing_is_noop(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    _write_verification(ws, status="FAIL", tier1=0, total=10, ratio=None)
    r = pc.checkpoint_and_advance(ws)
    assert r.action == Action.NOOP
    assert not pc.has_checkpoint(ws)


def test_first_clean_pass_creates_checkpoint(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "v1", "k.cpp": "impl1"})
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.4)
    r = pc.checkpoint_and_advance(ws)
    assert r.action == Action.CHECKPOINT_CREATED
    assert r.baseline_tier1_pass == 10
    assert pc.has_checkpoint(ws)
    # baseline dir exists with kernel files copied
    baseline = ws / "kernel" / pc.BASELINE_DIRNAME
    assert (baseline / "k.h").read_text() == "v1"
    assert (baseline / "k.cpp").read_text() == "impl1"
    # verification snapshot stashed
    assert (baseline / "verification.json").is_file()


def test_pass_within_tolerance_also_checkpoints(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    _write_verification(ws, status="PASS_WITHIN_TOLERANCE", tier1=8, total=10, ratio=0.5)
    r = pc.checkpoint_and_advance(ws)
    assert r.action == Action.CHECKPOINT_CREATED


# ---------------------------------------------------------------------------
# State 1: precision regression → revert
# ---------------------------------------------------------------------------

def test_regression_reverts_kernel(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "GOOD", "k.cpp": "GOOD_IMPL"})
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.4)
    pc.checkpoint_and_advance(ws)  # checkpoint GOOD

    # Simulate a perf re-emit that broke precision (NaN → FAIL) + overwrote files
    (ws / "kernel" / "k.h").write_text("BROKEN")
    (ws / "kernel" / "k.cpp").write_text("BROKEN_IMPL")
    _write_verification(ws, status="FAIL", tier1=0, total=10, ratio=None)

    r = pc.checkpoint_and_advance(ws)
    assert r.action == Action.REVERTED_REGRESSION
    assert r.reverted
    assert r.consumes_budget
    assert r.rollback_kind == "perf_regression_revert"
    # kernel files restored to GOOD (#55: known-good not lost)
    assert (ws / "kernel" / "k.h").read_text() == "GOOD"
    assert (ws / "kernel" / "k.cpp").read_text() == "GOOD_IMPL"
    # verification.json restored to the passing baseline
    vj = json.loads((ws / "verification.json").read_text())
    assert vj["precision"]["status"] == "PASS"
    assert r.consecutive_no_improve == 1


def test_regression_fewer_tier1_reverts(tmp_path):
    """Still 'PASS' status but fewer tier1 cases → counts as regression."""
    ws = _mk_workspace(tmp_path, {"k.h": "GOOD"})
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.4)
    pc.checkpoint_and_advance(ws)

    (ws / "kernel" / "k.h").write_text("WORSE")
    _write_verification(ws, status="PASS", tier1=7, total=10, ratio=0.9)  # faster but fewer pass
    r = pc.checkpoint_and_advance(ws)
    assert r.action == Action.REVERTED_REGRESSION
    assert (ws / "kernel" / "k.h").read_text() == "GOOD"


# ---------------------------------------------------------------------------
# State 2: faster → UPDATE checkpoint (the one main flagged "don't drop")
# ---------------------------------------------------------------------------

def test_faster_updates_checkpoint(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.4)
    pc.checkpoint_and_advance(ws)  # baseline ratio 0.4

    # Re-emit: still PASS, FASTER (0.4 → 0.7)
    (ws / "kernel" / "k.h").write_text("v2_faster")
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.7)
    r = pc.checkpoint_and_advance(ws)
    assert r.action == Action.ADVANCED_FASTER
    assert not r.reverted
    assert r.baseline_ratio == 0.7
    assert r.consecutive_no_improve == 0  # progress resets ladder
    # checkpoint now holds the faster kernel
    assert (ws / "kernel" / pc.BASELINE_DIRNAME / "k.h").read_text() == "v2_faster"
    # current kernel kept (NOT reverted)
    assert (ws / "kernel" / "k.h").read_text() == "v2_faster"


def test_faster_then_regression_reverts_to_faster_baseline(tmp_path):
    """After advancing to a faster best, a later regression must restore the
    FASTER baseline (not the original) — hill-climbing checkpoint is the best.
    """
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.4)
    pc.checkpoint_and_advance(ws)

    (ws / "kernel" / "k.h").write_text("v2_faster")
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.7)
    pc.checkpoint_and_advance(ws)  # advance to v2_faster

    (ws / "kernel" / "k.h").write_text("v3_broken")
    _write_verification(ws, status="FAIL", tier1=0, total=10, ratio=None)
    r = pc.checkpoint_and_advance(ws)
    assert r.action == Action.REVERTED_REGRESSION
    assert (ws / "kernel" / "k.h").read_text() == "v2_faster"  # restored to BEST, not v1


# ---------------------------------------------------------------------------
# State 3: not faster → revert (keep best-known-good)
# ---------------------------------------------------------------------------

def test_not_faster_reverts(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.5)
    pc.checkpoint_and_advance(ws)

    (ws / "kernel" / "k.h").write_text("v2_slower")
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.45)  # slower
    r = pc.checkpoint_and_advance(ws)
    assert r.action == Action.REVERTED_NO_IMPROVE
    assert r.reverted
    assert r.consumes_budget
    assert r.rollback_kind == "perf_regression_revert"
    assert (ws / "kernel" / "k.h").read_text() == "v1"  # kept best-known-good
    assert r.consecutive_no_improve == 1


def test_equal_ratio_is_not_faster(tmp_path):
    """Same ratio (within noise eps) does not count as improvement → revert."""
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.5)
    pc.checkpoint_and_advance(ws)
    (ws / "kernel" / "k.h").write_text("v2")
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.5)  # identical
    r = pc.checkpoint_and_advance(ws)
    assert r.action == Action.REVERTED_NO_IMPROVE


# ---------------------------------------------------------------------------
# N-consecutive-non-improve ladder
# ---------------------------------------------------------------------------

def test_consecutive_non_improve_ladder(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.5)
    pc.checkpoint_and_advance(ws)
    assert not pc.should_accept_best_known_good(ws, limit=2)

    # 1st non-improve
    (ws / "kernel" / "k.h").write_text("v2")
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.4)
    pc.checkpoint_and_advance(ws)
    assert not pc.should_accept_best_known_good(ws, limit=2)

    # 2nd non-improve → hit limit
    (ws / "kernel" / "k.h").write_text("v3")
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.45)
    pc.checkpoint_and_advance(ws)
    assert pc.should_accept_best_known_good(ws, limit=2)


def test_faster_resets_ladder(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.5)
    pc.checkpoint_and_advance(ws)
    (ws / "kernel" / "k.h").write_text("v2")
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.4)
    pc.checkpoint_and_advance(ws)  # non-improve #1
    # now a faster one
    (ws / "kernel" / "k.h").write_text("v3")
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.9)
    pc.checkpoint_and_advance(ws)  # faster → resets
    assert not pc.should_accept_best_known_good(ws, limit=2)


# ---------------------------------------------------------------------------
# atomic snapshot/restore with subdirs (multi-file, build/ excluded)
# ---------------------------------------------------------------------------

def test_snapshot_excludes_build_dir(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    (ws / "kernel" / "build").mkdir()
    (ws / "kernel" / "build" / "huge.so").write_text("BINARY" * 1000)
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.4)
    pc.checkpoint_and_advance(ws)
    # build/ NOT snapshotted
    assert not (ws / "kernel" / pc.BASELINE_DIRNAME / "build").exists()


def test_restore_preserves_build_dir(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "GOOD"})
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.4)
    pc.checkpoint_and_advance(ws)
    # add a build dir after checkpoint, then trigger a regression revert
    (ws / "kernel" / "build").mkdir()
    (ws / "kernel" / "build" / "out.so").write_text("SO")
    (ws / "kernel" / "k.h").write_text("BROKEN")
    _write_verification(ws, status="FAIL", tier1=0, total=10, ratio=None)
    pc.checkpoint_and_advance(ws)
    # k.h restored, build/ untouched by revert
    assert (ws / "kernel" / "k.h").read_text() == "GOOD"
    assert (ws / "kernel" / "build" / "out.so").read_text() == "SO"


def test_snapshot_handles_subdirs(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    (ws / "kernel" / "arch35").mkdir()
    (ws / "kernel" / "arch35" / "sub.h").write_text("subv1")
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.4)
    pc.checkpoint_and_advance(ws)
    assert (ws / "kernel" / pc.BASELINE_DIRNAME / "arch35" / "sub.h").read_text() == "subv1"
    # break subdir file, regress, revert
    (ws / "kernel" / "arch35" / "sub.h").write_text("broken")
    _write_verification(ws, status="FAIL", tier1=0, total=10, ratio=None)
    pc.checkpoint_and_advance(ws)
    assert (ws / "kernel" / "arch35" / "sub.h").read_text() == "subv1"


# ---------------------------------------------------------------------------
# robustness
# ---------------------------------------------------------------------------

def test_missing_verification_is_noop(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    r = pc.checkpoint_and_advance(ws)
    assert r.action == Action.NOOP


def test_malformed_verification_is_noop(tmp_path):
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    (ws / "verification.json").write_text("{not json")
    r = pc.checkpoint_and_advance(ws)
    assert r.action == Action.NOOP


def test_null_ratio_not_faster_reverts(tmp_path):
    """If re-emit perf ratio is null (unmeasured), treat as NOT faster → revert."""
    ws = _mk_workspace(tmp_path, {"k.h": "v1"})
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=0.5)
    pc.checkpoint_and_advance(ws)
    (ws / "kernel" / "k.h").write_text("v2")
    _write_verification(ws, status="PASS", tier1=10, total=10, ratio=None)
    r = pc.checkpoint_and_advance(ws)
    assert r.action == Action.REVERTED_NO_IMPROVE
    assert (ws / "kernel" / "k.h").read_text() == "v1"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
