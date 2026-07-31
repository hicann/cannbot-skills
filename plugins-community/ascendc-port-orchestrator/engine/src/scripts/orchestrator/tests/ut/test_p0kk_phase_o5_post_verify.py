# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0kk (2026-05-06): Phase O5 independent post-verify.

Plan: docs/design/CONTRACT_AND_MATURITY_NOTES.md#fsm-phase-gap-fix-plan Step 2.

Re-measures kernel against edge_dataset.pt CPU truth, compares to
worker's claimed pass counts. Closes the loophole that P0ee couldn't:
worker writes PASS counts that don't match the actual kernel run.

Architecture is model-agnostic per user direction (2026-05-06):
- Comparison logic is pure Python (no LLM)
- Runner that executes verifier scripts is pluggable callable
- Default runner stubbed until Step 2.1 (SSH-based real runner)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import phase_o5  # noqa: E402


@pytest.fixture(autouse=True)
def _backward_scope(tmp_path):
    (tmp_path / ".opgen_state.json").write_text(json.dumps({
        "schema_version": 1,
        "op": "test_op",
        "opgen_mode": "backward",
    }))


def _seed_verification(ws: Path, *, prec: dict):
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "verification.json").write_text(json.dumps({
        "precision": prec, "performance": {"status": "PASS", "ratio": 1.5},
    }))


def test_skip_returns_immediately(tmp_path):
    """skip=True bypasses everything; runner is never invoked."""
    _seed_verification(tmp_path, prec={"status": "PASS",
                                        "pass_a": {"status": "PASS",
                                                    "tier1_pass": 50, "total": 50}})
    rep = phase_o5.post_verify_for_finalize(tmp_path, "test_op", skip=True)
    assert rep.verdict == "SKIPPED"


def test_missing_verification_runner_failed(tmp_path):
    """No verification.json → RUNNER_FAILED with clear message."""
    rep = phase_o5.post_verify_for_finalize(tmp_path, "test_op")
    assert rep.verdict == "RUNNER_FAILED"
    assert "verification.json missing" in rep.summary


def test_malformed_verification_runner_failed(tmp_path):
    (tmp_path / "verification.json").write_text("{ not json")
    rep = phase_o5.post_verify_for_finalize(tmp_path, "test_op")
    assert rep.verdict == "RUNNER_FAILED"


def test_backward_no_passes_to_verify_fails_closed(tmp_path):
    _seed_verification(tmp_path, prec={
        "status": "PASS",
        "pass_a": {"status": "N/A"},
        "pass_b": {"status": "N/A"},
    })
    rep = phase_o5.post_verify_for_finalize(tmp_path, "test_op")
    assert rep.verdict == "RUNNER_FAILED"


def test_default_runner_returns_runner_failed(tmp_path):
    """Default runner is stubbed until Step 2.1 — caller sees RUNNER_FAILED."""
    _seed_verification(tmp_path, prec={
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
    })
    rep = phase_o5.post_verify_for_finalize(tmp_path, "test_op")
    assert rep.verdict == "RUNNER_FAILED"
    assert "default runner not yet implemented" in rep.summary


def test_verified_when_runner_matches_claim(tmp_path):
    """Pluggable runner returns counts matching claim → VERIFIED."""
    _seed_verification(tmp_path, prec={
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
        "pass_b": {"status": "PASS", "tier1_pass": 11, "total": 11},
    })

    def matching_runner(ws, op, lane=0):
        return phase_o5.MeasuredResult(
            pass_a={"tier1_pass": 50, "total": 50},
            pass_b={"tier1_pass": 11, "total": 11},
        )
    rep = phase_o5.post_verify_for_finalize(tmp_path, "test_op",
                                             runner=matching_runner)
    assert rep.verdict == "VERIFIED"


def test_mismatch_when_runner_disagrees(tmp_path):
    """Worker claimed 50/50 but runner measured 47/50 → MISMATCH."""
    _seed_verification(tmp_path, prec={
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
    })

    def lying_worker_runner(ws, op, lane=0):
        return phase_o5.MeasuredResult(
            pass_a={"tier1_pass": 47, "total": 50},
        )
    rep = phase_o5.post_verify_for_finalize(tmp_path, "test_op",
                                             runner=lying_worker_runner)
    assert rep.verdict == "MISMATCH"
    assert any("tier1_pass" in m and "47" in m and "50" in m for m in rep.mismatches)


def test_mismatch_dramatic_zero_pass(tmp_path):
    """The 30_NMS scenario: claimed 50/50, measured 0/50."""
    _seed_verification(tmp_path, prec={
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 31, "total": 31},
    })

    def zero_runner(ws, op, lane=0):
        return phase_o5.MeasuredResult(pass_a={"tier1_pass": 0, "total": 31})
    rep = phase_o5.post_verify_for_finalize(tmp_path, "test_op",
                                             runner=zero_runner)
    assert rep.verdict == "MISMATCH"


def test_runner_error_propagates_as_runner_failed(tmp_path):
    """Runner reports its own error → RUNNER_FAILED, not MISMATCH."""
    _seed_verification(tmp_path, prec={
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
    })

    def erroring_runner(ws, op, lane=0):
        return phase_o5.MeasuredResult(runner_error="A5 SSH timeout")
    rep = phase_o5.post_verify_for_finalize(tmp_path, "test_op",
                                             runner=erroring_runner)
    assert rep.verdict == "RUNNER_FAILED"
    assert "A5 SSH timeout" in rep.summary


def test_runner_exception_caught(tmp_path):
    """Runner raises an exception → RUNNER_FAILED, not crash."""
    _seed_verification(tmp_path, prec={
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
    })

    def crash_runner(ws, op, lane=0):
        raise ConnectionError("A5 unreachable")
    rep = phase_o5.post_verify_for_finalize(tmp_path, "test_op",
                                             runner=crash_runner)
    assert rep.verdict == "RUNNER_FAILED"
    assert "A5 unreachable" in rep.summary


def test_runner_partial_returns_caught_per_pass(tmp_path):
    """Runner returns pass_a but not pass_b → flag the pass_b mismatch."""
    _seed_verification(tmp_path, prec={
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
        "pass_b": {"status": "PASS", "tier1_pass": 11, "total": 11},
    })

    def partial_runner(ws, op, lane=0):
        return phase_o5.MeasuredResult(
            pass_a={"tier1_pass": 50, "total": 50},
            # pass_b deliberately None
        )
    rep = phase_o5.post_verify_for_finalize(tmp_path, "test_op",
                                             runner=partial_runner)
    assert rep.verdict == "MISMATCH"
    assert any("pass_b" in m and "not measured" in m for m in rep.mismatches)


def test_format_block_message_includes_counts(tmp_path):
    """Block message UX: caller pastes it and sees claimed vs measured."""
    _seed_verification(tmp_path, prec={
        "status": "PASS",
        "pass_a": {"status": "PASS", "tier1_pass": 50, "total": 50},
    })

    def bad_runner(ws, op, lane=0):
        return phase_o5.MeasuredResult(pass_a={"tier1_pass": 0, "total": 31})
    rep = phase_o5.post_verify_for_finalize(tmp_path, "test_op",
                                             runner=bad_runner)
    msg = phase_o5.format_block_message("test_op", rep)
    assert "test_op" in msg
    assert "50" in msg  # claimed
    assert "0" in msg   # measured
    assert (
        "MISMATCH" in msg
        or "claim doesn't match" in msg.lower()
        or "Refuse finalize" in msg
        or "Worker claimed" in msg
    )
