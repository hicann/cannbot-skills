# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Rollback-history loop break-out + signature semantics.

Origin: 3_FusionAttention 2026-05-07 hit an infinite-loop pattern where the
finalize coverage gate correctly rejected `pass_a.total=1 vs benchmark 50`
and rolled back to await_worker, but the next worker spawn read the same
generic brief and emitted the same shape, looping until the global spawn
cap fired with exit 6.

Fix: every rollback writes a `(gate, rollback_state)` signature to
`workspace/.rollback_history.jsonl`. If the last 2 entries share a
signature, the orchestrator routes to `await_user_decision` instead of
`await_worker`, capping pathological loops at 2 spawns.

The signature is computed from the `gate` field (a `GateID` enum value),
NOT from reason-text matching. Reason text varies across retries (counts,
paths, error messages) but the gate identity is what kw needs to fix.

Tag for archeology: P0abe (2026-05-07).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import finalize_pipeline as fp  # noqa: E402
from finalize_pipeline import GateID  # noqa: E402


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "test_op"
    ws.mkdir()
    return ws


def test_record_rollback_writes_jsonl_entry(workspace: Path) -> None:
    entry = fp.record_rollback(
        workspace,
        rollback_state="await_worker",
        reason="pass_a.total=1 but benchmark has 50 cases",
        gate=GateID.PASS_A_COVERAGE.value,
    )
    assert entry["gate"] == GateID.PASS_A_COVERAGE.value
    assert entry["rollback_state"] == "await_worker"
    # Signature is (gate, rollback_state), no reason-text dependency
    assert entry["signature"] == f"{GateID.PASS_A_COVERAGE.value}::await_worker"
    history = workspace / ".rollback_history.jsonl"
    assert history.exists()
    lines = history.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["gate"] == GateID.PASS_A_COVERAGE.value


def test_signature_invariant_under_reason_text_changes() -> None:
    """Signature must NOT change when reason text differs (different counts,
    different paths, different error message). Two rollbacks for the same
    gate at the same rollback_state must produce identical signatures.
    """
    sig1 = getattr(fp, '_rollback_signature')(
        GateID.PASS_A_COVERAGE.value, "await_worker"
    )
    sig2 = getattr(fp, '_rollback_signature')(
        GateID.PASS_A_COVERAGE.value, "await_worker"
    )
    assert sig1 == sig2


def test_signature_differs_for_different_gates() -> None:
    a = getattr(fp, '_rollback_signature')(GateID.PASS_A_COVERAGE.value, "await_worker")
    b = getattr(fp, '_rollback_signature')(GateID.KB_WRITEUP.value, "await_worker")
    assert a != b


def test_signature_differs_for_different_rollback_states() -> None:
    a = getattr(fp, '_rollback_signature')(GateID.PERSIST_EVIDENCE.value, "await_probe")
    b = getattr(fp, '_rollback_signature')(GateID.PERSIST_EVIDENCE.value, "await_optimizer")
    assert a != b


def test_loop_break_not_triggered_by_single_entry(workspace: Path) -> None:
    fp.record_rollback(workspace, rollback_state="await_worker",
                       reason="first rejection",
                       gate=GateID.KB_WRITEUP.value)
    assert fp.detect_loop_break(workspace) is None


def test_loop_break_not_triggered_by_different_gates(workspace: Path) -> None:
    fp.record_rollback(workspace, rollback_state="await_worker",
                       reason="coverage 1/50",
                       gate=GateID.PASS_A_COVERAGE.value)
    fp.record_rollback(workspace, rollback_state="await_worker",
                       reason="knowledge_update.md missing",
                       gate=GateID.KB_WRITEUP.value)
    assert fp.detect_loop_break(workspace) is None


def test_loop_break_triggered_when_last_two_share_signature(
    workspace: Path,
) -> None:
    fp.record_rollback(
        workspace, rollback_state="await_worker",
        reason="pass_a.total=1 but benchmark has 50 cases",
        gate=GateID.PASS_A_COVERAGE.value,
    )
    # Reason text differs but gate is the same — must still trigger loop-break
    fp.record_rollback(
        workspace, rollback_state="await_worker",
        reason="pass_a.total=2 but benchmark has 50 cases",
        gate=GateID.PASS_A_COVERAGE.value,
    )
    brk = fp.detect_loop_break(workspace)
    assert brk is not None
    assert brk["loop_detected_at_count"] == 3  # n+1 counts both repeats


def test_loop_break_recovery_when_gate_changes(workspace: Path) -> None:
    """If the worker FINALLY emits a different shape (different gate fires),
    loop-break clears.
    """
    fp.record_rollback(workspace, rollback_state="await_worker",
                       reason="coverage 1/50",
                       gate=GateID.PASS_A_COVERAGE.value)
    fp.record_rollback(workspace, rollback_state="await_worker",
                       reason="coverage 1/50",
                       gate=GateID.PASS_A_COVERAGE.value)
    assert fp.detect_loop_break(workspace) is not None
    fp.record_rollback(workspace, rollback_state="await_worker",
                       reason="knowledge_update.md missing",
                       gate=GateID.KB_WRITEUP.value)
    # Tail signature differs from prev → loop-break NOT detected
    assert fp.detect_loop_break(workspace) is None


def test_rollback_context_block_in_brief_when_history_exists(
    workspace: Path,
) -> None:
    """The kw_brief should include a CRITICAL section pointing to the prior
    rollback signature. Without this, kw-N+1 has no in-context knowledge of
    what kw-N was rejected for.
    """
    sys.path.insert(0, str(_HERE.parent.parent / "briefs"))
    from briefs._common import rollback_context_block

    assert rollback_context_block(workspace) == ""

    fp.record_rollback(
        workspace, rollback_state="await_worker",
        reason="pass_a.total=1 but benchmark has 50 cases",
        gate=GateID.PASS_A_COVERAGE.value,
    )
    block = rollback_context_block(workspace)
    assert "CRITICAL: Previous spawn rejected" in block
    assert GateID.PASS_A_COVERAGE.value in block
    assert "await_worker" in block


def test_rollback_context_block_warns_at_repeat_count(workspace: Path) -> None:
    sys.path.insert(0, str(_HERE.parent.parent / "briefs"))
    from briefs._common import rollback_context_block

    fp.record_rollback(workspace, rollback_state="await_worker",
                       reason="coverage 1/50",
                       gate=GateID.PASS_A_COVERAGE.value)
    fp.record_rollback(workspace, rollback_state="await_worker",
                       reason="coverage 1/50",
                       gate=GateID.PASS_A_COVERAGE.value)
    block = rollback_context_block(workspace)
    assert "ATTEMPT" in block.upper()
    assert "loop" in block.lower() or "await_user_decision" in block


# ---------------------------------------------------------------------------
# DEBT-192 (2026-07-03): finalize loop-convergence guard — perf-methodology
# alternation resilience. detect_loop_break only fires on byte-identical
# consecutive signatures; the observed live loop (gelu gate #2) oscillated
# between P141 PERF_METHODOLOGY_ASYMMETRY and the independent_re_measure
# POST_WORKER_AUDIT reject, so no two consecutive signatures matched and the op
# looped to the spawn cap. detect_nonconvergent_loop must catch this.
# ---------------------------------------------------------------------------
def _perf_asymmetry(workspace: Path) -> None:
    fp.record_rollback(
        workspace, rollback_state="await_worker",
        reason=("P141 PERF_METHODOLOGY_ASYMMETRY (port_a3): perf_counter wrap "
                "around a3=aclnn-pipeline + a5=ACLRT_LAUNCH_KERNEL"),
        gate=GateID.PERF_METHODOLOGY_ASYMMETRY.value,
    )


def _irm_missing(workspace: Path) -> None:
    fp.record_rollback(
        workspace, rollback_state="await_worker",
        reason=("performance.status=PASS but performance.independent_re_measure "
                "missing or empty — NEVER trust skill-reported performance"),
        gate=GateID.POST_WORKER_AUDIT.value,
    )


def test_debt192_same_signature_still_detected(workspace: Path) -> None:
    """detect_nonconvergent_loop must subsume P0abe: two identical signatures
    still trip it (pattern=same_signature).
    """
    _perf_asymmetry(workspace)
    _perf_asymmetry(workspace)
    loop = fp.detect_nonconvergent_loop(workspace)
    assert loop is not None
    assert loop["pattern"] == "same_signature"


def test_debt192_alternating_perf_gates_detected(workspace: Path) -> None:
    """The bug: worker oscillates between PERF_METHODOLOGY_ASYMMETRY and the
    independent_re_measure POST_WORKER_AUDIT reject. No two CONSECUTIVE
    signatures match, so plain detect_loop_break stays blind — but the
    perf-family guard must fire after K=3.
    """
    _perf_asymmetry(workspace)   # 1
    _irm_missing(workspace)      # 2 (differs from prev → detect_loop_break None)
    _perf_asymmetry(workspace)   # 3
    # Plain P0abe detector is blind to the alternation:
    assert fp.detect_loop_break(workspace) is None
    # DEBT-192 guard catches it:
    loop = fp.detect_nonconvergent_loop(workspace)
    assert loop is not None
    assert loop["pattern"] == "perf_family"
    assert loop["count"] == 3


def test_debt192_not_triggered_below_k(workspace: Path) -> None:
    """Two alternating perf rollbacks (< K) must NOT trip the guard — the
    worker still gets its genuine attempts.
    """
    _perf_asymmetry(workspace)
    _irm_missing(workspace)
    assert fp.detect_nonconvergent_loop(workspace) is None


def test_debt192_mixed_nonperf_gate_breaks_family(workspace: Path) -> None:
    """A non-perf gate in the window means real progress/movement occurred —
    the perf-family loop must NOT be declared.
    """
    _perf_asymmetry(workspace)
    fp.record_rollback(workspace, rollback_state="await_worker",
                       reason="knowledge_update.md missing",
                       gate=GateID.KB_WRITEUP.value)
    _perf_asymmetry(workspace)
    # Last-2 differ (KB then PERF) → no same_signature; window has a non-perf
    # entry → no perf_family.
    assert fp.detect_nonconvergent_loop(workspace) is None


def test_debt192_classify_coerce_na_for_precision_pass_port_a3(
    workspace: Path,
) -> None:
    """A precision-PASS port_a3 op stuck on a perf loop → recommend
    coerce_perf_na (retract the unmeasured perf claim to N/A), NOT fail_fast.
    """
    (workspace / "verification.json").write_text(json.dumps({
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {"status": "PASS"},
    }))
    _perf_asymmetry(workspace)
    _irm_missing(workspace)
    _perf_asymmetry(workspace)
    loop = fp.detect_nonconvergent_loop(workspace)
    assert loop is not None
    action = fp.classify_loop_break_action(workspace, loop)
    assert action["action"] == "coerce_perf_na"
    assert action["perf_family"] is True
    assert action["precision_pass"] is True
    assert action["port_a3"] is True


def test_debt192_classify_fail_fast_for_backward_mode(workspace: Path) -> None:
    """Same perf loop but a backward op (not port_a3) → fail_fast. The
    coerce-N/A fail-safe is scoped to precision-focused port_a3 only.
    """
    (workspace / "verification.json").write_text(json.dumps({
        "mode": "backward",
        "precision": {"status": "PASS"},
        "performance": {"status": "PASS"},
    }))
    _perf_asymmetry(workspace)
    _irm_missing(workspace)
    _perf_asymmetry(workspace)
    loop = fp.detect_nonconvergent_loop(workspace)
    assert loop is not None
    action = fp.classify_loop_break_action(workspace, loop)
    assert action["action"] == "fail_fast"
    assert action["port_a3"] is False


def test_debt192_classify_fail_fast_when_precision_not_pass(
    workspace: Path,
) -> None:
    """port_a3 but precision not PASS → fail_fast (don't coerce a non-clean
    deliverable through on a perf technicality).
    """
    (workspace / "verification.json").write_text(json.dumps({
        "mode": "port_a3_to_a5",
        "precision": {"status": "PARTIAL"},
        "performance": {"status": "PASS"},
    }))
    _perf_asymmetry(workspace)
    _irm_missing(workspace)
    _perf_asymmetry(workspace)
    loop = fp.detect_nonconvergent_loop(workspace)
    assert loop is not None
    action = fp.classify_loop_break_action(workspace, loop)
    assert action["action"] == "fail_fast"
