# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""DEBT-108 regression: _is_legitimate_pipeline_exhaustion treats await_worker
iter_cap as legitimate exhaustion when verification.json shows pass_a PASS
AND the last rollback signature is an infra-debt gate (NOT algorithm work).

Caught 2026-05-20 on gather_elements_v2 task #51: 10 spawns hit iter_cap.
kw-1 produced pass_a PASS 8/8 byte-identical-preserved; kw-2..10 fixed
infra gates only (binary_provenance, P96, delegation_scan, kb_writeup,
post_worker_audit, ssh timeout). iter_cap=9 penalized infra debt path.
Required manual --bump-cap to drive through.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
from orchestrator import _is_legitimate_pipeline_exhaustion  # noqa: E402


def _make_workspace(tmp_path: Path, *, precision_status: str,
                    last_gate: str | None) -> Path:
    """Build a fake workspace with verification.json + .rollback_history.jsonl."""
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    (ws / "verification.json").write_text(json.dumps({
        "precision": {"status": precision_status},
    }))
    if last_gate is not None:
        rh = ws / ".rollback_history.jsonl"
        rh.write_text(json.dumps({
            "ts": "2026-05-20T10:00:00Z",
            "gate": last_gate,
            "rollback_state": "await_worker",
            "reason": "test fixture",
            "signature": f"{last_gate}::await_worker",
        }) + "\n")
    return ws


@pytest.mark.parametrize("gate", [
    "binary_provenance",
    "infra_baseline_paper_over",
    "kb_writeup",
    "post_worker_audit",
    "phase_o5_runner_failed",
    "phase_o5_mismatch",
    "perf_methodology_asymmetry",
])
def test_await_worker_pass_plus_infra_gate_is_legitimate(tmp_path, gate):
    """When pass_a PASS + last rollback is infra-debt, exhaustion is legitimate."""
    ws = _make_workspace(tmp_path, precision_status="PASS", last_gate=gate)
    assert _is_legitimate_pipeline_exhaustion(ws, "await_worker") is True


@pytest.mark.parametrize("gate", [
    "verification_file_missing",
    "verification_malformed",
    "model_py_shape",
    "pass_a_coverage",
    "pass_count",
    "persist_evidence",
])
def test_await_worker_algorithm_gate_is_not_legitimate(tmp_path, gate):
    """Algorithm gates always require more worker iters; never exhaust legit."""
    ws = _make_workspace(tmp_path, precision_status="PASS", last_gate=gate)
    assert _is_legitimate_pipeline_exhaustion(ws, "await_worker") is False


def test_await_worker_precision_fail_not_legitimate(tmp_path):
    """If precision != PASS, infra-gate isn't enough."""
    ws = _make_workspace(tmp_path, precision_status="FAIL",
                         last_gate="binary_provenance")
    assert _is_legitimate_pipeline_exhaustion(ws, "await_worker") is False


def test_await_worker_no_rollback_history_not_legitimate(tmp_path):
    """Without rollback history we can't classify the gate — be safe."""
    ws = _make_workspace(tmp_path, precision_status="PASS", last_gate=None)
    assert _is_legitimate_pipeline_exhaustion(ws, "await_worker") is False


def test_await_worker_no_verification_not_legitimate(tmp_path):
    """No verification.json → can't check pass_a status → return False."""
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    assert _is_legitimate_pipeline_exhaustion(ws, "await_worker") is False


def test_pass_within_tolerance_also_legitimate(tmp_path):
    """PASS_WITHIN_TOLERANCE accepted same as PASS (v2.1 verdict equivalence)."""
    ws = _make_workspace(tmp_path, precision_status="PASS_WITHIN_TOLERANCE",
                         last_gate="binary_provenance")
    assert _is_legitimate_pipeline_exhaustion(ws, "await_worker") is True


def test_await_researcher_unaffected(tmp_path):
    """DEBT-108 fix scoped to await_worker; await_researcher logic untouched."""
    ws = tmp_path / "workspace" / "op"
    ws.mkdir(parents=True)
    # No researcher output → existing await_researcher branch returns False
    assert _is_legitimate_pipeline_exhaustion(ws, "await_researcher") is False
