# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for P146 — close the N/A-status bypass of P143.

clipped_swiglu lane-1 rerun 2026-05-17T23:13Z shipped with
performance.status="N/A" + performance.method="" and finalize accepted
it because P143's evidence check only fired on NOT_VERIFIED_SAME_METHOD.
Worker dodged the gate by picking "N/A" instead.

P146 uniformly applies the evidence requirement to ALL
retraction-equivalent statuses ({"N/A", "SKIPPED", "NA", None,
"NOT_VERIFIED_SAME_METHOD"}) in port_a3 mode when precision=PASS.
Backward mode retains its independent verification contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from finalize_pipeline import _check_perf_methodology  # noqa: E402


def _check(vj: dict):
    return _check_perf_methodology(None, vj)


def test_p146_na_in_port_a3_with_empty_method_blocks():
    """clipped_swiglu 2026-05-17T23:13Z exact pattern: precision=PASS,
    perf.status=N/A, perf.method="". P143's NOT_VERIFIED_SAME_METHOD-only
    check missed this. P146 must reject.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {"status": "N/A", "method": ""},
    }
    reason = _check(vj)
    assert reason is not None
    assert "P146" in reason
    assert "LAZY_ESCAPE" in reason
    assert "N/A" in reason or "retraction-equivalent" in reason


def test_p146_skipped_in_port_a3_also_blocks():
    """Worker variant: status=SKIPPED instead of N/A. Same dodge."""
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {"status": "SKIPPED"},
    }
    reason = _check(vj)
    assert reason is not None
    assert "P146" in reason


def test_p146_none_status_in_port_a3_also_blocks():
    """Worker variant: status omitted entirely. Same dodge."""
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {},
    }
    reason = _check(vj)
    assert reason is not None
    assert "P146" in reason


def test_p146_backward_mode_na_is_out_of_scope():
    """P146 only tightens arch22→arch35 migration evidence."""
    vj = {
        "mode": "backward",
        "precision": {"status": "PASS"},
        "performance": {"status": "N/A"},
    }
    assert _check(vj) is None


def test_p146_na_with_full_evidence_allows():
    """Legitimate N/A with per-option infeasibility evidence in
    retraction.reason. Gate must accept.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "N/A",
            "retraction": {
                "reason": (
                    "option_1_infeasible_because: A3 runner is static-linked "
                    "binary with no exposed stream handle. "
                    "option_2_infeasible_because: A3 runner has no Python "
                    "bindings."
                ),
            },
        },
    }
    assert _check(vj) is None


def test_p146_clipped_swiglu_lane1_rerun_signature():
    """The literal clipped_swiglu lane-1 2026-05-17T23:13Z archive shape
    (precision PASS bit-exact + perf N/A + empty method). MUST be rejected
    by P146; pins regression for this exact dodge pattern.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "tier": "L1",
        "precision": {
            "status": "PASS",
            "pass_a": {"status": "PASS", "n_pass": 8, "total": 8},
        },
        "performance": {
            "status": "N/A",
            "method": "",
            "ratio": None,
        },
    }
    reason = _check(vj)
    assert reason is not None, "lazy N/A escape must be rejected by P146"
    assert "P146" in reason


def test_p146_does_not_block_when_precision_fails():
    """If precision is FAIL, perf check is irrelevant — gate returns None.
    P146 must not regress this.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "FAIL"},
        "performance": {"status": "N/A"},
    }
    assert _check(vj) is None
