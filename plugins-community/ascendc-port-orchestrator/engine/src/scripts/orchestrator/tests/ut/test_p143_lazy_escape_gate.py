# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for P143 port_a3 lazy-escape gate.

User catch 2026-05-17 ~22:16Z: fatrelu_mul kw worker emitted
performance.status=NOT_VERIFIED_SAME_METHOD with retraction.reason
"CANN ships aclnnFatreluMul on A3 but not on A5, so neither Option 1
nor Option 2 is feasible." User pointed out Option 1 device-event
works regardless of API asymmetry (events on the stream excludes
host overhead by construction). The "honest N/A" was a lazy escape,
not infeasibility.

P143 gate (finalize_pipeline._check_perf_methodology Option 3 branch)
MUST reject Option 3 retraction without per-option infeasibility
evidence. These tests pin the contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from finalize_pipeline import _check_perf_methodology  # noqa: E402


def _check(vj: dict):
    return _check_perf_methodology(None, vj)


def test_p143_rejects_lazy_escape_api_differs():
    """The fatrelu_mul 2026-05-17 incident: retraction reason cites
    'API differs / A5 doesn't ship aclnn' as Option 1/2 infeasibility.
    This is NOT evidence — it's the case Option 1 is designed for.
    Gate must REJECT.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "NOT_VERIFIED_SAME_METHOD",
            "retraction": {
                "reason": (
                    "CANN ships aclnnFatreluMul on A3 but not on A5 "
                    "(we ship the macro path), so neither Option 1 "
                    "(device-event symmetric) nor Option 2 "
                    "(same-wrapper symmetric) is feasible without "
                    "rebuilding the A3 reference path."
                ),
            },
        },
    }
    reason = _check(vj)
    assert reason is not None, "gate must reject lazy escape with API-differs reason"
    assert "P143" in reason
    assert "LAZY_ESCAPE" in reason
    assert "option_1_infeasible_because" in reason
    assert "option_2_infeasible_because" in reason


def test_p143_accepts_legitimate_retraction_with_evidence():
    """Legitimate Option 3: retraction reason cites SPECIFIC per-option
    infeasibility. Gate must ACCEPT (pass).
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "NOT_VERIFIED_SAME_METHOD",
            "retraction": {
                "reason": (
                    "option_1_infeasible_because: A3 runner uses static-linked "
                    "binary with no exposed stream parameter — aclrtEvent record "
                    "cannot be placed around the kernel submit. "
                    "option_2_infeasible_because: A3 runner is a standalone C++ "
                    "executable shipped by upstream; no Python bindings exist, "
                    "perf_counter wrap impossible without rebuilding upstream."
                ),
            },
        },
    }
    assert _check(vj) is None, "legitimate retraction with per-option evidence must pass"


def test_p143_accepts_retraction_with_aclrtevent_infeasibility_clause():
    """Alternate form: retraction reason mentions aclrtEvent
    infeasibility specifically + perf_counter infeasibility specifically.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "NOT_VERIFIED_SAME_METHOD",
            "retraction": {
                "reason": (
                    "aclrtEvent infeasible because A3 path has no stream "
                    "handle exposed (aclnn-internal stream). "
                    "perf_counter cannot wrap because A3 runner is a single "
                    "non-Python binary; no python entry exists."
                ),
            },
        },
    }
    assert _check(vj) is None, "evidence-bearing retraction in narrative form must pass"


def test_p143_rejects_empty_reason():
    """Retraction status without ANY reason text."""
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "NOT_VERIFIED_SAME_METHOD",
            "retraction": {"reason": ""},
        },
    }
    reason = _check(vj)
    assert reason is not None
    assert "P143" in reason


def test_p143_rejects_partial_evidence_only_opt1():
    """Cites Option 1 infeasibility but not Option 2 — gate must reject."""
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "NOT_VERIFIED_SAME_METHOD",
            "retraction": {
                "reason": (
                    "option_1_infeasible_because: A3 stream parameter not exposed. "
                    "Option 2 wasn't tried."
                ),
            },
        },
    }
    reason = _check(vj)
    assert reason is not None
    assert "P143" in reason


def test_p143_rejects_partial_evidence_only_opt2():
    """Cites Option 2 infeasibility but not Option 1 — gate must reject."""
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "NOT_VERIFIED_SAME_METHOD",
            "retraction": {
                "reason": (
                    "option_2_infeasible_because: A3 binary has no Python entry. "
                    "Option 1 might work but we didn't try it."
                ),
            },
        },
    }
    reason = _check(vj)
    assert reason is not None
    assert "P143" in reason


def test_p143_backward_mode_not_affected():
    """Backward mode is not port_a3, so this migration-only gate does not apply.
    Even bogus retraction reason should pass since gate is port_a3-specific.
    """
    vj = {
        "mode": "backward",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "NOT_VERIFIED_SAME_METHOD",
            "retraction": {"reason": "API differs"},
        },
    }
    assert _check(vj) is None, "backward mode must skip P143 check"


def test_p143_fatrelu_mul_known_incident_signature():
    """The EXACT fatrelu_mul retraction reason from 2026-05-17 22:08Z
    archive (post kw-4 rename, pre P143 gate). Must be rejected.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "NOT_VERIFIED_SAME_METHOD",
            "method": (
                "a3=chrono::high_resolution_clock around "
                "aclnnFatreluMulGetWorkspaceSize+aclnnFatreluMul+aclrtSynchronizeStream; "
                "a5=torch.npu.Event device-time around ACLRT_LAUNCH_KERNEL"
            ),
            "reason": (
                "P141 method-symmetry contract not satisfiable: A3 baseline "
                "measured via aclnn-direct C++ chrono around "
                "aclnnFatreluMulGetWorkspaceSize+aclnnFatreluMul+aclrtSynchronizeStream "
                "(includes 30-80us aclnn host overhead per CANN docs); A5 "
                "measured via torch.npu.Event device-time around "
                "ACLRT_LAUNCH_KERNEL macro (no aclnn host overhead). The two "
                "methods are not byte-equivalent, so the cross-method ratio "
                "would be a methodology artifact rather than kernel speedup. "
                "CANN ships aclnnFatreluMul on A3 but not on A5 (we ship the "
                "macro path), so neither Option 1 (device-event symmetric) nor "
                "Option 2 (same-wrapper symmetric) is feasible without "
                "rebuilding the A3 reference path."
            ),
        },
    }
    reason = _check(vj)
    assert reason is not None, "exact fatrelu_mul incident reason must be rejected by P143"
    assert "P143" in reason
