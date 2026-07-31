# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Regression tests for P141 port_a3 perf-methodology gate.

User catch 2026-05-17 ~20:14Z: "虚假的测试数据对于整个harness是毁灭性的，
因为这个流程都不可信。这种事情今后必须杜绝。我们的安全网一再被突破".

Independent audit confirmed clipped_swiglu 4.5× + expand_into_jagged_permute
3.77× are methodology asymmetry artifacts: A3 timed window included full
aclnn pipeline (torch_npu.npu_<op> wrapper / aclnn pair GetWorkspaceSize +
Execute + workspace alloc + executor build); A5 timed window included only
direct ACLRT_LAUNCH_KERNEL macro launch. Same `perf_counter` + `sync`
wrapper but wrapping non-equivalent callables.

P141 gate (finalize_pipeline._check_perf_methodology) MUST reject this
exact pattern at finalize time. These tests pin the contract so the gate
can't regress.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))

from finalize_pipeline import _check_perf_methodology  # noqa: E402


def _check(vj: dict):
    return _check_perf_methodology(None, vj)


def test_p141_rejects_bogus_perf_counter_asymmetric_pattern():
    """The exact 2026-05-17 incident pattern: perf_counter+sync wrapper,
    A3 path uses torch_npu wrapper / aclnn, A5 path uses ACLRT_LAUNCH_KERNEL.
    Must REJECT.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "method": "perf_counter+sync wrapper; A3=torch_npu.npu_clipped_swiglu; A5=ACLRT_LAUNCH_KERNEL via pybind",
            "ratio": 4.5,
            "status": "PASS",
        },
    }
    reason = _check(vj)
    assert reason is not None, "gate must reject asymmetric perf_counter pattern"
    assert "PERF_METHODOLOGY_ASYMMETRY" in reason
    assert "port_a3" in reason.lower()


def test_p141_accepts_explicit_not_verified_same_method_retraction():
    """Archive that explicitly declares NOT_VERIFIED_SAME_METHOD with ratio
    omitted is the honest retraction path — BUT per P143 (2026-05-17 followup)
    must ALSO cite per-option infeasibility evidence (`option_1_infeasible_because`
    + `option_2_infeasible_because`). The retraction status alone is not enough.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "status": "NOT_VERIFIED_SAME_METHOD",
            "method": "asymmetric — retracted per P141 contract",
            "retraction": {
                "reason": (
                    "option_1_infeasible_because: A3 runner is statically-linked "
                    "third-party binary with no exposed stream handle. "
                    "option_2_infeasible_because: A3 runner is shipped as a "
                    "standalone executable without Python bindings."
                ),
            },
        },
    }
    assert _check(vj) is None, "explicit retraction WITH per-option evidence must bypass gate"


def test_p141_accepts_option_1_device_event_timing():
    """Option 1 of P141 contract: aclrtEvent on A3 + torch.npu.Event on A5,
    explicit method_symmetric=true. This is the device-event timing pattern
    that excludes host overhead by construction on both sides.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "method": (
                "a3=aclrtEventElapsedTime around aclnn execute call; "
                "a5=torch.npu.Event around ACLRT_LAUNCH_KERNEL launch; "
                "method_symmetric=true device-event"
            ),
            "ratio": 2.0,
            "status": "PASS",
        },
    }
    assert _check(vj) is None, "Option 1 device-event symmetric must pass"


def test_p141_rejects_bare_ratio_without_method_declaration():
    """Bare ratio + empty method = unauditable + retraction-eligible."""
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {"ratio": 3.5, "status": "PASS"},
    }
    reason = _check(vj)
    assert reason is not None
    assert "PERF_METHODOLOGY_ASYMMETRY" in reason
    assert "method is empty" in reason or "method_low" in reason or "performance.method" in reason


def test_p141_accepts_backward_mode_torch_npu_profiler():
    """Backward mode is outside the migration-specific asymmetry gate;
    a valid profiler claim remains accepted.
    """
    vj = {
        "mode": "backward",
        "precision": {"status": "PASS"},
        "performance": {
            "method": "torch_npu.profiler warmup=5 active=5",
            "ratio": 1.2,
            "status": "PASS",
        },
    }
    assert _check(vj) is None, "valid backward perf method must pass"


def test_p141_accepts_option_2_same_wrapper_symmetric():
    """Option 2 of P141 contract: same Python+pybind wrapper on both sides
    (e.g. A3 wraps its aclnn call in the same pybind+torch::empty layer
    that A5 uses). Method declares method_symmetric=true / same_wrapper.
    """
    vj = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "method": "both sides perf_counter wrap pybind ACLRT_LAUNCH_KERNEL same_wrapper",
            "ratio": 1.2,
            "status": "PASS",
        },
    }
    assert _check(vj) is None, "Option 2 same-wrapper symmetric must pass"


def test_p141_known_incident_signatures_retracted():
    """The two SHAs of today's retraction (clipped_swiglu 4.5×,
    expand_into_jagged_permute 3.77×) must both be rejected if presented
    raw. This pins the regression for the EXACT incident text.
    """
    # clipped_swiglu shape
    vj_clipped = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "method": (
                "median over 5 trials per case via torch.npu.synchronize; "
                "A3 baseline from a3_baseline_perf.json; A3 uses "
                "torch_npu.npu_clipped_swiglu (perf_counter wrap), "
                "A5 uses ACLRT_LAUNCH_KERNEL via pybind"
            ),
            "ratio": 4.5,
            "status": "PASS",
        },
    }
    assert _check(vj_clipped) is not None, "clipped_swiglu pattern must regress-reject"

    # expand_into_jagged_permute shape (A3 path subprocess to cpp runner,
    # but cpp uses aclnn pair; A5 path Python+pybind+macro). Same asymmetry.
    vj_expand = {
        "mode": "port_a3_to_a5",
        "precision": {"status": "PASS"},
        "performance": {
            "method": (
                "median over 5 trials per case via torch.npu.synchronize; "
                "A3 baseline from a3_baseline_perf.json via aclnn pair "
                "subprocess cpp runner; A5 via ACLRT_LAUNCH_KERNEL pybind "
                "(perf_counter wrap)"
            ),
            "ratio": 3.77,
            "status": "PASS",
        },
    }
    assert _check(vj_expand) is not None, "expand_into_jagged_permute pattern must regress-reject"
