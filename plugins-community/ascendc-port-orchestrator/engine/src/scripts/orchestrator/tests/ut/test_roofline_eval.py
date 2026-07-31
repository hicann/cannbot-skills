# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Unit tests for roofline_eval.py — NODE-9 Phase B1."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))

from roofline_eval import (
    SOC_A3,
    SOC_A5,
    Bound,
    RooflineResult,
    analyze,
    attach_to_verification,
    from_msprof_summary,
    should_skip_optimizer,
    soc_for_target,
)


# ---- SoC resolution ----

def test_soc_a3_exact():
    assert soc_for_target("a3").name == "Ascend910_9382"


def test_soc_a5_exact():
    assert soc_for_target("a5").name == "Ascend950PR"


def test_soc_a3_ds_stripped():
    assert soc_for_target("a3-ds").name == "Ascend910_9382"


def test_soc_unknown_falls_back():
    assert soc_for_target("ascend910c_unknown").name == "Ascend950PR"


# ---- SoC parameter sanity ----

def test_a5_ridge_fp16():
    ridge = SOC_A5.ridge_fp16
    assert 35 < ridge < 40, f"expected ridge ~37.3, got {ridge}"


def test_a3_peak_fp16_positive():
    assert SOC_A3.peak_fp16_tflops > 0


# ---- Matmul OI + efficiency ----

def test_matmul_fp16_a5_memory_bound():
    """Small matmul (M=32,K=32,N=32) — OI ~10.7, below ridge 37.3 → memory-bound."""
    r = analyze(
        op_type="matmul", soc_target="a5", dtype_itemsize=2,
        measured_time_s=10e-6,
        op_shape={"M": 32, "K": 32, "N": 32},
    )
    assert r.actionable
    assert r.bound == Bound.MEMORY
    assert r.oi_flop_per_byte > 0
    assert r.efficiency_pct > 0


def test_matmul_fp16_a5_compute_bound():
    """Large matmul (M=8192,K=8192,N=8192) — OI ~2730, well above ridge → compute-bound."""
    r = analyze(
        op_type="matmul", soc_target="a5", dtype_itemsize=2,
        measured_time_s=0.5,
        op_shape={"M": 8192, "K": 8192, "N": 8192},
    )
    assert r.actionable
    assert r.bound == Bound.COMPUTE
    assert r.oi_flop_per_byte > SOC_A5.ridge_fp16


def test_matmul_unknown_shape():
    r = analyze(op_type="matmul", soc_target="a5")
    assert not r.actionable
    assert "missing op_shape" in r.recommendation.lower() or "unknown" in r.recommendation.lower()


# ---- Elementwise ----

def test_elementwise_fp16():
    """Elementwise: OI = 2/(3*2) = 0.33 — deeply memory-bound."""
    r = analyze(
        op_type="elementwise", soc_target="a5", dtype_itemsize=2,
        measured_time_s=10e-6,
        op_shape={"N": 65536},
    )
    assert r.actionable
    assert r.bound == Bound.MEMORY
    assert 0.3 < r.oi_flop_per_byte < 0.35


# ---- Reduction ----

def test_reduction_fp32():
    r = analyze(
        op_type="reduction", soc_target="a5", dtype_itemsize=4,
        measured_time_s=5e-6,
        op_shape={"N": 4096},
    )
    assert r.actionable
    assert r.oi_flop_per_byte > 0


# ---- Softmax ----

def test_softmax_fp16():
    r = analyze(
        op_type="softmax", soc_target="a5", dtype_itemsize=2,
        measured_time_s=100e-6,
        op_shape={"N": 512, "D": 64},
    )
    assert r.actionable
    assert r.bound in (Bound.MEMORY, Bound.COMPUTE)


# ---- Attention ----

def test_attention_fp16():
    r = analyze(
        op_type="attention", soc_target="a5", dtype_itemsize=2,
        measured_time_s=1e-3,
        op_shape={"Sq": 64, "Sk": 64, "D": 64},
    )
    assert r.actionable
    assert r.oi_flop_per_byte > 0


# ---- Efficiency thresholds ----

def test_efficiency_above_80_recommends_skip():
    r = analyze(
        op_type="matmul", soc_target="a5", dtype_itemsize=2,
        measured_time_s=200e-6,
        op_shape={"M": 512, "K": 512, "N": 512},
    )
    # Force high efficiency by using a very fast measured time
    if r.efficiency_pct > 80:
        assert "near hardware limit" in r.recommendation.lower() or "skip" in r.recommendation.lower()


def test_unknown_op_type_returns_safe_default():
    r = analyze(op_type="conv2d", soc_target="a5")
    assert not r.actionable
    assert "unknown" in r.recommendation.lower()


# ---- should_skip_optimizer ----

def test_should_skip_when_above_threshold():
    r = RooflineResult(
        op_type="matmul", efficiency_pct=85.0,
        actionable=True, recommendation="near limit",
        bound=Bound.COMPUTE, soc_name="Ascend950PR",
    )
    assert should_skip_optimizer(r)


def test_should_not_skip_when_below_threshold():
    r = RooflineResult(
        op_type="matmul", efficiency_pct=50.0,
        actionable=True, recommendation="optimize",
        bound=Bound.MEMORY, soc_name="Ascend950PR",
    )
    assert not should_skip_optimizer(r)


def test_should_not_skip_when_not_actionable():
    r = RooflineResult(op_type="unknown", actionable=False, recommendation="fallback")
    assert not should_skip_optimizer(r)


def test_custom_skip_threshold():
    r = RooflineResult(
        op_type="matmul", efficiency_pct=75.0,
        actionable=True, recommendation="light",
        bound=Bound.MEMORY, soc_name="Ascend950PR",
    )
    assert should_skip_optimizer(r, threshold_pct=70.0)


# ---- attach_to_verification ----

def test_attach_to_verification_fields():
    r = analyze(
        op_type="matmul", soc_target="a5", dtype_itemsize=2,
        measured_time_s=100e-6,
        op_shape={"M": 256, "K": 256, "N": 256},
    )
    d = attach_to_verification(r)
    assert "roofline_efficiency_pct" in d
    assert "roofline_bound" in d
    assert "roofline_oi_flop_per_byte" in d
    assert "roofline_recommendation" in d
    assert "roofline_soc" in d
    assert isinstance(d["roofline_efficiency_pct"], float)


# ---- from_msprof_summary ----

def test_from_msprof_summary_file_not_found(tmp_path):
    r = from_msprof_summary(tmp_path / "nonexistent.json")
    assert r is None


def test_from_msprof_summary_valid(tmp_path):
    import json
    data = {
        "op_type": "matmul", "dtype_itemsize": 2,
        "measured_time_s": 1e-3, "op_shape": {"M": 128, "K": 128, "N": 128},
    }
    p = tmp_path / "msprof.json"
    p.write_text(json.dumps(data))
    r = from_msprof_summary(p, soc_target="a5")
    assert r is not None
    assert r.actionable


def test_from_msprof_summary_invalid_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json")
    r = from_msprof_summary(p)
    assert r is None


# ---- Boundary: zero measured time ----

def test_zero_measured_time():
    r = analyze(
        op_type="matmul", soc_target="a5", dtype_itemsize=2,
        measured_time_s=0.0,
        op_shape={"M": 128, "K": 128, "N": 128},
    )
    assert r.actionable
    assert r.efficiency_pct == 0.0
    assert "no measured_time" in r.recommendation.lower()


# ---- msprof-based bound classification ----

def test_msprof_memory_bound_override():
    r = analyze(
        op_type="matmul", soc_target="a5", dtype_itemsize=2,
        measured_time_s=100e-6,
        op_shape={"M": 128, "K": 128, "N": 128},
        msprof_vec_ratio=0.2,
        msprof_mte2_ratio=0.8,
        msprof_hbm_bw_util_pct=80.0,
    )
    assert r.actionable
    assert r.bound == Bound.MEMORY


def test_msprof_unclear_bottleneck():
    r = analyze(
        op_type="matmul", soc_target="a5", dtype_itemsize=2,
        measured_time_s=100e-6,
        op_shape={"M": 128, "K": 128, "N": 128},
        msprof_vec_ratio=0.1,
        msprof_mte2_ratio=0.1,
        msprof_hbm_bw_util_pct=10.0,
    )
    assert r.actionable
    assert r.bound == Bound.UNCLEAR
    assert "bottleneck unclear" in r.recommendation.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


# ---- CUBE vs VEC peak selection (2026-06-12, A5 cube-peak calibration) ----

def test_a5_cube_fields_populated():
    """A5 cube peaks measured on .171 NPU1 — must be the cube values, not vec."""
    assert SOC_A5.peak_cube_fp16_tflops == pytest.approx(373.0)
    assert SOC_A5.peak_cube_fp32_tflops == pytest.approx(24.0)
    # VEC peaks unchanged (ROOFLINE_MODEL.md).
    assert SOC_A5.peak_fp16_tflops == pytest.approx(56.0)


def test_peak_tflops_matmul_uses_cube():
    """matmul/attention → CUBE peak; elementwise/reduction/softmax → VEC peak."""
    from roofline_eval import _peak_tflops
    # Half-precision inputs use two-byte elements.
    assert _peak_tflops(SOC_A5, "matmul", 2) == pytest.approx(373.0)
    assert _peak_tflops(SOC_A5, "attention", 2) == pytest.approx(373.0)
    assert _peak_tflops(SOC_A5, "elementwise", 2) == pytest.approx(56.0)
    assert _peak_tflops(SOC_A5, "softmax", 2) == pytest.approx(56.0)
    assert _peak_tflops(SOC_A5, "reduction", 2) == pytest.approx(56.0)
    # Single-precision inputs use four-byte elements; the cube and vector
    # calibrations remain distinct even though their values are close.
    assert _peak_tflops(SOC_A5, "matmul", 4) == pytest.approx(24.0)
    assert _peak_tflops(SOC_A5, "elementwise", 4) == pytest.approx(28.0)


def test_peak_tflops_falls_back_to_vec_when_uncalibrated():
    """A SoC with no cube calibration (peak_cube_*==0) falls back to VEC peak."""
    from roofline_eval import SocRoofline, _peak_tflops
    uncal = SocRoofline(
        name="X", peak_fp16_tflops=50.0, peak_fp32_tflops=25.0,
        peak_bw_gb_s=1000.0, ub_per_core_kb=192, l2_cache_mb=48, core_count=40,
    )
    assert uncal.peak_cube_fp16_tflops == 0.0
    assert _peak_tflops(uncal, "matmul", 2) == pytest.approx(50.0)  # vec fallback


def test_a5_matmul_efficiency_uses_cube_ceiling():
    """Regression: the bug fixed here. A compute-bound matmul's roofline ceiling
    must be the CUBE peak (373e12), not the VEC peak (56e12). Before the fix the
    VEC peak inflated efficiency_pct ~6.7× → perf_gate wrongly skipped ko.
    """
    r = analyze(
        op_type="matmul", soc_target="a5", dtype_itemsize=2,
        measured_time_s=0.5, op_shape={"M": 8192, "K": 8192, "N": 8192},
    )
    assert r.bound == Bound.COMPUTE
    assert r.roofline_throughput == pytest.approx(373.0e12)  # cube, NOT 56e12
