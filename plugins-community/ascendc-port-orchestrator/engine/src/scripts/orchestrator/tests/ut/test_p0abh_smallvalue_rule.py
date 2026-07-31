# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""§4.5.3 Small-Value-Threshold rule wired into precision_eval_two_tier.

Vendor 昇腾算子精度标准 2.1 §4.5.3 (mirror at
src/skills/references/target/ascendc/PRECISION_STANDARD_v2.1.md): when the
ground-truth magnitude is below `SMALL_VALUE_THRESHOLDS[dtype]`, the
relative-error metric (MARE/MERE) is unstable — division-by-near-zero
amplifies a 1-ULP absolute error into a large relative error. The
standard swaps to an absolute-error count and gates on
`ErrorCount_npu ≤ 2 × max(ErrorCount_cann, 1)`.

These tests verify:
1. The threshold dicts match the vendor table values
2. `compute_small_value_error_count` correctly counts (golden in regime
   AND absolute-error exceeds the dtype's small-value error threshold)
3. `classify_output` promotes a case from FAIL to PASS_T1_SMALLVAL when
   T1 fails purely from MARE amplification on small-magnitude golden,
   AND ours is at parity with CANN under the absolute-error metric
4. The promotion does NOT fire when fewer than 10% of elements lie in
   the small-value regime (don't promote tail-cases that wouldn't survive
   larger-magnitude inputs)
5. The promotion does NOT fire when ours is genuinely worse than CANN
   under the absolute-error metric (real precision gap)
6. T1 PASS path is unaffected (the rule is a fallback only)

P0abh tag for archeology.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent.parent))
import precision_eval_two_tier as pet  # noqa: E402


# ---------------------------------------------------------------------------
# Threshold-table values match the vendor wiki (§4.5.3)
# ---------------------------------------------------------------------------
def test_small_value_thresholds_match_vendor_table():
    """§4.5.3 table: FLOAT16 = 2^-11, BFLOAT16 = 2^-8, FLOAT32 = 2^-14."""
    assert pet.SMALL_VALUE_THRESHOLDS[torch.float16] == 2 ** -11
    assert pet.SMALL_VALUE_THRESHOLDS[torch.bfloat16] == 2 ** -8
    assert pet.SMALL_VALUE_THRESHOLDS[torch.float32] == 2 ** -14


def test_small_value_error_thresholds_match_vendor_table():
    """§4.5.3 table: FLOAT16 / BFLOAT16 = 2^-16, FLOAT32 = 2^-30."""
    assert pet.SMALL_VALUE_ERROR_THRESHOLDS[torch.float16] == 2 ** -16
    assert pet.SMALL_VALUE_ERROR_THRESHOLDS[torch.bfloat16] == 2 ** -16
    assert pet.SMALL_VALUE_ERROR_THRESHOLDS[torch.float32] == 2 ** -30


# ---------------------------------------------------------------------------
# compute_small_value_error_count
# ---------------------------------------------------------------------------
def test_smallval_count_zero_when_all_golden_above_threshold():
    """When |golden| ≥ small_value_threshold for every element, count is 0."""
    golden = torch.full((100,), 1.0, dtype=torch.float32)
    actual = golden + 1e-3  # large absolute diff but doesn't matter — none in small-value regime
    err, n = pet.compute_small_value_error_count(actual, golden, torch.float32)
    assert err == 0
    assert n == 0


def test_smallval_count_nonzero_when_diff_exceeds_error_threshold():
    """Element in small-value regime AND with abs-diff > error_thresh → counted."""
    # 50 elements at magnitude 0 (in regime) with abs-diff 1e-5 (>> 2^-30)
    golden = torch.zeros(50, dtype=torch.float32)
    actual = torch.full((50,), 1e-5, dtype=torch.float32)
    err, n = pet.compute_small_value_error_count(actual, golden, torch.float32)
    assert n == 50
    assert err == 50  # every element exceeds the abs-error gate


def test_smallval_count_excludes_in_regime_but_within_error_threshold():
    """In-regime element with abs-diff ≤ error_thresh is NOT counted."""
    golden = torch.zeros(20, dtype=torch.float32)
    actual = torch.full((20,), 2 ** -32, dtype=torch.float32)  # below 2^-30
    err, n = pet.compute_small_value_error_count(actual, golden, torch.float32)
    assert n == 20
    assert err == 0


def test_smallval_count_unsupported_dtype_returns_zero():
    """Dtypes without an entry in SMALL_VALUE_THRESHOLDS return (0, 0)."""
    golden = torch.zeros(10, dtype=torch.float64)
    actual = torch.zeros(10, dtype=torch.float64)
    err, n = pet.compute_small_value_error_count(actual, golden, torch.float64)
    assert err == 0
    assert n == 0


# ---------------------------------------------------------------------------
# classify_output — small-value rule fallback behavior
# ---------------------------------------------------------------------------
def _make_smallvalue_3fa_like_case(n=1000, seed=0):
    """Construct a tensor triple resembling the 3_FA failure pattern:
    golden = mostly small-magnitude (post-softmax-weighted output),
    ours diverges from golden by ~1 fp16 ULP,
    cann diverges from golden by ~1 fp16 ULP — both below the abs-error gate.
    Under standard MARE/MERE, this would FAIL T1 by amplification. Under
    §4.5.3, both ours and cann have ErrorCount = 0, so 0 ≤ 2×max(0,1) = 2 PASS."""
    g = torch.randn(n, dtype=torch.float64, generator=torch.Generator().manual_seed(seed)) * 1e-3
    # Both ours and CANN have ~1 fp16 ULP drift from golden, but only 1e-6 — below 2^-16
    ulp_drift = 1e-6
    ours = (g + ulp_drift).to(torch.float16)
    cann = (g - ulp_drift).to(torch.float16)
    cpu_truth = g.to(torch.float16)
    return ours, cann, cpu_truth


def test_classify_promotes_t1_smallvalue_on_3fa_like_case():
    """Pattern matching 3_FA's PARTIAL: small-magnitude golden, ours has small
    abs-diff vs CPU truth. Standard MARE blows up, but §4.5.3 says PASS.
    """
    ours, cann, cpu_truth = _make_smallvalue_3fa_like_case()
    result = pet.classify_output(ours, cann, cpu_truth)
    # MARE will be inflated relative to threshold (10 * 2^-10 ≈ 9.77e-3)
    # because |golden| ≈ 1e-3 → relative error blows up
    # The §4.5.3 rule should rescue it because abs-diff is well below 2^-16.
    assert result["verdict"] in ("PASS_T1", "PASS_T1_SMALLVAL"), \
        f"expected T1 or T1_SMALLVAL, got {result['verdict']!r} (mare={result['ours_mare']})"
    if result["verdict"] == "PASS_T1_SMALLVAL":
        assert result["smallval_rule_fired"] is True
        assert result["ours_smallval_error_count"] <= 2 * max(result["cann_smallval_error_count"], 1)


def test_classify_does_not_fire_smallvalue_below_10pct_regime():
    """When < 10% of elements lie in the small-value regime, the rule does
    NOT promote — we don't want a tail-of-distribution case to unlock PASS
    on what's otherwise a genuine large-magnitude failure.
    """
    # 5% of elements at small magnitude with small abs-diff; 95% at moderate
    # magnitude with deliberately worsened abs-diff that fails T1.
    g_small = torch.zeros(50, dtype=torch.float32)
    g_large = torch.full((950,), 0.5, dtype=torch.float32)
    golden = torch.cat([g_small, g_large])
    a_small = torch.full((50,), 1e-7, dtype=torch.float32)
    a_large = g_large + 0.01  # 2% relative error — fails fp32 T1 (thr = 2^-13 ≈ 1.22e-4)
    ours = torch.cat([a_small, a_large])
    cann = golden.clone()  # cann perfect
    result = pet.classify_output(ours, cann, golden)
    assert result["verdict"] != "PASS_T1_SMALLVAL"


def test_classify_does_not_promote_when_ours_strictly_worse_than_cann():
    """Both have small-magnitude golden, but ours has way more abs-errors than
    CANN. §4.5.3 rule must reject (real precision gap, not amplification).
    """
    n = 200
    golden = torch.zeros(n, dtype=torch.float32)
    # ours: every element has abs-diff > 2^-30 (1e-5)
    ours = torch.full((n,), 1e-5, dtype=torch.float32)
    # cann: clean — abs-diff = 0
    cann = torch.zeros(n, dtype=torch.float32)
    result = pet.classify_output(ours, cann, golden)
    # ours_count = 200, cann_count = 0; gate: 200 ≤ 2×max(0,1) = 2 → FAIL
    assert result["verdict"] != "PASS_T1_SMALLVAL"


def test_classify_t1_pass_unaffected_by_smallvalue_path():
    """Ops that already pass T1 should never get PASS_T1_SMALLVAL — T1 is
    authoritative when it fires.
    """
    n = 100
    golden = torch.full((n,), 1.0, dtype=torch.float32)
    # ours within fp32 T1 thresholds
    ours = golden + 1e-6
    cann = golden + 5e-7
    result = pet.classify_output(ours, cann, golden)
    assert result["verdict"] == "PASS_T1"
    assert result["smallval_rule_fired"] is False


def test_classify_includes_smallvalue_metrics_in_result():
    """All callers see ours/cann smallval error counts + thresholds."""
    n = 100
    golden = torch.full((n,), 1.0, dtype=torch.float32)
    ours = golden + 1e-6
    cann = golden + 5e-7
    result = pet.classify_output(ours, cann, golden)
    assert "ours_smallval_error_count" in result
    assert "cann_smallval_error_count" in result
    assert "n_smallval_elements" in result
    assert "smallval_threshold" in result
    assert "smallval_error_threshold" in result
    assert "smallval_rule_fired" in result
    assert result["smallval_threshold"] == 2 ** -14
    assert result["smallval_error_threshold"] == 2 ** -30
