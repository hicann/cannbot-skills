# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""T2 (double-baseline ratio) regression tests for backward CDV grading.

Owner-requested "需要补" (2026-06-14): a SEPARATE, ADDITIVE T2 tier on top of the
strict T1 same-dtype-threshold verdict. T2 reuses the CANONICAL
contract_validator_poc.v_double_baseline_ratio + RATIO_THRESH (v2.1 §4.5.1) against a
SAME-PRECISION-CPU competitor (competitor_kind=torch_same_dtype_cpu).

Guardrails these tests pin:
  * ratio ≈ 1 (ours ≈ competitor) → T2 PASS
  * ratio ≫ thresh (ours much worse than competitor) → T2 FAIL
  * competitor absent → T2 N/A (graceful), and T1 still graded
  * T1 path UNCHANGED by T2 (T2 never flips T1's per-dtype verdict / overall verdict)
  * thresholds are the canonical RATIO_THRESH (no invented ratio/threshold)

Pure CPU; no hardware.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backward_cdv_grade import grade, grade_t2_double_baseline  # noqa: E402
from contract_validator_poc import RATIO_THRESH  # noqa: E402


def _rec(dtype, ours, comp, *, profile="randn", competitor=True):
    """One per-case record. ours/comp = dict with mere/mare/rmse."""
    r = {
        "profile": profile, "dtype": dtype, "output": "q",
        "ours_mere": ours["mere"], "ours_mare": ours["mare"],
        "ours_rmse": ours["rmse"], "ours_max_abs_diff": ours.get("mad", ours["rmse"]),
    }
    if competitor:
        r["competitor_kind"] = "torch_same_dtype_cpu"
        r["baseline_mere"] = comp["mere"]
        r["baseline_mare"] = comp["mare"]
        r["baseline_rmse"] = comp["rmse"]
    return r


def _recs(dtype, ours, comp, n=20, **kw):
    return [_rec(dtype, ours, comp, **kw) for _ in range(n)]


# tiny errors that comfortably PASS T1 (fp16 thr = 2^-10 ≈ 9.77e-4)
TINY = {"mere": 5e-5, "mare": 3e-4, "rmse": 1e-4}


# ---------------- T2 PASS: ratio ≈ 1 ----------------
def test_t2_ratio_about_one_passes():
    recs = _recs("float16", ours=TINY, comp=TINY)  # ours == competitor → ratio 1.0
    t2 = grade_t2_double_baseline(recs, "fag", "L1")
    assert t2["status"] == "PASS"
    assert t2["per_dtype"]["float16"]["is_pass"] is True
    for m in ("mare", "mere", "rmse"):
        assert t2["per_dtype"]["float16"]["bootstrap_median_ratio"][m]["ci"][1] <= RATIO_THRESH["L1"][
            {"mare": 0, "mere": 1, "rmse": 2}[m]]


def test_t2_ours_slightly_better_passes():
    # ours half the competitor error → ratio 0.5 < all L1 caps
    ours = {"mere": 2.5e-5, "mare": 1.5e-4, "rmse": 5e-5}
    recs = _recs("float16", ours=ours, comp=TINY)
    t2 = grade_t2_double_baseline(recs, "fag", "L1")
    assert t2["status"] == "PASS"


# ---------------- T2 FAIL: ratio ≫ thresh ----------------
def test_t2_ratio_much_worse_fails():
    # ours 100x worse than competitor → ratio 100 ≫ L1 caps (5/1.5/1.5)
    bad = {"mere": 5e-3, "mare": 3e-2, "rmse": 1e-2}
    recs = _recs("float16", ours=bad, comp=TINY)
    t2 = grade_t2_double_baseline(recs, "fag", "L1")
    assert t2["status"] == "FAIL"
    assert t2["per_dtype"]["float16"]["is_pass"] is False


# ---------------- competitor absent → N/A graceful ----------------
def test_t2_competitor_absent_is_na():
    recs = _recs("float16", ours=TINY, comp=TINY, competitor=False)
    t2 = grade_t2_double_baseline(recs, "fag", "L1")
    assert t2["status"] == "N/A"
    assert "no representative records" in t2["reason"]


def test_grade_emits_na_when_no_competitor_but_t1_still_graded():
    recs = _recs("float16", ours=TINY, comp=TINY, competitor=False)
    out = grade(recs, "fag", "numerically_hard", "L1", ["fp64_golden"])
    assert out["pass_a_t2"]["status"] == "N/A"
    # T1 still produced a per-dtype verdict
    assert "float16" in out["representative_statistical"]["per_dtype"]


# ---------------- T1 UNCHANGED by T2 ----------------
def test_t2_does_not_flip_t1_pass():
    """T1 passes, T2 passes — verdict driven by T1; both present."""
    recs = _recs("float16", ours=TINY, comp=TINY)
    out = grade(recs, "fag", "numerically_hard", "L1", ["fp64_golden"])
    assert out["verdict"] == "PASS"               # T1 primary
    assert out["pass_a_t2"]["status"] == "PASS"   # T2 additive evidence


def test_t2_pass_does_not_mask_t1_fail():
    """The load-bearing guardrail: ours ≈ competitor (T2 PASS via ratio≈1) but BOTH miss the
    strict same-dtype absolute threshold → T1 must STILL be FAIL. T2 must NOT rescue it.
    """
    # fp32 thr = 2^-13 ≈ 1.22e-4; mare_thr = 10·thr ≈ 1.22e-3.
    # ours == competitor (ratio 1 → T2 PASS), but mare 5e-3 > mare_thr → T1 FAIL.
    near = {"mere": 1e-7, "mare": 5e-3, "rmse": 1e-5}
    recs = _recs("float32", ours=near, comp=near)
    out = grade(recs, "fag", "numerically_hard", "L1", ["fp64_golden"])
    assert out["pass_a_t2"]["status"] == "PASS"        # competitor-equivalent
    assert out["representative_statistical"]["per_dtype"]["float32"]["is_pass"] is False  # T1 strict FAIL
    assert out["verdict"] == "FAIL"                     # T2 PASS did NOT flip the overall T1 verdict


def test_t1_verdict_identical_with_and_without_competitor():
    """Adding competitor data (enabling T2) must NOT change the T1 verdict/counts at all."""
    out_no = grade(_recs("float16", ours=TINY, comp=TINY, competitor=False),
                   "fag", "numerically_hard", "L1", ["fp64_golden"])
    out_yes = grade(_recs("float16", ours=TINY, comp=TINY, competitor=True),
                    "fag", "numerically_hard", "L1", ["fp64_golden"])
    assert out_no["verdict"] == out_yes["verdict"]
    assert (out_no["representative_statistical"]["per_dtype"]["float16"]["is_pass"]
            == out_yes["representative_statistical"]["per_dtype"]["float16"]["is_pass"])


# ---------------- canonical thresholds (no invented ratio) ----------------
def test_t2_uses_canonical_ratio_thresh():
    recs = _recs("float16", ours=TINY, comp=TINY)
    t2 = grade_t2_double_baseline(recs, "fag", "L1")
    assert t2["per_dtype"]["float16"]["ratio_thresholds"] == {
        "mare": RATIO_THRESH["L1"][0], "mere": RATIO_THRESH["L1"][1], "rmse": RATIO_THRESH["L1"][2]}
    assert t2["validator"] == "double_baseline_ratio"
    assert t2["competitor_kind"] == "torch_same_dtype_cpu"


def test_t2_tier_changes_thresholds_not_validator():
    recs = _recs("float16", ours=TINY, comp=TINY)
    l0 = grade_t2_double_baseline(recs, "fag", "L0")
    l2 = grade_t2_double_baseline(recs, "fag", "L2")
    assert l0["validator"] == l2["validator"] == "double_baseline_ratio"
    assert l0["per_dtype"]["float16"]["ratio_thresholds"] != l2["per_dtype"]["float16"]["ratio_thresholds"]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
