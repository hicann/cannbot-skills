# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Precision comparison utilities — v2 schema.

Provides three-way symmetric comparison (ans, ref, golden) with
comparison groups, ULP metrics (float), and element statistics.

Float types get the full 12-field comparison (ae, re, rmse, svec, ulp).
Integer types get a minimal comparison (ae, mismatch_rate only).
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch

from constants import DEFAULT_TOLERANCES
from ulp import ulp_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ElementStats:
    nan: Optional[int]       # None for int types
    inf: Optional[int]       # None for int types
    subnormal: Optional[int]  # None for int types
    small: Optional[int]      # None for int types (no sv_th)
    total: int

    def to_dict(self) -> dict:
        return {
            "nan": self.nan,
            "inf": self.inf,
            "subnormal": self.subnormal,
            "small": self.small,
            "total": self.total,
        }


@dataclass
class ComparisonGroup:
    """Comparison metrics between two tensors.

    Float types populate all fields.
    Integer types set re_max, re_mean, rmse, svec, ulp_* to None.
    """
    ae_max: float
    ae_mean: float
    re_max: Optional[float]
    re_mean: Optional[float]
    rmse: Optional[float]
    mean: float
    std: float
    svec: Optional[int]
    ulp_mean: Optional[float]
    ulp_max: Optional[int]
    ulp_miss_rate: Optional[float]
    mismatch_rate: float

    def to_dict(self) -> dict:
        return {
            "ae_max": self.ae_max,
            "ae_mean": self.ae_mean,
            "re_max": self.re_max,
            "re_mean": self.re_mean,
            "rmse": self.rmse,
            "mean": self.mean,
            "std": self.std,
            "svec": self.svec,
            "ulp_mean": self.ulp_mean,
            "ulp_max": self.ulp_max,
            "ulp_miss_rate": self.ulp_miss_rate,
            "mismatch_rate": self.mismatch_rate,
        }


@dataclass
class Ratios:
    """Ratio metrics (ans_vs_golden / ref_vs_golden). Float-only."""
    max_re: float
    mean_re: float
    rmse: float
    svec: float

    def to_dict(self) -> dict:
        return {
            "max_re": self.max_re,
            "mean_re": self.mean_re,
            "rmse": self.rmse,
            "svec": self.svec,
        }


@dataclass
class ComponentResult:
    """Measurement result with pass/fail judgment and optional diagnosis."""
    ans_counts: ElementStats
    ref_counts: ElementStats
    golden_counts: ElementStats
    ans_vs_golden: ComparisonGroup
    ref_vs_golden: ComparisonGroup
    ans_vs_ref: ComparisonGroup
    ratios: Optional[Ratios]
    passed: bool = True                    # NEW: mismatch_rate-based
    diagnosis: Optional[dict] = None       # NEW: from CAKE2

    def to_dict(self) -> dict:
        d = {
            "ans": self.ans_counts.to_dict(),
            "ref": self.ref_counts.to_dict(),
            "golden": self.golden_counts.to_dict(),
            "ans_vs_golden": self.ans_vs_golden.to_dict(),
            "ref_vs_golden": self.ref_vs_golden.to_dict(),
            "ans_vs_ref": self.ans_vs_ref.to_dict(),
            "ratios": self.ratios.to_dict() if self.ratios is not None else None,
        }
        d["passed"] = self.passed
        if self.diagnosis is not None:
            d["diagnosis"] = self.diagnosis
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _count_element_stats(t: torch.Tensor, sv_th=None) -> ElementStats:
    """Count element statistics: NaN, Inf, subnormal, small-value, total.

    For int types (sv_th is None): subnormal and small are None.
    For float types: subnormal counts denormals, small counts |x| < sv_th.
    """
    if t.is_floating_point():
        abs_t = torch.abs(t)
        is_finite = torch.isfinite(t)
        tiny = torch.finfo(t.dtype).tiny
        subnormal = int(((abs_t > 0) & (abs_t < tiny) & is_finite).sum().item())
        if sv_th is not None and sv_th > 0:
            small = int((abs_t < sv_th).sum().item())
        else:
            small = 0
        nan = int(torch.isnan(t).sum().item())
        inf = int(torch.isinf(t).sum().item())
    else:
        # Integer/bool types: no NaN/Inf/subnormal/small-value concepts
        nan = None
        inf = None
        subnormal = None
        small = None

    return ElementStats(
        nan=nan,
        inf=inf,
        subnormal=subnormal,
        small=small,
        total=t.numel(),
    )


# ---------------------------------------------------------------------------
# Float comparison (full 12-field)
# ---------------------------------------------------------------------------

def _compute_comparison_float(
    a: torch.Tensor, b: torch.Tensor, golden: torch.Tensor,
    sv_th: float, sv_err: float,
    atol: float, rtol: float, ulp_tol: int,
    dtype: torch.dtype,
    ulp_method: str = "bitwise",
    include_subnormal: bool = True,
) -> ComparisonGroup:
    """Full precision comparison for float types.

    golden is used only to determine the large/small value regions.
    """
    a64 = a.double()
    b64 = b.double()
    g64 = golden.double()

    # Signed and absolute error — all elements
    diff = a64 - b64
    ae = torch.abs(diff)
    ae_max = torch.max(ae).item() if ae.numel() > 0 else 0.0
    ae_mean = torch.mean(ae).item() if ae.numel() > 0 else 0.0
    mean_val = torch.mean(diff).item() if diff.numel() > 0 else 0.0
    std_val = torch.std(diff).item() if diff.numel() > 1 else 0.0

    # Large-value region: |golden| >= sv_th
    abs_golden = torch.abs(g64)
    large_mask = abs_golden >= sv_th

    n_large = large_mask.sum().item()
    if n_large > 0:
        re = ae[large_mask] / abs_golden[large_mask]
        re_max = torch.max(re).item()
        re_mean = torch.mean(re).item()
        rmse = torch.sqrt(torch.mean(ae[large_mask] ** 2)).item()
    else:
        re_max = 0.0
        re_mean = 0.0
        rmse = 0.0

    # Small-value region: |golden| < sv_th, error > sv_err
    small_mask = ~large_mask
    if small_mask.sum().item() > 0:
        svec = int((ae[small_mask] > sv_err).sum().item())
    else:
        svec = 0

    # ULP metrics (only for floating-point types)
    if dtype in [torch.bfloat16, torch.float16, torch.float32]:
        ulp = ulp_metrics(a, b, dtype, ulp_tol=ulp_tol, method=ulp_method,
                          include_subnormal=include_subnormal)
    else:
        # Integer and other non-float types: skip ULP computation
        ulp = {"ulp_mean": 0.0, "ulp_max": 0, "ulp_miss_rate": 0.0}

    # Mismatch rate
    close = torch.isclose(a.float(), b.float(), atol=atol, rtol=rtol)
    mismatch_rate = 1.0 - close.sum().item() / max(close.numel(), 1)

    return ComparisonGroup(
        ae_max=ae_max,
        ae_mean=ae_mean,
        re_max=re_max,
        re_mean=re_mean,
        rmse=rmse,
        mean=mean_val,
        std=std_val,
        svec=svec,
        ulp_mean=ulp["ulp_mean"],
        ulp_max=ulp["ulp_max"],
        ulp_miss_rate=ulp["ulp_miss_rate"],
        mismatch_rate=mismatch_rate,
    )


# ---------------------------------------------------------------------------
# Integer comparison (ae + mismatch_rate only)
# ---------------------------------------------------------------------------

def _compute_comparison_int(
    a: torch.Tensor, b: torch.Tensor,
    atol: float, rtol: float,
) -> ComparisonGroup:
    """Minimal precision comparison for integer types.

    Only computes ae_max, ae_mean, mismatch_rate.
    re/rmse/svec/ulp fields are set to None (not applicable).
    """
    # Use int64 to avoid overflow on subtraction
    diff = a.long() - b.long()
    ae = torch.abs(diff)
    ae_max = torch.max(ae).item() if ae.numel() > 0 else 0
    ae_mean = torch.mean(ae.float()).item() if ae.numel() > 0 else 0.0
    mean_val = torch.mean(diff.float()).item() if diff.numel() > 0 else 0.0
    std_val = torch.std(diff.float()).item() if diff.numel() > 1 else 0.0

    # Mismatch rate: exact equality for zero tolerance, double-cast otherwise
    # (float32 loses precision for int32 > 2^24 / int64 > 2^53)
    if atol == 0 and rtol == 0.0:
        close = (a == b)
    else:
        close = torch.isclose(a.double(), b.double(), atol=atol, rtol=rtol)
    mismatch_rate = 1.0 - close.sum().item() / max(close.numel(), 1)

    return ComparisonGroup(
        ae_max=float(ae_max),
        ae_mean=ae_mean,
        re_max=None,
        re_mean=None,
        rmse=None,
        mean=mean_val,
        std=std_val,
        svec=None,
        ulp_mean=None,
        ulp_max=None,
        ulp_miss_rate=None,
        mismatch_rate=mismatch_rate,
    )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def _compute_comparison(
    a: torch.Tensor, b: torch.Tensor, golden: torch.Tensor,
    sv_th, sv_err,
    atol: float, rtol: float, ulp_tol,
    dtype: torch.dtype,
    ulp_method: str = "bitwise",
    include_subnormal: bool = True,
) -> ComparisonGroup:
    """Dispatch to float or int comparison based on dtype.

    For float types, sv_th/sv_err/ulp_tol must not be None.
    For int types, they are ignored.
    """
    if dtype.is_floating_point:
        if sv_th is None:
            raise ValueError("sv_th required for float comparison")
        if sv_err is None:
            raise ValueError("sv_err required for float comparison")
        if ulp_tol is None:
            raise ValueError("ulp_tol required for float comparison")
        return _compute_comparison_float(
            a, b, golden, sv_th, sv_err, atol, rtol, ulp_tol, dtype,
            ulp_method=ulp_method, include_subnormal=include_subnormal,
        )
    else:
        return _compute_comparison_int(a, b, atol, rtol)


# ---------------------------------------------------------------------------
# Diagnosis engine
# ---------------------------------------------------------------------------

DIAGNOSIS_PATTERNS = [
    {
        "id": "nan_inf_introduced",
        "detect": lambda r: (
            ((r.ans_counts.nan or 0) > 0 or (r.ans_counts.inf or 0) > 0)
            and (r.golden_counts.nan or 0) == 0 and (r.golden_counts.inf or 0) == 0
            and (r.ref_counts.nan or 0) == 0 and (r.ref_counts.inf or 0) == 0
        ),
        "likelihood": "high",
        "category": "numerical_stability",
        "description": "Custom kernel introduces NaN/Inf that don't exist in golden or ref",
        "suggestions": [
            "Check for division by zero in kernel code",
            "Check log/sqrt inputs — ensure they receive positive values",
            "Check exp inputs — large values cause overflow to Inf",
            "Add numerical guards: clamp inputs before unstable operations",
        ],
    },
    {
        "id": "zero_output",
        "detect": lambda r: (
            r.ans_vs_golden.mismatch_rate > 0.9
            and r.ans_vs_golden.ae_mean < 1e-6
        ),
        "likelihood": "high",
        "category": "incomplete_implementation",
        "description": "Output is near-zero everywhere — likely placeholder or uninitialized output",
        "suggestions": [
            "Verify Process() implements the full algorithm, not just Duplicate(output, 0.0f, size)",
            "Check that all input tensors are loaded and used in computation",
            "Verify DataCopy moves data from GM to local buffer before computing",
        ],
    },
    {
        "id": "localized_outlier",
        "detect": lambda r: (
            not r.passed
            and r.ratios is not None and r.ratios.max_re > r.ratios.mean_re * 5
            and r.ratios is not None and r.ratios.mean_re <= 2.0
        ),
        "likelihood": "high",
        "category": "boundary_handling",
        "description": "max_re ratio is high but mean_re is acceptable — few extreme outliers",
        "suggestions": [
            "Check kernel handling of boundary/edge-case inputs (near-zero, very large, subnormal)",
            "Review tiling boundary: last tile may have fewer elements than expected",
            "Check DataCopyPad usage for non-32B-aligned data sizes",
        ],
    },
    {
        "id": "systematic_drift",
        "detect": lambda r: (
            not r.passed and r.ratios is not None and r.ratios.mean_re > 2.0
        ),
        "likelihood": "high",
        "category": "algorithm_error",
        "description": "Systematic precision drift — mean relative error ratio exceeded",
        "suggestions": [
            "Verify the algorithm matches the DSL/reference exactly",
            "Check for dtype cast precision loss (e.g., fp32 intermediate cast to fp16 too early)",
            "Check cross-tile accumulation — results may only reflect the last tile",
            "Verify all input tensors are used — missing an input causes wrong computation",
        ],
    },
    {
        "id": "small_value_error",
        "detect": lambda r: (
            not r.passed and r.ratios is not None and r.ratios.svec > 2.0
        ),
        "likelihood": "medium",
        "category": "small_value_handling",
        "description": "Small-value region errors exceed tolerance",
        "suggestions": [
            "Check computation near zero — small values may underflow or lose precision",
            "Verify subnormal handling in the kernel",
            "Consider using higher-precision intermediate computation for small inputs",
        ],
    },
    {
        "id": "uneven_distribution",
        "detect": lambda r: (
            not r.passed
            and r.ratios is not None and r.ratios.rmse > 2.0
            and r.ratios is not None and r.ratios.max_re <= 10.0
            and r.ratios is not None and r.ratios.mean_re <= 2.0
        ),
        "likelihood": "medium",
        "category": "error_distribution",
        "description": "RMSE ratio exceeded while max/mean RE are acceptable — uneven error distribution",
        "suggestions": [
            "Check for input-dependent precision patterns (certain value ranges trigger larger errors)",
            "Review vectorized computation — some lanes may compute differently",
            "Profile with different input distributions to identify problematic ranges",
        ],
    },
]


INT_DIAGNOSIS_PATTERNS = [
    {
        "id": "int_total_mismatch",
        "detect": lambda r: r.ans_vs_golden.mismatch_rate > 0.9,
        "likelihood": "high",
        "category": "incomplete_implementation",
        "description": "Nearly all integer outputs differ from golden — algorithm likely wrong or output uninitialized",
        "suggestions": [
            "Verify Process() implements the full algorithm, not a placeholder",
            "Check that output tensor is written with correct DataCopy",
            "Verify input tensors are loaded before computation",
        ],
    },
    {
        "id": "int_overflow",
        "detect": lambda r: r.ans_vs_golden.ae_max > 128,
        "likelihood": "high",
        "category": "integer_overflow",
        "description": "Large absolute error suggests integer overflow or truncation during dtype cast",
        "suggestions": [
            "Check for int8/int16 overflow — use wider intermediate type (int32/int64)",
            "Verify Cast instructions use correct rounding mode",
            "Check multiplication results fit in the target integer range",
        ],
    },
    {
        "id": "int_sign_flip",
        "detect": lambda r: (
            abs(r.ans_vs_golden.mean) < r.ans_vs_golden.std * 0.1
            and r.ans_vs_golden.std > 1.0
            and r.ans_vs_golden.mismatch_rate > 0.01
        ),
        "likelihood": "medium",
        "category": "sign_error",
        "description": "Mean error near zero with high std suggests sign-flip errors on some elements",
        "suggestions": [
            "Check signed/unsigned type mismatch in kernel",
            "Verify sign-extension when casting between integer widths",
            "Check Abs/Neg operations handle sign bit correctly",
        ],
    },
    {
        "id": "int_off_by_one",
        "detect": lambda r: (
            r.ans_vs_golden.ae_max <= 1
            and r.ans_vs_golden.mismatch_rate > 0.0
        ),
        "likelihood": "high",
        "category": "rounding_error",
        "description": "All errors are off-by-one — likely a rounding or floor/ceil direction issue",
        "suggestions": [
            "Check rounding mode: floor vs ceil vs round-to-nearest",
            "Verify float-to-int conversion uses the same rounding as reference",
            "Check boundary values where rounding direction matters (x.5 cases)",
        ],
    },
    {
        "id": "int_scattered_error",
        "detect": lambda r: (
            r.ans_vs_golden.ae_max <= 128
            and 0.01 < r.ans_vs_golden.mismatch_rate <= 0.9
        ),
        "likelihood": "medium",
        "category": "partial_tile_error",
        "description": "Moderate mismatch rate with bounded error — some tiles or branches compute incorrectly",
        "suggestions": [
            "Check tiling boundary handling — last tile may process wrong element count",
            "Verify branch conditions in multi-path kernels (e.g., TopK, Sort)",
            "Check index arithmetic for off-by-one in loop bounds",
        ],
    },
]


def _collect_failed_checks(result):
    """Collect the names of failed precision checks for a ComponentResult."""
    failed_checks = []
    if result.ratios is not None:
        if result.ratios.max_re > 10.0:
            failed_checks.append("max_re")
        if result.ratios.mean_re > 2.0:
            failed_checks.append("mean_re")
        if result.ratios.rmse > 2.0:
            failed_checks.append("rmse")
        if result.ratios.svec > 2.0:
            failed_checks.append("svec")
    else:
        if result.ans_vs_golden.mismatch_rate > 0.0:
            failed_checks.append("mismatch")
        if result.ans_vs_golden.ae_max > 0:
            failed_checks.append("ae_max")

    golden_ref_clean = (
        (result.golden_counts.nan or 0) == 0 and (result.golden_counts.inf or 0) == 0
        and (result.ref_counts.nan or 0) == 0 and (result.ref_counts.inf or 0) == 0
    )
    if golden_ref_clean and ((result.ans_counts.nan or 0) > 0 or (result.ans_counts.inf or 0) > 0):
        failed_checks.append("nan_inf")
    return failed_checks


def _match_diagnosis_patterns(result, is_float):
    """Run all applicable diagnosis patterns against result, return matched ones."""
    patterns = DIAGNOSIS_PATTERNS if is_float else INT_DIAGNOSIS_PATTERNS
    matched = []
    for pat in patterns:
        try:
            if pat["detect"](result):
                matched.append({
                    "likelihood": pat["likelihood"],
                    "category": pat["category"],
                    "description": pat["description"],
                    "suggestions": pat["suggestions"],
                })
        except Exception as exc:
            logger.debug("diagnosis pattern %s detect failed: %s", pat.get("id"), exc)
            continue
    return matched


def _diagnose(result):
    """Match diagnostic patterns against a ComponentResult."""
    is_float = result.ratios is not None
    failed_checks = _collect_failed_checks(result)

    if result.passed:
        return {"verdict": "pass", "failed_checks": [], "pattern": None, "root_causes": []}

    matched = _match_diagnosis_patterns(result, is_float)
    pattern_id = matched[0]["category"] if matched else "unknown"
    return {
        "verdict": "fail",
        "failed_checks": failed_checks,
        "pattern": pattern_id,
        "root_causes": matched if matched else [{
            "likelihood": "low",
            "category": "unknown",
            "description": "No known pattern matched — manual investigation needed",
            "suggestions": [
                "Compare ans_vs_golden and ref_vs_golden metrics side by side",
                "Check the precision JSON for detailed per-element statistics",
                "Generate a dashboard with op-dashboard for visual analysis",
            ],
        }],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _compute_ratios(ans_vs_golden, ref_vs_golden, dtype):
    """Compute answer-vs-reference ratios for float dtypes (None for int types)."""
    if not dtype.is_floating_point:
        return None
    ref_floor = 1e-7
    return Ratios(
        max_re=ans_vs_golden.re_max / max(ref_vs_golden.re_max, ref_floor),
        mean_re=ans_vs_golden.re_mean / max(ref_vs_golden.re_mean, ref_floor),
        rmse=ans_vs_golden.rmse / max(ref_vs_golden.rmse, ref_floor),
        svec=ans_vs_golden.svec / max(ref_vs_golden.svec, 1),
    )


def _has_new_nan_inf(ans_counts, ref_counts, golden_counts) -> bool:
    """True if the answer introduces NaN/Inf that neither reference nor golden has."""
    if (golden_counts.nan or 0) == 0 and (ref_counts.nan or 0) == 0 and (ans_counts.nan or 0) > 0:
        return True
    if (golden_counts.inf or 0) == 0 and (ref_counts.inf or 0) == 0 and (ans_counts.inf or 0) > 0:
        return True
    return False


def dual_inspect(y_ans: torch.Tensor, y_ref: torch.Tensor,
                 y_golden: torch.Tensor, name: str, *,
                 tolerances: Optional[dict] = None,
                 dtype: Optional[torch.dtype] = None) -> ComponentResult:
    """Three-way precision comparison producing ComponentResult.

    Args:
        y_ans: Answer tensor (custom NPU kernel output).
        y_ref: Reference tensor (NPU eager baseline).
        y_golden: Golden tensor (CPU fp64 baseline).
        name: Component name (e.g. "Output", "Grad-X").
        tolerances: Override tolerance dict. If None, uses DEFAULT_TOLERANCES
                    for the given dtype.
        dtype: Data type for ULP calculation and default tolerances.
               If None, inferred from y_ans.dtype.
    """
    if dtype is None:
        dtype = y_ans.dtype

    # Resolve tolerances
    if tolerances is None:
        tol = DEFAULT_TOLERANCES.get(dtype, DEFAULT_TOLERANCES[torch.bfloat16])
    else:
        tol = tolerances

    atol = tol["atol"]
    rtol = tol["rtol"]
    # These are None for int types (not applicable)
    ulp_tol = tol.get("ulp_tol")
    sv_th = tol.get("sv_th")
    sv_err = tol.get("sv_err")
    ulp_method = tol.get("ulp_method", "bitwise")
    include_subnormal = tol.get("include_subnormal", True)

    # 1. Element statistics
    ans_counts = _count_element_stats(y_ans, sv_th)
    ref_counts = _count_element_stats(y_ref, sv_th)
    golden_counts = _count_element_stats(y_golden, sv_th)

    # 2. Three comparison groups (dispatches to float or int internally)
    _kw = dict(sv_th=sv_th, sv_err=sv_err, atol=atol, rtol=rtol,
               ulp_tol=ulp_tol, dtype=dtype, ulp_method=ulp_method,
               include_subnormal=include_subnormal)
    ans_vs_golden = _compute_comparison(y_ans, y_golden, y_golden, **_kw)
    ref_vs_golden = _compute_comparison(y_ref, y_golden, y_golden, **_kw)
    ans_vs_ref = _compute_comparison(y_ans, y_ref, y_golden, **_kw)

    # 3. Ratios (only for float types where re/rmse are available)
    ratios = _compute_ratios(ans_vs_golden, ref_vs_golden, dtype)

    # --- Pass/Fail: mismatch_rate based ---
    nan_inf_fail = _has_new_nan_inf(ans_counts, ref_counts, golden_counts)
    passed = not nan_inf_fail and ans_vs_golden.mismatch_rate == 0.0

    result = ComponentResult(
        ans_counts=ans_counts,
        ref_counts=ref_counts,
        golden_counts=golden_counts,
        ans_vs_golden=ans_vs_golden,
        ref_vs_golden=ref_vs_golden,
        ans_vs_ref=ans_vs_ref,
        ratios=ratios,
        passed=passed,
    )

    # --- Diagnosis ---
    if not passed:
        result.diagnosis = _diagnose(result)

    return result
