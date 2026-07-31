#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""precision_tier1 — Tier-1 (strict ecosystem) precision classification + the shared
precision constants, extracted from precision_eval_two_tier.py (behavior-neutral, 2026-07-05).
LEAF (holds the shared PRECISION_THRESHOLDS/EPS/etc.). precision_eval_two_tier is a __main__
subprocess grader, so tier2 + parent import these FROM THIS LEAF, never re-importing the
__main__ parent (avoids circular-import/dup hazard)."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any

import torch
try:
    import torch_npu  # noqa: F401
    HAS_NPU = True
except ImportError:
    HAS_NPU = False

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference_provider"))
from verify import _coarser_float_dtype  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent / "orchestrator" / "precision"))
from cannbench_grader import compare_tensors as _eco_compare_tensors  # noqa: E402


def _ecosystem_t1(ours: torch.Tensor, cpu_truth: torch.Tensor, eff_dtype: torch.dtype,
                  native_output: "torch.Tensor | None" = None):
    """生态 T1 via the vendored cann-bench compare.py.

    golden = fp64 CPU truth; output = ours cast back to its NATIVE (eff) dtype (so compare.py picks
    the right per-dtype threshold + small-value/cancellation carve-out); native_output = the REAL
    CPU-same-precision reference (reference fn re-run at native dtype on CPU) for the carve-out
    baseline. native_output=None ⇒ compare.py's stricter carve-out (NPU must be exact in the
    small-value region — never looser); provisioning native is what faithfully relaxes fp32 near-zero.
    Returns (passed: bool, CompareResult).
    """
    out_t = ours.detach().cpu()
    if eff_dtype.is_floating_point and out_t.dtype != eff_dtype:
        out_t = out_t.to(eff_dtype)
    gold_t = cpu_truth.detach().cpu()
    if gold_t.is_floating_point():
        gold_t = gold_t.double()
    nat_t = None
    if native_output is not None:
        nat_t = native_output.detach().cpu()
        if eff_dtype.is_floating_point and nat_t.dtype != eff_dtype:
            nat_t = nat_t.to(eff_dtype)
    dtype_str = str(eff_dtype).replace("torch.", "")
    res = _eco_compare_tensors(out_t, gold_t, dtype=dtype_str, native_output=nat_t)
    return bool(res.passed), res


# ---------------------------------------------------------------------------
# Thresholds — identical to production SKILL
# ---------------------------------------------------------------------------
EPS = 1e-7
PRECISION_THRESHOLDS = {
    torch.float16: 2 ** -10,    # ≈ 9.77e-4
    torch.bfloat16: 2 ** -7,    # ≈ 7.81e-3
    torch.float32: 2 ** -13,    # ≈ 1.22e-4
}
INT_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool)

# Quantized integer output LSB tolerance: fp32→int8/int16 rounding can differ
# by ±1 between CPU PyTorch .round() and NPU AscendC Cast(CAST_RINT) due to
# structural fp32 1-ULP differences in the pre-quant arithmetic chain (Sigmoid
# transcendental, FMA-contraction grouping).  Aligned with the inline verifier
# vendor/AscendOpGenAgent/utils/verification_ascendc.py INT_LSB_TOLERANCE.
INT_LSB_TOLERANCE = {
    torch.int8: 1,
    torch.int16: 1,
}

# Vendor standard 昇腾算子精度标准 2.1 §4.5.3 small-value rule.
#
# When the magnitude of the ground-truth value is below `SMALL_VALUE_THRESHOLDS[dtype]`,
# the relative error metric (MARE/MERE) is unstable — division-by-near-zero
# blows up the relative error even though the absolute error is at the dtype's
# precision floor. Vendor's rule swaps to an absolute-error count for those
# cases and compares against the declared independent reference's count.
#
# Source: src/skills/references/target/ascendc/PRECISION_STANDARD_v2.1.md §4.5.3
# Threshold table (verified against vendor wiki 2026-05-08):
#   dtype          | small_value_thresh | error_thresh
#   FLOAT16        | 2^-11              | 2^-16
#   BFLOAT16       | 2^-8               | 2^-16
#   FLOAT32        | 2^-14              | 2^-30
#   HiFLOAT32      | 2^-12              | 2^-28
#   FLOAT8 E4M3    | 2^-4               | 2^-6
#   FLOAT8 E5M2    | 2^-3               | 2^-5
SMALL_VALUE_THRESHOLDS = {
    torch.float16: 2 ** -11,    # ≈ 4.88e-4
    torch.bfloat16: 2 ** -8,    # ≈ 3.91e-3
    torch.float32: 2 ** -14,    # ≈ 6.10e-5
}
SMALL_VALUE_ERROR_THRESHOLDS = {
    torch.float16: 2 ** -16,    # ≈ 1.53e-5
    torch.bfloat16: 2 ** -16,   # ≈ 1.53e-5
    torch.float32: 2 ** -30,    # ≈ 9.31e-10
}


def compute_mere_mare(
    actual: torch.Tensor,
    golden: torch.Tensor,
    threshold: float = 0.0,
) -> tuple[float, float]:
    """Return (MERE, MARE) with vendor §4.5.3 small-value rule.

    Per vendor/AscendOpGenAgent/utils/verification_ascendc.py `_compute_mere`
    and `_compute_mare`: elements whose absolute error is already within
    `threshold * 1e-3` have their relative error suppressed to 0 (they don't
    contribute to mean/max). This is the small-value rule baked into
    MERE/MARE itself, NOT a separate gate.

    `threshold` defaults to 0 for backward compatibility (no mask applied).
    Callers using this for T1 verification MUST pass the per-dtype threshold
    so the mask is applied. classify_output() does this correctly.
    """
    if actual.shape != golden.shape:
        raise ValueError(f"shape mismatch: actual={tuple(actual.shape)} golden={tuple(golden.shape)}")
    a = actual.detach().to(torch.float64).cpu()
    g = golden.detach().to(torch.float64).cpu()
    diff = (a - g).abs()
    rel = diff / (g.abs() + EPS)
    if threshold > 0:
        # Vendor §4.5.3: elements with diff already within threshold*1e-3
        # are not counted as errors (rel -> 0 for mean/max)
        rel = torch.where(diff < threshold * 1e-3, torch.zeros_like(rel), rel)
    if rel.numel() == 0:
        return 0.0, 0.0
    return float(rel.mean().item()), float(rel.max().item())


def compute_small_value_error_count(
    actual: torch.Tensor,
    golden: torch.Tensor,
    dtype: torch.dtype,
) -> tuple[int, int]:
    """Return (ErrorCount, n_smallval) for the §4.5.3 small-value rule.

    ErrorCount = number of elements where (|golden| < SMALL_VALUE_THRESHOLD)
                 AND (|actual - golden| > SMALL_VALUE_ERROR_THRESHOLD).
    n_smallval = number of elements where |golden| < SMALL_VALUE_THRESHOLD
                 (the cardinality of the regime in which the rule fires).

    Both counts are 0 when the dtype has no small-value entry. The caller
    interprets ErrorCount in the gate `target_count ≤ 2 × max(reference_count, 1)`.
    """
    if dtype not in SMALL_VALUE_THRESHOLDS:
        return 0, 0
    a = actual.detach().to(torch.float64).cpu()
    g = golden.detach().to(torch.float64).cpu()
    g_thr = SMALL_VALUE_THRESHOLDS[dtype]
    e_thr = SMALL_VALUE_ERROR_THRESHOLDS[dtype]
    smallval_mask = g.abs() < g_thr
    error_mask = (a - g).abs() > e_thr
    return int((smallval_mask & error_mask).sum().item()), int(smallval_mask.sum().item())


# ---------------------------------------------------------------------------
# Per-output two-tier classifier
# ---------------------------------------------------------------------------
def classify_output(
    ours: torch.Tensor,
    cann: torch.Tensor,
    cpu_truth: torch.Tensor,
    cand_orig_dtype: "torch.dtype | None" = None,
    *,
    native_output: "torch.Tensor | None" = None,
    route: str = "ecosystem",
) -> dict[str, Any]:
    """Compute the forward precision verdict for one (ours, cann, cpu_truth) tensor triple.

    DEFAULT (route="ecosystem", owner-directed 2026-06-30): the FLOAT T1 verdict is the VERBATIM
    vendored cann-bench compare.py (生态 standard) — golden=cpu_truth(fp64), output=ours@native dtype,
    native_output=CPU-same-precision reference. This replaces the drifted per-dtype tolerance
    classifier that false-FAILed fp32 near-zero (the carve-out in compare.py handles small-value/
    cancellation faithfully). `cann` (the A3-CANN competitor) is IGNORED for the 生态 verdict; it is
    only used by the OPTIONAL 商用 route (route="commercial") as the T2 parity fallback, and its
    metrics are always reported for audit.

    `cand_orig_dtype` (2026-05-29): the caller in evaluate() casts `ours` to cpu_truth.dtype BEFORE
    this call (for comparable error MATH), so `ours.dtype` here is already the truth dtype — the
    fp16-aware coarser-dtype selection MUST use the candidate's ORIGINAL (pre-cast) dtype, passed
    here. When None, falls back to ours.dtype.
    `native_output`: REAL CPU-same-precision reference for compare.py's carve-out (None ⇒ stricter).
    `route`: "ecosystem" (default 生态/compare.py) | "commercial" (opt-in T2-vs-cann parity fallback).
    """
    dtype = cpu_truth.dtype

    # Integer / bool: bit-exact with LSB-tolerance for quantized int8/int16.
    if dtype in INT_DTYPES:
        lsb_tol = INT_LSB_TOLERANCE.get(dtype)
        if lsb_tol is not None:
            # Quantized integer outputs (int8/int16): allow ±1 LSB per element.
            # Matches vendor/AscendOpGenAgent/utils/verification_ascendc.py rule.
            diff = (ours.cpu().to(torch.int32) - cpu_truth.cpu().to(torch.int32)).abs()
            ours_match = bool((diff.max().item() if diff.numel() else 0) <= lsb_tol)
            diff_cann = (cann.cpu().to(torch.int32) - cpu_truth.cpu().to(torch.int32)).abs()
            cann_match = bool((diff_cann.max().item() if diff_cann.numel() else 0) <= lsb_tol)
            rule_str = f"bit-exact-±{lsb_tol}LSB"
        else:
            # Non-quantized integers (int32/int64/bool): strict bit-exact.
            ours_match = bool(torch.equal(ours.cpu(), cpu_truth.cpu()))
            cann_match = bool(torch.equal(cann.cpu(), cpu_truth.cpu()))
            rule_str = "bit-exact"
        verdict = "PASS_T1" if ours_match else (
            "PASS_T2" if (not ours_match) and (not cann_match) else "FAIL"
        )
        return {
            "dtype": str(dtype).replace("torch.", ""),
            "rule": rule_str,
            "verdict": verdict,
            "ours_match": ours_match,
            "cann_match": cann_match,
            "ours_mere": 0.0 if ours_match else float("inf"),
            "ours_mare": 0.0 if ours_match else float("inf"),
            "cann_mere": 0.0 if cann_match else float("inf"),
            "cann_mare": 0.0 if cann_match else float("inf"),
            "threshold": 0.0,
            "mare_threshold": 0.0,
        }

    # task#15(b) / #262 fp16-aware: int dtypes returned above (bit-exact). For
    # the FLOAT path, when ours is a LOWER float precision than the cpu_truth
    # oracle (fp16-FA kernel vs fp32 oracle — OL-68 Case-A), key the precision
    # threshold to the COARSER dtype. A low-precision output cannot be held to a
    # high-precision threshold — doing so spuriously FAILs a numerically-correct
    # fp16 kernel (the canonical-O5 0/61 vs worker 22/22 gap). Reassigning `dtype`
    # here propagates to the threshold lookup, small-value mask, and reported dtype.
    # Use the candidate's ORIGINAL pre-cast dtype (the caller casts ours→cpu_truth
    # dtype before this call; #269 read the already-cast ours.dtype = no-op).
    eff_cand_dtype = cand_orig_dtype if cand_orig_dtype is not None else ours.dtype
    if eff_cand_dtype.is_floating_point:
        dtype = _coarser_float_dtype(eff_cand_dtype, dtype)

    # Float: per-dtype threshold + Tier 2 fallback to vs-CANN.
    if dtype not in PRECISION_THRESHOLDS:
        return {
            "dtype": str(dtype).replace("torch.", ""),
            "rule": "unknown-dtype",
            "verdict": "EVAL_ERR",
            "error": f"no threshold for {dtype}",
        }

    thr = PRECISION_THRESHOLDS[dtype]
    mare_thr = 10 * thr

    # Reported metrics (always computed for audit; the VERDICT comes from compare.py for 生态).
    ours_mere, ours_mare = compute_mere_mare(ours, cpu_truth, threshold=thr)
    cann_mere, cann_mare = compute_mere_mare(cann, cpu_truth, threshold=thr)
    beats_cann = (ours_mere < cann_mere) and (ours_mare <= cann_mare)
    ours_smallval_err, n_smallval = compute_small_value_error_count(ours, cpu_truth, dtype)
    cann_smallval_err, _ = compute_small_value_error_count(cann, cpu_truth, dtype)

    # ===================================================================
    # 生态 T1 — VERBATIM vendored cann-bench compare.py (the real standard).
    # output=ours@native dtype, golden=cpu_truth(fp64), native=CPU-same-precision.
    # ===================================================================
    eco_pass, eco_res = _ecosystem_t1(ours, cpu_truth, dtype, native_output=native_output)

    # OPTIONAL 商用 T2 (route="commercial"): parity-or-better vs the A3-CANN competitor.
    pass_t2 = (ours_mere <= cann_mere) and (ours_mare <= cann_mare)

    if eco_pass:
        verdict = "PASS_T1"            # 生态 compare.py passed (incl. its small-value/cancel carve-out)
    elif route == "commercial" and pass_t2:
        verdict = "PASS_T2"           # opt-in: at least as good as the A3-CANN competitor
    else:
        verdict = "FAIL"

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "rule": ("生态 compare.py (vendored cann-bench@007855b: MERE/MARE Stage-1 + small-value/"
                 "cancellation Stage-2 with CPU-same-precision native) [T1]; "
                 "ours≤CANN MERE&MARE [opt-in 商用 T2]"),
        "verdict": verdict,
        "route": route,
        "eco_pass": eco_pass,
        "eco_native_used": native_output is not None,
        "eco_mere": float(eco_res.mere),
        "eco_mare": float(eco_res.mare),
        "beats_cann": beats_cann,
        "ours_mere": ours_mere,
        "ours_mare": ours_mare,
        "cann_mere": cann_mere,
        "cann_mare": cann_mare,
        "threshold": thr,
        "mare_threshold": mare_thr,
        "ours_smallval_error_count": ours_smallval_err,
        "cann_smallval_error_count": cann_smallval_err,
        "n_smallval_elements": n_smallval,
        "smallval_threshold": SMALL_VALUE_THRESHOLDS.get(dtype),
        "smallval_error_threshold": SMALL_VALUE_ERROR_THRESHOLDS.get(dtype),
        # compare.py owns the small-value carve-out now (not the ≥10% §4.5.3 gate).
        "smallval_rule_fired": bool(getattr(eco_res, "small_value_total_count", 0) > 0),
        # option-b carve-out signal (rides through classify_port_a3_case's `m` → per_case) so the
        # op-level native-provisioning gate can distinguish a native-DEPENDENT carve-out FAIL (which a
        # real cpu_same_precision native could relax) from a normal-region kernel error.
        "eco_passed": bool(eco_pass),
        "eco_small_value_passed": bool(getattr(eco_res, "small_value_passed", True)),
        "eco_cancel_passed": bool(getattr(eco_res, "cancel_passed", True)),
        "eco_small_value_total_count": int(getattr(eco_res, "small_value_total_count", 0)),
        "eco_cancel_total_count": int(getattr(eco_res, "cancel_total_count", 0)),
        # TRUE region signal — the exact predicate compare.py uses at :610 (normal_mismatch_count > 0).
        # A FAIL with normal-region over-threshold points is a genuine kernel error the native cannot
        # excuse; a FAIL with ONLY small-value/cancellation-region points is native-DEPENDENT. This
        # supersedes the small_value_passed/cancel_passed proxy, which mis-attributes a MIXED-region
        # kernel bug (normal AND small-value error) as native-dependent.
        # R1 hardening (round-4): a compare.py-INTERNAL exception on already-shape-matched tensors
        # returns passed=False with error_msg SET and normal_mismatch_count=0 (default). That is an
        # EVALUATION error, NOT a native-provisioning gap — route it to normal-region (kernel_fail),
        # never native_provision_failed.
        "eco_normal_region_fail": bool(getattr(eco_res, "normal_mismatch_count", 0) > 0
                                       or getattr(eco_res, "error_msg", None)),
    }
