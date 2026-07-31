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

"""precision_tier2 — Tier-2/3 (ref-self-drift + list) precision classification, extracted from
precision_eval_two_tier.py (behavior-neutral, 2026-07-05). Depends on precision_tier1 (leaf)
for compute_mere_mare + the shared constants."""
from __future__ import annotations
from typing import Any

import torch

from precision_tier1 import compute_mere_mare, INT_DTYPES, INT_LSB_TOLERANCE, PRECISION_THRESHOLDS


def compute_ref_self_drift(ref_runs: list[torch.Tensor]) -> dict[str, float]:
    """Measure how non-deterministic the reference is across N runs.

    Returns:
        {
          "self_mere_max": max pairwise MERE between any two runs,
          "self_mare_max": max pairwise MARE between any two runs,
          "deterministic": bool — True iff all pairs bit-equal,
        }
    """
    if len(ref_runs) < 2:
        return {"self_mere_max": 0.0, "self_mare_max": 0.0, "deterministic": True}
    a_cpu = ref_runs[0].detach().cpu()
    if all(torch.equal(a_cpu, r.detach().cpu()) for r in ref_runs[1:]):
        return {"self_mere_max": 0.0, "self_mare_max": 0.0, "deterministic": True}
    self_mere = 0.0
    self_mare = 0.0
    for i, left_run in enumerate(ref_runs):
        for right_run in ref_runs[i + 1:]:
            try:
                m, a = compute_mere_mare(left_run, right_run)
            except (ValueError, RuntimeError):
                continue
            self_mere = max(self_mere, m)
            self_mare = max(self_mare, a)
    return {
        "self_mere_max": self_mere,
        "self_mare_max": self_mare,
        "deterministic": False,
    }


def classify_output_t3(
    ours: torch.Tensor,
    ref_runs: list[torch.Tensor],
) -> dict[str, Any]:
    """Compute Tier 3 verdict — ours judged against CANN-on-NPU reference.

    No CPU truth axis. Threshold defaults to T1 (strict); if reference itself
    is non-deterministic across the 3 runs, the threshold is loosened to
    encompass the ref's own drift band (`max(thresh_t1, ref_self_drift_max)`).

    Reference choice for the comparison: ref_runs[0] (first run).
    """
    if len(ref_runs) == 0:
        return {
            "rule": "T3-no-reference",
            "verdict": "EVAL_ERR",
            "error": "no reference runs supplied for T3 eval",
        }

    cann_ref = ref_runs[0]
    dtype = cann_ref.dtype

    self_drift = compute_ref_self_drift(ref_runs)

    # Integer / bool: bit-exact with LSB-tolerance for quantized int8/int16.
    if dtype in INT_DTYPES:
        lsb_tol = INT_LSB_TOLERANCE.get(dtype)
        if lsb_tol is not None:
            # Quantized int8/int16: ±1 LSB tolerance per element.
            def _int_match(a: torch.Tensor, b: torch.Tensor) -> bool:
                d = (a.cpu().to(torch.int32) - b.cpu().to(torch.int32)).abs()
                return bool((d.max().item() if d.numel() else 0) <= lsb_tol)
            all_match = all(_int_match(cann_ref, r.detach()) for r in ref_runs[1:])
            ref_deterministic = all_match
            ours_match = _int_match(ours, cann_ref)
        else:
            all_match = all(torch.equal(cann_ref.cpu(), r.detach().cpu()) for r in ref_runs[1:])
            ref_deterministic = all_match
            ours_match = bool(torch.equal(ours.cpu(), cann_ref.cpu()))
        if ref_deterministic:
            verdict = "PASS_T3" if ours_match else "FAIL"
        else:
            ours_in_ref = any(
                _int_match(ours, r.detach()) if lsb_tol is not None
                else torch.equal(ours.cpu(), r.detach().cpu())
                for r in ref_runs
            )
            verdict = "PASS_T3" if ours_in_ref else "FAIL"
        return {
            "dtype": str(dtype).replace("torch.", ""),
            "rule": f"T3-bit-exact-±{lsb_tol}LSB-int" if lsb_tol is not None else "T3-bit-exact-int",
            "verdict": verdict,
            "ours_match": ours_match,
            "ref_self_drift": self_drift,
            "reference_source": "NPU+CANN",
            "tier_axis": "T3",
        }

    if dtype not in PRECISION_THRESHOLDS:
        return {
            "dtype": str(dtype).replace("torch.", ""),
            "rule": "T3-unknown-dtype",
            "verdict": "EVAL_ERR",
            "error": f"no T3 threshold for {dtype}",
        }

    thr_t1 = PRECISION_THRESHOLDS[dtype]
    mare_thr_t1 = 10 * thr_t1

    # Threshold loosening: if reference itself drifts, expand band.
    thr_t3 = max(thr_t1, self_drift["self_mere_max"])
    mare_thr_t3 = max(mare_thr_t1, self_drift["self_mare_max"])

    ours_mere, ours_mare = compute_mere_mare(ours, cann_ref, threshold=thr_t1)
    pass_t3 = (ours_mere < thr_t3) and (ours_mare < mare_thr_t3)

    return {
        "dtype": str(dtype).replace("torch.", ""),
        "rule": (
            f"T3-MERE<{thr_t3:.2e} AND MARE<{mare_thr_t3:.2e} vs CANN-on-NPU "
            f"(thresh loosened from T1 {thr_t1:.2e}/{mare_thr_t1:.2e} by ref-self-drift)"
            if not self_drift["deterministic"]
            else f"T3-MERE<{thr_t1:.2e} AND MARE<{mare_thr_t1:.2e} vs CANN-on-NPU (ref deterministic, T1 thresholds)"
        ),
        "verdict": "PASS_T3" if pass_t3 else "FAIL",
        "ours_mere": ours_mere,
        "ours_mare": ours_mare,
        "threshold": thr_t3,
        "mare_threshold": mare_thr_t3,
        "ref_self_drift": self_drift,
        "reference_source": "NPU+CANN",
        "tier_axis": "T3",
    }


def classify_index_list_t3(
    ours_indices: torch.Tensor,
    ref_runs_indices: list[torch.Tensor],
    *,
    ours_scores: torch.Tensor | None = None,
    ref_runs_scores: list[torch.Tensor] | None = None,
    overlap_floor: float = 0.95,
    tied_score_tolerance: float | None = None,
) -> dict[str, Any]:
    """P0zz Phase 2(a): T3 verdict for index-list outputs (NMS, topK, etc).

    Distance metric is set-overlap (NOT MERE/MARE — index magnitudes don't
    reflect semantic similarity). Tied-score-drift admission is primary:
    when kernel and ref disagree on which index to keep but the disagreement
    is at indices with tied scores (within `tied_score_tolerance`), admit
    as PASS_T3.

    Args:
        ours_indices:    1D LongTensor of indices our kernel kept.
        ref_runs_indices: list of 1D LongTensors, one per ref run.
        ours_scores:     optional 1D FloatTensor of scores (full input space,
                         not selected). Required for tied-score admission.
        ref_runs_scores: optional same. Allows admitting case where ref's
                         own scores differ slightly across runs.
        overlap_floor:  fraction of ref's set ours must cover after tied
                        admission. Default 0.95; per OL-109 should NEVER be
                        > 1.0 even when ref deterministic — set-overlap is
                        the metric, exact match is not required.
        tied_score_tolerance: |scores[k] - scores[r]| ≤ this admits the
                               disagreement as tie-drift, not divergence.
                               Defaults: 1 ULP at fp16 ≈ 9.77e-4 if scores
                               are fp16, else 1e-6 (fp32-safe).

    Returns dict with verdict, set_overlap, admitted_ties, mismatches.
    """
    # Resolve default tied-score tolerance from scores dtype if available
    if tied_score_tolerance is None:
        if ours_scores is not None and ours_scores.dtype == torch.float16:
            tied_score_tolerance = 9.77e-4
        elif ours_scores is not None and ours_scores.dtype == torch.bfloat16:
            tied_score_tolerance = 7.81e-3
        else:
            tied_score_tolerance = 1e-6

    if not ref_runs_indices:
        return {
            "rule": "T3-index-list-no-reference",
            "verdict": "EVAL_ERR",
            "error": "no reference runs supplied for index-list T3 eval",
        }

    # Convert to sets of ints (sort-invariant comparison)
    ours_set = set(int(i) for i in ours_indices.flatten().cpu().tolist())

    # Reference self-overlap: pairwise across ref runs
    ref_sets = [set(int(i) for i in r.flatten().cpu().tolist()) for r in ref_runs_indices]
    ref_self_overlap = 1.0
    if len(ref_sets) >= 2:
        pairs = []
        for i, left_set in enumerate(ref_sets):
            for right_set in ref_sets[i + 1:]:
                if not right_set:
                    continue
                pairs.append(len(left_set & right_set) / len(right_set))
        ref_self_overlap = (sum(pairs) / len(pairs)) if pairs else 1.0

    # Compare ours against each ref run; admit ties; pick best overlap
    best_overlap = 0.0
    best_admitted_ties = 0
    best_mismatches: list[dict] = []
    best_ref_idx = 0

    for ridx, ref_set in enumerate(ref_sets):
        if not ref_set:
            continue

        # Initial overlap
        intersection = ours_set & ref_set
        ours_only = ours_set - ref_set       # kernel kept, ref dropped
        ref_only = ref_set - ours_set        # ref kept, kernel dropped

        # Tied-score-drift admission: for each disagreement, check whether
        # the disagreed indices have scores within tied_score_tolerance.
        # When admitted, the disagreement is reclassified as a "tie-drift"
        # and counts toward overlap.
        admitted_ties = 0
        unresolved_mismatches: list[dict] = []
        if ours_scores is not None:
            scores_arr = ours_scores.flatten().cpu()

            # For each idx in ours_only, find the nearest-score idx in ref_only
            # and check if it's a tied-score swap.
            ref_only_list = list(ref_only)
            for k in list(ours_only):
                if not ref_only_list:
                    unresolved_mismatches.append({"idx": k, "side": "ours_only", "reason": "no_ref_only_to_pair"})
                    continue
                if k >= len(scores_arr):
                    unresolved_mismatches.append({"idx": k, "side": "ours_only", "reason": "score_oob"})
                    continue
                k_score = float(scores_arr[k].item())
                # Find ref_only_idx with closest score
                paired_r = None
                paired_dist = float("inf")
                for r in ref_only_list:
                    if r >= len(scores_arr):
                        continue
                    r_score = float(scores_arr[r].item())
                    d = abs(k_score - r_score)
                    if d < paired_dist:
                        paired_dist = d
                        paired_r = r
                if paired_r is not None and paired_dist <= tied_score_tolerance:
                    # tied-score swap: admit
                    admitted_ties += 1
                    ref_only_list.remove(paired_r)  # consume the pair
                else:
                    unresolved_mismatches.append({
                        "idx": k, "side": "ours_only",
                        "score_dist_to_nearest_ref_only": paired_dist,
                        "tied_tol": tied_score_tolerance,
                    })
            # Whatever stays in ref_only_list = unmatched ref-only indices
            for r in ref_only_list:
                unresolved_mismatches.append({"idx": r, "side": "ref_only"})
        else:
            # Without scores, can't admit ties; all disagreements are unresolved
            for k in ours_only:
                unresolved_mismatches.append({"idx": k, "side": "ours_only"})
            for r in ref_only:
                unresolved_mismatches.append({"idx": r, "side": "ref_only"})

        # Compute overlap = (intersection + admitted_ties) / |ref_set|
        admitted_intersection = len(intersection) + admitted_ties
        overlap = admitted_intersection / len(ref_set) if ref_set else 1.0

        if overlap > best_overlap:
            best_overlap = overlap
            best_admitted_ties = admitted_ties
            best_mismatches = unresolved_mismatches
            best_ref_idx = ridx

    # Effective floor: never accept worse than ref's own consistency
    effective_floor = max(overlap_floor, ref_self_overlap)
    pass_t3 = best_overlap >= effective_floor

    return {
        "rule": (
            f"T3-index-set-overlap≥{effective_floor:.3f} (ref_self_overlap={ref_self_overlap:.3f}, "
            f"floor={overlap_floor:.3f}); tied_score_tol={tied_score_tolerance:.2e}"
        ),
        "verdict": "PASS_T3" if pass_t3 else "FAIL",
        "set_overlap": best_overlap,
        "ref_self_overlap": ref_self_overlap,
        "admitted_ties": best_admitted_ties,
        "matched_ref_run_idx": best_ref_idx,
        "ref_set_size": len(ref_sets[best_ref_idx]) if ref_sets else 0,
        "ours_set_size": len(ours_set),
        "unresolved_mismatches": best_mismatches[:10],  # cap for log brevity
        "n_unresolved_mismatches": len(best_mismatches),
        "tied_score_tolerance": tied_score_tolerance,
        "effective_floor": effective_floor,
        "reference_source": "NPU+CANN",
        "tier_axis": "T3",
    }
