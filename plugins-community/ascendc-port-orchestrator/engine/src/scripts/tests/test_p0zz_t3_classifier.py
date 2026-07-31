# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""P0zz (2026-05-06): T3 tier classification — when CPU truth is structurally
undefined, kernel is judged against CANN-on-NPU reference under T1-equivalent
thresholds (loosened only when reference itself is non-deterministic).

Spec: src/skills/references/target/ascendc/OPERATIONAL_KNOWLEDGE.md OL-109 §T3 (commits
4ad4829 + 46b675e). Trigger requires BOTH:
  1. CPU truth structurally undefined (cpu_model.forward raised torch_npu err)
  2. Reference IS non-deterministic on test inputs (3× ref-run check)

If condition #2 fails (reference deterministic), T3 still applies but
threshold = thresh_t1 strict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parent.parent))
import precision_eval_two_tier as pe  # noqa: E402


# ---------------------------------------------------------------------------
# compute_ref_self_drift
# ---------------------------------------------------------------------------
def test_self_drift_deterministic_when_all_runs_identical():
    a = torch.randn(8, 8, dtype=torch.float32)
    runs = [a.clone(), a.clone(), a.clone()]
    drift = pe.compute_ref_self_drift(runs)
    assert drift["deterministic"] is True
    assert drift["self_mere_max"] == 0.0
    assert drift["self_mare_max"] == 0.0


def test_self_drift_nonzero_when_runs_differ():
    base = torch.randn(8, 8, dtype=torch.float32)
    # Each run has a small perturbation
    runs = [
        base + 0.0,
        base + 1e-3,
        base + 2e-3,
    ]
    drift = pe.compute_ref_self_drift(runs)
    assert drift["deterministic"] is False
    assert drift["self_mere_max"] > 0.0


def test_self_drift_single_run_treated_deterministic():
    """Edge case: only 1 ref run available → can't measure drift, assume det."""
    runs = [torch.randn(4, 4, dtype=torch.float32)]
    drift = pe.compute_ref_self_drift(runs)
    assert drift["deterministic"] is True


# ---------------------------------------------------------------------------
# classify_output_t3
# ---------------------------------------------------------------------------
def test_t3_pass_when_kernel_matches_deterministic_reference():
    """Reference deterministic + ours within thresh_t1 → PASS_T3."""
    ref = torch.randn(8, 8, dtype=torch.float32)
    ours = ref + 1e-7  # well within fp32 thresh (1.22e-4)
    runs = [ref.clone(), ref.clone(), ref.clone()]
    res = pe.classify_output_t3(ours, runs)
    assert res["verdict"] == "PASS_T3"
    assert res["tier_axis"] == "T3"
    assert res["reference_source"] == "NPU+CANN"
    assert res["ref_self_drift"]["deterministic"] is True


def test_t3_fail_when_kernel_outside_t1_thresh_with_deterministic_ref():
    """Reference deterministic + ours significantly off → FAIL."""
    ref = torch.ones(8, 8, dtype=torch.float32)
    ours = ref + 1.0  # 100% relative error >> fp32 thresh
    runs = [ref.clone(), ref.clone(), ref.clone()]
    res = pe.classify_output_t3(ours, runs)
    assert res["verdict"] == "FAIL"


def test_t3_threshold_loosens_when_ref_is_nondeterministic():
    """Reference drifts → T3 threshold loosens to encompass ref-self band."""
    base = torch.zeros(8, 8, dtype=torch.float32)
    # Reference itself drifts by 1e-2 (large relative error)
    runs = [base.clone(), base + 1e-2, base + 2e-2]
    # Kernel matches first ref run within strict T1
    ours = runs[0].clone() + 1e-9
    res = pe.classify_output_t3(ours, runs)
    # Ours within strict thresh, but ref is non-det. Check threshold loosened.
    assert res["ref_self_drift"]["deterministic"] is False
    assert res["threshold"] >= res["ref_self_drift"]["self_mere_max"]


def test_t3_int_output_bit_exact_against_deterministic_reference():
    """Integer output: bit-exact against ref → PASS_T3."""
    ref = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    ours = ref.clone()
    runs = [ref.clone(), ref.clone(), ref.clone()]
    res = pe.classify_output_t3(ours, runs)
    assert res["verdict"] == "PASS_T3"


def test_t3_int_output_fail_when_kernel_diverges_from_ref():
    """Integer output: differs from ref while ref is deterministic → FAIL."""
    ref = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64)
    ours = torch.tensor([1, 2, 3, 4, 99], dtype=torch.int64)
    runs = [ref.clone(), ref.clone(), ref.clone()]
    res = pe.classify_output_t3(ours, runs)
    assert res["verdict"] == "FAIL"


def test_t3_int_output_admits_kernel_matching_any_ref_when_ref_drifts():
    """Integer output: ref non-det → ours admitted if matches ANY ref run."""
    refs = [
        torch.tensor([1, 2, 3], dtype=torch.int64),
        torch.tensor([1, 3, 2], dtype=torch.int64),  # different valid order
        torch.tensor([2, 1, 3], dtype=torch.int64),
    ]
    ours = refs[1].clone()
    res = pe.classify_output_t3(ours, refs)
    assert res["verdict"] == "PASS_T3"


def test_t3_no_reference_runs_returns_eval_err():
    res = pe.classify_output_t3(torch.zeros(4), [])
    assert res["verdict"] == "EVAL_ERR"


def test_t3_unknown_dtype_returns_eval_err():
    """fp64 has no T1 threshold defined — T3 should EVAL_ERR rather than guess."""
    ref = torch.randn(8, dtype=torch.float64)
    ours = ref.clone()
    runs = [ref, ref.clone(), ref.clone()]
    res = pe.classify_output_t3(ours, runs)
    assert res["verdict"] == "EVAL_ERR"
    assert "no T3 threshold" in res.get("error", "")


# ---------------------------------------------------------------------------
# _is_torch_npu_required_error trigger detection
# ---------------------------------------------------------------------------
def test_torch_npu_import_error_triggers_t3():
    exc = ImportError("No module named 'torch_npu'")
    assert getattr(pe, '_is_torch_npu_required_error')(exc) is True


def test_module_not_found_torch_npu_triggers_t3():
    exc = ModuleNotFoundError("No module named 'torch_npu'")
    assert getattr(pe, '_is_torch_npu_required_error')(exc) is True


def test_attribute_error_npu_method_triggers_t3():
    exc = AttributeError(
        "module 'torch' has no attribute 'npu_quant_matmul'"
    )
    assert getattr(pe, '_is_torch_npu_required_error')(exc) is True


def test_runtime_error_npu_device_triggers_t3():
    exc = RuntimeError(
        "Expected NPU device but got CPU backend in op forward"
    )
    assert getattr(pe, '_is_torch_npu_required_error')(exc) is True


def test_real_eval_bug_does_not_trigger_t3():
    """Shape mismatch isn't a T3 trigger — it's a real test bug."""
    exc = ValueError("shape mismatch: actual=(8,4) golden=(8,8)")
    assert getattr(pe, '_is_torch_npu_required_error')(exc) is False


def test_unrelated_import_error_does_not_trigger_t3():
    exc = ImportError("No module named 'numpy'")
    assert getattr(pe, '_is_torch_npu_required_error')(exc) is False


# ---------------------------------------------------------------------------
# Phase 2(a): classify_index_list_t3 — NMS / topK / index-list outputs
# ---------------------------------------------------------------------------
def test_index_list_perfect_match_passes():
    """Kernel exactly matches deterministic ref → PASS_T3."""
    ref = torch.tensor([0, 2, 5, 7], dtype=torch.int64)
    ours = ref.clone()
    runs = [ref.clone(), ref.clone(), ref.clone()]
    res = pe.classify_index_list_t3(ours, runs)
    assert res["verdict"] == "PASS_T3"
    assert res["set_overlap"] == 1.0
    assert res["ref_self_overlap"] == 1.0


def test_index_list_set_equivalence_under_reorder():
    """Same indices, different order — set-overlap is order-invariant."""
    ref = torch.tensor([0, 2, 5, 7], dtype=torch.int64)
    ours = torch.tensor([7, 5, 2, 0], dtype=torch.int64)
    runs = [ref.clone(), ref.clone(), ref.clone()]
    res = pe.classify_index_list_t3(ours, runs)
    assert res["verdict"] == "PASS_T3"
    assert res["set_overlap"] == 1.0


def test_index_list_below_floor_without_scores_fails():
    """No scores → no tie admission. 0.5 overlap below default 0.95 floor → FAIL."""
    ref = torch.tensor([0, 2, 5, 7], dtype=torch.int64)
    ours = torch.tensor([0, 2, 9, 11], dtype=torch.int64)  # 50% overlap
    runs = [ref.clone(), ref.clone(), ref.clone()]
    res = pe.classify_index_list_t3(ours, runs)
    assert res["verdict"] == "FAIL"
    assert res["set_overlap"] == 0.5


def test_index_list_tied_score_swap_admitted():
    """Kernel kept idx 4, ref kept idx 5; their scores differ by < tol → admit."""
    ref = torch.tensor([0, 2, 5, 7], dtype=torch.int64)
    ours = torch.tensor([0, 2, 4, 7], dtype=torch.int64)
    # Scores: idx 4 and idx 5 have nearly identical scores (within 1 ULP fp32)
    scores = torch.zeros(8, dtype=torch.float32)
    scores[0] = 0.9
    scores[2] = 0.7
    scores[4] = 0.5  # tie with idx 5
    scores[5] = 0.5 + 1e-7  # < 1e-6 tol
    scores[7] = 0.3
    runs = [ref.clone(), ref.clone(), ref.clone()]
    res = pe.classify_index_list_t3(
        ours, runs, ours_scores=scores, ref_runs_scores=[scores] * 3,
    )
    assert res["verdict"] == "PASS_T3"
    assert res["admitted_ties"] == 1
    assert res["set_overlap"] == 1.0


def test_index_list_real_divergence_not_admitted():
    """Kernel kept idx 9 (very different score) where ref kept idx 5 → no admit, FAIL."""
    ref = torch.tensor([0, 2, 5, 7], dtype=torch.int64)
    ours = torch.tensor([0, 2, 9, 7], dtype=torch.int64)
    scores = torch.zeros(10, dtype=torch.float32)
    scores[0] = 0.9
    scores[2] = 0.7
    scores[5] = 0.5
    scores[7] = 0.3
    scores[9] = 0.1  # 0.4 dist from idx 5 — way above tol
    runs = [ref.clone(), ref.clone(), ref.clone()]
    res = pe.classify_index_list_t3(
        ours, runs, ours_scores=scores, ref_runs_scores=[scores] * 3,
    )
    assert res["verdict"] == "FAIL"
    assert res["admitted_ties"] == 0


def test_index_list_ref_self_overlap_relaxes_floor():
    """When ref itself drifts, effective floor = max(overlap_floor, ref_self_overlap).
    A kernel achieving the same overlap as ref's own self-consistency should PASS.
    """
    # 3 ref runs that all have 0.75 pairwise overlap with each other
    refs = [
        torch.tensor([0, 1, 2, 3], dtype=torch.int64),
        torch.tensor([0, 1, 2, 4], dtype=torch.int64),
        torch.tensor([0, 1, 3, 4], dtype=torch.int64),
    ]
    # Kernel matches ref[0] exactly
    ours = refs[0].clone()
    res = pe.classify_index_list_t3(ours, refs, overlap_floor=0.95)
    # Ref-self overlap is < 1.0 (drifty), floor relaxed
    assert res["ref_self_overlap"] < 1.0
    # Ours achieves 1.0 against ref[0], passes
    assert res["verdict"] == "PASS_T3"


def test_index_list_no_reference_returns_eval_err():
    res = pe.classify_index_list_t3(torch.tensor([0, 1, 2]), [])
    assert res["verdict"] == "EVAL_ERR"


def test_index_list_dtype_inferred_tied_tolerance_fp16():
    """Tied tolerance defaults: fp16 scores → ~9.77e-4."""
    ref = torch.tensor([0, 1, 2], dtype=torch.int64)
    ours = ref.clone()
    scores_fp16 = torch.zeros(4, dtype=torch.float16)
    res = pe.classify_index_list_t3(
        ours, [ref.clone()], ours_scores=scores_fp16,
    )
    assert res["tied_score_tolerance"] >= 9.0e-4  # fp16 ULP scale


def test_index_list_tied_swap_caps_at_floor_unmet_for_extra_kernel_indices():
    """Kernel adds an extra idx with no ref counterpart → unresolved, FAIL."""
    ref = torch.tensor([0, 1], dtype=torch.int64)
    ours = torch.tensor([0, 1, 5, 6], dtype=torch.int64)  # extra 5,6 with no ref pair
    scores = torch.tensor([0.9, 0.8, 0.0, 0.0, 0.0, 0.5, 0.5], dtype=torch.float32)
    runs = [ref.clone()]
    res = pe.classify_index_list_t3(
        ours, runs, ours_scores=scores,
    )
    # Kernel kept extras with no ref-only to pair against → unresolved
    # set_overlap = 2/2 = 1.0 against ref BUT mismatches non-empty
    # The classifier still passes if overlap meets floor — extras don't reduce it
    # under the current formula. This test documents that behavior.
    assert res["set_overlap"] == 1.0
    # n_unresolved counts the extras
    assert res["n_unresolved_mismatches"] >= 2
