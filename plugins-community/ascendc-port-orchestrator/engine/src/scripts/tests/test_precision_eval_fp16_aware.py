# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""task#15(b): precision_eval_two_tier.classify_output fp16-aware threshold keying.

Root cause (canonical O5 0/61 vs worker 22/22): classify_output keyed the precision
threshold off cpu_truth.dtype. For fp16-FA-kernel vs fp32-oracle it applied the fp32
threshold (1.22e-4) to an fp16 output → spurious FAIL. Fix: key to the COARSER dtype
(reuse verify.py _coarser_float_dtype). NOT bar-lowering — over-tol fp16 still FAILs.
"""
import pathlib
import sys

import torch

_S = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_S))
import precision_eval_two_tier as p2t  # noqa: E402


def test_import_reuses_verify_helper():
    # single-source: the helper is the one from reference_provider/verify.py
    assert getattr(p2t, '_coarser_float_dtype')(torch.float16, torch.float32) == torch.float16


def test_fp16_ours_vs_fp32_truth_passes_t1():
    # Correct fp16 kernel = fp16-quantization of the fp32 oracle truth. Diff is
    # ~fp16-eps: FAILS the fp32 threshold (1.22e-4) but within fp16 (9.77e-4).
    torch.manual_seed(0)
    cpu_truth = (torch.rand(512, dtype=torch.float32) + 0.5)   # [0.5,1.5), away from 0
    ours = cpu_truth.to(torch.float16)                          # correct fp16 output
    cann = cpu_truth.clone()                                    # CANN == truth (T2 moot)
    r = p2t.classify_output(ours, cann, cpu_truth)
    assert r["verdict"].startswith("PASS_T1"), f"fp16-vs-fp32-oracle should PASS_T1: {r}"
    assert r["dtype"] == "float16", f"threshold dtype should be coarser (fp16): {r}"


def test_fp16_over_tolerance_still_fails():
    # Real error 5e-2 >> fp16 threshold → must NOT pass T1 (no bar-lowering).
    torch.manual_seed(1)
    cpu_truth = (torch.rand(512, dtype=torch.float32) + 0.5)
    ours = (cpu_truth + 5e-2).to(torch.float16)
    cann = cpu_truth.clone()  # cann accurate → ours strictly worse → not even T2
    r = p2t.classify_output(ours, cann, cpu_truth)
    assert r["verdict"] not in ("PASS_T1", "PASS_T1_SMALLVAL"), f"over-tol fp16 must not PASS_T1: {r}"


def test_same_dtype_fp32_unchanged():
    # fp32 ours vs fp32 truth: coarser(fp32,fp32)=fp32 → behavior unchanged.
    torch.manual_seed(2)
    cpu_truth = (torch.rand(256, dtype=torch.float32) + 1.0)
    ours = cpu_truth.clone()
    r = p2t.classify_output(ours, cpu_truth.clone(), cpu_truth)
    assert r["verdict"].startswith("PASS_T1") and r["dtype"] == "float32", r


def test_int_dtype_unaffected_by_coarser():
    # int truth → bit-exact branch (returns before the float coarser-dtype reassignment).
    cpu_truth = torch.arange(16, dtype=torch.int32)
    r = p2t.classify_output(cpu_truth.clone(), cpu_truth.clone(), cpu_truth)
    assert r["dtype"] == "int32" and r["verdict"] == "PASS_T1", r


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
