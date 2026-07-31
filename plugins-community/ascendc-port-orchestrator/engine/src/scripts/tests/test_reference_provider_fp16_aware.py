# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""task#15(iii) regression: fp16-aware canonical evaluator (reference_provider/verify.py).

Locks the fix: a lower-precision kernel output (fp16-FA) compared against a
higher-precision oracle (fp32) must NOT hard-fail on dtype mismatch and must use
the COARSER dtype's tolerance — otherwise a numerically-correct fp16 kernel is
spuriously FAILed (the contested FA-A3 pass count: 0 when ref stored fp32 vs 1
when stored fp16). NOT bar-lowering: over-tolerance fp16 errors still FAIL, and
float-vs-int mismatch still hard-fails.
"""
import logging
import pathlib
import sys

import torch

_RP = pathlib.Path(__file__).resolve().parents[1] / "reference_provider"
sys.path.insert(0, str(_RP))
import verify  # noqa: E402


def _run(cand_t, ref_t, tolerance_mode=True):
    def model_new():
        return {"out": cand_t}

    case = {"inputs": {}, "outputs": {"out": ref_t}}
    return verify.run_one_case(model_new, case, torch.device("cpu"),
                               "positional", tolerance_mode=tolerance_mode)


def test_coarser_float_dtype():
    cf = getattr(verify, '_coarser_float_dtype')
    assert cf(torch.float16, torch.float32) == torch.float16
    assert cf(torch.float32, torch.float16) == torch.float16
    assert cf(torch.bfloat16, torch.float32) == torch.bfloat16
    assert cf(torch.float16, torch.bfloat16) == torch.bfloat16
    assert cf(torch.float32, torch.float32) == torch.float32


def test_fp16_cand_vs_fp32_ref_within_tol_passes():
    # FA-A3 scenario: fp16 kernel output ≈ fp16-quantization of the fp32 oracle
    # truth. Within the fp16 band, but FAR outside the fp32 band (1e-5/1e-6) the
    # pre-fix code applied via the ref dtype — and pre-fix it never even reached
    # the tolerance compare (hard-failed on dtype mismatch). Post-fix → PASS.
    torch.manual_seed(0)
    ref = (torch.rand(256, dtype=torch.float32) + 0.5)  # [0.5, 1.5)
    cand = ref.to(torch.float16)                         # pure fp16 quantization
    ok, msg = _run(cand, ref)
    assert ok, f"fp16-quantized output vs fp32-oracle should PASS within fp16-tol, got FAIL: {msg}"


def test_fp16_cand_vs_fp32_ref_over_tol_fails():
    # Real error: 5e-2 >> fp16 tol → must still FAIL (no bar-lowering).
    base = torch.randn(256, dtype=torch.float32).abs() + 0.5
    ref = base.to(torch.float32)
    cand = (base + 5e-2).to(torch.float16)
    ok, msg = _run(cand, ref)
    assert not ok, f"fp16 over-tolerance error must FAIL, got PASS: {msg}"


def test_float_vs_int_still_hardfails():
    cand = torch.zeros(8, dtype=torch.float16)
    ref = torch.zeros(8, dtype=torch.int64)
    ok, msg = _run(cand, ref)
    assert not ok and "dtype mismatch" in msg, f"float-vs-int must hard-fail: {msg}"


def test_same_dtype_unchanged():
    # fp32 cand vs fp32 ref, tiny diff within fp32 tol → PASS (regression guard:
    # same-dtype path behavior is unchanged by the cross-precision branch).
    base = torch.randn(64, dtype=torch.float32).abs() + 1.0
    ok, msg = _run(base, base + 1e-7)
    assert ok, f"same-dtype within-tol should PASS unchanged: {msg}"


if __name__ == "__main__":
    test_coarser_float_dtype()
    test_fp16_cand_vs_fp32_ref_within_tol_passes()
    test_fp16_cand_vs_fp32_ref_over_tol_fails()
    test_float_vs_int_still_hardfails()
    test_same_dtype_unchanged()
    logging.info("ALL PASS")
