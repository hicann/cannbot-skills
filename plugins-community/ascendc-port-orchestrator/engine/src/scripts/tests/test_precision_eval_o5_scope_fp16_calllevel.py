# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""task#15 fix(a)+corrected-fix(b) — CALL-LEVEL test through evaluate().

The #269 lesson: an isolation test of classify_output(fp16,...) passed but the fix
was a NO-OP because the evaluate() call-site pre-casts ours->cpu_truth.dtype. These
tests run the REAL evaluate() path on a synthetic archive, asserting:
  (b) fp16 candidate vs fp32 oracle is scored with fp16 tolerance (PASS_T1), NOT
      fp32-strict (the no-op symptom would FAIL it).
  (a) a kernel-declared typed `_OutOfScope` raise -> SKIP_OOS (not EVAL_ERR/FAIL);
      a NON-sentinel exception -> EVAL_ERR (coverage-fraud guard: fail != skip).
  honesty: summary records n_skip_oos / n_in_scope / coverage_pct / in_scope_only,
      and an in-scope OVERALL with skips is flagged *_IN_SCOPE_ONLY (anti-masquerade).
"""
import importlib.util
import pathlib
import sys
import textwrap

import torch

_S = pathlib.Path(__file__).resolve().parents[1]


def _load_eval():
    sys.path.insert(0, str(_S))
    spec = importlib.util.spec_from_file_location("_p2t", _S / "precision_eval_two_tier.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_MODEL_PY = textwrap.dedent('''
    import torch
    class Model(torch.nn.Module):
        def forward(self, x):
            return x * 2.0          # fp32 oracle truth
    def get_input_groups():
        torch.manual_seed(0)
        small = torch.rand(64, dtype=torch.float32) + 0.5     # in-scope (x2 ~ fp16-quant clean-pass)
        return [
            (small.clone(),),                                  # case 0: in-scope
            (small.clone() + 0.1,),                            # case 1: in-scope
            (torch.full((9999,), 1.0, dtype=torch.float32),),  # case 2: OOS (kernel raises _OutOfScope)
            (torch.full((50,), 2.0, dtype=torch.float32),),    # case 3: real bug (non-sentinel raise)
    ]
''')

_MODELNEW_PY = textwrap.dedent('''
    import torch
    class ModelNew(torch.nn.Module):
        def forward(self, x):
            if x.numel() > 4096:                 # declared physical scope cap
                raise RuntimeError("_OutOfScope: numel %d > UB cap 4096" % x.numel())
            if torch.allclose(x, torch.full_like(x, 2.0)):   # case 3 sentinel: a real bug path
                raise RuntimeError("genuine shape bug — not out of scope")
            return (x * 2.0).to(torch.float16)   # correct fp16 kernel output
''')


def _mk_archive(tmp_path):
    (tmp_path / "model.py").write_text(_MODEL_PY)
    (tmp_path / "model_new_ascendc.py").write_text(_MODELNEW_PY)
    return tmp_path


def test_calllevel_fp16_inscope_pass_oos_skip_realbug_evalerr(tmp_path):
    m = _load_eval()
    arch = _mk_archive(tmp_path)
    summ = m.evaluate(arch, verbose=False)
    by_case = {r["case"]: r["verdict"] for r in summ["results"]}

    # (b) in-scope fp16-vs-fp32-oracle scored with fp16 tolerance -> PASS_T1
    #     (no-op symptom would FAIL these on fp32-strict 1.22e-4).
    assert by_case[0].startswith("PASS_T1"), f"case0 in-scope fp16 should PASS_T1: {by_case}"
    assert by_case[1].startswith("PASS_T1"), f"case1 in-scope fp16 should PASS_T1: {by_case}"
    # (a) typed _OutOfScope -> SKIP_OOS ; non-sentinel raise -> EVAL_ERR
    assert by_case[2] == "SKIP_OOS", f"case2 _OutOfScope should SKIP_OOS: {by_case}"
    assert by_case[3] == "EVAL_ERR", f"case3 real bug must be EVAL_ERR (not skipped): {by_case}"
    # honesty data
    assert summ["n_skip_oos"] == 1, summ
    assert summ["n_in_scope"] == 3, summ
    assert summ["in_scope_only"] is True, summ
    assert 0 < summ["coverage_pct"] < 100, summ


def test_calllevel_no_op_regression_guard(tmp_path):
    """Direct proof the fp16-aware fix fires THROUGH evaluate(): if the candidate
    dtype were read post-cast (the #269 no-op), case0/1 would be scored fp32-strict
    and FAIL. They PASS -> the corrected cand_orig_dtype threading works in the
    real call path.
    """
    m = _load_eval()
    summ = m.evaluate(_mk_archive(tmp_path), verbose=False)
    inscope = [r for r in summ["results"] if r["verdict"] != "SKIP_OOS" and r["case"] in (0, 1)]
    assert all(r["verdict"].startswith("PASS_T1") for r in inscope), \
        f"fp16-aware must fire through evaluate() call-site (not no-op): {[(r['case'], r['verdict']) for r in inscope]}"


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
