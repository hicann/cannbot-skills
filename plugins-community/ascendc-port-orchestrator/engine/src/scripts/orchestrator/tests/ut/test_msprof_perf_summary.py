# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""0-d scalar tensor is not an empty case (2026-08-25, codex review F5).

vendor/msprof_perf_summary._case_has_empty_tensor must match the runner's
numel()==0 semantics: a 0-d scalar (shape == [], numel == 1) is a legal
measurable input, and a missing shape is unspecified — neither is empty.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_VENDOR = _HERE.parents[2] / "vendor" / "msprof_perf_summary.py"
_spec = importlib.util.spec_from_file_location("msprof_perf_summary", _VENDOR)
msprof = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(msprof)
sys.modules.setdefault("msprof_perf_summary", msprof)

# Resolved by name rather than imported: ``msprof_perf_summary`` is loaded above
# from an explicit file location, so no top-of-module import statement can bind it.
_case_has_empty_tensor = getattr(msprof, "_case_has_empty_tensor")


def _case(*inputs):
    return {"inputs": list(inputs)}


def test_zero_dim_scalar_is_not_empty():
    case = _case({"type": "tensor", "shape": [], "dtype": "float32"})
    assert _case_has_empty_tensor(case) is False


def test_shape_with_zero_dim_is_empty():
    case = _case({"type": "tensor", "shape": [0, 4], "dtype": "float32"})
    assert _case_has_empty_tensor(case) is True


def test_missing_shape_is_not_empty():
    case = _case({"type": "tensor", "dtype": "float32"})
    assert _case_has_empty_tensor(case) is False


def test_none_shape_is_not_empty():
    case = _case({"type": "tensor", "shape": None, "dtype": "float32"})
    assert _case_has_empty_tensor(case) is False
