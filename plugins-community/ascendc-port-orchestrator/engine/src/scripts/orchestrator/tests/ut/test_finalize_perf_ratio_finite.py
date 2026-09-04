# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software; you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Non-finite perf ratio rejection (2026-08-25, codex review F4).

A measured-class perf status (PASS/PASS_WITHIN_TOLERANCE/FAIL/
BELOW_THRESHOLD) must carry a FINITE numeric ratio. nan/inf floats, bools,
and non-finite numeric strings ("NaN"/"inf"/"Infinity") are not measurements
and fall into the existing "not a measured number" hard fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()
_ORCH = _HERE.parents[2]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import finalize_pipeline  # noqa: E402,F401  (initialize the finalize import cycle)
# Bind the checker directly: the module-under-test's helper is protected, and a
# `from ... import` binding keeps the call sites free of cross-module `_name` access.
from finalize_checks_structural import _check_pass_performance_metadata  # noqa: E402


def _vj(ratio):
    return {"performance": {"status": "PASS", "ratio": ratio}}


@pytest.mark.parametrize("ratio", [float("nan"), float("inf"), float("-inf"), True])
def test_non_finite_or_bool_ratio_rejected(ratio):
    reason = _check_pass_performance_metadata("PASS", _vj(ratio), None)
    assert reason is not None
    assert "not a measured number" in reason


@pytest.mark.parametrize("ratio", ["NaN", "nan", "inf", "-inf", "Infinity"])
def test_non_finite_string_ratio_rejected(ratio):
    reason = _check_pass_performance_metadata("PASS", _vj(ratio), None)
    assert reason is not None
    assert "not a measured number" in reason


@pytest.mark.parametrize("ratio", [1.5, 0, 2, "1.05", "0"])
def test_finite_ratio_accepted(ratio):
    assert _check_pass_performance_metadata("PASS", _vj(ratio), None) is None
