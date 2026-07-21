# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

"""Tests for NaN/Inf handling in ULP distance calculations."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Add evaluation scripts to path (this file lives in <skill>/tests/, scripts in <skill>/scripts/)
SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from ulp import (
    ulp_distance_bitwise,
    ulp_distance_native,
    ulp_metrics,
)

DTYPES = [torch.bfloat16, torch.float16, torch.float32]


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

class TestNaN:
    """Both NaN → 0, one NaN → inf."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_both_nan_bitwise(self, dtype):
        a = torch.tensor([float('nan')])
        b = torch.tensor([float('nan')])
        dist = ulp_distance_bitwise(a, b, dtype)
        assert dist[0] == 0.0

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_both_nan_native(self, dtype):
        a = torch.tensor([float('nan')])
        b = torch.tensor([float('nan')])
        dist = ulp_distance_native(a, b, dtype)
        assert dist[0] == 0.0

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_one_nan_a_bitwise(self, dtype):
        a = torch.tensor([float('nan')])
        b = torch.tensor([1.0])
        dist = ulp_distance_bitwise(a, b, dtype)
        assert np.isinf(dist[0])

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_one_nan_b_native(self, dtype):
        a = torch.tensor([1.0])
        b = torch.tensor([float('nan')])
        dist = ulp_distance_native(a, b, dtype)
        assert np.isinf(dist[0])


# ---------------------------------------------------------------------------
# Inf handling
# ---------------------------------------------------------------------------

class TestInf:
    """Same-sign Inf → 0, diff-sign or one Inf → inf."""

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_both_pos_inf_bitwise(self, dtype):
        a = torch.tensor([float('inf')])
        b = torch.tensor([float('inf')])
        dist = ulp_distance_bitwise(a, b, dtype)
        assert dist[0] == 0.0

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_both_neg_inf_native(self, dtype):
        a = torch.tensor([float('-inf')])
        b = torch.tensor([float('-inf')])
        dist = ulp_distance_native(a, b, dtype)
        assert dist[0] == 0.0

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_diff_sign_inf_bitwise(self, dtype):
        a = torch.tensor([float('inf')])
        b = torch.tensor([float('-inf')])
        dist = ulp_distance_bitwise(a, b, dtype)
        assert np.isinf(dist[0])

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_one_inf_native(self, dtype):
        a = torch.tensor([float('inf')])
        b = torch.tensor([1.0])
        dist = ulp_distance_native(a, b, dtype)
        assert np.isinf(dist[0])


# ---------------------------------------------------------------------------
# Mixed values (normal + NaN + Inf)
# ---------------------------------------------------------------------------

class TestMixed:
    """Ensure NaN/Inf handling doesn't break normal values."""

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("method", ["bitwise", "native"])
    def test_identical_normal_values(self, dtype, method):
        a = torch.tensor([1.0, 2.0, 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        m = ulp_metrics(a, b, dtype, method=method)
        assert m["ulp_mean"] == 0.0
        assert m["nan_mismatch"] == 0

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_mixed_nan_normal(self, dtype):
        a = torch.tensor([1.0, float('nan'), 3.0])
        b = torch.tensor([1.0, float('nan'), 3.0])
        m = ulp_metrics(a, b, dtype, method="bitwise")
        assert m["nan_mismatch"] == 0  # both NaN → match
        assert m["ulp_miss_rate"] == 0.0

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_nan_mismatch_counted(self, dtype):
        a = torch.tensor([1.0, float('nan'), 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        m = ulp_metrics(a, b, dtype, method="bitwise")
        assert m["nan_mismatch"] == 1
        assert m["ulp_miss_rate"] > 0.0

    @pytest.mark.parametrize("dtype", DTYPES)
    def test_nan_mismatch_in_native(self, dtype):
        a = torch.tensor([1.0, float('nan'), 3.0])
        b = torch.tensor([1.0, 2.0, 3.0])
        m = ulp_metrics(a, b, dtype, method="native")
        assert m["nan_mismatch"] == 1
        assert m["ulp_max"] == float('inf')
