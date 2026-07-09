#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""Tests for the mul operator (dav-3510 Reg reference): z = x * y, float32."""

import logging
import os

import pytest
import torch
import torch_npu

logger = logging.getLogger(__name__)

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.environ.get("OP_MUL_LIB", os.path.join(_HERE, "build", "libop_mul.so"))
torch.ops.load_library(_LIB)


def test_mul_interface_exist():
    assert hasattr(torch.ops.op_mul, "mul"), \
        "The 'mul' operator is not registered in the 'torch.ops.op_mul' namespace."


SHAPES = [
    (1,), (7,), (8,), (1024,), (2049,), (10000,), (65536,), (32, 33), (4, 3, 64, 64),
]
DTYPES = [torch.float32]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_mul_operator(shape, dtype):
    x = torch.randn(*shape, dtype=dtype)
    y = torch.randn(*shape, dtype=dtype)
    expected = x * y
    result = torch.ops.op_mul.mul(x.npu(), y.npu()).cpu()
    assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), \
        f"mul failed for shape {shape}. Max diff: {torch.max(torch.abs(result - expected)):.6f}"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
