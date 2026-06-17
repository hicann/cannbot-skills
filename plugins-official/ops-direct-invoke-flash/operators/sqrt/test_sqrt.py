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

import logging
import os

import pytest
import torch
import torch_npu

logger = logging.getLogger(__name__)

# The standalone library is produced by `cmake --build build` next to this file.
_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = os.environ.get("OP_SQRT_LIB", os.path.join(_HERE, "build", "libop_sqrt.so"))
torch.ops.load_library(_LIB)


def test_sqrt_interface_exist():
    """The 'sqrt' op must be discoverable under the torch.ops.op_sqrt namespace."""
    logger.info(torch.ops.op_sqrt.sqrt)
    assert hasattr(torch.ops.op_sqrt, "sqrt"), \
        "The 'sqrt' operator is not registered in the 'torch.ops.op_sqrt' namespace."


SHAPES = [
    (1,),
    (7,),
    (64,),
    (512,),
    (4096,),
    (65536,),
    (8, 8),
    (16, 16),
    (64, 64),
    (16, 256),
    (256, 16),
    (128, 384),
    (3, 7, 11),
    (8, 16, 32),
    (16, 48, 96),
    (2, 3, 16, 16),
    (2, 3, 48, 48),
    (4, 3, 96, 96),
    (512, 512),
]

DTYPES = [
    torch.float32,
    torch.float16,
    torch.bfloat16,
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_sqrt_operator(shape, dtype):
    """Compare op_sqrt.sqrt against torch reference across shapes and dtypes."""
    # sqrt is defined for non-negative inputs.
    x = torch.rand(*shape, dtype=dtype).abs() + 0.01

    expected = torch.sqrt(x)
    result = torch.ops.op_sqrt.sqrt(x.npu()).cpu()

    # fp16/bf16 go through an fp32 cast path; loosen tolerance for the low-precision dtypes.
    rtol, atol = (1e-4, 1e-4) if dtype == torch.float32 else (1e-2, 1e-2)
    assert torch.allclose(result, expected, rtol=rtol, atol=atol), \
        f"Sqrt failed for shape {shape}, dtype {dtype}. " \
        f"Max diff: {torch.max(torch.abs(result.float() - expected.float())):.6f}"

    logger.info("Test passed: shape=%s, dtype=%s", shape, dtype)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
