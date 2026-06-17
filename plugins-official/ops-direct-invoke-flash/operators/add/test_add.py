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
_LIB = os.environ.get("OP_ADD_LIB", os.path.join(_HERE, "build", "libop_add.so"))
torch.ops.load_library(_LIB)


def test_add_interface_exist():
    """The 'add' op must be discoverable under the torch.ops.op_add namespace.

    Guards against schema/registration drift that would hide the operator from
    Python even when the C++ side compiled and linked.
    """
    logger.info(torch.ops.op_add.add)
    assert hasattr(torch.ops.op_add, "add"), \
        "The 'add' operator is not registered in the 'torch.ops.op_add' namespace."


SHAPES = [
    (1,),
    (3,),
    (10,),
    (100,),
    (1024,),
    (10000,),
    (10, 10),
    (32, 32),
    (100, 100),
    (10, 100),
    (100, 10),
    (256, 512),
    (5, 10, 15),
    (16, 32, 64),
    (32, 64, 128),
    (1, 3, 32, 32),
    (4, 3, 64, 64),
    (8, 3, 128, 128),
    (1000, 1000),
]

DTYPES = [
    torch.float32,
    torch.float16,
    torch.int32,
]


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_add_operator(shape, dtype):
    """Compare op_add.add against torch reference across shapes and dtypes."""
    if dtype in [torch.int32]:
        a = torch.randint(-100, 100, shape, dtype=dtype)
        b = torch.randint(-100, 100, shape, dtype=dtype)
    else:
        a = torch.randn(*shape, dtype=dtype)
        b = torch.randn(*shape, dtype=dtype)

    expected = a + b
    result = torch.ops.op_add.add(a.npu(), b.npu()).cpu()

    if dtype in [torch.int32]:
        assert torch.equal(result, expected), \
            f"Add failed for shape {shape}, dtype {dtype}. " \
            f"Expected {expected}, but got {result}"
    else:
        assert torch.allclose(result, expected, rtol=1e-4, atol=1e-4), \
            f"Add failed for shape {shape}, dtype {dtype}. " \
            f"Max diff: {torch.max(torch.abs(result - expected)):.6f}"

    logger.info("Test passed: shape=%s, dtype=%s", shape, dtype)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
