#!/usr/bin/env python3
# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

import sys
from pathlib import Path

import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.reshape_matmul_rowwise_quant_int8 import (
    reshape_matmul_rowwise_quant_int8 as tl_reshape_matmul_rowwise_quant_int8,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, h: torch.Tensor):
        if x.ndim != 2 or h.ndim != 2:
            raise ValueError(f"Expected 2D inputs, got x:{x.ndim}D h:{h.ndim}D")
        if x.dtype != torch.bfloat16 or h.dtype != torch.bfloat16:
            raise ValueError(f"Expected bfloat16 inputs, got x:{x.dtype} h:{h.dtype}")
        k = h.shape[0]
        if x.shape[1] % k != 0:
            raise ValueError(f"x.shape[1]={x.shape[1]} must be divisible by k={k}")
        if h.shape[0] != k or h.shape[1] != k:
            raise ValueError(f"h must be square (k,k), got {h.shape}")
        if x.shape[0] % 128 != 0:
            raise ValueError(f"x.shape[0]={x.shape[0]} must be divisible by 128")
        if x.shape[1] % 256 != 0:
            raise ValueError(f"x.shape[1]={x.shape[1]} must be divisible by 256")
        if k % 256 != 0:
            raise ValueError(f"k={k} must be divisible by 256")

        kernel = self._build_kernel(x, h)
        return kernel(x, h)

    def _build_kernel(self, x: torch.Tensor, h: torch.Tensor):
        m, n = x.shape
        k = h.shape[0]
        if h.shape != (k, k):
            raise ValueError(f"h must be square (k,k), got {h.shape}")
        dtype = str(x.dtype).split(".")[-1]
        return tl_reshape_matmul_rowwise_quant_int8(
            m,
            n,
            k,
            dtype=dtype,
            accum_dtype="float",
        )
