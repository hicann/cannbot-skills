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

from design.tile_level.matmul_leakyrelu import (
    matmul_leakyrelu as tl_matmul_leakyrelu,
)


class ModelNew(nn.Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, a: torch.Tensor, b: torch.Tensor):
        if a.ndim != 2 or b.ndim != 2:
            raise ValueError(f"Expected 2D inputs, got a:{a.ndim}D b:{b.ndim}D")
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"K dimension mismatch: a.shape[1]={a.shape[1]} != b.shape[0]={b.shape[0]}")
        if a.dtype != b.dtype:
            raise ValueError(f"dtype mismatch: a:{a.dtype} vs b:{b.dtype}")
        if a.dtype not in (torch.float16, torch.int8):
            raise ValueError(f"Unsupported dtype {a.dtype}, expected float16 or int8")
        kernel = self._build_kernel(a, b)
        return kernel(a, b)

    def _build_kernel(self, a: torch.Tensor, b: torch.Tensor):
        m, k = a.shape
        _, n = b.shape
        dtype = str(a.dtype).split(".")[-1]
        accum_dtype = "int32" if a.dtype == torch.int8 else "float"
        return tl_matmul_leakyrelu(
            m,
            n,
            k,
            dtype=dtype,
            accum_dtype=accum_dtype,
            negative_slope=self.negative_slope,
        )
