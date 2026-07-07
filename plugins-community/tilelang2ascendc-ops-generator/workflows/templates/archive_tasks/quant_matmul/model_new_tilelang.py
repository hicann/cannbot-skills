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

from design.tile_level.int8_matmul_scale import (
    int8_matmul_scale as tl_int8_matmul_scale,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor):
        if a.ndim != 2 or b.ndim != 2 or scale.ndim != 1:
            raise ValueError(f"Expected a:2D b:2D scale:1D, got a:{a.ndim}D b:{b.ndim}D scale:{scale.ndim}D")
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"K dimension mismatch: a.shape[1]={a.shape[1]} != b.shape[0]={b.shape[0]}")
        if b.shape[1] != scale.shape[0]:
            raise ValueError(f"N dimension mismatch: b.shape[1]={b.shape[1]} != scale.shape[0]={scale.shape[0]}")
        if a.dtype != torch.int8 or b.dtype != torch.int8:
            raise ValueError(f"Expected int8 inputs, got a:{a.dtype} b:{b.dtype}")
        if scale.dtype != torch.float32:
            raise ValueError(f"Expected float32 scale, got {scale.dtype}")

        kernel = self._build_kernel(a, b)
        return kernel(a, b, scale)

    def _build_kernel(self, a: torch.Tensor, b: torch.Tensor):
        m, k = a.shape
        _, n = b.shape
        return tl_int8_matmul_scale(
            m,
            n,
            k,
            dtype="int8",
            accum_dtype="int32",
        )
