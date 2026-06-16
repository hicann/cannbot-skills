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

from design.tile_level.rms_norm import rms_norm as tl_rms_norm


class ModelNew(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, gamma: torch.Tensor):
        if gamma.ndim != 1:
            raise ValueError(f"gamma must be 1D, got {gamma.ndim}D")
        if x.ndim < 2:
            raise ValueError(f"x must be at least 2D, got {x.ndim}D")
        if x.shape[-1] != gamma.shape[0]:
            raise ValueError(f"Last dim mismatch: x.shape[-1]={x.shape[-1]} != gamma.shape[0]={gamma.shape[0]}")
        if x.dtype != gamma.dtype:
            raise ValueError(f"dtype mismatch: x:{x.dtype} vs gamma:{gamma.dtype}")
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError(f"Unsupported dtype {x.dtype}")

        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        gamma_1d = gamma.contiguous()

        kernel = self._build_kernel(x_2d)
        y_2d, inv_rms_1d = kernel(x_2d, gamma_1d)
        return (
            y_2d.reshape(original_shape),
            inv_rms_1d.reshape(*original_shape[:-1], 1),
        )

    def _build_kernel(self, x: torch.Tensor):
        m, n = x.shape
        return tl_rms_norm(
            m,
            n,
            eps=self.eps,
            dtype=str(x.dtype).split(".")[-1],
        )
