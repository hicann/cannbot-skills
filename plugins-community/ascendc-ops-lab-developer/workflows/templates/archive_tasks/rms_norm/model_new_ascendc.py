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

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_LIB_PATTERN = str(_KERNEL_BUILD / "rms_norm_ext*")

# Prefer whl import (triggers TORCH_LIBRARY registration in register.cpp)
try:
    import rms_norm_ext  
except ImportError:
    # Fallback: direct .so loading from kernel/build/
    if _LIB_PATTERN not in "".join(sys.path):
        import glob as _glob

        _libs = _glob.glob(_LIB_PATTERN)
        if _libs:
            torch.ops.load_library(_libs[0])


class ModelNew(nn.Module):
    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, gamma: torch.Tensor):
        if x.ndim < 2:
            raise ValueError("x must be at least 2D")
        if gamma.ndim != 1:
            raise ValueError("gamma must be 1D")
        if x.shape[-1] != gamma.shape[0]:
            raise ValueError("gamma shape mismatch")
        if x.dtype != gamma.dtype:
            raise ValueError("x and gamma dtype must match")
        if x.dtype not in (torch.float16, torch.float32, torch.bfloat16):
            raise ValueError("unsupported dtype")
        if not x.is_contiguous() or not gamma.is_contiguous():
            raise ValueError("inputs must be contiguous")

        original_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        gamma_1d = gamma.contiguous()
        y_2d, inv_rms_1d = torch.ops.npu.rms_norm(x_2d, gamma_1d, self.eps)
        return (
            y_2d.reshape(original_shape),
            inv_rms_1d.reshape(*original_shape[:-1], 1),
        )
