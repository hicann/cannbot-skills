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

import ctypes
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch_npu

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

_KERNEL_BUILD = Path(__file__).resolve().parent / "kernel" / "build"
_so_path = _KERNEL_BUILD / "flash_attention_ext.cpython-311-x86_64-linux-gnu.so"
if _so_path.is_file():
    ctypes.CDLL(str(_so_path), mode=ctypes.RTLD_GLOBAL)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if q.dtype == torch.float16:
            return torch.ops.npu.flash_attention_fp16(q, k, v)
        elif q.dtype == torch.bfloat16:
            return torch.ops.npu.flash_attention_bf16(q, k, v)
        else:
            raise ValueError(f"Unsupported dtype: {q.dtype}")


def get_init_inputs():
    return []
