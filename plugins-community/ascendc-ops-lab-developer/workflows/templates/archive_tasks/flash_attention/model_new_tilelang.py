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
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from design.tile_level.flash_attention import (
    FlashAttentionConfig,
    flash_attention_fwd as tl_flash_attention_fwd,
)

BLOCK_M = 64
BLOCK_N = 64


def _build_kernel(cfg: FlashAttentionConfig):
    return tl_flash_attention_fwd(cfg)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
            raise ValueError(
                f"Expected 4D tensors, got q.ndim={q.ndim}, k.ndim={k.ndim}, v.ndim={v.ndim}"
            )
        if not (q.dtype == k.dtype == v.dtype):
            raise ValueError(
                f"All input dtypes must match, got q={q.dtype}, k={k.dtype}, v={v.dtype}"
            )

        dtype_str = str(q.dtype).split(".")[-1]
        if dtype_str not in ("float16", "bfloat16"):
            raise ValueError(f"Unsupported dtype: {dtype_str}, expected float16 or bfloat16")

        batch, heads, q_seq_len, dim = q.shape
        kv_seq_len = k.shape[2]

        cfg = FlashAttentionConfig(
            batch=batch,
            heads=heads,
            q_seq_len=q_seq_len,
            kv_seq_len=kv_seq_len,
            dim=dim,
            dtype=dtype_str,
        )
        kernel = _build_kernel(cfg)
        output = kernel(q, k, v)
        return output


def get_init_inputs():
    return []
