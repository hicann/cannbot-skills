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

# Template for: flash_attention_gqa
import torch
import torch.nn as nn

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))

from design.tile_level.flash_attention_gqa import (
    flash_attention_gqa_fwd as tl_flash_attention_gqa_fwd,
)
from model import _validate_gqa_inputs


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        _validate_gqa_inputs(q, k, v)
        if (q.shape[1] // k.shape[1]) * q.shape[2] % 64 != 0:
            result_num = (q.shape[1] // k.shape[1]) * q.shape[2]
            raise ValueError(f"group_size * q_seq_len must be divisible by 64, got {result_num}")
        if k.shape[2] % 64 != 0:
            raise ValueError(f"kv_seq_len must be divisible by 64, got {k.shape[2]}")
        if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
            raise ValueError(f"Expected float16, got q:{q.dtype} k:{k.dtype} v:{v.dtype}")

        self._cached_batch = q.shape[0]
        packed_q = self._pack_q(q, k.shape[1])
        flat_k = self._flatten_kv(k)
        flat_v = self._flatten_kv(v)
        kernel = self._build_kernel(q, k)
        packed_o = kernel(packed_q, flat_k, flat_v)
        return self._unpack_o(packed_o, q.shape[1], q.shape[2])

    def _build_kernel(self, q: torch.Tensor, k: torch.Tensor):
        batch, heads, q_seq_len, dim = q.shape
        _, kv_heads, kv_seq_len, _ = k.shape
        return tl_flash_attention_gqa_fwd(
            batch,
            q_seq_len,
            kv_seq_len,
            heads,
            kv_heads,
            dim,
        )

    def _pack_q(self, q: torch.Tensor, kv_heads: int) -> torch.Tensor:
        batch, heads, q_seq_len, dim = q.shape
        group_size = heads // kv_heads
        return q.reshape(batch, kv_heads, group_size, q_seq_len, dim).reshape(
            batch * kv_heads,
            group_size * q_seq_len,
            dim,
        )

    def _flatten_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch, kv_heads, seq_len, dim = x.shape
        return x.reshape(batch * kv_heads, seq_len, dim)

    def _unpack_o(self, o: torch.Tensor, heads: int, q_seq_len: int) -> torch.Tensor:
        bh2, _, dim = o.shape
        kv_heads = bh2 // self._cached_batch
        group_size = heads // kv_heads
        return o.reshape(self._cached_batch, kv_heads, group_size, q_seq_len, dim).reshape(
            self._cached_batch,
            heads,
            q_seq_len,
            dim,
        )
