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

from design.tile_level.sparse_flash_attention import (
    sparse_flash_attention_fwd as tl_sparse_flash_attention_fwd,
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sparse_indices: torch.Tensor,
    ) -> torch.Tensor:
        if not all(t.ndim == 4 for t in (q, k, v, sparse_indices)):
            raise ValueError("q, k, v, sparse_indices must all be 4D tensors")
        if k.shape != v.shape:
            raise ValueError(f"k and v must have identical shapes, got {tuple(k.shape)} vs {tuple(v.shape)}")

        batch, q_seq_len, heads, dim = q.shape
        kb, kv_seq_len, kv_heads, kd = k.shape
        ib, iq_seq_len, ikv_heads, sparse_size = sparse_indices.shape

        if (kb, kd) != (batch, dim):
            raise ValueError(
                f"q and k/v batch or hidden size mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}"
            )
        if (ib, iq_seq_len, ikv_heads) != (batch, q_seq_len, kv_heads):
            raise ValueError(
                "sparse_indices shape must be [B, S1, N2, sparse_size], "
                f"got {tuple(sparse_indices.shape)} for q={tuple(q.shape)}, k={tuple(k.shape)}"
            )
        if heads % kv_heads != 0:
            raise ValueError(f"GQA requires N1 % N2 == 0, got N1={heads}, N2={kv_heads}")
        if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
            raise ValueError("q, k, v must be float16")

        group_size = heads // kv_heads
        block_group = max(16, ((group_size + 15) // 16) * 16)

        packed_q = self._pack_q(q, kv_heads)
        flat_k = self._flatten_kv(k)
        flat_v = self._flatten_kv(v)
        packed_indices = self._pack_indices(sparse_indices).to(torch.int32)

        padded_q = self._pad_group(packed_q, block_group)
        kernel = self._build_kernel(q, k, sparse_indices)
        packed_o = kernel(padded_q, flat_k, flat_v, packed_indices)
        return self._unpack_output(packed_o, batch, q_seq_len, kv_heads, heads, group_size).to(torch.float16)

    def _build_kernel(self, q: torch.Tensor, k: torch.Tensor, sparse_indices: torch.Tensor):
        batch, q_seq_len, heads, dim = q.shape
        _, kv_seq_len, kv_heads, _ = k.shape
        sparse_size = sparse_indices.shape[-1]
        return tl_sparse_flash_attention_fwd(
            batch,
            q_seq_len,
            kv_seq_len,
            heads,
            kv_heads,
            dim,
            sparse_size,
        )

    def _pack_q(self, q: torch.Tensor, kv_heads: int) -> torch.Tensor:
        batch, q_seq_len, heads, dim = q.shape
        group_size = heads // kv_heads
        return q.reshape(batch, q_seq_len, kv_heads, group_size, dim).permute(0, 2, 1, 3, 4).reshape(
            batch * kv_heads,
            q_seq_len,
            group_size,
            dim,
        )

    def _flatten_kv(self, x: torch.Tensor) -> torch.Tensor:
        batch, kv_seq_len, kv_heads, dim = x.shape
        return x.permute(0, 2, 1, 3).reshape(batch * kv_heads, kv_seq_len, dim).contiguous()

    def _pack_indices(self, sparse_indices: torch.Tensor) -> torch.Tensor:
        batch, q_seq_len, kv_heads, sparse_size = sparse_indices.shape
        return sparse_indices.permute(0, 2, 1, 3).reshape(batch * kv_heads, q_seq_len, sparse_size).contiguous()

    def _pad_group(self, packed_q: torch.Tensor, block_group: int) -> torch.Tensor:
        bh2, q_seq_len, group_size, dim = packed_q.shape
        padded = torch.zeros(
            (bh2, q_seq_len, block_group, dim),
            dtype=packed_q.dtype,
            device=packed_q.device,
        )
        padded[:, :, :group_size, :] = packed_q
        return padded.contiguous()

    def _unpack_output(self, *args) -> torch.Tensor:
        packed_o, batch, q_seq_len, kv_heads, heads, group_size = args
        return packed_o[:, :, :group_size, :].reshape(batch, kv_heads, q_seq_len, group_size, -1).permute(
            0,
            2,
            1,
            3,
            4,
        ).reshape(batch, q_seq_len, heads, -1)
