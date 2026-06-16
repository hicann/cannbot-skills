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

import torch
import torch.nn as nn


def _process_kv_head(*args):
    q, k, v, sparse_indices, out, sm_scale, batch_idx, query_idx, kv_head_idx, s2, group_size = args
    """Process attention for one kv_head and all its GQA groups."""
    token_index = sparse_indices[batch_idx, query_idx, kv_head_idx]
    if token_index.numel() == 0:
        return out
    if torch.any(token_index < 0) or torch.any(token_index >= s2):
        raise IndexError(f"token index out of range for S2={s2}: {token_index.tolist()}")

    k_sel = k[batch_idx, token_index, kv_head_idx, :]
    v_sel = v[batch_idx, token_index, kv_head_idx, :]
    k_sel_fp32 = k_sel.to(torch.float32)
    v_sel_fp32 = v_sel.to(torch.float32)

    for group_idx in range(group_size):
        q_head_idx = kv_head_idx * group_size + group_idx
        q_vec = q[batch_idx, query_idx, q_head_idx, :].to(torch.float32)
        scores = torch.matmul(q_vec, k_sel_fp32.transpose(0, 1)) * sm_scale
        attn = torch.softmax(scores, dim=-1)
        out[batch_idx, query_idx, q_head_idx, :] = torch.matmul(attn, v_sel_fp32)
    return out


def sparse_flash_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sparse_indices: torch.Tensor,
) -> torch.Tensor:
    if not all(t.ndim == 4 for t in (q, k, v, sparse_indices)):
        raise ValueError("q, k, v, sparse_indices must all be 4D tensors")
    if k.shape != v.shape:
        raise ValueError(f"k and v must have identical shapes, got {tuple(k.shape)} vs {tuple(v.shape)}")

    b, s1, n1, d = q.shape
    kb, s2, n2, kd = k.shape
    ib, is1, in2, _ = sparse_indices.shape

    if (kb, kd) != (b, d):
        raise ValueError(
            f"q and k/v batch or hidden size mismatch: q={tuple(q.shape)}, k={tuple(k.shape)}"
        )
    if (ib, is1, in2) != (b, s1, n2):
        raise ValueError(
            "sparse_indices shape must be [B, S1, N2, sparse_size], "
            f"got {tuple(sparse_indices.shape)} for q={tuple(q.shape)}, k={tuple(k.shape)}"
        )
    if n1 % n2 != 0:
        raise ValueError(f"GQA requires N1 % N2 == 0, got N1={n1}, N2={n2}")

    group_size = n1 // n2
    sm_scale = d ** -0.5
    out = torch.zeros((b, s1, n1, d), dtype=torch.float32, device=q.device)

    sparse_indices = sparse_indices.to(torch.int64)

    for batch_idx in range(b):
        for query_idx in range(s1):
            for kv_head_idx in range(n2):
                out = _process_kv_head(
                    q, k, v, sparse_indices, out, sm_scale,
                    batch_idx, query_idx, kv_head_idx, s2, group_size)

    return out.to(torch.float16)


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sparse_indices: torch.Tensor,
    ) -> torch.Tensor:
        return sparse_flash_attention(q, k, v, sparse_indices)


def get_input_groups():
    cases = [
        (1, 4, 256, 8, 2, 64, 128),
        (1, 8, 128, 4, 2, 32, 8),
    ]

    input_groups = []
    for b, s1, s2, n1, n2, d, sparse_size in cases:
        if sparse_size > s2:
            raise ValueError(
                f"sparse_size must be <= selectable tokens, got sparse_size={sparse_size}, S2={s2}"
            )

        torch.manual_seed(42 + sparse_size)
        q = torch.randn((b, s1, n1, d), dtype=torch.float16)
        k = torch.randn((b, s2, n2, d), dtype=torch.float16)
        v = torch.randn((b, s2, n2, d), dtype=torch.float16)
        sparse_indices = torch.stack(
            [
                torch.randperm(s2)[:sparse_size].sort().values
                for _ in range(b * s1 * n2)
            ]
        ).reshape(b, s1, n2, sparse_size).to(torch.int32)
        input_groups.append([q, k, v, sparse_indices])

    return input_groups


def get_init_inputs():
    return []
