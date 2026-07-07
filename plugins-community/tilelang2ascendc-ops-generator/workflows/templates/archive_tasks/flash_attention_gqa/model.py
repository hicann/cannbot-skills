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

# Template for: flash_attention_gqa
import torch
# Template for: flash_attention_gqa
import torch.nn as nn


def _validate_gqa_inputs(q, k, v):
    """Validate flash_attention_gqa forward inputs. Raises ValueError on mismatch."""
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError(f"Expected 4D inputs, got q:{q.ndim}D k:{k.ndim}D v:{v.ndim}D")
    if k.shape != v.shape:
        raise ValueError(f"k and v shapes must match, got k{k.shape} v{v.shape}")
    if q.shape[0] != k.shape[0]:
        raise ValueError(f"Batch size mismatch: q{q.shape[0]} vs k{k.shape[0]}")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError(f"Head dim mismatch: q{q.shape[-1]} k{k.shape[-1]} v{v.shape[-1]}")
    if q.shape[1] % k.shape[1] != 0:
        raise ValueError(f"q heads must be divisible by kv heads: {q.shape[1]} % {k.shape[1]} != 0")


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        _validate_gqa_inputs(q, k, v)

        group_size = q.shape[1] // k.shape[1]
        k_expanded = k.repeat_interleave(group_size, dim=1)
        v_expanded = v.repeat_interleave(group_size, dim=1)

        acc = torch.einsum("bhsd,bhkd->bhsk", q, k_expanded) * (1.0 / q.shape[-1]) ** 0.5
        acc = acc.softmax(dim=-1)
        o = torch.einsum("bhsk,bhkd->bhsd", acc, v_expanded)
        return o.to(torch.float16)


def get_input_groups():
    cases = [
        (2, 2048, 4096, 32, 8, 128),
    ]
    input_groups = []
    for b, q_s, kv_s, h, kv_h, d in cases:
        q = torch.rand((b, h, q_s, d), dtype=torch.float16)
        k = torch.rand((b, kv_h, kv_s, d), dtype=torch.float16)
        v = torch.rand((b, kv_h, kv_s, d), dtype=torch.float16)
        input_groups.append([q, k, v])
    return input_groups


def get_init_inputs():
    return []
