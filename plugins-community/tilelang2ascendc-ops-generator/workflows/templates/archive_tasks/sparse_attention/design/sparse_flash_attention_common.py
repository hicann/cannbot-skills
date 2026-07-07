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

"""Shared config and helpers for sparse_flash_attention block-level and tile-level designs."""

from dataclasses import dataclass
from types import SimpleNamespace

import tilelang


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@dataclass
class SparseAttnConfig:
    batch: int
    q_seq_len: int
    kv_seq_len: int
    heads: int
    kv_heads: int
    dim: int
    sparse_size: int


def kernel_setup(cfg):
    """Return common block/tile sizing and scheduling parameters for sparse attention."""
    batch, q_seq_len, kv_seq_len, heads, kv_heads, dim, sparse_size = (
        cfg.batch, cfg.q_seq_len, cfg.kv_seq_len, cfg.heads, cfg.kv_heads, cfg.dim, cfg.sparse_size
    )
    result = SimpleNamespace()
    result.group_size = heads // kv_heads
    result.block_group = max(16, ((result.group_size + 15) // 16) * 16)
    result.block_sparse = max(16, ((sparse_size + 15) // 16) * 16)
    result.dtype = "float16"
    result.accum_dtype = "float"
    result.total_bkvh = batch * kv_heads
    result.block_num = result.total_bkvh * q_seq_len
    result.q_shape = [result.total_bkvh, q_seq_len, result.block_group, dim]
    result.kv_shape = [result.total_bkvh, kv_seq_len, dim]
    result.sparse_index_shape = [result.total_bkvh, q_seq_len, sparse_size]
    result.output_shape = [result.total_bkvh, q_seq_len, result.block_group, dim]
    return result
