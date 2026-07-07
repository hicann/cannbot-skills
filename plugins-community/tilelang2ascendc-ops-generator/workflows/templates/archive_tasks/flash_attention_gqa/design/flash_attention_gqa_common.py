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

"""Shared config and helpers for flash_attention_gqa block-level and tile-level designs."""

from dataclasses import dataclass
from types import SimpleNamespace

import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


@dataclass
class GQAConfig:
    batch: int
    q_seq_len: int
    kv_seq_len: int
    heads: int
    kv_heads: int
    dim: int


@dataclass
class ProdWorkspace:
    acc_s_ub: ...
    acc_s_half: ...
    workspace_2: ...
    workspace_meta: ...
    m_i_prev: ...
    sumexp_i_ub: ...
    cid: ...
    vid: ...
    block_q_packed: ...
    slot_prod: ...


def _kernel_setup(cfg):
    """Return common block/tile sizing and scheduling parameters for GQA."""
    batch, q_seq_len, kv_seq_len, heads, kv_heads, dim = (
        cfg.batch, cfg.q_seq_len, cfg.kv_seq_len, cfg.heads, cfg.kv_heads, cfg.dim
    )
    result = SimpleNamespace()
    result.block_q_packed = 64
    result.block_kv_seq = 64
    result.block_bkvh = 2
    result.prelaunch = 2
    result.ring_slots = result.prelaunch + 1
    result.dtype = "float16"
    result.accum_dtype = "float"
    result.group_size = heads // kv_heads
    result.packed_q_seq_len = result.group_size * q_seq_len
    result.q_shape = [batch * kv_heads, result.packed_q_seq_len, dim]
    result.kv_shape = [batch * kv_heads, kv_seq_len, dim]
    result.output_shape = [batch * kv_heads, result.packed_q_seq_len, dim]
    result.total_bkvh = batch * kv_heads
    result.q_blocks_per_head = result.packed_q_seq_len // result.block_q_packed
    result.q_blocks = result.block_bkvh * result.q_blocks_per_head
    result.bkvh_blocks = T.ceildiv(result.total_bkvh, result.block_bkvh)
    result.block_num = result.bkvh_blocks
    result.kv_loops = T.ceildiv(kv_seq_len, result.block_kv_seq)
    return result


def _store_prod_workspace(ws):
    """Store producer workspace data: copy acc_s to workspace_2 and metadata to workspace_meta."""
    T.copy(ws.acc_s_ub, ws.acc_s_half)
    T.copy(
        ws.acc_s_half,
        ws.workspace_2[
            ws.cid,
            ws.slot_prod,
            ws.vid * ws.block_q_packed // 2:ws.vid * ws.block_q_packed // 2 + ws.block_q_packed // 2,
            :,
        ],
    )
    for h_i in range(ws.block_q_packed // 2):
        ws.workspace_meta[
            ws.cid,
            ws.slot_prod,
            ws.vid * ws.block_q_packed // 2 + h_i,
            0,
        ] = ws.m_i_prev[h_i]
        ws.workspace_meta[
            ws.cid,
            ws.slot_prod,
            ws.vid * ws.block_q_packed // 2 + h_i,
            1,
        ] = ws.sumexp_i_ub[h_i]
    T.set_cross_flag("MTE3", 1)
