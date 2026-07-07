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

"""Shared config and helpers for flash_attention block-level and tile-level designs."""

from collections import namedtuple
from dataclasses import dataclass

import tilelang
import tilelang.language as T

KernelConfig = namedtuple("KernelConfig", [
    "block_m", "block_n", "prelaunch", "ring_slots",
    "dtype", "accum_dtype", "shape", "block_num", "kv_loops",
])

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
}


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
    block_m: ...
    slot_prod: ...


def _kernel_setup(batch, seq_len, heads, dim):
    """Return common block/tile sizing and scheduling parameters."""
    block_m, block_n = 64, 64
    prelaunch = 2
    ring_slots = prelaunch + 1
    dtype = "float16"
    accum_dtype = "float"
    shape = [batch, heads, seq_len, dim]
    block_num = seq_len // block_m * heads * batch
    kv_loops = T.ceildiv(seq_len, block_n)
    return KernelConfig(block_m, block_n, prelaunch, ring_slots,
                        dtype, accum_dtype, shape, block_num, kv_loops)


def _kernel_indices(cid, seq_len, block_m, heads, batch):
    """Decompose a linear kernel instance index into (bx, by, bz)."""
    bx = cid % (seq_len // block_m)
    by = cid // (seq_len // block_m) % heads
    bz = cid // (seq_len // block_m) // heads % batch
    return bx, by, bz


def _store_prod_workspace(ws):
    """Store producer workspace data: copy acc_s to workspace_2 and metadata to workspace_meta."""
    T.copy(ws.acc_s_ub, ws.acc_s_half)
    T.copy(
        ws.acc_s_half,
        ws.workspace_2[
            ws.cid,
            ws.slot_prod,
            ws.vid * ws.block_m // 2:ws.vid * ws.block_m // 2 + ws.block_m // 2,
            :,
        ],
    )
    for h_i in range(ws.block_m // 2):
        ws.workspace_meta[
            ws.cid,
            ws.slot_prod,
            ws.vid * ws.block_m // 2 + h_i,
            0,
        ] = ws.m_i_prev[h_i]
        ws.workspace_meta[
            ws.cid,
            ws.slot_prod,
            ws.vid * ws.block_m // 2 + h_i,
            1,
        ] = ws.sumexp_i_ub[h_i]
    T.set_cross_flag("MTE3", 1)
