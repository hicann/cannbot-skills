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

"""Shared config and helpers for gather_elements_v2 block-level and tile-level designs."""

from dataclasses import dataclass
from types import SimpleNamespace

import tilelang
import tilelang.language as T


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@dataclass
class RowProcessCtx:
    bx: ...
    vid: ...
    sub_block_m: ...
    block_m: ...
    m_size: ...
    x_in: ...
    index_in: ...
    y_out: ...
    i_g: ...
    x_ub: ...
    index_ub: ...
    out_ub: ...
    row_map: ... = None


def _kernel_setup(m_size):
    """Select block_m and compute derived grid parameters."""
    block_m = 32
    if m_size < 32:
        block_m = 16
    if m_size < 16:
        block_m = 8
    if m_size < 8:
        block_m = 4
    if m_size < 4:
        block_m = 2
    if m_size < 2:
        block_m = 1

    cfg = SimpleNamespace()
    cfg.block_m = block_m
    cfg.m_num = T.ceildiv(m_size, block_m)
    num_physical_cores = 20
    cfg.used_core_num = min(num_physical_cores, cfg.m_num)
    cfg.tasks_per_core = T.ceildiv(cfg.m_num, cfg.used_core_num)
    vec_num = 2
    cfg.sub_block_m = max(block_m // vec_num, 1)
    return cfg


def _gather_cols(i_g, index_ub, out_ub, x_ub):
    """Gather columns from x_ub into out_ub using index_ub."""
    for col in T.serial(i_g):
        gather_idx = T.cast(index_ub[0, col], "int32")
        out_ub[0, col] = x_ub[0, gather_idx]


def _process_rows_for_bx(ctx):
    """Process rows for a given block index bx and vector id vid."""
    for row in T.serial(ctx.sub_block_m):
        row_idx = ctx.bx * ctx.block_m + ctx.vid * ctx.sub_block_m + row
        if row_idx < ctx.m_size:
            if ctx.row_map is not None:
                x_row = T.cast(ctx.row_map[row_idx], "int32")
            else:
                x_row = row_idx
            T.copy(ctx.x_in[x_row, :], ctx.x_ub)
            T.copy(ctx.index_in[row_idx, :], ctx.index_ub)
            _gather_cols(ctx.i_g, ctx.index_ub, ctx.out_ub, ctx.x_ub)
            T.copy(ctx.out_ub, ctx.y_out[row_idx, :])
