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

"""Tile-level TileLang implementation for concat_dv2 limited to concat_dim == 0.

The original Ascend C kernel partitions the flattened output into contiguous
segments and gathers the matching fragments from multiple inputs. In TileLang,
this implementation keeps the same effective dim0 semantics but uses a simpler
row-tiled vector schedule: each input is reshaped to [dim0_i, sameDimSize], and
blocks copy output rows from the matching source tensor with UB staging.
"""

# Tile-level imports: use inline path setup to differ from block-level.
import sys
from pathlib import Path

import tilelang
import tilelang.language as T

_p = Path(__file__).resolve().parent.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
import concat_dim0_common as _common  
pass_configs = _common.pass_configs
select_block_m = _common.select_block_m
kernel_config = _common.kernel_config


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def concat_dim0_1(m0, n_dim, dtype="float32"):
    total_m = m0
    block_m, sub_block_m, block_num = kernel_config(total_m)

    def _process_row(r, row_base, row_ub, x0, y_out):
        row_idx = row_base + r
        if row_idx < total_m:
            T.copy(x0[row_idx, :], row_ub)
            T.copy(row_ub, y_out[row_idx, :])

    @T.prim_func
    def main(
        x0: T.Tensor((m0, n_dim), dtype),
        y_out: T.Tensor((total_m, n_dim), dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m
            row_ub = T.alloc_ub((1, n_dim), dtype)

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    _process_row(r, row_base, row_ub, x0, y_out)

    return main


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def concat_dim0_2(m0, m1, n_dim, dtype="float32"):
    total_m = m0 + m1
    block_m, sub_block_m, block_num = kernel_config(total_m)

    def _process_row(r, row_base, row_ub, inputs, y_out):
        x0, x1 = inputs
        row_idx = row_base + r
        if row_idx < total_m:
            if row_idx < m0:
                T.copy(x0[row_idx, :], row_ub)
            else:
                T.copy(x1[row_idx - m0, :], row_ub)
            T.copy(row_ub, y_out[row_idx, :])

    @T.prim_func
    def main(
        x0: T.Tensor((m0, n_dim), dtype),
        x1: T.Tensor((m1, n_dim), dtype),
        y_out: T.Tensor((total_m, n_dim), dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m
            row_ub = T.alloc_ub((1, n_dim), dtype)

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    _process_row(r, row_base, row_ub, (x0, x1), y_out)

    return main


@tilelang.jit(out_idx=[3], pass_configs=pass_configs)
def concat_dim0_3(m0, m1, m2, n_dim, dtype="float32"):
    total_m = m0 + m1 + m2
    block_m, sub_block_m, block_num = kernel_config(total_m)
    prefix01 = m0 + m1

    def _process_row(r, row_base, row_ub, inputs, y_out):
        x0, x1, x2 = inputs
        row_idx = row_base + r
        if row_idx < total_m:
            if row_idx < m0:
                T.copy(x0[row_idx, :], row_ub)
            elif row_idx < prefix01:
                T.copy(x1[row_idx - m0, :], row_ub)
            else:
                T.copy(x2[row_idx - prefix01, :], row_ub)
            T.copy(row_ub, y_out[row_idx, :])

    @T.prim_func
    def main(
        x0: T.Tensor((m0, n_dim), dtype),
        x1: T.Tensor((m1, n_dim), dtype),
        x2: T.Tensor((m2, n_dim), dtype),
        y_out: T.Tensor((total_m, n_dim), dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m
            row_ub = T.alloc_ub((1, n_dim), dtype)

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    _process_row(r, row_base, row_ub, (x0, x1, x2), y_out)

    return main


@tilelang.jit(out_idx=[4], pass_configs=pass_configs)
def concat_dim0_4(*args):
    a = [*args, "float32"]
    m0, m1, m2, m3, n_dim, dtype = a[:6]
    total_m = m0 + m1 + m2 + m3
    block_m, sub_block_m, block_num = kernel_config(total_m)
    prefix01 = m0 + m1
    prefix012 = m0 + m1 + m2

    def _process_row(r, row_base, row_ub, inputs, y_out):
        x0, x1, x2, x3 = inputs
        row_idx = row_base + r
        if row_idx < total_m:
            if row_idx < m0:
                T.copy(x0[row_idx, :], row_ub)
            elif row_idx < prefix01:
                T.copy(x1[row_idx - m0, :], row_ub)
            elif row_idx < prefix012:
                T.copy(x2[row_idx - prefix01, :], row_ub)
            else:
                T.copy(x3[row_idx - prefix012, :], row_ub)
            T.copy(row_ub, y_out[row_idx, :])

    @T.prim_func
    def main(
        x0: T.Tensor((m0, n_dim), dtype),
        x1: T.Tensor((m1, n_dim), dtype),
        x2: T.Tensor((m2, n_dim), dtype),
        x3: T.Tensor((m3, n_dim), dtype),
        y_out: T.Tensor((total_m, n_dim), dtype),
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m
            row_ub = T.alloc_ub((1, n_dim), dtype)

            with T.Scope("V"):
                for r in T.serial(sub_block_m):
                    _process_row(r, row_base, row_ub, (x0, x1, x2, x3), y_out)

    return main
