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

"""Block-level TileLang design for avg_pool3_d.

This file mirrors tile-level mode selection and scheduling, but keeps only
coarse-grained block partition and Vector-side pipeline skeleton.
Fine-grained compute details are intentionally left as TODO comments.
"""

import sys
from pathlib import Path

import tilelang
import tilelang.language as T

# Import shared types and utilities from the common module.
_COMMON_DIR = Path(__file__).resolve().parent.parent
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from avg_pool3_d_common import (  
    pass_configs,
    PoolSpatialWindow,
    KernelImpls,
    decompose_spatial,
    is_reduce_d_fast_path,
    resolve_kernel_mode,
    parse_block_args,
)


# NOTE: Each @T.prim_func below (generic_3d, reduce_d, split_c, split_w, multi_w)
# shares the same pipeline boilerplate (Kernel setup, row/block decomposition,
# spatial-dimension flattening).  This is intentional: T.lang prim_funcs compile
# their body into separate compute graphs and *cannot* share DSL code via a
# Python helper.  The common structure is listed in full for clarity of each
# variant's inner logic.


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def avg_pool3_d(*args):
    (
        n_batch, c_dim, d_dim, h_dim, w_dim, out_d, out_h, out_w,
        k_d, k_h, k_w, s_d, s_h, s_w, p_d, p_h, p_w,
        count_include_pad, divisor_override,
        block_c, dtype, split_mode, split_w_tile_kw, multi_w_window_w_num,
        in_spatial, out_spatial, m_out, block_m, m_num,
        used_core_num_m, tasks_per_core_m, vec_num, sub_block_m,
        pool_size, hw, split_block_c, c_num,
        total_blocks_mc, used_core_num_mc, tasks_per_core_mc,
        split_w_step, multi_w_window,
    ) = parse_block_args(*args)

    def _process_rows(x_in, y_out, row_base, sub_block_m):
        for m in T.serial(sub_block_m):
            out_row = row_base + m
            n_idx, od, oh, ow = decompose_spatial(out_row, out_spatial, out_h, out_w)
            _ = (x_in, y_out, n_idx, od, oh, ow)

    def _process_split_c_rows(x_in, y_out, row_base, c_base, sub_block_m):
        for m in T.serial(sub_block_m):
            out_row = row_base + m
            n_idx, od, oh, ow = decompose_spatial(out_row, out_spatial, out_h, out_w)
            _ = (x_in, y_out, c_base, n_idx, od, oh, ow)

    def _scope_process_rows(x_in, y_out, bx, vid):
        with T.Scope("V"):
            if bx < m_num:
                _process_rows(x_in, y_out, bx * block_m + vid * sub_block_m, sub_block_m)

    def _scope_process_split_c_rows(*args):
        x_in, y_out, task_id, bx, bc, vid = args
        with T.Scope("V"):
            if task_id < total_blocks_mc:
                _process_split_c_rows(x_in, y_out, bx * block_m + vid * sub_block_m,
                                      bc * split_block_c, sub_block_m)

    def _scope_process_multi_w_rows(x_in, y_out, bx, vid):
        with T.Scope("V"):
            if bx < m_num:
                _process_multi_w_rows(x_in, y_out, bx * block_m + vid * sub_block_m, sub_block_m)

    def _process_multi_w_aligned(*args):
        x_in, y_out, n_idx, od, oh, ow, m, sub_block_m = args
        for local_w in T.serial(multi_w_window):
            if local_w < sub_block_m - m and local_w < out_w - ow:
                _ = (x_in, y_out, n_idx, od, oh, ow + local_w)

    def _process_multi_w_prefix(*args):
        x_in, y_out, n_idx, od, oh, ow, m, sub_block_m, prefix_len = args
        for local_w in T.serial(multi_w_window):
            if local_w < prefix_len and local_w < sub_block_m - m and local_w < out_w - ow:
                _ = (x_in, y_out, n_idx, od, oh, ow + local_w)

    def _process_multi_w_rows(x_in, y_out, row_base, sub_block_m):
        for m in T.serial(sub_block_m):
            out_row = row_base + m
            n_idx, od, oh, ow = decompose_spatial(out_row, out_spatial, out_h, out_w)
            if multi_w_window > 1 and ow % multi_w_window == 0:
                _process_multi_w_aligned(x_in, y_out, n_idx, od, oh, ow, m, sub_block_m)
            elif m == 0:
                prefix_len = multi_w_window - (ow % multi_w_window)
                _process_multi_w_prefix(x_in, y_out, n_idx, od, oh, ow, m, sub_block_m, prefix_len)

    @T.prim_func
    def generic_3d(
        x_in: T.Tensor((n_batch * in_spatial, c_dim), dtype),
        y_out: T.Tensor((m_out, c_dim), dtype),
    ):
        with T.Kernel(used_core_num_m, is_npu=True) as (cid, vid):
            core_idx = cid
            for local_idx in T.serial(tasks_per_core_m):
                bx = core_idx * tasks_per_core_m + local_idx
                _scope_process_rows(x_in, y_out, bx, vid)

    @T.prim_func
    def reduce_d(
        x_in: T.Tensor((n_batch * in_spatial, c_dim), dtype),
        y_out: T.Tensor((m_out, c_dim), dtype),
    ):
        with T.Kernel(used_core_num_m, is_npu=True) as (cid, vid):
            core_idx = cid
            for local_idx in T.serial(tasks_per_core_m):
                bx = core_idx * tasks_per_core_m + local_idx
                _scope_process_rows(x_in, y_out, bx, vid)

    @T.prim_func
    def split_c(
        x_in: T.Tensor((n_batch * in_spatial, c_dim), dtype),
        y_out: T.Tensor((m_out, c_dim), dtype),
    ):
        with T.Kernel(used_core_num_mc, is_npu=True) as (cid, vid):
            core_idx = cid
            for local_idx in T.serial(tasks_per_core_mc):
                task_id = core_idx * tasks_per_core_mc + local_idx
                bc = task_id // m_num
                bx = task_id % m_num
                _scope_process_split_c_rows(x_in, y_out, task_id, bx, bc, vid)

    @T.prim_func
    def split_w(
        x_in: T.Tensor((n_batch * in_spatial, c_dim), dtype),
        y_out: T.Tensor((m_out, c_dim), dtype),
    ):
        with T.Kernel(used_core_num_m, is_npu=True) as (cid, vid):
            core_idx = cid
            for local_idx in T.serial(tasks_per_core_m):
                bx = core_idx * tasks_per_core_m + local_idx
                _scope_process_rows(x_in, y_out, bx, vid)

    @T.prim_func
    def multi_w(
        x_in: T.Tensor((n_batch * in_spatial, c_dim), dtype),
        y_out: T.Tensor((m_out, c_dim), dtype),
    ):
        with T.Kernel(used_core_num_m, is_npu=True) as (cid, vid):
            core_idx = cid
            for local_idx in T.serial(tasks_per_core_m):
                bx = core_idx * tasks_per_core_m + local_idx
                _scope_process_multi_w_rows(x_in, y_out, bx, vid)

    split_c_ready = block_c > 0 and c_dim % block_c == 0
    window = PoolSpatialWindow(k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w, p_h=p_h, p_w=p_w)
    impls = KernelImpls(generic_3d=generic_3d, reduce_d=reduce_d,
                        split_c=split_c, split_w=split_w, multi_w=multi_w)
    return resolve_kernel_mode(window, impls,
                                split_mode=split_mode, split_c_ready=split_c_ready,
                                multi_w_window=multi_w_window)
