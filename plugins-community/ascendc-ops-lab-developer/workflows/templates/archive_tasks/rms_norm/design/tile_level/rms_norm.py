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

"""Unified TileLang design for RMSNorm.

This file keeps three specialized prim_funcs:
- merge_n: preferred when N <= 1024
- single_row: preferred when 1024 < N <= 8192
- splitd: preferred when N > 8192

The kernel uses one external dtype for input/output tensors and accumulates in
float32 internally, matching the AscendC interface.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs)
def rms_norm(m_size, n_size, eps=1e-5, dtype="float32"):
    block_m = 64
    block_n = 1024
    num_physical_cores = 20
    m_num = T.ceildiv(m_size, block_m)
    used_core_num = min(num_physical_cores, m_num)
    tasks_per_core = T.ceildiv(m_num, used_core_num)
    vec_num = 2
    sub_block_m = block_m // vec_num

    row_factor = 8
    row_loops = T.ceildiv(sub_block_m, row_factor)
    n_num = T.ceildiv(n_size, block_n)
    need_cast = dtype != "float32"
    out_cast_mode = "CAST_ROUND" if dtype == "bfloat16" else "CAST_NONE"

    eps_const = T.float32(eps)
    inv_n_const = T.float32(1.0 / n_size)

    def _load_with_cast(src_slice, typed_ub, float_ub, count):
        if need_cast:
            T.copy(src_slice, typed_ub)
            T.tile.cast(float_ub, typed_ub, mode="CAST_NONE", count=count)
        else:
            T.copy(src_slice, float_ub)

    def _store_inv_rms_with_cast(float_ub, typed_ub, dst_slice, count):
        if need_cast:
            T.tile.cast(typed_ub, float_ub, mode=out_cast_mode, count=count)
            T.copy(typed_ub[:, 0], dst_slice)
        else:
            T.copy(float_ub[:, 0], dst_slice)

    def _store_y_with_cast(float_ub, typed_ub, dst_slice, count):
        if need_cast:
            T.tile.cast(typed_ub, float_ub, mode=out_cast_mode, count=count)
            T.copy(typed_ub, dst_slice)
        else:
            T.copy(float_ub, dst_slice)

    @T.prim_func
    def merge_n(  
        x_in: T.Tensor((m_size, n_size), dtype),
        gamma: T.Tensor((n_size,), dtype),
        y_out: T.Tensor((m_size, n_size), dtype),
        inv_rms: T.Tensor((m_size,), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            gamma_in_ub = T.alloc_ub((1, n_size), dtype)
            x_in_rows_ub = T.alloc_ub((row_factor, n_size), dtype)
            out_cast_rows_ub = T.alloc_ub((row_factor, n_size), dtype)
            inv_rms_cast_ub = T.alloc_ub((row_factor, 1), dtype)
            single_x_in_row_ub = T.alloc_ub((1, n_size), dtype)
            single_out_cast_row_ub = T.alloc_ub((1, n_size), dtype)
            single_inv_rms_cast_ub = T.alloc_ub((1, 1), dtype)
            x_ub = T.alloc_ub((row_factor, n_size), "float32")
            x_sq_ub = T.alloc_ub((row_factor, n_size), "float32")
            gamma_ub = T.alloc_ub((1, n_size), "float32")
            gamma_broad_ub = T.alloc_ub((row_factor, n_size), "float32")
            sum_sq_ub = T.alloc_ub((row_factor, 1), "float32")
            inv_rms_ub = T.alloc_ub((row_factor, 1), "float32")
            rstd_broad_ub = T.alloc_ub((row_factor, n_size), "float32")
            out_ub = T.alloc_ub((row_factor, n_size), "float32")

            single_x_ub = T.alloc_ub((1, n_size), "float32")
            single_x_sq_ub = T.alloc_ub((1, n_size), "float32")
            single_inv_rms_ub = T.alloc_ub((1, 1), "float32")
            single_out_ub = T.alloc_ub((1, n_size), "float32")

            inv_n_ub = T.alloc_ub((row_factor, 1), "float32")
            eps_ub = T.alloc_ub((row_factor, 1), "float32")

            reduce_tmp = T.alloc_ub((2 * row_factor * n_size,), "uint8")
            gamma_bcast_tmp = T.alloc_ub((2 * row_factor, n_size), "uint8")
            rstd_bcast_tmp = T.alloc_ub((2 * row_factor, n_size), "uint8")
            single_reduce_tmp = T.alloc_ub((2 * n_size,), "uint8")

            def _process_partial_rows(row_base):
                for rr in T.serial(row_factor):
                    row_idx = row_base + rr
                    if row_idx < m_size:
                        _load_with_cast(x_in[row_idx, :], single_x_in_row_ub, single_x_ub, n_size)
                        T.tile.mul(single_x_sq_ub, single_x_ub, single_x_ub)
                        T.reduce_sum(single_x_sq_ub, single_x_sq_ub[:, 0], single_reduce_tmp, dim=-1)
                        single_sum_sq = single_x_sq_ub[0, 0] * inv_n_const + eps_const
                        single_x_sq_ub[0, 0] = single_sum_sq
                        T.tile.rsqrt(single_inv_rms_ub[:, 0], single_x_sq_ub[:, 0])
                        single_rstd = single_inv_rms_ub[0, 0]
                        T.tile.mul(single_out_ub, single_x_ub, single_rstd)
                        T.tile.mul(single_out_ub, single_out_ub, gamma_ub)
                        _store_inv_rms_with_cast(single_inv_rms_ub, single_inv_rms_cast_ub,
                                                inv_rms[row_idx:row_idx + 1], 1)
                        _store_y_with_cast(single_out_ub, single_out_cast_row_ub, y_out[row_idx, :], n_size)

            def _process_rows_for_bx(bx, vid):
                for r in T.serial(row_loops):
                    row_base = bx * block_m + vid * sub_block_m + r * row_factor
                    if row_base + row_factor <= m_size:
                        _load_with_cast(x_in[row_base:row_base + row_factor, :],
                                        x_in_rows_ub, x_ub, row_factor * n_size)
                        T.tile.mul(x_sq_ub, x_ub, x_ub)
                        T.reduce_sum(x_sq_ub, sum_sq_ub, reduce_tmp, dim=-1)
                        T.tile.mul(sum_sq_ub, sum_sq_ub, inv_n_ub)
                        T.tile.add(sum_sq_ub, sum_sq_ub, eps_ub)
                        T.tile.rsqrt(inv_rms_ub, sum_sq_ub)
                        _store_inv_rms_with_cast(inv_rms_ub, inv_rms_cast_ub,
                                                inv_rms[row_base:row_base + row_factor], row_factor)
                        T.tile.broadcast(rstd_broad_ub, inv_rms_ub, rstd_bcast_tmp)
                        T.tile.mul(out_ub, x_ub, rstd_broad_ub)
                        T.tile.mul(out_ub, out_ub, gamma_broad_ub)
                        _store_y_with_cast(out_ub, out_cast_rows_ub,
                                           y_out[row_base:row_base + row_factor, :],
                                           row_factor * n_size)
                    else:
                        _process_partial_rows(row_base)

            def _process_all_tasks():
                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        _process_rows_for_bx(bx, vid)

            with T.Scope("V"):
                _load_with_cast(gamma[0], gamma_in_ub, gamma_ub, n_size)
                T.tile.broadcast(gamma_broad_ub, gamma_ub, gamma_bcast_tmp)
                T.tile.fill(inv_n_ub, inv_n_const)
                T.tile.fill(eps_ub, eps_const)
                _process_all_tasks()

    @T.prim_func
    def single_row(  
        x_in: T.Tensor((m_size, n_size), dtype),
        gamma: T.Tensor((n_size,), dtype),
        y_out: T.Tensor((m_size, n_size), dtype),
        inv_rms: T.Tensor((m_size,), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            gamma_in_row_ub = T.alloc_ub((1, n_size), dtype)
            x_in_row_ub = T.alloc_ub((1, n_size), dtype)
            out_cast_row_ub = T.alloc_ub((1, n_size), dtype)
            inv_rms_cast_ub = T.alloc_ub((1, 1), dtype)
            x_ub = T.alloc_ub((1, n_size), "float32")
            x_sq_ub = T.alloc_ub((1, n_size), "float32")
            gamma_ub = T.alloc_ub((1, n_size), "float32")
            inv_rms_ub = T.alloc_ub((1, 1), "float32")
            out_ub = T.alloc_ub((1, n_size), "float32")

            reduce_tmp = T.alloc_ub((2 * n_size,), "uint8")

            def _process_rows_for_bx(bx, vid):
                for row in T.serial(sub_block_m):
                    row_idx = bx * block_m + vid * sub_block_m + row
                    if row_idx < m_size:
                        _load_with_cast(x_in[row_idx, :], x_in_row_ub, x_ub, n_size)
                        T.tile.mul(x_sq_ub, x_ub, x_ub)
                        T.reduce_sum(x_sq_ub, x_sq_ub[:, 0], reduce_tmp, dim=-1)
                        sum_sq = x_sq_ub[0, 0] * inv_n_const + eps_const
                        x_sq_ub[0, 0] = sum_sq
                        T.tile.rsqrt(inv_rms_ub[:, 0], x_sq_ub[:, 0])
                        inv_rms_val = inv_rms_ub[0, 0]
                        T.tile.mul(out_ub, x_ub, inv_rms_val)
                        T.tile.mul(out_ub, out_ub, gamma_ub)
                        _store_inv_rms_with_cast(inv_rms_ub, inv_rms_cast_ub, inv_rms[row_idx:row_idx + 1], 1)
                        _store_y_with_cast(out_ub, out_cast_row_ub, y_out[row_idx, :], n_size)

            def _process_all_tasks():
                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        _process_rows_for_bx(bx, vid)

            with T.Scope("V"):
                _load_with_cast(gamma[0], gamma_in_row_ub, gamma_ub, n_size)
                _process_all_tasks()

    def _load_x_tile(x_in, row_idx, x_in_ub, x_ub, by):
        """Load one tile of X from GM into UB, with optional cast to float32."""
        col_base = by * block_n
        valid_n = T.if_then_else(col_base + block_n <= n_size, block_n, n_size - col_base)
        if need_cast:
            T.copy(
                x_in[row_idx:row_idx + 1, col_base:col_base + valid_n],
                x_in_ub[:, 0:valid_n],
            )
            T.tile.cast(x_ub, x_in_ub, mode="CAST_NONE", count=valid_n)
        else:
            T.copy(
                x_in[row_idx:row_idx + 1, col_base:col_base + valid_n],
                x_ub[:, 0:valid_n],
            )
        return col_base, valid_n

    @T.prim_func
    def splitd(  
        x_in: T.Tensor((m_size, n_size), dtype),
        gamma: T.Tensor((n_size,), dtype),
        y_out: T.Tensor((m_size, n_size), dtype),
        inv_rms: T.Tensor((m_size,), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            x_in_ub = T.alloc_ub((1, block_n), dtype)
            gamma_in_ub = T.alloc_ub((1, block_n), dtype)
            x_ub = T.alloc_ub((1, block_n), "float32")
            x_sq_ub = T.alloc_ub((1, block_n), "float32")
            gamma_ub = T.alloc_ub((1, block_n), "float32")
            inv_rms_ub = T.alloc_ub((1, 1), "float32")
            inv_rms_cast_ub = T.alloc_ub((1, 1), dtype)
            out_ub = T.alloc_ub((1, block_n), "float32")
            out_cast_ub = T.alloc_ub((1, block_n), dtype)

            reduce_tmp = T.alloc_ub((2 * block_n,), "uint8")

            def _process_pass1(row_idx):
                T.tile.fill(out_ub, T.float32(0))
                for by in T.serial(n_num):
                    col_base, valid_n = _load_x_tile(x_in, row_idx, x_in_ub, x_ub, by)
                    T.tile.mul(x_sq_ub[:, 0:valid_n], x_ub[:, 0:valid_n], x_ub[:, 0:valid_n])
                    T.tile.add(out_ub[:, 0:valid_n], out_ub[:, 0:valid_n], x_sq_ub[:, 0:valid_n])

            def _process_pass2(row_idx, inv_rms_val):
                for by in T.serial(n_num):
                    col_base, valid_n = _load_x_tile(x_in, row_idx, x_in_ub, x_ub, by)
                    if need_cast:
                        T.copy(gamma[col_base:col_base + valid_n], gamma_in_ub[0, 0:valid_n])
                        T.tile.cast(gamma_ub, gamma_in_ub, mode="CAST_NONE", count=valid_n)
                    else:
                        T.copy(gamma[col_base:col_base + valid_n], gamma_ub[0, 0:valid_n])
                    T.tile.mul(out_ub[:, 0:valid_n], x_ub[:, 0:valid_n], inv_rms_val)
                    T.tile.mul(out_ub[:, 0:valid_n], out_ub[:, 0:valid_n], gamma_ub[:, 0:valid_n])
                    if need_cast:
                        T.tile.cast(out_cast_ub, out_ub, mode=out_cast_mode, count=valid_n)
                        T.copy(out_cast_ub[:, 0:valid_n], y_out[row_idx:row_idx + 1, col_base:col_base + valid_n])
                    else:
                        T.copy(out_ub[:, 0:valid_n], y_out[row_idx:row_idx + 1, col_base:col_base + valid_n])

            def _process_rows_for_bx(bx, vid):
                for row in T.serial(sub_block_m):
                    row_idx = bx * block_m + vid * sub_block_m + row
                    if row_idx < m_size:
                        _process_pass1(row_idx)
                        T.reduce_sum(out_ub, out_ub[:, 0], reduce_tmp, dim=-1)
                        x_sq_ub[0, 0] = out_ub[0, 0] * inv_n_const + eps_const
                        T.tile.rsqrt(inv_rms_ub[:, 0], x_sq_ub[:, 0])
                        inv_rms_val = inv_rms_ub[0, 0]
                        _store_inv_rms_with_cast(inv_rms_ub, inv_rms_cast_ub, inv_rms[row_idx:row_idx + 1], 1)
                        _process_pass2(row_idx, inv_rms_val)

            def _process_all_tasks():
                for local_idx in T.serial(tasks_per_core):
                    bx = cid * tasks_per_core + local_idx
                    if bx < m_num:
                        _process_rows_for_bx(bx, vid)

            with T.Scope("V"):
                _process_all_tasks()

    if n_size <= 1024:
        return merge_n
    if n_size > 8192:
        return splitd
    return single_row
