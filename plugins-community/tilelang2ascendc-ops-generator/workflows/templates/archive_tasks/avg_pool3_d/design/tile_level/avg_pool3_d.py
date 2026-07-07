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

import tilelang
import tilelang.language as T

# Tile-level path setup differs from block-level.
_p = Path(__file__).resolve().parent.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
import avg_pool3_d_common as _common  
pass_configs = _common.pass_configs
PoolSpatialWindow = _common.PoolSpatialWindow
KernelImpls = _common.KernelImpls
decompose_spatial = _common.decompose_spatial
flatten_5d = _common.flatten_5d
compute_pool_divisor = _common.compute_pool_divisor
is_reduce_d_fast_path = _common.is_reduce_d_fast_path
resolve_kernel_mode = _common.resolve_kernel_mode
parse_block_args = _common.parse_block_args

# NOTE: Each @T.prim_func below (generic_3d, reduce_d, split_c, split_w, multi_w)
# shares the same pipeline boilerplate (Kernel setup, UB allocation, row/block
# decomposition, spatial-dimension flattening).  This is intentional: T.lang
# prim_funcs compile their body into separate compute graphs and *cannot* share
# DSL code via a Python helper.  The common structure is listed in full for
# clarity of each variant's inner logic.


@tilelang.jit(out_idx=[1], pass_configs=pass_configs)
def avg_pool3_d(*args):
    a = parse_block_args(*args)
    (
        n_batch, c_dim, d_dim, h_dim, w_dim, out_d, out_h, out_w,
        k_d, k_h, k_w, s_d, s_h, s_w, p_d, p_h, p_w,
        count_include_pad, divisor_override,
        block_c, dtype, split_mode, split_w_tile_kw, multi_w_window_w_num,
        in_spatial, out_spatial, m_out, block_m, m_num,
        _, _, vec_num, sub_block_m,
        pool_size, hw, split_block_c, c_num,
        _, _, _,
        split_w_step, multi_w_window,
    ) = a

    def _process_scope(*args, mode="generic"):
        if mode == "split_c":
            x_in, y_out, acc_ub, inp_ub, out_ub, row_base, c_base = args
        else:
            x_in, y_out, acc_ub, inp_ub, out_ub, row_base = args
            c_base = 0
        with T.Scope("V"):
            for m in T.serial(sub_block_m):
                _process_scope_row(
                    x_in, y_out, acc_ub, inp_ub, out_ub,
                    row_base + m, c_base, mode,
                )

    def _process_scope_row(*args):
        x_in, y_out, acc_ub, inp_ub, out_ub, out_row, c_base, mode = args
        n_idx, od, oh, ow = decompose_spatial(out_row, out_spatial, out_h, out_w)
        T.tile.fill(acc_ub, T.float32(0.0))
        for kd in T.serial(k_d):
            if mode == "reduce_d":
                _accumulate_reduce_d_slice(x_in, acc_ub, inp_ub, n_idx, in_spatial, hw, od, oh, ow, kd)
            elif mode == "split_w":
                _accumulate_split_w_slice(x_in, acc_ub, inp_ub, n_idx, in_spatial, hw, od, oh, ow, kd)
            else:
                _accumulate_depth_slice(x_in, acc_ub, inp_ub, n_idx, in_spatial, hw, od, oh, ow, kd, col=c_base)
        if divisor_override > 0:
            T.tile.mul(out_ub, acc_ub, T.float32(1.0 / divisor_override))
        else:
            T.tile.mul(out_ub, acc_ub, T.float32(1.0 / pool_size))
        T.copy(out_ub, y_out[out_row, c_base])

    @T.prim_func
    def generic_3d(
        x_in: T.Tensor((n_batch * in_spatial, c_dim), dtype),
        y_out: T.Tensor((m_out, c_dim), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m
            acc_ub = T.alloc_ub((1, c_dim), dtype)
            inp_ub = T.alloc_ub((1, c_dim), dtype)
            out_ub = T.alloc_ub((1, c_dim), dtype)
            _process_scope(x_in, y_out, acc_ub, inp_ub, out_ub, row_base, mode="generic")

    @T.prim_func
    def reduce_d(
        x_in: T.Tensor((n_batch * in_spatial, c_dim), dtype),
        y_out: T.Tensor((m_out, c_dim), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m
            acc_ub = T.alloc_ub((1, c_dim), dtype)
            inp_ub = T.alloc_ub((1, c_dim), dtype)
            out_ub = T.alloc_ub((1, c_dim), dtype)
            _process_scope(x_in, y_out, acc_ub, inp_ub, out_ub, row_base, mode="reduce_d")

    @T.prim_func
    def split_c(
        x_in: T.Tensor((n_batch * in_spatial, c_dim), dtype),
        y_out: T.Tensor((m_out, c_dim), dtype),
    ):
        with T.Kernel(m_num * c_num, is_npu=True) as (cid, vid):
            bx = cid % m_num
            bc = cid // m_num
            row_base = bx * block_m + vid * sub_block_m
            c_base = bc * split_block_c
            acc_ub = T.alloc_ub((1, split_block_c), dtype)
            inp_ub = T.alloc_ub((1, split_block_c), dtype)
            out_ub = T.alloc_ub((1, split_block_c), dtype)
            _process_scope(x_in, y_out, acc_ub, inp_ub, out_ub, row_base, c_base, mode="split_c")

    @T.prim_func
    def split_w(
        x_in: T.Tensor((n_batch * in_spatial, c_dim), dtype),
        y_out: T.Tensor((m_out, c_dim), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m
            acc_ub = T.alloc_ub((1, c_dim), dtype)
            inp_ub = T.alloc_ub((1, c_dim), dtype)
            out_ub = T.alloc_ub((1, c_dim), dtype)
            _process_scope(x_in, y_out, acc_ub, inp_ub, out_ub, row_base, mode="split_w")

    def _accumulate_reduce_d_slice(*args, col=0):
        x_in, acc_ub, inp_ub, n_idx, in_spatial, hw, od, oh, ow, kd = args
        id_val = od * s_d - p_d + kd
        if 0 <= id_val < d_dim:
            in_row = flatten_5d(n_idx, in_spatial, id_val, hw, oh, w_dim, ow)
            T.copy(x_in[in_row, col], inp_ub)
            T.tile.add(acc_ub, acc_ub, inp_ub)

    def _accumulate_width_slice(*args, col=0):
        x_in, acc_ub, inp_ub, n_idx, in_spatial, hw, id_val, ih_val, cur_ow = args
        for kw in T.serial(k_w):
            iw_val = cur_ow * s_w - p_w + kw
            if 0 <= iw_val < w_dim:
                in_row = flatten_5d(
                    n_idx, in_spatial,
                    id_val, hw, ih_val,
                    w_dim, iw_val)
                T.copy(x_in[in_row, col], inp_ub)
                T.tile.add(acc_ub, acc_ub, inp_ub)

    def _accumulate_depth_slice(*args, col=0):
        x_in, acc_ub, inp_ub, n_idx, in_spatial, hw, od, oh, cur_ow, kd = args
        id_val = od * s_d - p_d + kd
        if 0 <= id_val < d_dim:
            for kh in T.serial(k_h):
                ih_val = oh * s_h - p_h + kh
                if 0 <= ih_val < h_dim:
                    _accumulate_width_slice(x_in, acc_ub, inp_ub, n_idx, in_spatial, hw,
                                            id_val, ih_val, cur_ow, col=col)

    def _accumulate_split_w_element(*args):
        x_in, acc_ub, inp_ub, n_idx, in_spatial, hw, id_val, ih_val, iw_val = args
        if 0 <= iw_val < w_dim:
            in_row = flatten_5d(
                n_idx, in_spatial,
                id_val, hw, ih_val,
                w_dim, iw_val)
            T.copy(x_in[in_row, 0], inp_ub)
            T.tile.add(acc_ub, acc_ub, inp_ub)

    def _accumulate_split_w_chunk(*args):
        x_in, acc_ub, inp_ub, n_idx, in_spatial, hw, id_val, ih_val, ow = args
        for kw_base in T.serial((k_w + split_w_step - 1) // split_w_step):
            for kw_local in T.serial(split_w_step):
                kw = kw_base * split_w_step + kw_local
                if kw < k_w:
                    iw_val = ow * s_w - p_w + kw
                    _accumulate_split_w_element(x_in, acc_ub, inp_ub, n_idx, in_spatial, hw,
                                                id_val, ih_val, iw_val)

    def _accumulate_split_w_slice(*args):
        x_in, acc_ub, inp_ub, n_idx, in_spatial, hw, od, oh, ow, kd = args
        id_val = od * s_d - p_d + kd
        if 0 <= id_val < d_dim:
            for kh in T.serial(k_h):
                ih_val = oh * s_h - p_h + kh
                if 0 <= ih_val < h_dim:
                    _accumulate_split_w_chunk(x_in, acc_ub, inp_ub, n_idx, in_spatial, hw,
                                              id_val, ih_val, ow)

    def _accumulate_window(*args):
        x_in, acc_ub, inp_ub, out_ub, y_out, out_row, local_w, n_idx, in_spatial, hw, od, oh, cur_ow = args
        T.tile.fill(acc_ub, T.float32(0.0))
        for kd in T.serial(k_d):
            _accumulate_depth_slice(x_in, acc_ub, inp_ub, n_idx, in_spatial, hw, od, oh, cur_ow, kd)
        divisor = compute_pool_divisor(
            od, oh, cur_ow, k_d, k_h, k_w, s_d, s_h, s_w, p_d, p_h, p_w,
            d_dim, h_dim, w_dim, count_include_pad, divisor_override, pool_size)
        if divisor > 0:
            T.tile.mul(out_ub, acc_ub, T.float32(divisor))
        else:
            T.tile.fill(out_ub, T.float32(0.0))
        T.copy(out_ub, y_out[out_row + local_w, 0])

    def _process_multi_w_aligned(*args):
        x_in, y_out, acc_ub, inp_ub, out_ub, out_row, n_idx, in_spatial, hw, od, oh, ow, m, sub_block_m = args
        for local_w in T.serial(multi_w_window):
            if local_w < sub_block_m - m and local_w < out_w - ow:
                _accumulate_window(
                    x_in, acc_ub, inp_ub, out_ub, y_out, out_row, local_w,
                    n_idx, in_spatial, hw, od, oh, ow + local_w)

    def _process_multi_w_prefix(*args):
        x_in, y_out, acc_ub, inp_ub, out_ub, out_row, n_idx, in_spatial, hw, od, oh, ow, m, sub_block_m, prefix_len = (
            args
        )
        for local_w in T.serial(multi_w_window):
            if local_w < prefix_len and local_w < sub_block_m - m and local_w < out_w - ow:
                _accumulate_window(
                    x_in, acc_ub, inp_ub, out_ub, y_out, out_row, local_w,
                    n_idx, in_spatial, hw, od, oh, ow + local_w)

    def _process_multi_w_branch(*args):
        x_in, y_out, acc_ub, inp_ub, out_ub, out_row, n_idx, in_spatial, hw, od, oh, ow, m, sub_block_m = args
        if multi_w_window > 1 and ow % multi_w_window == 0:
            _process_multi_w_aligned(x_in, y_out, acc_ub, inp_ub, out_ub, out_row,
                                     n_idx, in_spatial, hw, od, oh, ow, m, sub_block_m)
        elif m == 0:
            prefix_len = multi_w_window - (ow % multi_w_window)
            _process_multi_w_prefix(x_in, y_out, acc_ub, inp_ub, out_ub, out_row,
                                    n_idx, in_spatial, hw, od, oh, ow, m, sub_block_m, prefix_len)

    def _process_multi_w_row(*args):
        x_in, y_out, acc_ub, inp_ub, out_ub, row_base = args
        with T.Scope("V"):
            for m in T.serial(sub_block_m):
                out_row = row_base + m
                n_idx, od, oh, ow = decompose_spatial(out_row, out_spatial, out_h, out_w)
                _process_multi_w_branch(x_in, y_out, acc_ub, inp_ub, out_ub, out_row,
                                        n_idx, in_spatial, hw, od, oh, ow, m, sub_block_m)

    @T.prim_func
    def multi_w(
        x_in: T.Tensor((n_batch * in_spatial, c_dim), dtype),
        y_out: T.Tensor((m_out, c_dim), dtype),
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            bx = cid
            row_base = bx * block_m + vid * sub_block_m
            acc_ub = T.alloc_ub((1, c_dim), dtype)
            inp_ub = T.alloc_ub((1, c_dim), dtype)
            out_ub = T.alloc_ub((1, c_dim), dtype)
            _process_multi_w_row(x_in, y_out, acc_ub, inp_ub, out_ub, row_base)

    split_c_ready = block_c > 0 and c_dim % block_c == 0
    window = PoolSpatialWindow(k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w, p_h=p_h, p_w=p_w)
    impls = KernelImpls(generic_3d=generic_3d, reduce_d=reduce_d,
                        split_c=split_c, split_w=split_w, multi_w=multi_w)
    return resolve_kernel_mode(split_mode, split_c_ready, multi_w_window,
                                window, impls, default=impls.generic_3d)
