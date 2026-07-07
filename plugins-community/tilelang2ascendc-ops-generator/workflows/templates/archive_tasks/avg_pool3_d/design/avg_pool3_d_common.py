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

"""Shared types and utilities for avg_pool3_d block-level and tile-level designs."""

from collections import namedtuple
from typing import NamedTuple

import tilelang


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


class PoolSpatialWindow(NamedTuple):
    """Spatial dimension parameters for a pooling kernel window."""
    k_h: int
    k_w: int
    s_h: int
    s_w: int
    p_h: int
    p_w: int


class KernelImpls(NamedTuple):
    """Collection of kernel implementation functions for branching dispatch."""
    generic_3d: callable
    reduce_d: callable
    split_c: callable
    split_w: callable
    multi_w: callable


class PoolParams(NamedTuple):
    """Pool kernel parameters for compute_pool_divisor."""
    k_d: int
    k_h: int
    k_w: int
    s_d: int
    s_h: int
    s_w: int
    p_d: int
    p_h: int
    p_w: int
    d_dim: int
    h_dim: int
    w_dim: int
    count_include_pad: int
    divisor_override: int
    pool_size: int


_BlockArgs = namedtuple("_BlockArgs", [
    "n_batch", "c_dim", "d_dim", "h_dim", "w_dim", "out_d", "out_h", "out_w",
    "k_d", "k_h", "k_w", "s_d", "s_h", "s_w", "p_d", "p_h", "p_w",
    "count_include_pad", "divisor_override",
    "block_c", "dtype", "split_mode", "split_w_tile_kw", "multi_w_window_w_num",
    "in_spatial", "out_spatial", "m_out", "block_m", "m_num",
    "used_core_num_m", "tasks_per_core_m", "vec_num", "sub_block_m",
    "pool_size", "hw", "split_block_c", "c_num",
    "total_blocks_mc", "used_core_num_mc", "tasks_per_core_mc",
    "split_w_step", "multi_w_window",
])


def parse_block_args(*args):
    """Parse variadic avg_pool3_d args and compute block-level tiling parameters."""
    a = [*args, 0, "float32", 0, 0, 1]
    (
        n_batch, c_dim, d_dim, h_dim, w_dim, out_d, out_h, out_w,
        k_d, k_h, k_w, s_d, s_h, s_w, p_d, p_h, p_w,
        count_include_pad, divisor_override,
        block_c, dtype, split_mode, split_w_tile_kw, multi_w_window_w_num,
    ) = a[:24]
    in_spatial = d_dim * h_dim * w_dim
    out_spatial = out_d * out_h * out_w
    m_out = n_batch * out_spatial

    block_m = 0
    for candidate in (64, 32, 16, 8, 4, 2):
        if candidate <= m_out and m_out % candidate == 0:
            block_m = candidate
            break
    if block_m == 0:
        raise ValueError(
            f"Unsupported output spatial size: M_out={m_out} is not divisible by any block_M in [2,4,8,16,32,64]"
        )

    m_num = m_out // block_m
    if m_num <= 0:
        raise ValueError(f"m_num must be positive, got {m_num}")

    num_physical_cores = 20
    used_core_num_m = min(num_physical_cores, m_num)
    tasks_per_core_m = (m_num + used_core_num_m - 1) // used_core_num_m

    vec_num = 2
    sub_block_m = block_m // vec_num

    pool_size = k_d * k_h * k_w
    hw = h_dim * w_dim

    split_block_c = block_c if block_c > 0 else 1
    c_num = c_dim // split_block_c
    total_blocks_mc = m_num * c_num
    used_core_num_mc = min(num_physical_cores, total_blocks_mc)
    tasks_per_core_mc = (total_blocks_mc + used_core_num_mc - 1) // used_core_num_mc

    split_w_step = split_w_tile_kw if split_w_tile_kw > 0 else k_w
    multi_w_window = multi_w_window_w_num if multi_w_window_w_num > 0 else 1

    return _BlockArgs(
        n_batch, c_dim, d_dim, h_dim, w_dim, out_d, out_h, out_w,
        k_d, k_h, k_w, s_d, s_h, s_w, p_d, p_h, p_w,
        count_include_pad, divisor_override,
        block_c, dtype, split_mode, split_w_tile_kw, multi_w_window_w_num,
        in_spatial, out_spatial, m_out, block_m, m_num,
        used_core_num_m, tasks_per_core_m, vec_num, sub_block_m,
        pool_size, hw, split_block_c, c_num,
        total_blocks_mc, used_core_num_mc, tasks_per_core_mc,
        split_w_step, multi_w_window,
    )


def decompose_spatial(out_row, out_spatial, out_h, out_w):
    """Decompose a flattened output row index into (n_idx, od, oh, ow)."""
    n_idx = out_row // out_spatial
    out_rem = out_row - n_idx * out_spatial
    od = out_rem // (out_h * out_w)
    od_rem = out_rem - od * (out_h * out_w)
    oh = od_rem // out_w
    ow = od_rem - oh * out_w
    return n_idx, od, oh, ow


def flatten_5d(*args):
    """Flatten a (batch, d, h, w) 5-D coordinate into a row offset."""
    n_idx, in_spatial, id_val, hw, h_val, w_dim, w_val = args
    return n_idx * in_spatial + id_val * hw + h_val * w_dim + w_val


def compute_pool_divisor(od, oh, ow, params: PoolParams):
    """Compute the reciprocal divisor for a pooling window at (od, oh, ow)."""
    if params.divisor_override > 0:
        return 1.0 / params.divisor_override
    if params.count_include_pad > 0:
        return 1.0 / params.pool_size
    d_begin = max(0, od * params.s_d - params.p_d)
    h_begin = max(0, oh * params.s_h - params.p_h)
    w_begin = max(0, ow * params.s_w - params.p_w)
    d_end = min(params.d_dim, od * params.s_d - params.p_d + params.k_d)
    h_end = min(params.h_dim, oh * params.s_h - params.p_h + params.k_h)
    w_end = min(params.w_dim, ow * params.s_w - params.p_w + params.k_w)
    valid = max(0, d_end - d_begin) * max(0, h_end - h_begin) * max(0, w_end - w_begin)
    return 1.0 / valid if valid > 0 else 0.0


def is_reduce_d_fast_path(window: PoolSpatialWindow):
    """Return True if only the depth dimension participates in pooling."""
    return (window.k_h == 1 and window.k_w == 1
            and window.s_h == 1 and window.s_w == 1
            and window.p_h == 0 and window.p_w == 0)


def resolve_kernel_mode(window, impls, **kwargs):
    """Resolve which kernel implementation to use based on input parameters."""
    split_mode = kwargs.get('split_mode', 0)
    split_c_ready = kwargs.get('split_c_ready', False)
    multi_w_window = kwargs.get('multi_w_window', 1)
    default = kwargs.get('default')
    if is_reduce_d_fast_path(window):
        return impls.reduce_d
    if split_mode == 1:
        if split_c_ready:
            return impls.split_c
        raise ValueError("split_c requires divisible channel tiles")
    if split_mode == 2:
        return impls.split_w
    if split_mode == 3:
        return impls.multi_w if multi_w_window > 1 else impls.generic_3d
    if split_c_ready:
        return impls.split_c
    return default
