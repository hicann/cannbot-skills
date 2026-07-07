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

from dataclasses import dataclass

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_zn_layout


pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@dataclass
class MatmulConfig:
    m_size: int
    n_size: int
    k_size: int
    dtype: str = "float16"
    accum_dtype: str = "float"
    negative_slope: float = 0.01


@tilelang.jit(out_idx=[2], workspace_idx=3, pass_configs=pass_configs)
def matmul_leakyrelu(cfg):
    m_size, n_size, k_size, dtype, accum_dtype, negative_slope = (
        cfg.m_size, cfg.n_size, cfg.k_size, cfg.dtype, cfg.accum_dtype, cfg.negative_slope
    )
    base_m, base_n, base_k, l1_prefetch = 128, 128, 128, 4
    num_physical_cores = 20
    if m_size % base_m != 0:
        raise ValueError(f"M={m_size} must be divisible by baseM={base_m}")
    if n_size % base_n != 0:
        raise ValueError(f"N={n_size} must be divisible by baseN={base_n}")
    if k_size % (l1_prefetch * base_k) != 0:
        raise ValueError(f"K={k_size} must be divisible by l1Prefetch*baseK={l1_prefetch * base_k}")
    m_num = m_size // base_m
    n_num = n_size // base_n
    total_blocks = m_num * n_num
    if total_blocks <= 0:
        raise ValueError(f"total_blocks must be positive, got {total_blocks}")
    used_core_num = min(num_physical_cores, total_blocks)
    tasks_per_core = (total_blocks + used_core_num - 1) // used_core_num
    vec_num = 2
    use_float_accum = accum_dtype == "float"
    negative_slope_const = T.float32(negative_slope)

    @T.prim_func
    def main(  
        a_mat: T.Tensor((m_size, k_size), dtype),
        b_mat: T.Tensor((k_size, n_size), dtype),
        c_mat: T.Tensor((m_size, n_size), "float32"),
        # Persistent-kernel workspace: one tile buffer per physical core.
        workspace: T.Tensor((used_core_num, base_m, base_n), accum_dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            core_idx = cid

            a_l1 = T.alloc_L1((base_m, l1_prefetch * base_k), dtype)
            b_l1 = T.alloc_L1((l1_prefetch * base_k, base_n), dtype)
            T.annotate_layout({
                a_l1: make_zn_layout(a_l1),
                b_l1: make_zn_layout(b_l1),
            })
            a_l0 = T.alloc_L0A((base_m, base_k), dtype)
            b_l0 = T.alloc_L0B((base_k, base_n), dtype)
            c_l0 = T.alloc_L0C((base_m, base_n), accum_dtype)

            c_accum_ub = T.alloc_ub((base_m // vec_num, base_n), accum_dtype)
            c_out_ub = T.alloc_ub((base_m // vec_num, base_n), "float32")

            # Fixed physical cores; each core walks a contiguous task range.
            def _process_k_tile(m_idx, n_idx):
                loop_k = k_size // (l1_prefetch * base_k)
                for k in T.serial(loop_k):
                    outer = k * l1_prefetch
                    T.copy(a_mat[m_idx * base_m, outer * base_k], a_l1)
                    T.copy(b_mat[outer * base_k, n_idx * base_n], b_l1)
                    for kk in T.serial(l1_prefetch):
                        T.copy(a_l1[0, kk * base_k], a_l0)
                        T.copy(b_l1[kk * base_k, 0], b_l0)
                        T.mma(
                            a_l0,
                            b_l0,
                            c_l0,
                            init=T.And(k == 0, kk == 0),
                        )

            def _process_compute_tile(task_id, m_idx, n_idx):
                with T.Scope("C"):
                    if task_id < total_blocks:
                        _process_k_tile(m_idx, n_idx)
                        T.copy(c_l0, workspace[core_idx, 0, 0])
                        T.set_cross_flag("FIX", 0)

            def _process_vector_tile(task_id, m_idx, n_idx):
                with T.Scope("V"):
                    if task_id < total_blocks:
                        T.wait_cross_flag(0)
                        T.copy(
                            workspace[core_idx, vid * base_m // vec_num, 0],
                            c_accum_ub,
                        )
                        if use_float_accum:
                            T.copy(c_accum_ub, c_out_ub)
                        else:
                            T.tile.cast(
                                c_out_ub,
                                c_accum_ub,
                                mode="CAST_NONE",
                                count=(base_m // vec_num) * base_n,
                            )
                        T.tile.leaky_relu(c_out_ub, c_out_ub, negative_slope_const)
                        T.copy(c_out_ub, c_mat[m_idx * base_m + vid * base_m // vec_num, n_idx * base_n])

            def _process_tile(local_idx):
                task_id = core_idx * tasks_per_core + local_idx
                m_idx = task_id // n_num
                n_idx = task_id % n_num
                _process_compute_tile(task_id, m_idx, n_idx)
                _process_vector_tile(task_id, m_idx, n_idx)

            for local_idx in T.serial(tasks_per_core):
                _process_tile(local_idx)

    return main
