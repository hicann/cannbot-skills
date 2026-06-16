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

"""Block-level TileLang design for matmul_leakyrelu.

This file keeps only block scheduling, pipeline skeleton, and cross-scope sync.
Fine-grained compute details are intentionally marked as TODO comments and should
be implemented in tile-level design.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True
}


@tilelang.jit(out_idx=[2], workspace_idx=3, pass_configs=pass_configs)
def matmul_leakyrelu(*args):
    a = [*args, "float16", "float", 0.01]
    m_size, n_size, k_size, dtype, accum_dtype, negative_slope = a[:6]
    block_m, block_n, block_k, k_l1 = 128, 256, 64, 256
    num_physical_cores = 20
    if m_size % block_m != 0:
        raise ValueError(f"M={m_size} must be divisible by block_M={block_m}")
    if n_size % block_n != 0:
        raise ValueError(f"N={n_size} must be divisible by block_N={block_n}")
    m_num = m_size // block_m
    n_num = n_size // block_n
    total_blocks = m_num * n_num
    if total_blocks <= 0:
        raise ValueError(f"total_blocks must be positive, got {total_blocks}")
    used_core_num = min(num_physical_cores, total_blocks)
    tasks_per_core = (total_blocks + used_core_num - 1) // used_core_num

    @T.prim_func
    def main(  
        a_mat: T.Tensor((m_size, k_size), dtype),
        b_mat: T.Tensor((k_size, n_size), dtype),
        c_mat: T.Tensor((m_size, n_size), dtype),
        workspace: T.Tensor((m_size, n_size), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            core_idx = cid

            # Block-level persistent-kernel pipeline:
            #   1) Fixed physical cores iterate over a contiguous task range.
            #   2) For each task, Cube scope produces one tile and Vector scope consumes it.
            def _process_tile(local_idx):
                task_id = core_idx * tasks_per_core + local_idx
                bx = task_id // n_num
                by = task_id % n_num
                with T.Scope("C"):
                    if task_id < total_blocks:
                        # TODO(tile-level):
                        # - implement block matmul (A @ B) for tile (bx, by)
                        # - hand off matmul tile to vector stage (via workspace/cross-scope sync)
                        # - ensure Cube-side pipeline/synchronization is correct
                        T.set_cross_flag("FIX", 0)
                with T.Scope("V"):
                    if task_id < total_blocks:
                        T.wait_cross_flag(0)
                        # TODO(tile-level):
                        # - wait for Cube-side matmul tile readiness
                        # - implement vector epilogue: leaky_relu(A @ B, negative_slope)
                        # - write final tile result to C
                        _ = negative_slope
                        _ = workspace
                        _ = a_mat
                        _ = b_mat
                        _ = c_mat
                        _ = bx
                        _ = by
                        _ = vid
                        pass

            for local_idx in T.serial(tasks_per_core):
                _process_tile(local_idx)

    return main
