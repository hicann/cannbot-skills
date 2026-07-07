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

"""Block-level TileLang design for RMSNorm.

The scheduling evolves with hidden size:
- `merge_n` for small N: UB can hold multiple full rows, so we process several
  rows together to maximize vector efficiency.
- `single_row` for medium N: a full row still fits in UB, but multiple rows no
  longer fit comfortably, so we reduce to one row at a time.
- `splitd` for very large N: even one full row no longer fits in UB, so the row
  must be split along the hidden dimension and processed in multiple passes.

NOTE: Each @T.prim_func below (merge_n, single_row, splitd) shares the same
pipeline boilerplate (Kernel setup, core_idx, task loop, Scope, row
decomposition).  This is intentional: T.lang prim_funcs compile into separate
compute graphs and *cannot* share DSL code via a Python helper.  The common
structure is listed in full for clarity of each variant's inner logic.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2], pass_configs=pass_configs)
def rms_norm(m_size, n_size, eps=1e-5, dtype="float32"):
    block_m = 64
    block_n = 1024
    num_physical_cores = 20
    if m_size % block_m != 0:
        raise ValueError(f"M={m_size} must be divisible by block_M={block_m}")
    m_num = m_size // block_m
    if m_num <= 0:
        raise ValueError(f"m_num must be positive, got {m_num}")
    used_core_num = min(num_physical_cores, m_num)
    tasks_per_core = (m_num + used_core_num - 1) // used_core_num
    vec_num = 2
    sub_block_m = block_m // vec_num

    row_factor = 8
    row_loops = sub_block_m // row_factor
    n_num = (n_size + block_n - 1) // block_n

    @T.prim_func
    def merge_n(
        x_in: T.Tensor((m_size, n_size), dtype),
        gamma: T.Tensor((n_size,), dtype),
        y_out: T.Tensor((m_size, n_size), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            core_idx = cid

            def _process_rows_for_bx(bx, vid):
                for r in T.serial(row_loops):
                    row_base = bx * block_m + vid * sub_block_m + r * row_factor
                    _ = (x_in, gamma, y_out, row_base)

            def _process_bx(bx, vid):
                with T.Scope("V"):
                    if bx < m_num:
                        _process_rows_for_bx(bx, vid)

            for local_idx in T.serial(tasks_per_core):
                bx = core_idx * tasks_per_core + local_idx
                _process_bx(bx, vid)

    @T.prim_func
    def single_row(
        x_in: T.Tensor((m_size, n_size), dtype),
        gamma: T.Tensor((n_size,), dtype),
        y_out: T.Tensor((m_size, n_size), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            core_idx = cid

            def _process_rows_for_bx(bx, vid):
                for row in T.serial(sub_block_m):
                    row_idx = bx * block_m + vid * sub_block_m + row
                    _ = (x_in, gamma, y_out, row_idx)

            def _process_bx(bx, vid):
                with T.Scope("V"):
                    if bx < m_num:
                        _process_rows_for_bx(bx, vid)

            for local_idx in T.serial(tasks_per_core):
                bx = core_idx * tasks_per_core + local_idx
                _process_bx(bx, vid)

    @T.prim_func
    def splitd(
        x_in: T.Tensor((m_size, n_size), dtype),
        gamma: T.Tensor((n_size,), dtype),
        y_out: T.Tensor((m_size, n_size), dtype),
    ):
        with T.Kernel(used_core_num, is_npu=True) as (cid, vid):
            core_idx = cid

            def _process_rows_for_bx(bx, vid):
                for row in T.serial(sub_block_m):
                    row_idx = bx * block_m + vid * sub_block_m + row
                    _ = (x_in, gamma, y_out, row_idx, n_num)

            def _process_bx(bx, vid):
                with T.Scope("V"):
                    if bx < m_num:
                        _process_rows_for_bx(bx, vid)

            for local_idx in T.serial(tasks_per_core):
                bx = core_idx * tasks_per_core + local_idx
                _process_bx(bx, vid)

    _ = eps

    if n_size <= 1024:
        return merge_n
    if n_size > 8192:
        return splitd
    return single_row
