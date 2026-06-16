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

"""Block-level TileLang design for Key-Value Sort.

Three tiling strategies based on data size:
- `fullload`: data fits entirely in UB, single-core single-pass sort.
- `singlecore`: data fits in one core's UB for sort, but needs workspace for
  multi-phase pipeline (sort → output).
- `multicore`: data must be partitioned across cores; each core sorts locally,
  then a multi-level merge tree produces the globally sorted result.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}


@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs)
def kv_sort(total_length, dtype="int32"):
    num_cores = 20
    ub_sort_capacity = 2048
    multi_core_per_loop = 512
    max_mrgsort_list = 4

    @T.prim_func
    def fullload(
        keys: T.Tensor((total_length,), dtype),
        values: T.Tensor((total_length,), dtype),
        sorted_keys: T.Tensor((total_length,), dtype),
        sorted_values: T.Tensor((total_length,), dtype),
    ):
        """FullLoad mode: entire key/value arrays fit in UB.

        Block-level view:
          - Single core (core 0) loads all data into UB.
          - One hardware Sort pass produces the sorted result.
          - Directly write back to GM outputs.
        """
        with T.Kernel(1, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                # TODO(tile-level):
                # - Load keys[0:total_length] and values[0:total_length] into UB
                # - Cast keys int32 → float32, negate for ascending via descending Sort
                # - Pad tail to 32-element alignment with MIN_FP32
                # - Concat → Sort<float,true> → Extract
                # - Negate + Cast back to int32
                # - Write sorted keys and values to GM
                _ = keys
                _ = values
                _ = sorted_keys
                _ = sorted_values

    need_cores = T.min(num_cores, T.ceildiv(total_length, multi_core_per_loop))
    per_core_elements = T.ceildiv(total_length, need_cores)
    last_core_elements = total_length - per_core_elements * (need_cores - 1)

    @T.prim_func
    def multicore(
        keys: T.Tensor((total_length,), dtype),
        values: T.Tensor((total_length,), dtype),
        sorted_keys: T.Tensor((total_length,), dtype),
        sorted_values: T.Tensor((total_length,), dtype),
    ):
        """MultiCore mode: data partitioned across multiple cores.

        Block-level view — three phases:

        Phase 1 (VBS): Vector Block Sort — all cores in parallel
          - Each core loads its partition of keys/values.
          - Splits into UB-sized blocks, sorts each block with hardware Sort.
          - Merges blocks within the core using MrgSort (≤4-way merge).
          - Writes one sorted segment per core to workspace GM (ping-pong buffer).

        Phase 2 (VMS): Vector Merge Sort — progressively fewer cores
          - Treats each core's sorted segment as one list.
          - Each round: groups of ≤4 lists are merged by one core.
          - Active cores decrease by 4× each round.
          - Continues until ≤4 lists remain.
          - SyncAll between rounds.

        Phase 3 (SortOut): Final merge — single core
          - Core 0 merges the remaining ≤4 lists.
          - Extract separates key/value from sort format.
          - Negates keys back and casts float32 → int32.
          - Writes final sorted_keys and sorted_values to output GM.
        """
        with T.Kernel(need_cores, is_npu=True) as (cid, vid):
            with T.Scope("V"):
                # --- Phase 1: VBS (each core sorts its partition) ---
                # Each core handles keys[cid*per_core_elements : (cid+1)*per_core_elements]
                # Sub-blocks sorted via Sort instruction, then merged within core

                # --- Phase 2: VMS (cross-core merge tree) ---
                # Round by round, groups of 4 sorted segments are merged
                # workspace ping-pong between two GM buffers
                # SyncAll after each round

                # --- Phase 3: SortOut (final merge on core 0) ---
                # ≤4 remaining segments merged, Extract + restore, write output

                _ = keys
                _ = values
                _ = sorted_keys
                _ = sorted_values
                _ = per_core_elements
                _ = last_core_elements
                _ = max_mrgsort_list

    if total_length <= ub_sort_capacity:
        return fullload
    if total_length <= ub_sort_capacity * 2:
        return fullload
    return multicore
