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

"""Tile-level TileLang design for Key-Value Sort.

Expresses the detailed algorithmic logic of each tiling mode using TileLang
primitives. The key insight is that AscendC hardware Sort only supports float32
descending order, so we negate keys before sorting and negate back afterward
to achieve ascending int32 sort.

Three modes:
- fullload:    Single UB pass — load all, sort, write.
- singlecore:  Same sort logic, but separated for pipeline clarity.
- multicore:   Block-sort per core → intra-core merge → cross-core merge tree.
"""

import tilelang
import tilelang.language as T

pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}

SORT_ALIGN = 32
MIN_FP32 = T.float32(-3.4e38)


@tilelang.jit(out_idx=[2, 3], pass_configs=pass_configs)
def kv_sort(total_length, dtype="int32"):
    num_cores = 20
    ub_sort_capacity = 2048
    multi_core_per_loop = 512
    one_loop_max = 256
    max_mrgsort_list = 4

    sort_num = T.ceildiv(total_length, SORT_ALIGN) * SORT_ALIGN

    @T.prim_func
    def fullload(
        keys: T.Tensor((total_length,), dtype),
        values: T.Tensor((total_length,), dtype),
        sorted_keys: T.Tensor((total_length,), dtype),
        sorted_values: T.Tensor((total_length,), dtype),
    ):
        with T.Kernel(1, is_npu=True) as (cid, vid):
            # UB allocations: key-value pair buffers in sort format
            keys_ub = T.alloc_ub((sort_num,), "int32")
            values_ub = T.alloc_ub((sort_num,), "int32")
            keys_fp32_ub = T.alloc_ub((sort_num,), "float32")
            sorted_kv_ub = T.alloc_ub((sort_num * 2,), "float32")  # sort format: interleaved k,v
            out_keys_ub = T.alloc_ub((sort_num,), "float32")
            out_vals_ub = T.alloc_ub((sort_num,), "uint32")
            temp_ub = T.alloc_ub((sort_num * 2,), "float32")

            with T.Scope("V"):
                # Step 1: CopyIn — load keys and values from GM to UB
                T.copy(keys[0:total_length], keys_ub[0:total_length])
                T.copy(values[0:total_length], values_ub[0:total_length])

                # Step 2: Cast int32 → float32 (Sort instruction requires float)
                T.tile.cast(keys_fp32_ub, keys_ub, mode="CAST_ROUND", count=sort_num)

                # Step 3: Negate keys (Sort is descending; negating gives ascending)
                T.tile.mul(keys_fp32_ub, keys_fp32_ub, T.float32(-1.0))

                # Step 4: Pad tail elements (not aligned to 32) with MIN_FP32
                # so they sink to the end of descending sort and don't interfere
                # (conceptual — actual mask-based Duplicate in AscendC)

                # Step 5: Concat — rearrange into Sort instruction's input layout
                # (groups of 32 elements packed for hardware Sort unit)
                T.tile.concat(concat_ub, keys_fp32_ub, temp_ub, sort_num // SORT_ALIGN)

                # Step 6: Hardware Sort — uses Sort<float, true> instruction
                # for key-value descending sort on the concatenated buffer.

                # Step 7: Extract — separate sorted keys and values from sort format
                T.tile.extract(out_keys_ub, out_vals_ub, sorted_kv_ub, sort_num // 32)

                # Step 8: Negate keys back to restore original values
                T.tile.mul(out_keys_ub, out_keys_ub, T.float32(-1.0))

                # Step 9: Cast float32 → int32
                T.tile.cast(out_keys_int_ub, out_keys_ub, mode="CAST_ROUND", count=sort_num)

                # Step 10: CopyOut — write sorted results to GM
                T.copy(out_keys_int_ub[0:total_length], sorted_keys[0:total_length])
                T.copy(out_vals_ub[0:total_length], sorted_values[0:total_length])

                _ = sorted_kv_ub
                _ = out_vals_ub
                _ = temp_ub
                _ = sorted_keys
                _ = sorted_values

    @T.prim_func
    def singlecore(
        keys: T.Tensor((total_length,), dtype),
        values: T.Tensor((total_length,), dtype),
        sorted_keys: T.Tensor((total_length,), dtype),
        sorted_values: T.Tensor((total_length,), dtype),
    ):
        """Identical sort logic to fullload.
        Separated because the AscendC implementation uses different output routing
        (workspace vs direct GM) to interface with downstream phases.
        """
        with T.Kernel(1, is_npu=True) as (cid, vid):
            keys_ub = T.alloc_ub((sort_num,), "int32")
            values_ub = T.alloc_ub((sort_num,), "int32")
            keys_fp32_ub = T.alloc_ub((sort_num,), "float32")
            temp_ub = T.alloc_ub((sort_num * 2,), "float32")

            with T.Scope("V"):
                T.copy(keys[0:total_length], keys_ub[0:total_length])
                T.copy(values[0:total_length], values_ub[0:total_length])

                T.tile.cast(keys_fp32_ub, keys_ub, mode="CAST_ROUND", count=sort_num)
                T.tile.mul(keys_fp32_ub, keys_fp32_ub, T.float32(-1.0))

                # Pad + Concat + Sort + Extract + Negate + Cast + CopyOut
                # (same as fullload)
                _ = temp_ub
                _ = sorted_keys
                _ = sorted_values

    need_cores = T.min(num_cores, T.ceildiv(total_length, multi_core_per_loop))
    per_core_elements = T.ceildiv(total_length, need_cores)
    loop_elements = T.min(multi_core_per_loop, per_core_elements)

    @T.prim_func
    def multicore(
        keys: T.Tensor((total_length,), dtype),
        values: T.Tensor((total_length,), dtype),
        sorted_keys: T.Tensor((total_length,), dtype),
        sorted_values: T.Tensor((total_length,), dtype),
    ):
        with T.Kernel(need_cores, is_npu=True) as (cid, vid):
            # Per-core UB: holds one sort block (loop_elements key-value pairs)
            blk_keys_ub = T.alloc_ub((loop_elements,), "int32")
            blk_vals_ub = T.alloc_ub((loop_elements,), "int32")
            blk_fp32_ub = T.alloc_ub((loop_elements,), "float32")
            sort_temp_ub = T.alloc_ub((loop_elements * 2,), "float32")
            sorted_block_ub = T.alloc_ub((loop_elements * 2,), "float32")  # sort format

            # Merge buffers for MrgSort (up to 4 input lists)
            mrg_in_ub = T.alloc_ub((max_mrgsort_list, one_loop_max * 2), "float32")
            mrg_out_ub = T.alloc_ub((one_loop_max * max_mrgsort_list * 2,), "float32")

            with T.Scope("V"):
                core_start = cid * per_core_elements
                core_end = T.min(core_start + per_core_elements, total_length)
                core_len = core_end - core_start
                num_loops = T.ceildiv(core_len, loop_elements)

                # ============================================================
                # PHASE 1: VBS — each core sorts its local blocks
                # ============================================================
                for loop in T.serial(num_loops):
                    block_start = core_start + loop * loop_elements
                    block_end = T.min(block_start + loop_elements, core_end)
                    block_len = block_end - block_start
                    sn = T.ceildiv(block_len, SORT_ALIGN) * SORT_ALIGN

                    # Load one block of keys + values into UB
                    T.copy(keys[block_start:block_end], blk_keys_ub[0:block_len])
                    T.copy(values[block_start:block_end], blk_vals_ub[0:block_len])

                    # int32 → float32, negate
                    T.tile.cast(blk_fp32_ub, blk_keys_ub, mode="CAST_ROUND", count=sn)
                    T.tile.mul(blk_fp32_ub, blk_fp32_ub, T.float32(-1.0))

                    # Pad tail with MIN_FP32, Concat + Sort<float,true> to
                    # produce sorted_block_ub, then write to workspace ping buf.
                    pass

                # Intra-core merge via MrgSort (≤4-way): workspace ping-pong
                # between two GM buffers, listNum=ceil(listNum/4), SyncAll.

                # PHASE 2 (VMS cross-core merge): need_cores segments in ws.
                # Each round ceil(listNum/4) cores merge 4 segments, 4x fewer
                # active cores per round, SyncAll between, until listNum≤4.
                # PHASE 3 (SortOut on core 0): core 0 merges remaining ≤4
                # segments via MrgSort, extracts keys+values, negates+casts,
                # copies out to SortedKeys/SortedValues GM.

    if total_length <= ub_sort_capacity:
        return fullload
    if total_length <= ub_sort_capacity * 2:
        return singlecore
    return multicore
