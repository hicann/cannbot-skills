/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <type_traits>

#include "kernel_tiling/kernel_tiling.h"
#include "basic_api/kernel_vec_intf.h"
#include "arch35/sort_tiling_key.h"
#include "arch35/sort_tiling_data.h"
#include "arch35/sort_radix_more_core.h"
#include "arch35/sort_radix_one_core.h"
#include "arch35/sort_merge_sort.h"
#include "arch35/sort_small_axis_insertion.h"
#include "arch35/sort_small_axis_two_stage.h"

using namespace AscendC;
using namespace Sort;

// 多核 Radix 的 schIndex=2：32 位计数额外置 bit 8，降序额外置 bit 16。
// 编码规则见 arch35/sort_tiling_key.h；输出索引 dtype 不决定这里的 isInt32。
// KERNEL_TASK_TYPE 会将 key 拼接为编译特性标识符，因此具名 key 必须保持数字宏。
#define SORT_RADIX_MORE_CORE_INT32_ASCENDING_TILING_KEY 258
#define SORT_RADIX_MORE_CORE_INT32_DESCENDING_TILING_KEY 65794
#define SORT_RADIX_MORE_CORE_INT64_ASCENDING_TILING_KEY 2
#define SORT_RADIX_MORE_CORE_INT64_DESCENDING_TILING_KEY 65538

template <typename Op>
__aicore__ inline void LaunchSortKernel(GM_ADDR input, GM_ADDR values, GM_ADDR indices, GM_ADDR userWorkspace,
                                        const SortRegBaseTilingData* sortTiling, TPipe* pipeline)
{
    Op op;
    op.Init(input, values, indices, userWorkspace, sortTiling, pipeline);
    op.Process();
}

template <uint64_t isInt32, uint64_t isDescend>
__aicore__ inline void LaunchRadixMoreCore(GM_ADDR radixInput, GM_ADDR radixValues, GM_ADDR radixIndices,
                                           GM_ADDR radixWorkspace, const SortRegBaseTilingData* radixTiling,
                                           TPipe* radixPipeline)
{
    using RadixType = std::conditional_t<
        sizeof(DTYPE_X) == sizeof(uint8_t), uint8_t,
        std::conditional_t<sizeof(DTYPE_X) == sizeof(uint16_t), uint16_t,
                           std::conditional_t<sizeof(DTYPE_X) == sizeof(uint32_t), uint32_t, uint64_t>>>;
    using CounterT = std::conditional_t<isInt32 != 0, uint32_t, int64_t>;
    LaunchSortKernel<SortRadixMoreCore<DTYPE_X, DTYPE_Y2, RadixType, CounterT, isDescend>>(
        radixInput, radixValues, radixIndices, radixWorkspace, radixTiling, radixPipeline);
}

template <uint64_t isDescend>
__aicore__ inline void LaunchMergeSortRoute(GM_ADDR mergeInput, GM_ADDR mergeValues, GM_ADDR mergeIndices,
                                            GM_ADDR mergeWorkspace, const SortRegBaseTilingData* mergeTiling,
                                            TPipe* mergePipeline)
{
    if constexpr (std::is_same_v<bfloat16_t, DTYPE_X>) {
        LaunchSortKernel<MergeSort<DTYPE_X, DTYPE_Y2, float, isDescend>>(mergeInput, mergeValues, mergeIndices,
                                                                         mergeWorkspace, mergeTiling, mergePipeline);
    } else if constexpr (std::is_same_v<half, DTYPE_X> || std::is_same_v<float, DTYPE_X>) {
        LaunchSortKernel<MergeSort<DTYPE_X, DTYPE_Y2, DTYPE_X, isDescend>>(mergeInput, mergeValues, mergeIndices,
                                                                           mergeWorkspace, mergeTiling, mergePipeline);
    }
}

template <uint64_t schId, uint64_t isInt32, uint64_t isDescend>
__global__ __aicore__ void sort(GM_ADDR input, GM_ADDR sortedValues, GM_ADDR sortedIndices, GM_ADDR workBuffer,
                                GM_ADDR tilingAddress)
{
    REGISTER_TILING_DEFAULT(SortRegBaseTilingData);
    GET_TILING_DATA_WITH_STRUCT(SortRegBaseTilingData, sortTilingData, tilingAddress);

    GM_ADDR sortWorkspace = AscendC::GetUserWorkspace(workBuffer);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KERNEL_TASK_TYPE(SORT_RADIX_MORE_CORE_INT32_ASCENDING_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(SORT_RADIX_MORE_CORE_INT32_DESCENDING_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(SORT_RADIX_MORE_CORE_INT64_ASCENDING_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
    KERNEL_TASK_TYPE(SORT_RADIX_MORE_CORE_INT64_DESCENDING_TILING_KEY, KERNEL_TYPE_MIX_AIV_1_0);
    TPipe sortPipeline;
    constexpr bool isDescending = (isDescend != 0);
    if constexpr (schId == SORT_SCHID_2) {
        LaunchRadixMoreCore<isInt32, isDescend>(input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData,
                                                &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_1) {
        LaunchSortKernel<SortRadixOneCore<DTYPE_X, DTYPE_Y2, isDescending>>(
            input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData, &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_0) {
        LaunchMergeSortRoute<isDescend>(input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData,
                                        &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_5 && std::is_same_v<bfloat16_t, DTYPE_X>) {
        LaunchSortKernel<Sort::SortSmallAxisInsertion<DTYPE_X, float, DTYPE_Y2, isDescending>>(
            input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData, &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_5) {
        LaunchSortKernel<Sort::SortSmallAxisInsertion<DTYPE_X, DTYPE_X, DTYPE_Y2, isDescending>>(
            input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData, &sortPipeline);
    } else if constexpr (schId == SORT_SCHID_6) {
        LaunchSortKernel<Sort::SortSmallAxisTwoStage<DTYPE_X, DTYPE_Y2, isDescending>>(
            input, sortedValues, sortedIndices, sortWorkspace, &sortTilingData, &sortPipeline);
    }
}
