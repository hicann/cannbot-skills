/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PING_PONG_MERGE_SORT_H
#define PING_PONG_MERGE_SORT_H

#include "kernel_operator.h"
#include "merge_sort_constants.h"

namespace PingPongMergeSortCommon {
using namespace AscendC;

using MergeSortConstants::DEALING_SORT_NUM_ONCE;
using MergeSortConstants::MERGE_LIST_MAX_NUM;

// 每轮四路归并：src 中每段长 queueLength，相邻四段合并至 dst。
// 完整分组由一次 MrgSort 批量处理，尾部不完整分组单独处理。
// 当前 arch35 的两份 UB 提案各占至少 8 字节/元素，即使全部 256 KiB 都用于提案，
// 元素数也不超过 16384。段长为 32×4^s，仍需归并时最大为 8192，不会截断 uint16 计数。
// 扩展 UB 或修改提案布局后，必须重新验证此上界。
template <typename T>
__aicore__ inline void MergeStage(LocalTensor<T> dst, LocalTensor<T> src, uint32_t totalElements, uint32_t queueLength,
                                  uint32_t queueCount)
{
    constexpr uint16_t allListsValid = (1U << MERGE_LIST_MAX_NUM) - 1U;
    // 完整分组：每组归并四段，每段 queueLength 个元素。
    uint32_t groupSpan = queueLength * MERGE_LIST_MAX_NUM;
    uint32_t fullGroupCount = queueCount / MERGE_LIST_MAX_NUM;
    uint32_t tailQueueCount = queueCount % MERGE_LIST_MAX_NUM;
    // 段数可被四整除时，最后一段仍可能不满；将最后一组留给尾部分支精确描述长度。
    if (tailQueueCount == 0U && totalElements != queueCount * queueLength) {
        --fullGroupCount;
        tailQueueCount = MERGE_LIST_MAX_NUM;
    }
    if (fullGroupCount > 0U) {
        uint16_t elementCounts[MERGE_LIST_MAX_NUM] = {
            static_cast<uint16_t>(queueLength), static_cast<uint16_t>(queueLength), static_cast<uint16_t>(queueLength),
            static_cast<uint16_t>(queueLength)};
        uint32_t sourceStride = GetSortOffset<T>(queueLength);
        MrgSortSrcList<T> sourceList(src, src[sourceStride], src[sourceStride * 2U], src[sourceStride * 3U]);
        uint32_t sortedCounts[MERGE_LIST_MAX_NUM];
        // false 关闭列表耗尽暂停，消费所有有效列表。
        MrgSort<T, false>(dst, sourceList, elementCounts, sortedCounts, allListsValid,
                          static_cast<int32_t>(fullGroupCount));
    }

    // 尾部剩余一至四段，第四段可能不足 queueLength。
    uint32_t tailOffset = fullGroupCount * groupSpan;
    uint32_t tailElements = totalElements - tailOffset;
    if (tailElements == 0U) {
        return;
    }
    uint32_t proposalTailOffset = GetSortOffset<T>(tailOffset);
    if (tailQueueCount == 1U) {
        DataCopy(dst[proposalTailOffset], src[proposalTailOffset], GetSortOffset<T>(tailElements));
        return;
    }

    uint16_t elementCounts[MERGE_LIST_MAX_NUM] = {0, 0, 0, 0};
    uint32_t sourceOffsets[MERGE_LIST_MAX_NUM] = {tailOffset, tailOffset, tailOffset, tailOffset};
    for (uint32_t list = 0; list < tailQueueCount; ++list) {
        uint32_t sourceOffset = tailOffset + list * queueLength;
        uint32_t sourceRemaining = totalElements - sourceOffset;
        sourceOffsets[list] = sourceOffset;
        elementCounts[list] = static_cast<uint16_t>(sourceRemaining < queueLength ? sourceRemaining : queueLength);
    }
    MrgSortSrcList<T> sourceList(src[GetSortOffset<T>(sourceOffsets[0])], src[GetSortOffset<T>(sourceOffsets[1])],
                                 src[GetSortOffset<T>(sourceOffsets[2])], src[GetSortOffset<T>(sourceOffsets[3])]);
    uint32_t sortedCounts[MERGE_LIST_MAX_NUM];
    uint16_t validBit = static_cast<uint16_t>((1U << tailQueueCount) - 1U);
    MrgSort<T, false>(dst[proposalTailOffset], sourceList, elementCounts, sortedCounts, validBit, 1);
}

// UB 内乒乓归并：Sort32 向 ping 写入 repeatTimes 个有序段。
// 随后每轮四路归并，在 ping/pong 间交替读写，段长每轮扩大四倍，直到只剩一段。
// 返回 true 表示最终结果在 ping，false 表示在 pong。
template <typename T>
__aicore__ inline bool SortToProposal(LocalTensor<T> ping, LocalTensor<T> pong, LocalTensor<T> srcValue,
                                      LocalTensor<uint32_t> baseIndex, uint32_t repeatTimes)
{
    Sort32<T>(ping, srcValue, baseIndex, repeatTimes);
    if (repeatTimes <= 1U) {
        return true;
    }
    uint32_t totalElements = repeatTimes * DEALING_SORT_NUM_ONCE;
    uint32_t queueLength = DEALING_SORT_NUM_ONCE;
    uint32_t queueCount = repeatTimes;
    bool sourceIsPing = true;
    while (queueCount > 1U) {
        if (sourceIsPing) {
            MergeStage(pong, ping, totalElements, queueLength, queueCount);
        } else {
            MergeStage(ping, pong, totalElements, queueLength, queueCount);
        }
        sourceIsPing = !sourceIsPing;
        queueLength *= MERGE_LIST_MAX_NUM;
        queueCount = (queueCount + MERGE_LIST_MAX_NUM - 1U) / MERGE_LIST_MAX_NUM;
    }
    return sourceIsPing;
}

} // namespace PingPongMergeSortCommon

#endif // PING_PONG_MERGE_SORT_H
