/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SMALL_AXIS_INSERTION_BASE_H
#define SMALL_AXIS_INSERTION_BASE_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "simt_api/asc_simt.h"
#include "util_type_simd.h"

namespace SmallAxisCommon {
using namespace AscendC;

constexpr uint32_t INSERTION_THREAD_NUM = 1024;
// DataCopy 的 blockCount 硬件上限。
constexpr uint32_t MAX_DATACOPY_BLOCK_COUNT = 4095;

// sch5 的逐行 SIMT 插入排序；每个线程独立处理一行。
template <typename T, typename IDX_T, bool IsDescend, bool OrderNan = false>
__simt_vf__ LAUNCH_BOUND(INSERTION_THREAD_NUM) __aicore__
    void SimtInsertionSortSegments(uint32_t validSegs, uint32_t segmentLen, uint32_t valueRowElems,
                                   uint32_t idxRowElems, __ubuf__ T* valueBase, __ubuf__ IDX_T* idxBase)
{
    for (int32_t seg = threadIdx.x; seg < static_cast<int32_t>(validSegs);
         seg += static_cast<int32_t>(INSERTION_THREAD_NUM)) {
        uint32_t segId = static_cast<uint32_t>(seg);
        uint32_t valueBaseOffset = segId * valueRowElems;
        uint32_t idxBaseOffset = segId * idxRowElems;
        for (uint32_t elem = 0; elem < segmentLen; ++elem) {
            idxBase[idxBaseOffset + elem] = static_cast<IDX_T>(elem);
        }

        for (uint32_t elem = 1; elem < segmentLen; ++elem) {
            T keyValue = valueBase[valueBaseOffset + elem];
            IDX_T keyIdx = idxBase[idxBaseOffset + elem];
            uint32_t insertPos = elem;
            while (insertPos > 0) {
                T prevValue = valueBase[valueBaseOffset + insertPos - 1];
                // 只移动严格逆序元素，等值元素（包括同为 NaN）保持输入次序。
                bool needMove;
                if constexpr (OrderNan && (IsSameType<T, float>::value || IsSameType<T, half>::value ||
                                           IsSameType<T, bfloat16_t>::value)) {
                    bool prevNan = prevValue != prevValue;
                    bool keyNan = keyValue != keyValue;
                    // 升序 NaN 在尾，降序在头；不改变 NaN 的原始值。
                    if (prevNan != keyNan) {
                        needMove = IsDescend ? keyNan : prevNan;
                    } else if (prevNan) {
                        needMove = false;
                    } else if constexpr (IsDescend) {
                        needMove = prevValue < keyValue;
                    } else {
                        needMove = prevValue > keyValue;
                    }
                } else if constexpr (IsDescend) {
                    needMove = prevValue < keyValue;
                } else {
                    needMove = prevValue > keyValue;
                }
                if (!needMove) {
                    break;
                }
                valueBase[valueBaseOffset + insertPos] = prevValue;
                idxBase[idxBaseOffset + insertPos] = idxBase[idxBaseOffset + insertPos - 1];
                --insertPos;
            }
            valueBase[valueBaseOffset + insertPos] = keyValue;
            idxBase[idxBaseOffset + insertPos] = keyIdx;
        }
    }
}

// 插入排序公共流水负责分批、UB 分配、加载和类型转换；派生类提供批次映射与写回。
// T 为存储类型，CONVERT_TYPE 为比较类型，IDX_T 为 UB 索引类型。
template <typename Derived, typename T, typename CONVERT_TYPE, typename IDX_T, bool IsDescend, bool OrderNan = false>
class SmallAxisInsertionBase {
public:
    __aicore__ inline void Process()
    {
        Derived* op = static_cast<Derived*>(this);
        if (op->IsProcessInvalid()) {
            return;
        }
        uint32_t batchesPerCore = (batchNum_ + blockDim_ - 1U) / blockDim_;
        uint32_t startBatch = blockIdx_ * batchesPerCore;
        uint32_t endBatch = batchNum_ < startBatch + batchesPerCore ? batchNum_ : startBatch + batchesPerCore;
        for (uint32_t batchId = startBatch; batchId < endBatch; ++batchId) {
            uint32_t validSegs = op->ComputeValidSegs(batchId);
            if (validSegs == 0U) {
                continue;
            }
            op->ProcessBatch(batchId, validSegs);
        }
    }

    __aicore__ inline void InitInsertionBuffers(TPipe* pipe)
    {
        pipe_ = pipe;
        if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
            castRowStride_ = RoundUpAlign(segmentLen_ * sizeof(T)) / sizeof(T);
            valueRowStride_ = castRowStride_;
        } else {
            valueRowStride_ = RoundUpAlign(segmentLen_ * sizeof(CONVERT_TYPE)) / sizeof(CONVERT_TYPE);
        }
        indexRowStride_ = RoundUpAlign(segmentLen_ * sizeof(IDX_T)) / sizeof(IDX_T);

        pipe_->InitBuffer(valueBuf_, segmentsPerBatch_ * valueRowStride_ * sizeof(CONVERT_TYPE));
        pipe_->InitBuffer(idxBuf_, segmentsPerBatch_ * indexRowStride_ * sizeof(IDX_T));
        if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
            pipe_->InitBuffer(castBuf_, segmentsPerBatch_ * castRowStride_ * sizeof(T));
        }

        values_ = valueBuf_.template Get<CONVERT_TYPE>();
        indices_ = idxBuf_.template Get<IDX_T>();
    }

    __aicore__ inline void LoadContiguousBatch(GlobalTensor<T>& inputGm, int64_t inputStart, uint32_t validSegs,
                                               bool padTail)
    {
        if (validSegs == 0U) {
            return;
        }
        uint32_t dstRowStride = IsSameType<T, CONVERT_TYPE>::value ? valueRowStride_ : castRowStride_;
        uint32_t copiedRowElems = padTail ? RoundUpAlign(segmentLen_ * sizeof(T)) / sizeof(T) : segmentLen_;
        uint32_t dstStrideBlocks = ((dstRowStride - copiedRowElems) * sizeof(T)) / UB_BLOCK_SIZE;
        DataCopyExtParams copyParam{static_cast<uint16_t>(validSegs), static_cast<uint32_t>(segmentLen_ * sizeof(T)), 0,
                                    dstStrideBlocks, 0};
        DataCopyPadExtParams<T> padParams{
            padTail, 0, static_cast<uint8_t>(padTail ? (copiedRowElems - segmentLen_) : 0U), static_cast<T>(0)};
        if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
            LocalTensor<T> loadBuf = castBuf_.template Get<T>();
            DataCopyPad(loadBuf, inputGm[inputStart], copyParam, padParams);
            event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
            Cast(values_, loadBuf, RoundMode::CAST_NONE, validSegs * valueRowStride_);
        } else {
            DataCopyPad(values_, inputGm[inputStart], copyParam, padParams);
            event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventId);
            WaitFlag<HardEvent::MTE2_V>(eventId);
        }
    }

    __aicore__ inline void SortBatch(uint32_t validSegs)
    {
        asc_vf_call<SimtInsertionSortSegments<CONVERT_TYPE, IDX_T, IsDescend, OrderNan>>(
            dim3(INSERTION_THREAD_NUM), validSegs, segmentLen_, valueRowStride_, indexRowStride_,
            (__ubuf__ CONVERT_TYPE*)values_.GetPhyAddr(), (__ubuf__ IDX_T*)indices_.GetPhyAddr());
    }

protected:
    TPipe* pipe_ = nullptr;

    TBuf<TPosition::VECCALC> valueBuf_; // 以 CONVERT_TYPE 保存参与比较的值。
    TBuf<TPosition::VECCALC> castBuf_;  // 以 T 保存 GM 搬运值，供 BF16/float 转换。
    TBuf<TPosition::VECCALC> idxBuf_;

    LocalTensor<CONVERT_TYPE> values_;
    LocalTensor<IDX_T> indices_;

    uint32_t blockIdx_ = 0;
    uint32_t blockDim_ = 0;
    uint32_t segmentLen_ = 0;
    uint32_t segmentsPerBatch_ = 0;
    uint32_t batchNum_ = 0;
    uint32_t valueRowStride_ = 0;
    uint32_t castRowStride_ = 0;
    uint32_t indexRowStride_ = 0;
};

} // namespace SmallAxisCommon

#endif
