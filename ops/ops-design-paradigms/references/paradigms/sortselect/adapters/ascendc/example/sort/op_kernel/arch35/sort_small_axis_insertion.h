/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_SMALL_AXIS_INSERTION_H
#define SORT_SMALL_AXIS_INSERTION_H

#include "basic_api/kernel_vec_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "sort_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "common/util_type_simd.h"
#include "common/small_axis_insertion_base.h"

namespace Sort {
using namespace AscendC;

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
class SortSmallAxisInsertion
    : public SmallAxisCommon::SmallAxisInsertionBase<SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>, T,
                                                     CONVERT_TYPE, OUT_IDX_T, IsDescend, true> {
    using Base = SmallAxisCommon::SmallAxisInsertionBase<SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>,
                                                         T, CONVERT_TYPE, OUT_IDX_T, IsDescend, true>;

public:
    __aicore__ inline SortSmallAxisInsertion() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace,
                                const SortRegBaseTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process()
    {
        Base::Process();
    }

    friend Base;

private:
    using Base::batchNum_;
    using Base::blockDim_;
    using Base::blockIdx_;
    using Base::castBuf_;
    using Base::castRowStride_;
    using Base::indexRowStride_;
    using Base::indices_;
    using Base::pipe_;
    using Base::segmentLen_;
    using Base::segmentsPerBatch_;
    using Base::valueRowStride_;
    using Base::values_;

    // 主流水函数，按调用顺序排列。
    __aicore__ inline bool IsProcessInvalid() const;
    __aicore__ inline uint32_t ComputeValidSegs(uint32_t batchId) const;
    __aicore__ inline void ProcessBatch(uint32_t batchId, uint32_t validSegs);
    __aicore__ inline void StoreBatch(int64_t segStart, uint32_t validSegs);

    const SortRegBaseTilingData* tilingData_ = nullptr;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<T> outValueGm_;
    GlobalTensor<OUT_IDX_T> outIdxGm_;

    int64_t totalSegs_ = 0;
};

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace, const SortRegBaseTilingData* tilingData, TPipe* pipe)
{
    (void)workspace;
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    segmentLen_ = tilingData_->numTileDataSize;
    segmentsPerBatch_ = tilingData_->keyParams0;
    batchNum_ = tilingData_->keyParams1;
    totalSegs_ = tilingData_->unsortedDimNum;
    inputXGm_.SetGlobalBuffer((__gm__ T*)x);
    outValueGm_.SetGlobalBuffer((__gm__ T*)y);
    outIdxGm_.SetGlobalBuffer((__gm__ OUT_IDX_T*)idx);

    if (IsProcessInvalid()) {
        return;
    }

    Base::InitInsertionBuffers(pipe);
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline bool SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::IsProcessInvalid() const
{
    return blockIdx_ >= blockDim_ || segmentLen_ == 0 || segmentsPerBatch_ == 0 ||
           segmentsPerBatch_ > SmallAxisCommon::MAX_DATACOPY_BLOCK_COUNT;
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline uint32_t SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::ComputeValidSegs(
    uint32_t batchId) const
{
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(segmentsPerBatch_);
    int64_t segRemain = totalSegs_ - segStart;
    if (segRemain <= 0) {
        return 0;
    }
    if (segRemain >= static_cast<int64_t>(segmentsPerBatch_)) {
        return segmentsPerBatch_;
    }
    return static_cast<uint32_t>(segRemain);
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::ProcessBatch(uint32_t batchId,
                                                                                                   uint32_t validSegs)
{
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(segmentsPerBatch_);
    Base::LoadContiguousBatch(inputXGm_, segStart * static_cast<int64_t>(segmentLen_), validSegs, false);
    Base::SortBatch(validSegs);
    StoreBatch(segStart, validSegs);
}

template <typename T, typename CONVERT_TYPE, typename OUT_IDX_T, bool IsDescend>
__aicore__ inline void SortSmallAxisInsertion<T, CONVERT_TYPE, OUT_IDX_T, IsDescend>::StoreBatch(int64_t segStart,
                                                                                                 uint32_t validSegs)
{
    if (validSegs == 0) {
        return;
    }
    // 插入排序写入 values_/indices_，写回阶段读取这两块 UB。
    event_t e = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(e);
    WaitFlag<HardEvent::V_MTE3>(e);

    int64_t gmOffset = segStart * segmentLen_;
    // 类型转换分支的 valueRowStride_ 等于 castRowStride_，使搬运缓冲与排序缓冲使用相同行跨度。
    uint32_t valueSrcStrideBlocks = ((valueRowStride_ - segmentLen_) * sizeof(T)) / UB_BLOCK_SIZE;
    DataCopyExtParams valueCopyParam{static_cast<uint16_t>(validSegs), static_cast<uint32_t>(segmentLen_ * sizeof(T)),
                                     valueSrcStrideBlocks, 0, 0};
    if constexpr (!IsSameType<T, CONVERT_TYPE>::value) {
        LocalTensor<T> storeBuf = castBuf_.template Get<T>();
        Cast(storeBuf, values_, RoundMode::CAST_RINT, validSegs * valueRowStride_);
        // Cast 在 Vector 写入 storeBuf；MTE3 随后读取并搬运至 GM。
        event_t e = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(e);
        WaitFlag<HardEvent::V_MTE3>(e);
        DataCopyPad(outValueGm_[gmOffset], storeBuf, valueCopyParam);
    } else {
        DataCopyPad(outValueGm_[gmOffset], values_, valueCopyParam);
    }

    uint32_t idxStrideBlocks = ((indexRowStride_ - segmentLen_) * sizeof(OUT_IDX_T)) / UB_BLOCK_SIZE;
    DataCopyExtParams idxCopyParam{static_cast<uint16_t>(validSegs),
                                   static_cast<uint32_t>(segmentLen_ * sizeof(OUT_IDX_T)), idxStrideBlocks, 0, 0};
    DataCopyPad(outIdxGm_[gmOffset], indices_, idxCopyParam);

    // 标量控制流须等待 MTE3 写回完成后再发起下一批。
    event_t eventId = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventId);
    WaitFlag<HardEvent::MTE3_S>(eventId);
}

} // namespace Sort

#endif
