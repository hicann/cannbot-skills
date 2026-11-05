/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_SMALL_AXIS_TWO_STAGE_H
#define SORT_SMALL_AXIS_TWO_STAGE_H

#include "basic_api/kernel_vec_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "sort_tiling_data.h"
#include "simt_api/asc_simt.h"
#include "common/small_axis_two_stage_base.h"

namespace Sort {
using namespace AscendC;

template <typename T, typename OutIdxT, bool IsDescend>
class SortSmallAxisTwoStage
    : public SmallAxisCommon::SmallAxisTwoStageBase<SortSmallAxisTwoStage<T, OutIdxT, IsDescend>, T, OutIdxT,
                                                    IsDescend> {
    using Base =
        SmallAxisCommon::SmallAxisTwoStageBase<SortSmallAxisTwoStage<T, OutIdxT, IsDescend>, T, OutIdxT, IsDescend>;

public:
    __aicore__ inline SortSmallAxisTwoStage() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx, GM_ADDR workspace,
                                const SortRegBaseTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

    friend Base;

private:
    using Base::batchNum_;
    using Base::batchSize_;
    using Base::blockDim_;
    using Base::blockIdx_;
    using Base::finalIdx_;
    using Base::finalValues_;
    using Base::maxFlatElems_;
    using Base::segmentLen_;

    __aicore__ inline bool IsProcessInvalid() const;
    __aicore__ inline uint32_t ComputeValidSegs(uint32_t batchId) const;
    __aicore__ inline void ProcessBatch(uint32_t batchId, uint32_t validSegs);
    __aicore__ inline void StoreBatch(int64_t segStart, uint32_t totalElems);

    const SortRegBaseTilingData* tilingData_ = nullptr;

    GlobalTensor<T> inputXGm_;
    GlobalTensor<T> outValueGm_;
    GlobalTensor<OutIdxT> outIdxGm_;

    int64_t totalSegs_ = 0;
};

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR idx,
                                                                          GM_ADDR workspace,
                                                                          const SortRegBaseTilingData* tilingData,
                                                                          TPipe* pipe)
{
    (void)workspace;
    if (tilingData == nullptr || pipe == nullptr) {
        return;
    }
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    blockDim_ = GetBlockNum();
    batchSize_ = tilingData_->keyParams0;
    batchNum_ = tilingData_->keyParams1;
    segmentLen_ = tilingData_->numTileDataSize;
    maxFlatElems_ = batchSize_ * segmentLen_;
    totalSegs_ = tilingData_->unsortedDimNum;

    inputXGm_.SetGlobalBuffer((__gm__ T*)x);
    outValueGm_.SetGlobalBuffer((__gm__ T*)y);
    outIdxGm_.SetGlobalBuffer((__gm__ OutIdxT*)idx);

    if (batchSize_ == 0 || segmentLen_ == 0 || maxFlatElems_ == 0) {
        return;
    }

    constexpr uint32_t kAliasElemBytes = static_cast<uint32_t>(sizeof(OutIdxT));
    Base::InitSortBuffers(pipe, maxFlatElems_, tilingData_->tmpUbSize, tilingData_->keyParams2 != 0U, kAliasElemBytes);
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline bool SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::IsProcessInvalid() const
{
    return blockIdx_ >= blockDim_ || batchSize_ == 0 || segmentLen_ == 0;
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::Process()
{
    Base::Process();
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline uint32_t SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::ComputeValidSegs(uint32_t batchId) const
{
    int64_t batchStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    int64_t remainingSegs = totalSegs_ - batchStart;
    if (remainingSegs <= 0) {
        return 0;
    }
    if (remainingSegs >= static_cast<int64_t>(batchSize_)) {
        return batchSize_;
    }
    return static_cast<uint32_t>(remainingSegs);
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::ProcessBatch(uint32_t batchId, uint32_t validSegs)
{
    uint32_t totalElems = validSegs * segmentLen_;
    int64_t segStart = static_cast<int64_t>(batchId) * static_cast<int64_t>(batchSize_);
    Base::LoadContiguousBatch(inputXGm_, segStart * static_cast<int64_t>(segmentLen_), totalElems);
    Base::RunTwoStageSort(totalElems);
    StoreBatch(segStart, totalElems);
}

template <typename T, typename OutIdxT, bool IsDescend>
__aicore__ inline void SortSmallAxisTwoStage<T, OutIdxT, IsDescend>::StoreBatch(int64_t segStart, uint32_t totalElems)
{
    int64_t gmOffset = segStart * static_cast<int64_t>(segmentLen_);
    DataCopyExtParams valueCopyParam{1, static_cast<uint32_t>(totalElems * sizeof(T)), 0, 0, 0};
    DataCopyExtParams idxCopyParam{1, static_cast<uint32_t>(totalElems * sizeof(OutIdxT)), 0, 0, 0};
    // 逆排名分支由 SIMT VF 生成最终缓冲，BuildOutputs 分支由向量 API 生成。
    // 写回 GM 前等待 VF 或向量生产者完成。
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
    DataCopyPad(outValueGm_[gmOffset], finalValues_, valueCopyParam);
    DataCopyPad(outIdxGm_[gmOffset], finalIdx_, idxCopyParam);

    // finalValues_ 复用 inputValues_；下一次 LoadBatch 会通过 MTE2 覆盖该 UB。
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    // 下一轮由标量控制流发起，因此等待 MTE3_S。
    event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
}

} // namespace Sort

#endif
