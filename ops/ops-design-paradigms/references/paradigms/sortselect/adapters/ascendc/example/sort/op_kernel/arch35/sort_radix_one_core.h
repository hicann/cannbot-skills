/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RADIX_SORT_ONE_CORE_H
#define RADIX_SORT_ONE_CORE_H

#include <cmath>
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"
#include "common/signed_zero_sort_utils.h"

namespace Sort {
using namespace AscendC;
template <typename T1, typename T2, bool isDescend>
class SortRadixOneCore {
public:
    __aicore__ inline SortRadixOneCore(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex, GM_ADDR workspace,
                                const SortRegBaseTilingData* __restrict tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessRadixSortOneCore(GlobalTensor<T1> inputXGm, int64_t gmOffset, uint32_t sortLoopRound);
    __aicore__ inline void ParseTilingData();

    GlobalTensor<T1> inputXGm_;
    GlobalTensor<T1> outValueGm_;
    GlobalTensor<uint32_t> outIdxGm_;

    TPipe* pipe_;
    const SortRegBaseTilingData* tilingData_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TBuf<TPosition::VECCALC> tmpUb_;
    TQue<QuePosition::VECOUT, 1> outIdxQueue_;
    TQue<QuePosition::VECOUT, 1> outValueQueue_;
    TBuf<TPosition::VECCALC> sourceOrderBuf_;
    LocalTensor<uint32_t> sourceOrder_;
    static constexpr SortConfig radixSortConfig{SortType::RADIX_SORT, isDescend};

    int64_t totalDataNum_ = 0;
    uint32_t numTileData_ = 0;
    int64_t unsortedDimNum_ = 0;
    uint32_t unsortedDimParallel_ = 0;
    uint32_t sortLoopTimes_ = 0;
    uint32_t tmpUbSize_ = 0;
    uint32_t blockIdx_ = 0;
    uint32_t realCoreNum_ = 0;
    uint32_t halfIndex_ = 0;
    uint32_t xUbSize_ = 0;
    uint32_t yUbSize_ = 0;
    uint32_t bufferNum_ = 1;
};

template <typename T1, typename T2, bool isDescend>
__aicore__ inline void SortRadixOneCore<T1, T2, isDescend>::Init(GM_ADDR x, GM_ADDR value, GM_ADDR sortIndex,
                                                                 GM_ADDR workspace,
                                                                 const SortRegBaseTilingData* __restrict tilingData,
                                                                 TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    pipe_ = pipe;
    tilingData_ = tilingData;
    ParseTilingData();
    inputXGm_.SetGlobalBuffer((__gm__ T1*)x);
    outValueGm_.SetGlobalBuffer((__gm__ T1*)value);
    outIdxGm_.SetGlobalBuffer((__gm__ uint32_t*)sortIndex);
    realCoreNum_ = GetBlockNum();

    pipe_->InitBuffer(inQueueX_, bufferNum_, xUbSize_);
    pipe_->InitBuffer(outValueQueue_, bufferNum_, xUbSize_);
    pipe_->InitBuffer(outIdxQueue_, bufferNum_, yUbSize_);
    pipe_->InitBuffer(tmpUb_, tmpUbSize_);
    if constexpr (SignedZeroSortCommon::IS_FLOATING_POINT_V<T1>) {
        pipe_->InitBuffer(sourceOrderBuf_, halfIndex_ * sizeof(uint32_t));
        sourceOrder_ = sourceOrderBuf_.Get<uint32_t>();
    }
}

template <typename T1, typename T2, bool isDescend>
__aicore__ inline void SortRadixOneCore<T1, T2, isDescend>::ParseTilingData()
{
    totalDataNum_ = tilingData_->lastAxisNum;                // 每行排序轴长 N
    numTileData_ = tilingData_->numTileDataSize;             // ub循环块大小
    unsortedDimNum_ = tilingData_->unsortedDimNum;           // 待排序行数 M
    unsortedDimParallel_ = tilingData_->unsortedDimParallel; // 每轮并行行数；本路线每行一个核
    sortLoopTimes_ = tilingData_->sortLoopTimes;             // 行批次数
    tmpUbSize_ = tilingData_->tmpUbSize;                     // 高级api需要用的ub大小
    xUbSize_ = tilingData_->keyParams0;                      // xInQue需要的ub大小
    yUbSize_ = tilingData_->keyParams1;                      // y2OutQue需要的ub大小
    halfIndex_ =
        tilingData_->keyParams2; // uint32 元素偏移 alignBlock(N*4)/4；int64 输出先将 B32 索引放后半区，再原地扩宽
    bufferNum_ = tilingData_->keyParams3 == 2 ? 2 : 1; // one-core队列buffer数
}

template <typename T1, typename T2, bool isDescend>
__aicore__ inline void SortRadixOneCore<T1, T2, isDescend>::ProcessRadixSortOneCore(GlobalTensor<T1> inputXGm,
                                                                                    int64_t gmOffset,
                                                                                    uint32_t sortLoopRound)
{
    uint32_t unsortDimIndex = blockIdx_ + sortLoopRound * unsortedDimParallel_;
    if (unsortDimIndex >= unsortedDimNum_) {
        return;
    }
    LocalTensor<T1> xLocal = inQueueX_.AllocTensor<T1>();
    uint32_t tileOffset = blockIdx_ * numTileData_;
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    DataCopyExtParams dataCopyParam;
    dataCopyParam.blockCount = 1;
    dataCopyParam.blockLen = numTileData_ * sizeof(T1);
    dataCopyParam.srcStride = 0;
    dataCopyParam.dstStride = 0;
    DataCopyPad(xLocal, inputXGm[tileOffset], dataCopyParam, padParams);
    inQueueX_.EnQue<T1>(xLocal);
    xLocal = inQueueX_.DeQue<T1>();
    LocalTensor<T1> sortValueUb = outValueQueue_.AllocTensor<T1>();
    LocalTensor<uint32_t> sortIdxUb = outIdxQueue_.AllocTensor<uint32_t>();
    LocalTensor<uint8_t> tmpUb = tmpUb_.Get<uint8_t>();
    bool hasNegativeZero = false;
    if constexpr (SignedZeroSortCommon::IS_FLOATING_POINT_V<T1>) {
        hasNegativeZero = SignedZeroSortCommon::HasNegativeZeroVec(xLocal, sourceOrder_, numTileData_);
        if (hasNegativeZero) {
            SignedZeroSortCommon::PrepareSignedZeroKeysVec(xLocal, sourceOrder_, numTileData_);
        }
    }
    // int64 输出复用索引队列：先写后半区 uint32 索引，再 Cast 到整块 int64 视图。
    if constexpr (sizeof(T2) == sizeof(int64_t)) {
        LocalTensor<uint32_t> sortIdxUbHalf = sortIdxUb[halfIndex_];
        if constexpr (SignedZeroSortCommon::IS_FLOATING_POINT_V<T1>) {
            AscendC::Sort<T1, false, radixSortConfig>(sortValueUb, sortIdxUbHalf, xLocal, tmpUb, numTileData_);
            if (hasNegativeZero) {
                SignedZeroSortCommon::RestoreSignedZeroValuesByIndexVec(sortValueUb, sortIdxUbHalf, sourceOrder_,
                                                                        numTileData_);
            }
        } else {
            AscendC::Sort<T1, false, radixSortConfig>(sortValueUb, sortIdxUbHalf, xLocal, tmpUb, numTileData_);
        }
        inQueueX_.FreeTensor(xLocal);
        LocalTensor<int64_t> sortIdxUbInt64 = sortIdxUb.template ReinterpretCast<int64_t>();
        LocalTensor<int32_t> sortIdxUbInt32 = sortIdxUbHalf.template ReinterpretCast<int32_t>();
        AscendC::Cast(sortIdxUbInt64, sortIdxUbInt32, RoundMode::CAST_NONE, numTileData_);
    } else {
        if constexpr (SignedZeroSortCommon::IS_FLOATING_POINT_V<T1>) {
            AscendC::Sort<T1, false, radixSortConfig>(sortValueUb, sortIdxUb, xLocal, tmpUb, numTileData_);
            if (hasNegativeZero) {
                SignedZeroSortCommon::RestoreSignedZeroValuesByIndexVec(sortValueUb, sortIdxUb, sourceOrder_,
                                                                        numTileData_);
            }
        } else {
            AscendC::Sort<T1, false, radixSortConfig>(sortValueUb, sortIdxUb, xLocal, tmpUb, numTileData_);
        }
        inQueueX_.FreeTensor(xLocal);
    }
    outValueQueue_.EnQue<T1>(sortValueUb);
    sortValueUb = outValueQueue_.DeQue<T1>();
    outIdxQueue_.EnQue<uint32_t>(sortIdxUb);
    sortIdxUb = outIdxQueue_.DeQue<uint32_t>();
    DataCopyPad(outValueGm_[gmOffset + tileOffset], sortValueUb, dataCopyParam);
    dataCopyParam.blockLen = numTileData_ * sizeof(T2);
    if constexpr (sizeof(T2) == sizeof(int32_t)) {
        DataCopyPad(outIdxGm_[gmOffset + tileOffset], sortIdxUb, dataCopyParam);
    } else {
        DataCopyPad(outIdxGm_[(gmOffset + tileOffset) * 2], sortIdxUb, dataCopyParam);
    }
    outValueQueue_.FreeTensor(sortValueUb);
    outIdxQueue_.FreeTensor(sortIdxUb);
}
template <typename T1, typename T2, bool isDescend>
__aicore__ inline void SortRadixOneCore<T1, T2, isDescend>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    for (uint32_t i = 0; i < sortLoopTimes_; i++) {
        int64_t loopOffset = i * unsortedDimParallel_ * totalDataNum_;
        ProcessRadixSortOneCore(inputXGm_[loopOffset], loopOffset, i);
    }
}
} // namespace Sort
#endif // RADIX_SORT_ONE_CORE_H
