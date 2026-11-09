/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file arg_max_with_value_copy.h
 * \brief arg_max_with_value_copy
 */

#ifndef ARG_MAX_WITH_VALUE_COPY_H
#define ARG_MAX_WITH_VALUE_COPY_H

#include "arg_max_with_value_base.h"

namespace ArgMaxWithValue {
using namespace AscendC;

template <typename T1, typename T2, bool withValue, bool isMin>
class ArgMaxWithValueCopy : public ArgMaxWithValueBase<T1, T2, T2, isMin> {
public:
    __aicore__ inline ArgMaxWithValueCopy(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace,
                                const ArgMaxWithValueTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyInX(uint64_t index, uint16_t copyNum);
    __aicore__ inline void CopyOut(uint64_t index, uint64_t copyNum);

    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint16_t ELEMENT_PER_BLCK = 32 / sizeof(T1);
    constexpr static uint32_t MAX_BLCOK_CUNT = 4095;

private:
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueIndice_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T2> indiceGm_;
    GlobalTensor<T1> valuesGm_;

    uint64_t processSize_ = 1024;
    uint32_t blockIdx_ = 0;
    uint64_t blockOffset_ = 0;
    uint64_t blkProcessNum_ = 0;
    // tiling params
    const ArgMaxWithValueTilingData* tiling_;
    // datacopy params
    DataCopyPadExtParams<T1> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyInParams_{1, 0, 0, 0, 0};
};

template <typename T1, typename T2, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueCopy<T1, T2, withValue, isMin>::Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values,
                                                                           GM_ADDR workspace,
                                                                           const ArgMaxWithValueTilingData* tilingData,
                                                                           TPipe* pipe)
{
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    indiceGm_.SetGlobalBuffer((__gm__ T2*)indice);
    valuesGm_.SetGlobalBuffer((__gm__ T1*)values);
    tiling_ = tilingData;

    uint64_t alignNum = CeilDivision(tiling_->cutASize, ELEMENT_PER_BLCK) * ELEMENT_PER_BLCK;
    pipe->InitBuffer(inQueueX_, BUFFER_NUM, alignNum * sizeof(T1));
    pipe->InitBuffer(outQueueIndice_, 1, processSize_ * sizeof(T2));

    blockIdx_ = GetBlockIdx();
    blkProcessNum_ = tiling_->blkFactor;
    blockOffset_ = blockIdx_ * tiling_->blkFactor;
    if (blockIdx_ < tiling_->blkTailFactor) {
        blkProcessNum_ += 1;
        blockOffset_ += blockIdx_;
    } else {
        blockOffset_ += tiling_->blkTailFactor;
    }
}

template <typename T1, typename T2, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueCopy<T1, T2, withValue, isMin>::Process()
{
    if (blockIdx_ >= tiling_->realCoreNum) {
        return;
    }
    // process max
    if constexpr (withValue) {
        uint64_t loop = blkProcessNum_ / tiling_->cutASize;
        uint64_t tail = blkProcessNum_ % tiling_->cutASize;
        for (uint64_t index = 0; index < loop; index++) {
            CopyInX(index, tiling_->cutASize);
            CopyOut(index, tiling_->cutASize);
        }
        if (tail != 0) {
            CopyInX(loop, tail);
            CopyOut(loop, tail);
        }
    }
    // process indices
    uint64_t loopIndices = blkProcessNum_ / processSize_;
    uint64_t tailIndices = blkProcessNum_ % processSize_;
    LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
    Duplicate<T2>(indiceUb, 0, processSize_);
    outQueueIndice_.EnQue(indiceUb);
    LocalTensor<T2> outIndice = outQueueIndice_.DeQue<T2>();

    DataCopyExtParams copyOutIndiceParams{1, 0, 0, 0, 0};
    copyOutIndiceParams.blockCount = 1;
    copyOutIndiceParams.blockLen = processSize_ * sizeof(T2);
    copyOutIndiceParams.srcStride = 0;
    copyOutIndiceParams.dstStride = 0;

    for (uint64_t index = 0; index < loopIndices; index++) {
        DataCopyPad(indiceGm_[blockOffset_ + index * processSize_], outIndice, copyOutIndiceParams);
    }

    if (tailIndices != 0) {
        copyOutIndiceParams.blockLen = tailIndices * sizeof(T2);
        DataCopyPad(indiceGm_[blockOffset_ + loopIndices * processSize_], outIndice, copyOutIndiceParams);
    }
    outQueueIndice_.FreeTensor(outIndice);
}

template <typename T1, typename T2, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueCopy<T1, T2, withValue, isMin>::CopyInX(uint64_t index, uint16_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
    // data copy params
    copyInParams_.blockCount = 1;
    copyInParams_.blockLen = copyNum * sizeof(T1);
    copyInParams_.srcStride = 0;
    copyInParams_.dstStride = 0;

    DataCopyPad(xUb, xGm_[blockOffset_ + index * tiling_->cutASize], copyInParams_, padParams_);
    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueCopy<T1, T2, withValue, isMin>::CopyOut(uint64_t index, uint64_t copyNum)
{
    LocalTensor<T1> outValues = inQueueX_.DeQue<T1>();
    event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
    WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
    DataCopyExtParams copyOutValuesParams{1, 0, 0, 0, 0};
    copyOutValuesParams.blockCount = 1;
    copyOutValuesParams.blockLen = copyNum * sizeof(T1);
    copyOutValuesParams.srcStride = 0;
    copyOutValuesParams.dstStride = 0;
    DataCopyPad(valuesGm_[blockOffset_ + index * tiling_->cutASize], outValues, copyOutValuesParams);

    inQueueX_.FreeTensor(outValues);
}
} // namespace ArgMaxWithValue

#endif // ARG_MAX_WITH_VALUE_COPY_H
