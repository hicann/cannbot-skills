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
 * \file arg_max_with_value_ar_gather.h
 * \brief arg_max_with_value_ar_gather
 */

#ifndef ARG_MAX_WITH_VALUE_AR_GATHER_H
#define ARG_MAX_WITH_VALUE_AR_GATHER_H

#include "arg_max_with_value_base.h"

namespace ArgMaxWithValue {
using namespace AscendC;

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
class ArgMaxWithValueArGather : public ArgMaxWithValueBase<T1, T2, T3, isMin> {
public:
    __aicore__ inline ArgMaxWithValueArGather(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace,
                                const ArgMaxWithValueTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint16_t copyNum);
    __aicore__ inline void Compute(uint64_t index, uint16_t processNum, LocalTensor<T2>& indiceUb,
                                   LocalTensor<T1>& valuesUb);
    __aicore__ inline void CopyOut(uint64_t offset, uint64_t copyNum);

    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint16_t ELEMENT_PER_BLCK = 32 / sizeof(T1);

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueIndice_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueValues_;
    TBuf<QuePosition::VECCALC> xBuf32;
    TBuf<QuePosition::VECCALC> outBuf32;
    TBuf<QuePosition::VECCALC> castIndice;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T3> indiceGm_;
    GlobalTensor<T1> valuesGm_;

    uint32_t blockIdx_ = 0;
    uint64_t blockOffset_ = 0;
    uint64_t loopOffset_ = 0;
    uint64_t blkProcessNum_ = 0;
    uint64_t loopNum_ = 0;
    uint16_t loopTail_ = 0;

    uint16_t vlSize_ = 1;
    uint64_t loopR_ = 0;
    uint64_t tailR_ = 0;
    // tiling params
    const ArgMaxWithValueTilingData* tiling_;
    uint64_t blkFactor_ = 0;
    uint64_t blkTailFactor_ = 0;
    uint64_t realCoreNum_ = 0;
    uint64_t rSize_ = 0;
    uint16_t cutASize_ = 0;

    // datacopy params
    DataCopyPadExtParams<T1> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyInParams_{1, 0, 0, 0, 0};
};

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArGather<T1, T2, T3, withValue, isMin>::Init(
    GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, const ArgMaxWithValueTilingData* tilingData,
    TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    indiceGm_.SetGlobalBuffer((__gm__ T3*)indice);
    valuesGm_.SetGlobalBuffer((__gm__ T1*)values);
    tiling_ = tilingData;
    blkFactor_ = tiling_->blkFactor;
    blkTailFactor_ = tiling_->blkTailFactor;
    realCoreNum_ = tiling_->realCoreNum;
    rSize_ = tiling_->rSize;
    cutASize_ = tiling_->cutASize;

    vlSize_ = this->GetVLSize();
    blkProcessNum_ = blkFactor_;
    blockOffset_ = blockIdx_ * blkFactor_;
    if (blockIdx_ < blkTailFactor_) {
        // 前面主块核处理数据为blkFactor+1
        blkProcessNum_ += 1;
        blockOffset_ += blockIdx_;
    } else {
        blockOffset_ += blkTailFactor_;
    }

    uint64_t alignCutANum = CeilDivision(cutASize_, ELEMENT_PER_BLCK) * ELEMENT_PER_BLCK;
    uint64_t alignNum = alignCutANum * rSize_;
    pipe->InitBuffer(inQueueX_, BUFFER_NUM, alignNum * sizeof(T1));

    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        pipe->InitBuffer(xBuf32, alignNum * sizeof(float));
        pipe->InitBuffer(outBuf32, alignCutANum * sizeof(float));
    }

    // 预留索引B32转int64的类型转换空间
    if constexpr (!IsSameType<T2, T3>::value) {
        pipe->InitBuffer(castIndice, alignCutANum * sizeof(T3));
    }

    pipe->InitBuffer(outQueueValues_, BUFFER_NUM, alignCutANum * sizeof(T1));
    pipe->InitBuffer(outQueueIndice_, BUFFER_NUM, alignCutANum * sizeof(T2));
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArGather<T1, T2, T3, withValue, isMin>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }

    loopNum_ = blkProcessNum_ / cutASize_;
    loopTail_ = blkProcessNum_ % cutASize_;
    // process main loop
    for (uint64_t index = 0; index < loopNum_; index++) {
        loopOffset_ = index * cutASize_;
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        CopyIn(cutASize_);
        Compute(index, cutASize_, indiceUb, valuesUb);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_, cutASize_);
    }

    // process tail loop
    if (loopTail_ > 0) {
        loopOffset_ = loopNum_ * cutASize_;
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        CopyIn(loopTail_);
        Compute(loopNum_, loopTail_, indiceUb, valuesUb);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_, loopTail_);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArGather<T1, T2, T3, withValue, isMin>::CopyIn(uint16_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
    // data copy params
    copyInParams_.blockCount = 1;
    copyInParams_.blockLen = copyNum * rSize_ * sizeof(T1);
    copyInParams_.srcStride = 0;
    copyInParams_.dstStride = 0;

    DataCopyPad(xUb, xGm_[(blockOffset_ + loopOffset_) * rSize_], copyInParams_, padParams_);
    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArGather<T1, T2, T3, withValue, isMin>::Compute(uint64_t index,
                                                                                      uint16_t processNum,
                                                                                      LocalTensor<T2>& indiceUb,
                                                                                      LocalTensor<T1>& valuesUb)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        LocalTensor<float> outUb32 = outBuf32.Get<float>();
        Cast(xUb32, srcUb, RoundMode::CAST_NONE, processNum * rSize_);
        this->template ArgMaxGatherRa<float, uint32_t, int32_t>(
            (__ubuf__ float*)outUb32.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
            (__ubuf__ float*)xUb32.GetPhyAddr(), processNum, rSize_);
    } else if constexpr (IsSameType<T1, half>::value) {
        this->template ArgMaxGatherRa<half, uint16_t, int16_t>((__ubuf__ T1*)valuesUb.GetPhyAddr(),
                                                               (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                                               (__ubuf__ T1*)srcUb.GetPhyAddr(), processNum, rSize_);
    } else if constexpr (IsSameType<T1, float>::value || IsSameType<T1, int32_t>::value) {
        this->template ArgMaxGatherRa<T1, uint32_t, int32_t>((__ubuf__ T1*)valuesUb.GetPhyAddr(),
                                                             (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                                             (__ubuf__ T1*)srcUb.GetPhyAddr(), processNum, rSize_);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxGatherRaInt64<T1, uint64_t, int64_t>((__ubuf__ T1*)valuesUb.GetPhyAddr(),
                                                                  (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                                                  (__ubuf__ T1*)srcUb.GetPhyAddr(), processNum, rSize_);
    }
    inQueueX_.FreeTensor(srcUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArGather<T1, T2, T3, withValue, isMin>::CopyOut(uint64_t offset, uint64_t copyNum)
{
    LocalTensor<T1> outValues = outQueueValues_.DeQue<T1>();
    if constexpr (withValue) {
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            this->CastToBF16(outValues, outBuf32.Get<float>(), copyNum);
        }
        this->CopyOutValues(valuesGm_, outValues, offset, copyNum);
    }
    outQueueValues_.FreeTensor(outValues);

    LocalTensor<T2> outIndice = outQueueIndice_.DeQue<T2>();
    if constexpr (!IsSameType<T2, T3>::value) {
        LocalTensor<T3> castUb = castIndice.Get<T3>();
        this->CopyOutIndice(indiceGm_, outIndice, offset, copyNum, castUb);
    } else {
        this->CopyOutIndice(indiceGm_, outIndice, offset, copyNum, outIndice);
    }
    outQueueIndice_.FreeTensor(outIndice);
}
} // namespace ArgMaxWithValue

#endif // ARG_MAX_WITH_VALUE_AR_GATHER_H
