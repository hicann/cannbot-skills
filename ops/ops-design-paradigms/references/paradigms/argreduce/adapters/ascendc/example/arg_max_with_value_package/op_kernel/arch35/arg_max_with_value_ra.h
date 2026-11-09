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
 * \file arg_max_with_value_ra.h
 * \brief arg_max_with_value_ra
 */

#ifndef ARG_MAX_WITH_VALUE_RA_H
#define ARG_MAX_WITH_VALUE_RA_H

#include "arg_max_with_value_base.h"

namespace ArgMaxWithValue {
using namespace AscendC;

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
class ArgMaxWithValueRa : public ArgMaxWithValueBase<T1, T2, T3, isMin> {
public:
    __aicore__ inline ArgMaxWithValueRa(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace,
                                const ArgMaxWithValueTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessWithCutA();
    __aicore__ inline void ProcessWithCutR();
    __aicore__ inline void ProcessMiddleLoop(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t aSize);
    __aicore__ inline void ProcessTailLoop(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t aSize);

    __aicore__ inline void CopyIn(uint64_t rIndex, uint16_t count, uint16_t copyNum);
    __aicore__ inline void Compute(uint64_t rNum, uint16_t offset, LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb,
                                   uint32_t startR);
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
    uint64_t nextASize_ = 0;
    uint16_t cutNextASize_ = 0;

    // datacopy params
    uint32_t ubFactorAlign_ = 0;
    uint64_t outASize_ = 0;
    uint64_t outAAlign_ = 0;
    DataCopyPadExtParams<T1> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyInParams_{1, 0, 0, 0, 0};
};

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueRa<T1, T2, T3, withValue, isMin>::Init(
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
    nextASize_ = tiling_->nextASize;
    cutNextASize_ = tiling_->cutNextASize;

    vlSize_ = this->GetVLSize();
    blkProcessNum_ = blkFactor_;
    blockOffset_ = blockIdx_ * blkFactor_;
    if (blockIdx_ < blkTailFactor_) {
        blkProcessNum_ += 1;
        blockOffset_ += blockIdx_;
    } else {
        blockOffset_ += blkTailFactor_;
    }

    uint64_t alignCutANum = CeilDivision(cutNextASize_, ELEMENT_PER_BLCK) * ELEMENT_PER_BLCK;
    uint64_t alignNum = alignCutANum * tiling_->cutRSize;
    ubFactorAlign_ = alignCutANum;
    outAAlign_ = alignCutANum;

    pipe->InitBuffer(inQueueX_, BUFFER_NUM, alignNum * sizeof(T1));
    if (tiling_->cutRSize != tiling_->rSize) {
        alignCutANum = alignCutANum * BUFFER_NUM;
    }

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
__aicore__ inline void ArgMaxWithValueRa<T1, T2, T3, withValue, isMin>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }

    if (tiling_->cutRSize != tiling_->rSize) {
        ProcessWithCutR();
    } else {
        ProcessWithCutA();
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueRa<T1, T2, T3, withValue, isMin>::ProcessWithCutA()
{
    loopNum_ = blkProcessNum_ / cutNextASize_;
    loopTail_ = blkProcessNum_ % cutNextASize_;
    outASize_ = cutNextASize_;
    // process main loop
    for (uint64_t index = 0; index < loopNum_; index++) {
        loopOffset_ = index * cutNextASize_;
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        CopyIn(0, rSize_, cutNextASize_);
        Compute(rSize_, 0, indiceUb, valuesUb, 0);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_, cutNextASize_);
    }

    // process tail loop
    if (loopTail_ > 0) {
        outASize_ = loopTail_;
        // 对齐值更新为loopTail_的对齐值
        ubFactorAlign_ = CeilDivision(loopTail_, ELEMENT_PER_BLCK) * ELEMENT_PER_BLCK;
        loopOffset_ = loopNum_ * cutNextASize_;
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        CopyIn(0, rSize_, loopTail_);
        Compute(rSize_, 0, indiceUb, valuesUb, 0);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_, loopTail_);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueRa<T1, T2, T3, withValue, isMin>::ProcessWithCutR()
{
    loopR_ = tiling_->rSize / tiling_->cutRSize;
    tailR_ = tiling_->rSize % tiling_->cutRSize;
    loopNum_ = blkProcessNum_ / cutNextASize_;
    loopTail_ = blkProcessNum_ % cutNextASize_;
    outASize_ = cutNextASize_;
    // process main loop
    for (uint64_t index = 0; index < loopNum_; index++) {
        loopOffset_ = index * cutNextASize_;
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        // process first r loop
        CopyIn(0, tiling_->cutRSize, cutNextASize_);
        Compute(tiling_->cutRSize, 0, indiceUb, valuesUb, 0);
        // process middle r loop
        ProcessMiddleLoop(indiceUb, valuesUb, cutNextASize_);
        // process last r loop
        ProcessTailLoop(indiceUb, valuesUb, cutNextASize_);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_, cutNextASize_);
    }

    // process tail loop
    if (loopTail_ > 0) {
        outASize_ = loopTail_;
        // 对齐值更新为loopTail_的对齐值
        loopOffset_ = loopNum_ * cutNextASize_;
        ubFactorAlign_ = CeilDivision(loopTail_, ELEMENT_PER_BLCK) * ELEMENT_PER_BLCK;
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        // process first r loop
        CopyIn(0, tiling_->cutRSize, loopTail_);
        Compute(tiling_->cutRSize, 0, indiceUb, valuesUb, 0);
        // process middle r loop
        ProcessMiddleLoop(indiceUb, valuesUb, loopTail_);
        // process last r loop
        ProcessTailLoop(indiceUb, valuesUb, loopTail_);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_, loopTail_);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueRa<T1, T2, T3, withValue, isMin>::ProcessMiddleLoop(LocalTensor<T2>& indiceUb,
                                                                                          LocalTensor<T1>& valuesUb,
                                                                                          uint64_t aSize)
{
    for (uint64_t index = 1; index < loopR_; index++) {
        uint64_t startR = index * tiling_->cutRSize;
        CopyIn(startR, tiling_->cutRSize, aSize);
        Compute(tiling_->cutRSize, outAAlign_, indiceUb, valuesUb, startR);
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            LocalTensor<float> outUb32 = outBuf32.Get<float>();
            this->template UpdateResult<float>(
                (__ubuf__ T2*)indiceUb[outAAlign_].GetPhyAddr(), (__ubuf__ float*)outUb32[outAAlign_].GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ float*)outUb32.GetPhyAddr(), aSize);
        } else {
            this->template UpdateResult<T1>(
                (__ubuf__ T2*)indiceUb[outAAlign_].GetPhyAddr(), (__ubuf__ T1*)valuesUb[outAAlign_].GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)valuesUb.GetPhyAddr(), aSize);
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueRa<T1, T2, T3, withValue, isMin>::ProcessTailLoop(LocalTensor<T2>& indiceUb,
                                                                                        LocalTensor<T1>& valuesUb,
                                                                                        uint64_t aSize)
{
    if (tailR_ > 0) {
        uint64_t startR = loopR_ * tiling_->cutRSize;
        CopyIn(startR, tailR_, aSize);
        Compute(tailR_, outAAlign_, indiceUb, valuesUb, startR);
        if constexpr (!IsSameType<T1, bfloat16_t>::value) {
            this->template UpdateResult<T1>(
                (__ubuf__ T2*)indiceUb[outAAlign_].GetPhyAddr(), (__ubuf__ T1*)valuesUb[outAAlign_].GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)valuesUb.GetPhyAddr(), aSize);
        } else {
            LocalTensor<float> outUb32 = outBuf32.Get<float>();
            this->template UpdateResult<float>(
                (__ubuf__ T2*)indiceUb[outAAlign_].GetPhyAddr(), (__ubuf__ float*)outUb32[outAAlign_].GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ float*)outUb32.GetPhyAddr(), aSize);
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueRa<T1, T2, T3, withValue, isMin>::CopyIn(uint64_t rIndex, uint16_t count,
                                                                               uint16_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
    // data copy params
    copyInParams_.blockCount = count;
    copyInParams_.blockLen = copyNum * sizeof(T1);
    copyInParams_.srcStride = (nextASize_ - copyNum) * sizeof(T1);
    copyInParams_.dstStride = 0;

    DataCopyPad(xUb, xGm_[blockOffset_ + loopOffset_ + rIndex * nextASize_], copyInParams_, padParams_);
    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueRa<T1, T2, T3, withValue, isMin>::Compute(uint64_t rNum, uint16_t offset,
                                                                                LocalTensor<T2>& indiceUb,
                                                                                LocalTensor<T1>& valuesUb,
                                                                                uint32_t startR)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        LocalTensor<float> outUb32 = outBuf32.Get<float>();
        Cast(xUb32, srcUb, RoundMode::CAST_NONE, ubFactorAlign_ * rNum);
        this->template ArgMaxRa<float, uint32_t>(
            (__ubuf__ float*)outUb32[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
            (__ubuf__ float*)xUb32.GetPhyAddr(), outASize_, ubFactorAlign_, rNum, startR);
    } else if constexpr (IsSameType<T1, half>::value) {
        this->template ArgMaxRa<half, uint16_t>(
            (__ubuf__ T1*)valuesUb[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
            (__ubuf__ T1*)srcUb.GetPhyAddr(), outASize_, ubFactorAlign_, rNum, startR);
    } else if constexpr (IsSameType<T1, float>::value || IsSameType<T1, int32_t>::value) {
        this->template ArgMaxRa<T1, uint32_t>(
            (__ubuf__ T1*)valuesUb[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
            (__ubuf__ T1*)srcUb.GetPhyAddr(), outASize_, ubFactorAlign_, rNum, startR);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxAraUnAlignInt64<int64_t, uint64_t>(
            (__ubuf__ T1*)valuesUb[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
            (__ubuf__ T1*)srcUb.GetPhyAddr(), 1, rNum, outASize_, ubFactorAlign_, ubFactorAlign_, startR);
    }
    inQueueX_.FreeTensor(srcUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueRa<T1, T2, T3, withValue, isMin>::CopyOut(uint64_t offset, uint64_t copyNum)
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

#endif // ARG_MAX_WITH_VALUE_RA_H
