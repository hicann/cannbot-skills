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
 * \file arg_max_with_value_ara_gather.h
 * \brief arg_max_with_value_ara_gather
 */

#ifndef ARG_MAX_WITH_VALUE_ARA_GATHER_H
#define ARG_MAX_WITH_VALUE_ARA_GATHER_H

#include "arg_max_with_value_base.h"

namespace ArgMaxWithValue {
using namespace AscendC;

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
class ArgMaxWithValueAraGather : public ArgMaxWithValueBase<T1, T2, T3, isMin> {
public:
    __aicore__ inline ArgMaxWithValueAraGather(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace,
                                const ArgMaxWithValueTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessCutAWithNDDMA();
    __aicore__ inline void ProcessWithCutNextA();
    __aicore__ inline void ProcessOneLoop(uint64_t index);
    __aicore__ inline void ProcessOneLoopTail(uint64_t index);

    __aicore__ inline void CopyInWithNDDMA(uint64_t copyNum);
    __aicore__ inline void CopyInWithMoveAlign(uint64_t copyNum);
    __aicore__ inline void CopyInCutNextA(uint64_t aIndex, uint64_t nextAIndex, uint64_t copyNum);

    __aicore__ inline void ComputeRA(uint16_t processNum, LocalTensor<T1>& srcUb, LocalTensor<T2>& indiceUb,
                                     LocalTensor<T1>& valuesUb);
    __aicore__ inline void ComputeAndCopyOutBigNA(uint64_t copyNum);
    __aicore__ inline void ComputeCutNextA(uint64_t index, LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb);
    __aicore__ inline void ComputeCutNextAWithTail(uint64_t aNum, LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb);

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
    uint64_t aLoopNum_ = 1;
    uint64_t aTail_ = 0;
    uint64_t copyOutNum_ = 0;

    // tiling params
    const ArgMaxWithValueTilingData* tiling_;
    uint64_t blkFactor_ = 0;
    uint64_t blkTailFactor_ = 0;
    uint64_t realCoreNum_ = 0;
    uint64_t rSize_ = 0;
    uint16_t cutASize_ = 0;
    uint64_t nextASize_ = 0;
    uint16_t cutNextASize_ = 0;

    // datacopy params
    uint32_t ubFactorAlignCutA_ = 0;
    uint32_t ubFactorAlignCutNextA_ = 0;
    DataCopyPadExtParams<T1> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyInParams_{1, 0, 0, 0, 0};
};

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::Init(
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
    nextASize_ = tiling_->nextASize;
    cutNextASize_ = tiling_->cutNextASize;

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

    uint64_t alignCutANum = CeilDivision(cutASize_ * cutNextASize_, ELEMENT_PER_BLCK) * ELEMENT_PER_BLCK;
    uint64_t alignNum = alignCutANum * rSize_;
    ubFactorAlignCutNextA_ = CeilDivision(cutNextASize_, ELEMENT_PER_BLCK) * ELEMENT_PER_BLCK;
    ubFactorAlignCutA_ = CeilDivision(cutASize_ * nextASize_, ELEMENT_PER_BLCK) * ELEMENT_PER_BLCK;
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
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    // 判断切分基于A轴还是A*nextA切分
    if (cutASize_ == 1) {
        // UBinner: 1 R cutNextASize
        ProcessWithCutNextA();
    } else {
        // UBinner: cutASize R nextASize
        loopNum_ = blkProcessNum_ / cutASize_;
        loopTail_ = blkProcessNum_ % cutASize_;
        // blockOffset需要乘以nextASize
        blockOffset_ = blockOffset_ * nextASize_;
        ProcessCutAWithNDDMA();
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::ProcessCutAWithNDDMA()
{
    // process main loop
    for (uint64_t index = 0; index < loopNum_; index++) {
        loopOffset_ = index * cutASize_ * nextASize_;
        CopyInWithNDDMA(cutASize_);
        ComputeAndCopyOutBigNA(cutASize_);
    }
    // process tail loop
    if (loopTail_ > 0) {
        ubFactorAlignCutA_ = CeilDivision(loopTail_ * nextASize_, ELEMENT_PER_BLCK) * ELEMENT_PER_BLCK;
        loopOffset_ = loopNum_ * cutASize_ * nextASize_;
        CopyInWithNDDMA(loopTail_);
        ComputeAndCopyOutBigNA(loopTail_);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::ProcessWithCutNextA()
{
    // 计算nextASize能分成多少个cutNextASize
    aLoopNum_ = CeilDivision(nextASize_, cutNextASize_);
    aTail_ = nextASize_ % cutNextASize_;
    // blockFactor传入每核处理的cutNextASize_个数
    loopNum_ = blkProcessNum_;

    if (aTail_ == 0) {
        // blockOffset为以cutNextASize为单位的偏移个数
        blockOffset_ = blockOffset_ * cutNextASize_;
        for (uint64_t index = 0; index < loopNum_; index++) {
            ProcessOneLoop(index);
        }
    } else {
        for (uint64_t index = 0; index < loopNum_; index++) {
            ProcessOneLoopTail(index);
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::ProcessOneLoop(uint64_t index)
{
    loopOffset_ = index * cutNextASize_;
    LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
    LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();

    // 计算出nextA轴上具体大小的offset
    uint64_t offset = blockOffset_ + loopOffset_;
    uint64_t aIndex = offset / nextASize_;
    uint64_t nextAIndex = offset % nextASize_;
    CopyInCutNextA(aIndex, nextAIndex, cutNextASize_);
    ComputeCutNextA(index, indiceUb, valuesUb);

    outQueueIndice_.EnQue(indiceUb);
    outQueueValues_.EnQue(valuesUb);
    CopyOut(blockOffset_ + loopOffset_, cutNextASize_);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::ProcessOneLoopTail(uint64_t index)
{
    copyOutNum_ = 0;
    loopOffset_ = index;
    LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
    LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();

    // 计算出nextA轴上以cutNextASize为单位的offset
    uint64_t offset = blockOffset_ + loopOffset_;
    uint64_t aIndex = offset / aLoopNum_;
    uint64_t nextAIndex = offset % aLoopNum_;
    if (nextAIndex == aLoopNum_ - 1) {
        CopyInCutNextA(aIndex, nextAIndex * cutNextASize_, aTail_);
        ComputeCutNextAWithTail(aTail_, indiceUb, valuesUb);
        copyOutNum_ += aTail_;
    } else {
        CopyInCutNextA(aIndex, nextAIndex * cutNextASize_, cutNextASize_);
        ComputeCutNextAWithTail(cutNextASize_, indiceUb, valuesUb);
        copyOutNum_ += cutNextASize_;
    }

    outQueueIndice_.EnQue(indiceUb);
    outQueueValues_.EnQue(valuesUb);
    nextAIndex = nextAIndex * cutNextASize_;
    CopyOut(aIndex * nextASize_ + nextAIndex, copyOutNum_);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::CopyInCutNextA(uint64_t aIndex,
                                                                                              uint64_t nextAIndex,
                                                                                              uint64_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
    // data copy params
    copyInParams_.blockCount = rSize_;
    copyInParams_.blockLen = copyNum * sizeof(T1);
    copyInParams_.srcStride = (nextASize_ - copyNum) * sizeof(T1);
    copyInParams_.dstStride = 0;

    DataCopyPad(xUb, xGm_[aIndex * rSize_ * nextASize_ + nextAIndex], copyInParams_, padParams_);
    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::ComputeCutNextAWithTail(
    uint64_t aNum, LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();
    ubFactorAlignCutNextA_ = CeilDivision(aNum, ELEMENT_PER_BLCK) * ELEMENT_PER_BLCK;
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        LocalTensor<float> outUb32 = outBuf32.Get<float>();
        Cast(xUb32, srcUb, RoundMode::CAST_NONE, ubFactorAlignCutNextA_ * rSize_);
        this->template ArgMaxRa<float, uint32_t>(
            (__ubuf__ float*)outUb32[copyOutNum_].GetPhyAddr(), (__ubuf__ T2*)indiceUb[copyOutNum_].GetPhyAddr(),
            (__ubuf__ float*)xUb32.GetPhyAddr(), aNum, ubFactorAlignCutNextA_, rSize_);
    } else if constexpr (IsSameType<T1, half>::value) {
        this->template ArgMaxRa<half, uint16_t>((__ubuf__ T1*)valuesUb[copyOutNum_].GetPhyAddr(),
                                                (__ubuf__ T2*)indiceUb[copyOutNum_].GetPhyAddr(),
                                                (__ubuf__ T1*)srcUb.GetPhyAddr(), aNum, ubFactorAlignCutNextA_, rSize_);
    } else if constexpr (IsSameType<T1, float>::value || IsSameType<T1, int32_t>::value) {
        this->template ArgMaxRa<T1, uint32_t>((__ubuf__ T1*)valuesUb[copyOutNum_].GetPhyAddr(),
                                              (__ubuf__ T2*)indiceUb[copyOutNum_].GetPhyAddr(),
                                              (__ubuf__ T1*)srcUb.GetPhyAddr(), aNum, ubFactorAlignCutNextA_, rSize_);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxAraUnAlignInt64<int64_t, uint64_t>(
            (__ubuf__ T1*)valuesUb[copyOutNum_].GetPhyAddr(), (__ubuf__ T2*)indiceUb[copyOutNum_].GetPhyAddr(),
            (__ubuf__ T1*)srcUb.GetPhyAddr(), 1, rSize_, aNum, ubFactorAlignCutNextA_, ubFactorAlignCutNextA_, 0);
    }
    inQueueX_.FreeTensor(srcUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::ComputeCutNextA(
    uint64_t index, LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        LocalTensor<float> outUb32 = outBuf32.Get<float>();
        Cast(xUb32, srcUb, RoundMode::CAST_NONE, ubFactorAlignCutNextA_ * rSize_);
        this->template ArgMaxRa<float, uint32_t>(
            (__ubuf__ float*)outUb32.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
            (__ubuf__ float*)xUb32.GetPhyAddr(), cutNextASize_, ubFactorAlignCutNextA_, rSize_);
    } else if constexpr (IsSameType<T1, half>::value) {
        this->template ArgMaxRa<half, uint16_t>((__ubuf__ T1*)valuesUb.GetPhyAddr(),
                                                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)srcUb.GetPhyAddr(),
                                                cutNextASize_, ubFactorAlignCutNextA_, rSize_);
    } else if constexpr (IsSameType<T1, float>::value || IsSameType<T1, int32_t>::value) {
        this->template ArgMaxRa<T1, uint32_t>((__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                              (__ubuf__ T1*)srcUb.GetPhyAddr(), cutNextASize_, ubFactorAlignCutNextA_,
                                              rSize_);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxAraUnAlignInt64<int64_t, uint64_t>(
            (__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)srcUb.GetPhyAddr(),
            1, rSize_, cutNextASize_, ubFactorAlignCutNextA_, ubFactorAlignCutNextA_, 0);
    }
    inQueueX_.FreeTensor(srcUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::CopyInWithNDDMA(uint64_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();

    // NDDMA loopInfo init: cutA R nextA ==> R cutA nextA
    NdDmaLoopInfo<NDDMA_DIM_NUM> loopInfo;
    for (int64_t i = 0; i < NDDMA_DIM_NUM; i++) {
        loopInfo.loopLpSize[i] = 0;
        loopInfo.loopRpSize[i] = 0;
    }
    loopInfo.loopSize[0] = nextASize_;
    loopInfo.loopSrcStride[0] = 1;
    loopInfo.loopDstStride[0] = 1;

    // 最大连续块的dstStride=cutASize*nextASize要做block对齐
    loopInfo.loopSize[1] = rSize_;
    loopInfo.loopSrcStride[1] = nextASize_;
    loopInfo.loopDstStride[1] = ubFactorAlignCutA_;

    loopInfo.loopSize[2] = copyNum;
    loopInfo.loopSrcStride[2] = rSize_ * nextASize_;
    loopInfo.loopDstStride[2] = nextASize_;

    T1 constValue = 0;
    static constexpr NdDmaConfig config = {false, 0, 0, false};
    NdDmaParams<T1, NDDMA_DIM_NUM> paramsMain = {loopInfo, constValue};
    DataCopy<T1, NDDMA_DIM_NUM, config>(xUb, xGm_[(blockOffset_ + loopOffset_) * rSize_], paramsMain);

    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::CopyInWithMoveAlign(uint64_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
    // data copy params
    copyInParams_.blockCount = 1;
    copyInParams_.blockLen = copyNum * rSize_ * nextASize_ * sizeof(T1);
    copyInParams_.srcStride = 0;
    copyInParams_.dstStride = 0;

    DataCopyPad(xUb, xGm_[(blockOffset_ + loopOffset_) * rSize_], copyInParams_, padParams_);
    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::ComputeAndCopyOutBigNA(uint64_t copyNum)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();

    LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
    LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
    // 对cutASize与nextASize合轴处理
    ComputeRA(copyNum * nextASize_, srcUb, indiceUb, valuesUb);
    outQueueIndice_.EnQue(indiceUb);
    outQueueValues_.EnQue(valuesUb);
    CopyOut(blockOffset_ + loopOffset_, copyNum * nextASize_);

    inQueueX_.FreeTensor(srcUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::ComputeRA(uint16_t processNum,
                                                                                         LocalTensor<T1>& srcUb,
                                                                                         LocalTensor<T2>& indiceUb,
                                                                                         LocalTensor<T1>& valuesUb)
{
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        LocalTensor<float> outUb32 = outBuf32.Get<float>();
        Cast(xBuf32.Get<float>(), srcUb, RoundMode::CAST_NONE, ubFactorAlignCutA_ * rSize_);
        this->template ArgMaxRa<float, uint32_t>(
            (__ubuf__ float*)outUb32.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
            (__ubuf__ float*)xUb32.GetPhyAddr(), processNum, ubFactorAlignCutA_, rSize_);
    } else if constexpr (IsSameType<T1, half>::value) {
        this->template ArgMaxRa<half, uint16_t>((__ubuf__ T1*)valuesUb.GetPhyAddr(),
                                                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)srcUb.GetPhyAddr(),
                                                processNum, ubFactorAlignCutA_, rSize_);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxAraUnAlignInt64<int64_t, uint64_t>(
            (__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)srcUb.GetPhyAddr(),
            1, rSize_, processNum, ubFactorAlignCutA_, ubFactorAlignCutA_, 0);
    } else {
        this->template ArgMaxRa<T1, uint32_t>((__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                              (__ubuf__ T1*)srcUb.GetPhyAddr(), processNum, ubFactorAlignCutA_, rSize_);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAraGather<T1, T2, T3, withValue, isMin>::CopyOut(uint64_t offset,
                                                                                       uint64_t copyNum)
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

#endif // ARG_MAX_WITH_VALUE_ARA_GATHER_H
