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
 * \file arg_max_with_value_ara.h
 * \brief arg_max_with_value_ara
 */

#ifndef ARG_MAX_WITH_VALUE_ARA_H
#define ARG_MAX_WITH_VALUE_ARA_H

#include "arg_max_with_value_base.h"

namespace ArgMaxWithValue {
using namespace AscendC;

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
class ArgMaxWithValueArA : public ArgMaxWithValueBase<T1, T2, T3, isMin> {
public:
    __aicore__ inline ArgMaxWithValueArA(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace,
                                const ArgMaxWithValueTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessWithCutA();
    __aicore__ inline void ProcessWithCutRA();
    __aicore__ inline void ProcessGatherARAWithCutA();
    __aicore__ inline void ProcessGatherARAWithCutAR();
    __aicore__ inline void ProcessCutRa(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t loopNum);
    __aicore__ inline void ProcessCutRaWithTail(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t loopNum);
    __aicore__ inline void ProcessOnce(uint64_t index, uint64_t loopNum);
    __aicore__ inline void ProcessOnceWithTail(uint64_t index, uint64_t loopNum);
    __aicore__ inline void ProcessOneRaWithGather(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb,
                                                  uint64_t aLoopNum);
    __aicore__ inline void CopyInXWithNDDMA(uint64_t copyNum);
    __aicore__ inline void CopyInXGather(uint64_t copyNum);
    __aicore__ inline void CopyInXWithCutR(uint64_t aIndex, uint64_t rIndex, uint64_t rNum);
    __aicore__ inline void CopyInXCutRa(uint64_t aIndex, uint64_t nextAIndex, uint64_t rIndex, uint64_t aNum,
                                        uint64_t rNum);
    __aicore__ inline void ComputeAndCopyOut(uint64_t copyNum);
    __aicore__ inline void ComputeAndCopyOutGatherV1(uint64_t copyNum, uint64_t loopNextA, uint64_t processASize);
    __aicore__ inline void ComputeCutRa(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t aNum,
                                        uint64_t rIndex, uint64_t rNum);
    __aicore__ inline void ComputeBigR(uint64_t index, uint16_t processNum, LocalTensor<T1>& srcUb,
                                       LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb);
    __aicore__ inline void ComputeGatherV1(uint64_t index, uint16_t processNum, LocalTensor<T1>& srcUb,
                                           LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb);
    __aicore__ inline void ComputeGatherV2(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t aNum,
                                           uint64_t rIndex, uint64_t rNum);
    __aicore__ inline void CopyInAndCompute(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t rIndex,
                                            uint64_t rNum, uint64_t loopNum);
    __aicore__ inline void CopyInAndComputeWithTail(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb,
                                                    uint64_t rIndex, uint64_t rNum, uint64_t loopNum);
    __aicore__ inline void CopyInAndComputeWithGather(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb,
                                                      uint64_t aLoopNum, uint64_t rIndex, uint64_t rNum);
    __aicore__ inline void CopyOut(uint64_t offset, uint64_t copyNum);

    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint16_t ELEMENT_PER_BLCK = 32 / sizeof(T1);
    constexpr static uint32_t ARA_MODE_101_NUM = 101;
    constexpr static uint32_t ARA_MODE_102_NUM = 102;
    constexpr static uint32_t ARA_MODE_103_NUM = 103;
    constexpr static uint32_t ARA_MODE_104_NUM = 104;

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueIndice_;
    TQue<QuePosition::VECOUT, 1> outQueueValues_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> valuesGm_;
    GlobalTensor<T3> indiceGm_;
    TBuf<QuePosition::VECCALC> tempValue;
    TBuf<QuePosition::VECCALC> tempIndice;
    TBuf<QuePosition::VECCALC> castIndice;
    TBuf<QuePosition::VECCALC> xBuf32;
    TBuf<QuePosition::VECCALC> outBuf32;

    uint64_t processSize_ = 1024;
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
    uint64_t rLoopNum_ = 1;
    uint64_t rTail_ = 0;
    uint64_t copyOutNum_ = 0;
    // tiling params
    const ArgMaxWithValueTilingData* tiling_;
    TPipe* pipe_;
    // datacopy params
    DataCopyPadExtParams<T1> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyInParams_{1, 0, 0, 0, 0};
};

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::Init(
    GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, const ArgMaxWithValueTilingData* tilingData,
    TPipe* pipe)
{
    pipe_ = pipe;
    indiceGm_.SetGlobalBuffer((__gm__ T3*)indice);
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    valuesGm_.SetGlobalBuffer((__gm__ T1*)values);
    tiling_ = tilingData;
    vlSize_ = this->GetVLSize();

    uint64_t alignNum = CeilDivision(tiling_->cutASize * tiling_->cutNextASize * tiling_->cutRSize, ELEMENT_PER_BLCK) *
                        ELEMENT_PER_BLCK;
    pipe_->InitBuffer(inQueueX_, BUFFER_NUM, alignNum * sizeof(T1));
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        pipe_->InitBuffer(xBuf32, alignNum * sizeof(float));
        pipe_->InitBuffer(outBuf32, processSize_ * BUFFER_NUM * sizeof(float));
    }
    if constexpr (!IsSameType<T2, T3>::value) {
        pipe_->InitBuffer(castIndice, processSize_ * sizeof(T3));
    }
    pipe_->InitBuffer(outQueueIndice_, 1, processSize_ * BUFFER_NUM * sizeof(T2));
    pipe_->InitBuffer(outQueueValues_, 1, processSize_ * BUFFER_NUM * sizeof(T1));

    blockIdx_ = GetBlockIdx();
    blockOffset_ = blockIdx_ * tiling_->blkFactor;
    blkProcessNum_ = tiling_->blkFactor;
    if (blockIdx_ < tiling_->blkTailFactor) {
        blkProcessNum_ += 1;
        blockOffset_ += blockIdx_;
    } else {
        blockOffset_ += tiling_->blkTailFactor;
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::Process()
{
    if (blockIdx_ >= tiling_->realCoreNum) {
        return;
    }

    switch (tiling_->aRaMode) {
        case (ARA_MODE_104_NUM):
            ProcessWithCutRA();
            break;
        case (ARA_MODE_102_NUM):
            ProcessWithCutA();
            break;
        case (ARA_MODE_101_NUM):
            ProcessGatherARAWithCutA();
            break;
        case (ARA_MODE_103_NUM):
            ProcessGatherARAWithCutAR();
            break;
        default:
            break;
    }
}

// aRaMode 101
template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ProcessGatherARAWithCutA()
{
    // 不切R场景，即R*nextA能在UB上放下，分核和核内切分均按照A进行
    blockOffset_ = blockOffset_ * tiling_->nextASize;
    uint64_t loop = blkProcessNum_ / tiling_->cutASize;
    uint64_t tail = blkProcessNum_ % tiling_->cutASize;

    uint64_t loopNextA = processSize_ / tiling_->nextASize;
    uint64_t processASize = loopNextA * tiling_->nextASize;
    // process main loop
    loopNum_ = tiling_->cutASize / loopNextA;
    loopTail_ = tiling_->cutASize % loopNextA;
    for (uint64_t index = 0; index < loop; index++) {
        loopOffset_ = index * tiling_->cutASize * tiling_->nextASize;
        CopyInXGather(tiling_->cutASize);
        ComputeAndCopyOutGatherV1(tiling_->cutASize, loopNextA, processASize);
    }
    // process tail loop
    if (tail > 0) {
        loopOffset_ = loop * tiling_->cutASize * tiling_->nextASize;
        loopNum_ = tail / loopNextA;
        loopTail_ = tail % loopNextA;
        CopyInXGather(tail);
        ComputeAndCopyOutGatherV1(tail, loopNextA, processASize);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::CopyInXGather(uint64_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
    // data copy params
    copyInParams_.blockCount = 1;
    copyInParams_.blockLen = copyNum * tiling_->rSize * tiling_->nextASize * sizeof(T1);
    copyInParams_.srcStride = 0;
    copyInParams_.dstStride = 0;

    DataCopyPad(xUb, xGm_[(blockOffset_ + loopOffset_) * tiling_->rSize], copyInParams_, padParams_);
    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ComputeAndCopyOutGatherV1(
    uint64_t copyNum, uint64_t loopNextA, uint64_t processAsize)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        Cast(xBuf32.Get<float>(), srcUb, RoundMode::CAST_NONE, copyNum * tiling_->rSize * tiling_->nextASize);
    }

    for (uint64_t idxI = 0; idxI < loopNum_; idxI++) {
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        ComputeGatherV1(idxI * loopNextA, loopNextA, srcUb, indiceUb, valuesUb);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_ + idxI * processAsize, processAsize);
    }

    if (loopTail_ != 0) {
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        ComputeGatherV1(loopNum_ * loopNextA, loopTail_, srcUb, indiceUb, valuesUb);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_ + loopNum_ * processAsize, loopTail_ * tiling_->nextASize);
    }
    inQueueX_.FreeTensor(srcUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ComputeGatherV1(
    uint64_t index, uint16_t processNum, LocalTensor<T1>& srcUb, LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb)
{
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        LocalTensor<float> outUb32 = outBuf32.Get<float>();
        this->template ArgMaxGatherV1<float, uint32_t, int32_t>(
            (__ubuf__ float*)outUb32.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
            (__ubuf__ float*)xUb32[index * tiling_->nextASize * tiling_->rSize].GetPhyAddr(), processNum,
            tiling_->cutRSize, tiling_->nextASize, 0);
    } else if constexpr (IsSameType<T1, half>::value) {
        this->template ArgMaxGatherV1<T1, uint16_t, int16_t>(
            (__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
            (__ubuf__ T1*)srcUb[index * tiling_->nextASize * tiling_->rSize].GetPhyAddr(), processNum,
            tiling_->cutRSize, tiling_->nextASize, 0);
    } else if constexpr (!IsSameType<T1, int64_t>::value) {
        this->template ArgMaxGatherV1<T1, uint32_t, int32_t>(
            (__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
            (__ubuf__ T1*)srcUb[index * tiling_->nextASize * tiling_->rSize].GetPhyAddr(), processNum,
            tiling_->cutRSize, tiling_->nextASize, 0);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxGatherV1Int64<T1, uint64_t, int64_t>(
            (__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
            (__ubuf__ T1*)srcUb[index * tiling_->nextASize * tiling_->rSize].GetPhyAddr(), processNum,
            tiling_->cutRSize, tiling_->nextASize, 0);
    }
}

// aRaMode 103
template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ProcessGatherARAWithCutAR()
{
    // 切R场景，即R * nextA不能在UB上放下
    // init two tmp buffer
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        pipe_->InitBuffer(tempValue, processSize_ * sizeof(float));
    } else {
        pipe_->InitBuffer(tempValue, processSize_ * sizeof(T1));
    }
    pipe_->InitBuffer(tempIndice, processSize_ * sizeof(T2));

    blockOffset_ = blockOffset_ * tiling_->nextASize;
    uint64_t loopNextA = processSize_ / tiling_->nextASize;
    uint64_t processASize = loopNextA * tiling_->nextASize;

    uint64_t loop = blkProcessNum_ / loopNextA;
    uint64_t tail = blkProcessNum_ % loopNextA;

    rLoopNum_ = tiling_->rSize / tiling_->cutRSize;
    rTail_ = tiling_->rSize % tiling_->cutRSize;

    // process main loop
    for (uint64_t index = 0; index < loop; index++) {
        loopOffset_ = index * processASize;
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        ProcessOneRaWithGather(indiceUb, valuesUb, loopNextA);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_, processASize);
    }
    // process tail loop
    if (tail > 0) {
        loopOffset_ = loop * processASize;
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        ProcessOneRaWithGather(indiceUb, valuesUb, tail);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_, tail * tiling_->nextASize);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ProcessOneRaWithGather(
    LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t aLoopNum)
{
    // process first r loop
    CopyInAndComputeWithGather(indiceUb, valuesUb, aLoopNum, 0, tiling_->cutRSize);
    // process middle r loop
    LocalTensor<T2> tempIndiceUb = tempIndice.Get<T2>();
    LocalTensor<T1> tempValueUb = tempValue.Get<T1>();
    for (uint64_t rIndex = 1; rIndex < rLoopNum_; rIndex++) {
        CopyInAndComputeWithGather(tempIndiceUb, tempValueUb, aLoopNum, rIndex, tiling_->cutRSize);
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            this->template UpdateResult<float>(
                (__ubuf__ T2*)tempIndiceUb.GetPhyAddr(), (__ubuf__ float*)tempValue.Get<float>().GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ float*)outBuf32.Get<float>().GetPhyAddr(),
                aLoopNum * tiling_->nextASize);
        } else {
            this->template UpdateResult<T1>((__ubuf__ T2*)tempIndiceUb.GetPhyAddr(),
                                            (__ubuf__ T1*)tempValueUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                            (__ubuf__ T1*)valuesUb.GetPhyAddr(), aLoopNum * tiling_->nextASize);
        }
    }
    // process last r loop
    if (rTail_ > 0) {
        CopyInAndComputeWithGather(tempIndiceUb, tempValueUb, aLoopNum, rLoopNum_, rTail_);
        if constexpr (!IsSameType<T1, bfloat16_t>::value) {
            this->template UpdateResult<T1>((__ubuf__ T2*)tempIndiceUb.GetPhyAddr(),
                                            (__ubuf__ T1*)tempValueUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                            (__ubuf__ T1*)valuesUb.GetPhyAddr(), aLoopNum * tiling_->nextASize);
        } else {
            this->template UpdateResult<float>(
                (__ubuf__ T2*)tempIndiceUb.GetPhyAddr(), (__ubuf__ float*)tempValue.Get<float>().GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ float*)outBuf32.Get<float>().GetPhyAddr(),
                aLoopNum * tiling_->nextASize);
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::CopyInAndComputeWithGather(
    LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t aLoopNum, uint64_t rIndex, uint64_t rNum)
{
    for (uint64_t aIndex = 0; aIndex < aLoopNum; aIndex++) {
        CopyInXWithCutR(aIndex, rIndex, rNum);
        ComputeGatherV2(indiceUb, valuesUb, aIndex, rIndex, rNum);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::CopyInXWithCutR(uint64_t aIndex,
                                                                                         uint64_t rIndex, uint64_t rNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
    // data copy params
    copyInParams_.blockCount = 1;
    copyInParams_.blockLen = rNum * tiling_->nextASize * sizeof(T1);
    copyInParams_.srcStride = 0;
    copyInParams_.dstStride = 0;

    DataCopyPad(xUb,
                xGm_[(blockOffset_ + loopOffset_ + aIndex * tiling_->nextASize) * tiling_->rSize +
                     rIndex * tiling_->cutRSize * tiling_->nextASize],
                copyInParams_, padParams_);
    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ComputeGatherV2(LocalTensor<T2>& indiceUb,
                                                                                         LocalTensor<T1>& valuesUb,
                                                                                         uint64_t aIndex,
                                                                                         uint64_t rIndex, uint64_t rNum)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        LocalTensor<float> outUb32 = outBuf32.Get<float>();
        Cast(xUb32, srcUb, RoundMode::CAST_NONE, tiling_->cutNextASize * rNum);
        if (rIndex == 0) {
            this->template ArgMaxGatherV1<float, uint32_t, int32_t>(
                (__ubuf__ float*)outUb32[aIndex * tiling_->nextASize].GetPhyAddr(),
                (__ubuf__ T2*)indiceUb[aIndex * tiling_->nextASize].GetPhyAddr(), (__ubuf__ float*)xUb32.GetPhyAddr(),
                1, rNum, tiling_->nextASize, rIndex * tiling_->cutRSize);
        } else {
            this->template ArgMaxGatherV1<float, uint32_t, int32_t>(
                (__ubuf__ float*)tempValue.Get<float>()[aIndex * tiling_->nextASize].GetPhyAddr(),
                (__ubuf__ T2*)indiceUb[aIndex * tiling_->nextASize].GetPhyAddr(), (__ubuf__ float*)xUb32.GetPhyAddr(),
                1, rNum, tiling_->nextASize, rIndex * tiling_->cutRSize);
        }
    } else if constexpr (IsSameType<T1, half>::value) {
        this->template ArgMaxGatherV1<T1, uint16_t, int16_t>(
            (__ubuf__ T1*)valuesUb[aIndex * tiling_->nextASize].GetPhyAddr(),
            (__ubuf__ T2*)indiceUb[aIndex * tiling_->nextASize].GetPhyAddr(), (__ubuf__ T1*)srcUb.GetPhyAddr(), 1, rNum,
            tiling_->nextASize, rIndex * tiling_->cutRSize);
    } else if constexpr (!IsSameType<T1, int64_t>::value) {
        this->template ArgMaxGatherV1<T1, uint32_t, int32_t>(
            (__ubuf__ T1*)valuesUb[aIndex * tiling_->nextASize].GetPhyAddr(),
            (__ubuf__ T2*)indiceUb[aIndex * tiling_->nextASize].GetPhyAddr(), (__ubuf__ T1*)srcUb.GetPhyAddr(), 1, rNum,
            tiling_->nextASize, rIndex * tiling_->cutRSize);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxGatherV1Int64<T1, uint64_t, int64_t>(
            (__ubuf__ T1*)valuesUb[aIndex * tiling_->nextASize].GetPhyAddr(),
            (__ubuf__ T2*)indiceUb[aIndex * tiling_->nextASize].GetPhyAddr(), (__ubuf__ T1*)srcUb.GetPhyAddr(), 1, rNum,
            tiling_->nextASize, rIndex * tiling_->cutRSize);
    }
    inQueueX_.FreeTensor(srcUb);
}

// aRaMode 104
template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ProcessWithCutRA()
{
    // init two temp buffer
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        pipe_->InitBuffer(tempValue, processSize_ * sizeof(float));
    } else {
        pipe_->InitBuffer(tempValue, processSize_ * sizeof(T1));
    }
    pipe_->InitBuffer(tempIndice, processSize_ * sizeof(T2));

    aLoopNum_ = CeilDivision(tiling_->nextASize, tiling_->cutNextASize);
    aTail_ = tiling_->nextASize % tiling_->cutNextASize;
    rLoopNum_ = tiling_->rSize / tiling_->cutRSize;
    rTail_ = tiling_->rSize % tiling_->cutRSize;
    loopNum_ = processSize_ / tiling_->cutNextASize;
    loopTail_ = processSize_ % tiling_->cutNextASize;
    uint64_t loop = blkProcessNum_ / loopNum_;
    uint64_t tail = blkProcessNum_ % loopNum_;

    if (aTail_ == 0) {
        blockOffset_ = blockOffset_ * tiling_->cutNextASize;
        for (uint64_t index = 0; index < loop; index++) {
            ProcessOnce(index, loopNum_);
        }
        if (tail > 0) {
            ProcessOnce(loop, tail);
        }
    } else {
        for (uint64_t index = 0; index < loop; index++) {
            ProcessOnceWithTail(index, loopNum_);
        }
        if (tail > 0) {
            ProcessOnceWithTail(loop, tail);
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ProcessOnce(uint64_t index, uint64_t loopNum)
{
    loopOffset_ = index * loopNum_ * tiling_->cutNextASize;
    LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
    LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
    ProcessCutRa(indiceUb, valuesUb, loopNum);
    outQueueIndice_.EnQue(indiceUb);
    outQueueValues_.EnQue(valuesUb);
    CopyOut(blockOffset_ + loopOffset_, loopNum * tiling_->cutNextASize);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ProcessOnceWithTail(uint64_t index,
                                                                                             uint64_t loopNum)
{
    copyOutNum_ = 0;
    loopOffset_ = index * loopNum_;
    LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
    LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
    ProcessCutRaWithTail(indiceUb, valuesUb, loopNum);
    outQueueIndice_.EnQue(indiceUb);
    outQueueValues_.EnQue(valuesUb);
    uint64_t offset = blockOffset_ + loopOffset_;
    uint64_t aIndex = offset / aLoopNum_;
    uint64_t nextAIndex = (offset % aLoopNum_) * tiling_->cutNextASize;

    CopyOut(aIndex * tiling_->nextASize + nextAIndex, copyOutNum_);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ProcessCutRa(LocalTensor<T2>& indiceUb,
                                                                                      LocalTensor<T1>& valuesUb,
                                                                                      uint64_t loopNum)
{
    // process first r loop
    uint64_t aNum = loopNum * tiling_->cutNextASize;
    CopyInAndCompute(indiceUb, valuesUb, 0, tiling_->cutRSize, loopNum);
    // process middle r loop
    LocalTensor<T2> tempIndiceUb = tempIndice.Get<T2>();
    LocalTensor<T1> tempValueUb = tempValue.Get<T1>();
    for (uint64_t rIndex = 1; rIndex < rLoopNum_; rIndex++) {
        CopyInAndCompute(tempIndiceUb, tempValueUb, rIndex * tiling_->cutRSize, tiling_->cutRSize, loopNum);
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            this->template UpdateResult<float>(
                (__ubuf__ T2*)tempIndiceUb.GetPhyAddr(), (__ubuf__ float*)tempValue.Get<float>().GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ float*)outBuf32.Get<float>().GetPhyAddr(), aNum);
        } else {
            this->template UpdateResult<T1>((__ubuf__ T2*)tempIndiceUb.GetPhyAddr(),
                                            (__ubuf__ T1*)tempValueUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                            (__ubuf__ T1*)valuesUb.GetPhyAddr(), aNum);
        }
    }
    // process last r loop
    if (rTail_ > 0) {
        CopyInAndCompute(tempIndiceUb, tempValueUb, rLoopNum_ * tiling_->cutRSize, rTail_, loopNum);
        if constexpr (!IsSameType<T1, bfloat16_t>::value) {
            this->template UpdateResult<T1>((__ubuf__ T2*)tempIndiceUb.GetPhyAddr(),
                                            (__ubuf__ T1*)tempValueUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                            (__ubuf__ T1*)valuesUb.GetPhyAddr(), aNum);
        } else {
            this->template UpdateResult<float>(
                (__ubuf__ T2*)tempIndiceUb.GetPhyAddr(), (__ubuf__ float*)tempValue.Get<float>().GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ float*)outBuf32.Get<float>().GetPhyAddr(), aNum);
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ProcessCutRaWithTail(LocalTensor<T2>& indiceUb,
                                                                                              LocalTensor<T1>& valuesUb,
                                                                                              uint64_t loopNum)
{
    // process first r loop
    CopyInAndComputeWithTail(indiceUb, valuesUb, 0, tiling_->cutRSize, loopNum);
    // process middle r loop
    LocalTensor<T2> tempIndiceUb = tempIndice.Get<T2>();
    LocalTensor<T1> tempValueUb = tempValue.Get<T1>();
    for (uint64_t rIndex = 1; rIndex < rLoopNum_; rIndex++) {
        CopyInAndComputeWithTail(tempIndiceUb, tempValueUb, rIndex * tiling_->cutRSize, tiling_->cutRSize, loopNum);
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            this->template UpdateResult<float>(
                (__ubuf__ T2*)tempIndiceUb.GetPhyAddr(), (__ubuf__ float*)tempValue.Get<float>().GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ float*)outBuf32.Get<float>().GetPhyAddr(), copyOutNum_);
        } else {
            this->template UpdateResult<T1>((__ubuf__ T2*)tempIndiceUb.GetPhyAddr(),
                                            (__ubuf__ T1*)tempValueUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                            (__ubuf__ T1*)valuesUb.GetPhyAddr(), copyOutNum_);
        }
    }
    // process last r loop
    if (rTail_ > 0) {
        CopyInAndComputeWithTail(tempIndiceUb, tempValueUb, rLoopNum_ * tiling_->cutRSize, rTail_, loopNum);
        if constexpr (!IsSameType<T1, bfloat16_t>::value) {
            this->template UpdateResult<T1>((__ubuf__ T2*)tempIndiceUb.GetPhyAddr(),
                                            (__ubuf__ T1*)tempValueUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                            (__ubuf__ T1*)valuesUb.GetPhyAddr(), copyOutNum_);
        } else {
            this->template UpdateResult<float>(
                (__ubuf__ T2*)tempIndiceUb.GetPhyAddr(), (__ubuf__ float*)tempValue.Get<float>().GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ float*)outBuf32.Get<float>().GetPhyAddr(), copyOutNum_);
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::CopyInAndCompute(
    LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t rIndex, uint64_t rNum, uint64_t loopNum)
{
    uint64_t offsetTmp = blockOffset_ + loopOffset_;
    for (uint64_t index = 0; index < loopNum; index++) {
        uint64_t offset = offsetTmp + index * tiling_->cutNextASize;
        uint64_t aIndex = offset / tiling_->nextASize;
        uint64_t nextAIndex = offset % tiling_->nextASize;
        copyOutNum_ = index * tiling_->cutNextASize;
        CopyInXCutRa(aIndex, nextAIndex, rIndex, tiling_->cutNextASize, rNum);
        ComputeCutRa(indiceUb, valuesUb, tiling_->cutNextASize, rIndex, rNum);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::CopyInAndComputeWithTail(
    LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t rIndex, uint64_t rNum, uint64_t loopNum)
{
    copyOutNum_ = 0;
    for (uint64_t index = 0; index < loopNum; index++) {
        uint64_t offset = blockOffset_ + loopOffset_ + index;
        uint64_t aIndex = offset / aLoopNum_;
        uint64_t nextAIndex = offset % aLoopNum_;
        if (nextAIndex == aLoopNum_ - 1) {
            CopyInXCutRa(aIndex, nextAIndex * tiling_->cutNextASize, rIndex, aTail_, rNum);
            ComputeCutRa(indiceUb, valuesUb, aTail_, rIndex, rNum);
            copyOutNum_ += aTail_;
        } else {
            CopyInXCutRa(aIndex, nextAIndex * tiling_->cutNextASize, rIndex, tiling_->cutNextASize, rNum);
            ComputeCutRa(indiceUb, valuesUb, tiling_->cutNextASize, rIndex, rNum);
            copyOutNum_ += tiling_->cutNextASize;
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::CopyInXCutRa(uint64_t aIndex,
                                                                                      uint64_t nextAIndex,
                                                                                      uint64_t rIndex, uint64_t aNum,
                                                                                      uint64_t rNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();

    // NDDMA loopInfo init
    NdDmaLoopInfo<NDDMA_DIM_NUM> loopInfo;
    for (int64_t i = 0; i < NDDMA_DIM_NUM; i++) {
        loopInfo.loopLpSize[i] = 0;
        loopInfo.loopRpSize[i] = 0;
    }
    loopInfo.loopSize[0] = aNum;
    loopInfo.loopSrcStride[0] = 1;
    loopInfo.loopDstStride[0] = rNum;

    loopInfo.loopSize[1] = rNum;
    loopInfo.loopSrcStride[1] = tiling_->nextASize;
    loopInfo.loopDstStride[1] = 1;

    loopInfo.loopSize[2] = 1;
    loopInfo.loopSrcStride[2] = tiling_->rSize * tiling_->nextASize;
    loopInfo.loopDstStride[2] = rNum * aNum;

    T1 constValue = 0;
    static constexpr NdDmaConfig config = {false, 0, 0, false};
    NdDmaParams<T1, NDDMA_DIM_NUM> paramsMain = {loopInfo, constValue};
    DataCopy<T1, NDDMA_DIM_NUM, config>(xUb, xGm_[(aIndex * tiling_->rSize + rIndex) * tiling_->nextASize + nextAIndex],
                                        paramsMain);

    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ComputeCutRa(LocalTensor<T2>& indiceUb,
                                                                                      LocalTensor<T1>& valuesUb,
                                                                                      uint64_t aNum, uint64_t rIndex,
                                                                                      uint64_t rNum)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        LocalTensor<float> outUb32 = outBuf32.Get<float>();
        Cast(xUb32, srcUb, RoundMode::CAST_NONE, tiling_->cutNextASize * rNum);
        if (rIndex == 0) {
            this->template ArgMaxV1<float, uint32_t>((__ubuf__ float*)outUb32[copyOutNum_].GetPhyAddr(),
                                                     (__ubuf__ T2*)indiceUb[copyOutNum_].GetPhyAddr(),
                                                     (__ubuf__ float*)xUb32.GetPhyAddr(), aNum, rNum, rIndex);
        } else {
            this->template ArgMaxV1<float, uint32_t>((__ubuf__ float*)tempValue.Get<float>()[copyOutNum_].GetPhyAddr(),
                                                     (__ubuf__ T2*)indiceUb[copyOutNum_].GetPhyAddr(),
                                                     (__ubuf__ float*)xUb32.GetPhyAddr(), aNum, rNum, rIndex);
        }
    } else if constexpr (IsSameType<T1, half>::value) {
        this->template ArgMaxV1<T1, uint16_t>((__ubuf__ T1*)valuesUb[copyOutNum_].GetPhyAddr(),
                                              (__ubuf__ T2*)indiceUb[copyOutNum_].GetPhyAddr(),
                                              (__ubuf__ T1*)srcUb.GetPhyAddr(), aNum, rNum, rIndex);
    } else if constexpr (!IsSameType<T1, int64_t>::value) {
        this->template ArgMaxV1<T1, uint32_t>((__ubuf__ T1*)valuesUb[copyOutNum_].GetPhyAddr(),
                                              (__ubuf__ T2*)indiceUb[copyOutNum_].GetPhyAddr(),
                                              (__ubuf__ T1*)srcUb.GetPhyAddr(), aNum, rNum, rIndex);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxV1Int64((__ubuf__ T1*)valuesUb[copyOutNum_].GetPhyAddr(),
                                     (__ubuf__ T2*)indiceUb[copyOutNum_].GetPhyAddr(), (__ubuf__ T1*)srcUb.GetPhyAddr(),
                                     aNum, rNum, rIndex);
    }
    inQueueX_.FreeTensor(srcUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ProcessWithCutA()
{
    blockOffset_ = blockOffset_ * tiling_->nextASize;
    uint64_t loop = blkProcessNum_ / tiling_->cutASize;
    uint64_t tail = blkProcessNum_ % tiling_->cutASize;
    loopNum_ = tiling_->cutASize * tiling_->nextASize / processSize_;
    loopTail_ = tiling_->cutASize * tiling_->nextASize % processSize_;
    // process main loop
    for (uint64_t index = 0; index < loop; index++) {
        loopOffset_ = index * tiling_->cutASize * tiling_->nextASize;
        CopyInXWithNDDMA(tiling_->cutASize);
        ComputeAndCopyOut(tiling_->cutASize);
    }
    // process tail loop
    if (tail > 0) {
        loopOffset_ = loop * tiling_->cutASize * tiling_->nextASize;
        loopNum_ = tail * tiling_->nextASize / processSize_;
        loopTail_ = tail * tiling_->nextASize % processSize_;
        CopyInXWithNDDMA(tail);
        ComputeAndCopyOut(tail);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::CopyInXWithNDDMA(uint64_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();

    // NDDMA loopInfo init
    NdDmaLoopInfo<NDDMA_DIM_NUM> loopInfo;
    for (int64_t i = 0; i < NDDMA_DIM_NUM; i++) {
        loopInfo.loopLpSize[i] = 0;
        loopInfo.loopRpSize[i] = 0;
    }
    loopInfo.loopSize[0] = tiling_->nextASize;
    loopInfo.loopSrcStride[0] = 1;
    loopInfo.loopDstStride[0] = tiling_->rSize;

    loopInfo.loopSize[1] = tiling_->rSize;
    loopInfo.loopSrcStride[1] = tiling_->nextASize;
    loopInfo.loopDstStride[1] = 1;

    loopInfo.loopSize[2] = copyNum;
    loopInfo.loopSrcStride[2] = tiling_->rSize * tiling_->nextASize;
    loopInfo.loopDstStride[2] = tiling_->rSize * tiling_->nextASize;

    T1 constValue = 0;
    static constexpr NdDmaConfig config = {false, 0, 0, false};
    NdDmaParams<T1, NDDMA_DIM_NUM> paramsMain = {loopInfo, constValue};
    DataCopy<T1, NDDMA_DIM_NUM, config>(xUb, xGm_[(blockOffset_ + loopOffset_) * tiling_->rSize], paramsMain);

    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ComputeAndCopyOut(uint64_t copyNum)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        Cast(xBuf32.Get<float>(), srcUb, RoundMode::CAST_NONE, copyNum * tiling_->rSize * tiling_->nextASize);
    }

    for (uint64_t idxI = 0; idxI < loopNum_; idxI++) {
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        ComputeBigR(idxI * processSize_, processSize_, srcUb, indiceUb, valuesUb);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_ + idxI * processSize_, processSize_);
    }

    if (loopTail_ != 0) {
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        ComputeBigR(loopNum_ * processSize_, loopTail_, srcUb, indiceUb, valuesUb);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_ + loopNum_ * processSize_, loopTail_);
    }
    inQueueX_.FreeTensor(srcUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::ComputeBigR(
    uint64_t index, uint16_t processNum, LocalTensor<T1>& srcUb, LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb)
{
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        LocalTensor<float> outUb32 = outBuf32.Get<float>();
        this->template ArgMaxV1<float, uint32_t>(
            (__ubuf__ float*)outUb32.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
            (__ubuf__ float*)xUb32[index * tiling_->rSize].GetPhyAddr(), processNum, tiling_->cutRSize);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxV1Int64((__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                     (__ubuf__ T1*)srcUb[index * tiling_->rSize].GetPhyAddr(), processNum,
                                     tiling_->cutRSize);
    } else if constexpr (IsSameType<T1, half>::value) {
        this->template ArgMaxV1<T1, uint16_t>((__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                              (__ubuf__ T1*)srcUb[index * tiling_->rSize].GetPhyAddr(), processNum,
                                              tiling_->cutRSize);
    } else {
        this->template ArgMaxV1<T1, uint32_t>((__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                              (__ubuf__ T1*)srcUb[index * tiling_->rSize].GetPhyAddr(), processNum,
                                              tiling_->cutRSize);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArA<T1, T2, T3, withValue, isMin>::CopyOut(uint64_t offset, uint64_t copyNum)
{
    LocalTensor<T2> outIndice = outQueueIndice_.DeQue<T2>();
    if constexpr (!IsSameType<T2, T3>::value) {
        LocalTensor<T3> castUb = castIndice.Get<T3>();
        this->CopyOutIndice(indiceGm_, outIndice, offset, copyNum, castUb);
    } else {
        this->CopyOutIndice(indiceGm_, outIndice, offset, copyNum, outIndice);
    }
    outQueueIndice_.FreeTensor(outIndice);

    LocalTensor<T1> outValues = outQueueValues_.DeQue<T1>();
    if constexpr (withValue) {
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            this->CastToBF16(outValues, outBuf32.Get<float>(), copyNum);
        }
        this->CopyOutValues(valuesGm_, outValues, offset, copyNum);
    }
    outQueueValues_.FreeTensor(outValues);
}
} // namespace ArgMaxWithValue

#endif // ARG_MAX_WITH_VALUE_ARA_H
