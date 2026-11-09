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
 * \file arg_max_with_value_ar.h
 * \brief arg_max_with_value_ar
 */

#ifndef ARG_MAX_WITH_VALUE_AR_H
#define ARG_MAX_WITH_VALUE_AR_H

#include "arg_max_with_value_base.h"

namespace ArgMaxWithValue {
using namespace AscendC;

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
class ArgMaxWithValueAr : public ArgMaxWithValueBase<T1, T2, T3, isMin> {
public:
    __aicore__ inline ArgMaxWithValueAr(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace,
                                const ArgMaxWithValueTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessWithoutCutR();
    __aicore__ inline void ProcessWithCutR();
    __aicore__ inline void ProcessBigR(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb);
    __aicore__ inline void ProcessCutR(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t loopSize);
    template <typename T4>
    __aicore__ inline void ProcessCutRTemp(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t loopSize);
    template <typename T4>
    __aicore__ inline void ProcessCutRB64(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t loopSize);
    __aicore__ inline void CopyInX(uint64_t index, uint16_t copyNum);
    __aicore__ inline void CopyInXCutR(uint64_t bIndex, uint64_t rIindex, uint16_t copyNum);
    __aicore__ inline void Compute(uint64_t index, uint16_t processNum, LocalTensor<T2>& indiceUb,
                                   LocalTensor<T1>& valuesUb);
    __aicore__ inline void ComputeCutR(uint64_t index, uint64_t dimR, LocalTensor<T2>& indiceUb,
                                       LocalTensor<T1>& valuesUb, uint32_t startR);
    __aicore__ inline void CopyOut(uint64_t offset, uint64_t copyNum);
    __aicore__ inline void CopyOutARIndice(GlobalTensor<T3>& indiceGm, LocalTensor<T2>& outIndice, uint64_t offset,
                                           uint64_t copyNum);

    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint16_t ELEMENT_PER_BLCK = 32 / sizeof(T1);

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueIndice_;
    TQue<QuePosition::VECOUT, 1> outQueueValues_;
    TBuf<QuePosition::VECCALC> xBuf32;
    TBuf<QuePosition::VECCALC> outBuf32;
    TBuf<QuePosition::VECCALC> castIndice;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T3> indiceGm_;
    GlobalTensor<T1> valuesGm_;

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
    // tiling params
    const ArgMaxWithValueTilingData* tiling_;
    // datacopy params
    DataCopyPadExtParams<T1> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyInParams_{1, 0, 0, 0, 0};
};

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::Init(
    GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, const ArgMaxWithValueTilingData* tilingData,
    TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    indiceGm_.SetGlobalBuffer((__gm__ T3*)indice);
    valuesGm_.SetGlobalBuffer((__gm__ T1*)values);
    tiling_ = tilingData;

    blkProcessNum_ = tiling_->blkFactor;
    blockOffset_ = blockIdx_ * tiling_->blkFactor;
    if (blockIdx_ < tiling_->blkTailFactor) {
        blkProcessNum_ += 1;
        blockOffset_ += blockIdx_;
    } else {
        blockOffset_ += tiling_->blkTailFactor;
    }

    vlSize_ = this->GetVLSize();
    uint64_t alignNum = CeilDivision(tiling_->cutASize * tiling_->cutRSize, ELEMENT_PER_BLCK) * ELEMENT_PER_BLCK +
                        vlSize_;
    pipe->InitBuffer(inQueueX_, BUFFER_NUM, alignNum * sizeof(T1));

    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        pipe->InitBuffer(xBuf32, alignNum * sizeof(float));
        pipe->InitBuffer(outBuf32, processSize_ * BUFFER_NUM * sizeof(float));
    }

    if constexpr (!IsSameType<T2, T3>::value) {
        pipe->InitBuffer(castIndice, processSize_ * sizeof(T3));
    }

    pipe->InitBuffer(outQueueValues_, BUFFER_NUM, processSize_ * sizeof(T1));
    if constexpr (IsSameType<T2, int64_t>::value || IsSameType<T3, int64_t>::value) {
        pipe->InitBuffer(outQueueIndice_, BUFFER_NUM, processSize_ * sizeof(int64_t));
    } else {
        pipe->InitBuffer(outQueueIndice_, BUFFER_NUM, processSize_ * sizeof(T2));
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::Process()
{
    if (blockIdx_ >= tiling_->realCoreNum) {
        return;
    }

    if (tiling_->cutRSize != tiling_->rSize) {
        ProcessWithCutR();
    } else {
        ProcessWithoutCutR();
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::ProcessWithCutR()
{
    loopR_ = tiling_->rSize / tiling_->cutRSize;
    tailR_ = tiling_->rSize % tiling_->cutRSize;

    uint64_t loopNum = blkProcessNum_ / processSize_;
    uint64_t tailSize = blkProcessNum_ % processSize_;
    for (uint64_t index = 0; index < loopNum; index++) {
        loopOffset_ = index * processSize_;
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        ProcessCutR(indiceUb, valuesUb, processSize_);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_, processSize_);
    }
    // process tail loop
    if (tailSize > 0) {
        loopOffset_ = loopNum * processSize_;
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        ProcessCutR(indiceUb, valuesUb, tailSize);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_, tailSize);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::ProcessWithoutCutR()
{
    uint64_t loop = blkProcessNum_ / processSize_;
    uint64_t tail = blkProcessNum_ % processSize_;
    loopNum_ = processSize_ / tiling_->cutASize;
    loopTail_ = processSize_ % tiling_->cutASize;
    // process main loop
    for (uint64_t index = 0; index < loop; index++) {
        loopOffset_ = index * processSize_;
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        ProcessBigR(indiceUb, valuesUb);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_, processSize_);
    }
    // process tail loop
    if (tail > 0) {
        loopOffset_ = loop * processSize_;
        loopNum_ = tail / tiling_->cutASize;
        loopTail_ = tail % tiling_->cutASize;
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        ProcessBigR(indiceUb, valuesUb);
        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOut(blockOffset_ + loopOffset_, tail);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::ProcessBigR(LocalTensor<T2>& indiceUb,
                                                                                    LocalTensor<T1>& valuesUb)
{
    // process main loop
    for (uint64_t index = 0; index < loopNum_; index++) {
        CopyInX(index, tiling_->cutASize);
        Compute(index, tiling_->cutASize, indiceUb, valuesUb);
    }
    // process tail data
    if (loopTail_ > 0) {
        CopyInX(loopNum_, loopTail_);
        Compute(loopNum_, loopTail_, indiceUb, valuesUb);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::ProcessCutR(LocalTensor<T2>& indiceUb,
                                                                                    LocalTensor<T1>& valuesUb,
                                                                                    uint64_t loopSize)
{
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        ProcessCutRTemp<float>(indiceUb, valuesUb, loopSize);
    } else {
        ProcessCutRTemp<T1>(indiceUb, valuesUb, loopSize);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
template <typename T4>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::ProcessCutRB64(LocalTensor<T2>& indiceUb,
                                                                                       LocalTensor<T1>& valuesUb,
                                                                                       uint64_t loopSize)
{
    for (uint64_t bIndex = 0; bIndex < loopSize; bIndex++) {
        T2 indice = 0;
        T4 values = this->template GetMaxOrMinValue<T4>();
        T4 valuesTmp = values;
        for (uint64_t rIindex = 0; rIindex < loopR_; rIindex++) {
            CopyInXCutR(bIndex, rIindex, tiling_->cutRSize);
            PipeBarrier<PIPE_ALL>();
            ComputeCutR(bIndex, tiling_->cutRSize, indiceUb, valuesUb, 0);
            PipeBarrier<PIPE_ALL>();
            valuesTmp = valuesUb.GetValue(bIndex);
            // min时取小的值，max时取大的值
            if ((isMin && valuesTmp < values) || (!isMin && valuesTmp > values)) {
                values = valuesTmp;
                indice = indiceUb.GetValue(bIndex) + rIindex * tiling_->cutRSize;
            }
        }
        if (tailR_ != 0) {
            CopyInXCutR(bIndex, loopR_, tailR_);
            PipeBarrier<PIPE_ALL>();
            ComputeCutR(bIndex, tailR_, indiceUb, valuesUb, 0);
            PipeBarrier<PIPE_ALL>();
            valuesTmp = valuesUb.GetValue(bIndex);
            if ((isMin && valuesTmp < values) || (!isMin && valuesTmp > values)) {
                values = valuesTmp;
                indice = indiceUb.GetValue(bIndex) + loopR_ * tiling_->cutRSize;
            }
        }
        valuesUb.SetValue(bIndex, values);
        indiceUb.SetValue(bIndex, indice);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
template <typename T4>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::ProcessCutRTemp(LocalTensor<T2>& indiceUb,
                                                                                        LocalTensor<T1>& valuesUb,
                                                                                        uint64_t loopSize)
{
    // first r loop
    for (uint64_t bIndex = 0; bIndex < loopSize; bIndex++) {
        CopyInXCutR(bIndex, 0, tiling_->cutRSize);
        ComputeCutR(bIndex, tiling_->cutRSize, indiceUb, valuesUb, 0);
    }
    // middle r loop
    for (uint64_t rIindex = 1; rIindex < loopR_; rIindex++) {
        for (uint64_t bIndex = 0; bIndex < loopSize; bIndex++) {
            CopyInXCutR(bIndex, rIindex, tiling_->cutRSize);
            ComputeCutR(bIndex + processSize_, tiling_->cutRSize, indiceUb, valuesUb, rIindex * tiling_->cutRSize);
        }
        if constexpr (!IsSameType<T1, bfloat16_t>::value) {
            this->template UpdateResult<T1>(
                (__ubuf__ T2*)indiceUb[processSize_].GetPhyAddr(), (__ubuf__ T1*)valuesUb[processSize_].GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)valuesUb.GetPhyAddr(), loopSize);
        } else {
            LocalTensor<float> outUb32 = outBuf32.Get<float>();
            this->template UpdateResult<float>(
                (__ubuf__ T2*)indiceUb[processSize_].GetPhyAddr(), (__ubuf__ float*)outUb32[processSize_].GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ float*)outUb32.GetPhyAddr(), loopSize);
        }
    }
    // tail r loop
    if (tailR_ != 0) {
        for (uint64_t bIndex = 0; bIndex < loopSize; bIndex++) {
            CopyInXCutR(bIndex, loopR_, tailR_);
            ComputeCutR(bIndex + processSize_, tailR_, indiceUb, valuesUb, loopR_ * tiling_->cutRSize);
        }
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            LocalTensor<float> outUb32 = outBuf32.Get<float>();
            this->template UpdateResult<float>(
                (__ubuf__ T2*)indiceUb[processSize_].GetPhyAddr(), (__ubuf__ float*)outUb32[processSize_].GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ float*)outUb32.GetPhyAddr(), loopSize);
        } else {
            this->template UpdateResult<T1>(
                (__ubuf__ T2*)indiceUb[processSize_].GetPhyAddr(), (__ubuf__ T1*)valuesUb[processSize_].GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)valuesUb.GetPhyAddr(), loopSize);
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::CopyInX(uint64_t index, uint16_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
    // data copy params
    copyInParams_.blockCount = 1;
    copyInParams_.blockLen = copyNum * tiling_->cutRSize * sizeof(T1);
    copyInParams_.srcStride = 0;
    copyInParams_.dstStride = 0;

    DataCopyPad(xUb, xGm_[(blockOffset_ + loopOffset_ + index * tiling_->cutASize) * tiling_->cutRSize], copyInParams_,
                padParams_);
    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::CopyInXCutR(uint64_t bIndex, uint64_t rIindex,
                                                                                    uint16_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
    // data copy params
    copyInParams_.blockCount = 1;
    copyInParams_.blockLen = copyNum * sizeof(T1);
    copyInParams_.srcStride = 0;
    copyInParams_.dstStride = 0;

    DataCopyPad(xUb, xGm_[(blockOffset_ + loopOffset_ + bIndex) * tiling_->rSize + rIindex * tiling_->cutRSize],
                copyInParams_, padParams_);
    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::Compute(uint64_t index, uint16_t processNum,
                                                                                LocalTensor<T2>& indiceUb,
                                                                                LocalTensor<T1>& valuesUb)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        LocalTensor<float> outUb32 = outBuf32.Get<float>();
        Cast(xUb32, srcUb, RoundMode::CAST_NONE, processNum * tiling_->cutRSize);
        this->template ArgMaxV1<float, uint32_t>((__ubuf__ float*)outUb32[index * tiling_->cutASize].GetPhyAddr(),
                                                 (__ubuf__ T2*)indiceUb[index * tiling_->cutASize].GetPhyAddr(),
                                                 (__ubuf__ float*)xUb32.GetPhyAddr(), processNum, tiling_->cutRSize);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxV1Int64((__ubuf__ T1*)valuesUb[index * tiling_->cutASize].GetPhyAddr(),
                                     (__ubuf__ T2*)indiceUb[index * tiling_->cutASize].GetPhyAddr(),
                                     (__ubuf__ T1*)srcUb.GetPhyAddr(), processNum, tiling_->cutRSize);
    } else if constexpr (IsSameType<T1, half>::value) {
        this->template ArgMaxV1<T1, uint16_t>((__ubuf__ T1*)valuesUb[index * tiling_->cutASize].GetPhyAddr(),
                                              (__ubuf__ T2*)indiceUb[index * tiling_->cutASize].GetPhyAddr(),
                                              (__ubuf__ T1*)srcUb.GetPhyAddr(), processNum, tiling_->cutRSize);
    } else {
        this->template ArgMaxV1<T1, uint32_t>((__ubuf__ T1*)valuesUb[index * tiling_->cutASize].GetPhyAddr(),
                                              (__ubuf__ T2*)indiceUb[index * tiling_->cutASize].GetPhyAddr(),
                                              (__ubuf__ T1*)srcUb.GetPhyAddr(), processNum, tiling_->cutRSize);
    }
    inQueueX_.FreeTensor(srcUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::ComputeCutR(uint64_t index, uint64_t dimR,
                                                                                    LocalTensor<T2>& indiceUb,
                                                                                    LocalTensor<T1>& valuesUb,
                                                                                    uint32_t startR)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();
    if constexpr (IsSameType<T1, half>::value) {
        this->template ArgMaxV1<T1, uint16_t>((__ubuf__ T1*)valuesUb[index].GetPhyAddr(),
                                              (__ubuf__ T2*)indiceUb[index].GetPhyAddr(),
                                              (__ubuf__ T1*)srcUb.GetPhyAddr(), 1, dimR, startR);
    } else if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        LocalTensor<float> outUb32 = outBuf32.Get<float>();
        Cast(xUb32, srcUb, RoundMode::CAST_NONE, dimR);
        this->template ArgMaxV1<float, uint32_t>((__ubuf__ float*)outUb32[index].GetPhyAddr(),
                                                 (__ubuf__ T2*)indiceUb[index].GetPhyAddr(),
                                                 (__ubuf__ float*)xUb32.GetPhyAddr(), 1, dimR, startR);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxV1Int64((__ubuf__ T1*)valuesUb[index].GetPhyAddr(),
                                     (__ubuf__ T2*)indiceUb[index].GetPhyAddr(), (__ubuf__ T1*)srcUb.GetPhyAddr(), 1,
                                     dimR, startR);
    } else {
        this->template ArgMaxV1<T1, uint32_t>((__ubuf__ T1*)valuesUb[index].GetPhyAddr(),
                                              (__ubuf__ T2*)indiceUb[index].GetPhyAddr(),
                                              (__ubuf__ T1*)srcUb.GetPhyAddr(), 1, dimR, startR);
    }
    inQueueX_.FreeTensor(srcUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::CopyOut(uint64_t offset, uint64_t copyNum)
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
        CopyOutARIndice(indiceGm_, outIndice, offset, copyNum);
    } else {
        CopyOutARIndice(indiceGm_, outIndice, offset, copyNum);
    }
    outQueueIndice_.FreeTensor(outIndice);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueAr<T1, T2, T3, withValue, isMin>::CopyOutARIndice(GlobalTensor<T3>& indiceGm,
                                                                                        LocalTensor<T2>& outIndice,
                                                                                        uint64_t offset,
                                                                                        uint64_t copyNum)
{
    DataCopyExtParams copyOutIndiceParams{1, 0, 0, 0, 0};
    copyOutIndiceParams.blockCount = 1;
    copyOutIndiceParams.blockLen = copyNum * sizeof(T3);
    copyOutIndiceParams.srcStride = 0;
    copyOutIndiceParams.dstStride = 0;
    if constexpr ((IsSameType<T1, int64_t>::value && !IsSameType<T3, int64_t>::value) ||
                  (!IsSameType<T1, int64_t>::value && IsSameType<T3, int64_t>::value)) {
        LocalTensor<T3> castUb = outQueueValues_.AllocTensor<T3>();
        Cast(castUb, outIndice, RoundMode::CAST_NONE, copyNum);
        outQueueValues_.EnQue(castUb);
        castUb = outQueueValues_.DeQue<T3>();
        DataCopyPad(indiceGm[offset], castUb, copyOutIndiceParams);
        outQueueValues_.FreeTensor(castUb);
    } else {
        DataCopyPad(indiceGm[offset], outIndice, copyOutIndiceParams);
    }
}
} // namespace ArgMaxWithValue

#endif // ARG_MAX_WITH_VALUE_AR_H
