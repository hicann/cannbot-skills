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
 * \file arg_max_with_value_ara_cut_a_and_next_a.h
 * \brief arg_max_with_value_ara_cut_a_and_next_a
 */

#ifndef ARG_MAX_WITH_VALUE_ARA_CUT_A_AND_NEXT_A_H
#define ARG_MAX_WITH_VALUE_ARA_CUT_A_AND_NEXT_A_H

#include "arg_max_with_value_base.h"
#include "op_kernel/math_util.h"

namespace ArgMaxWithValue {
using namespace AscendC;

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
class ArgMaxWithValueArACutAAndNextA : public ArgMaxWithValueBase<T1, T2, T3, isMin> {
public:
    __aicore__ inline ArgMaxWithValueArACutAAndNextA(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace,
                                const ArgMaxWithValueTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitAllGlobalBuffer(GM_ADDR x, GM_ADDR indice, GM_ADDR values);
    __aicore__ inline void InitAllBuffer();
    __aicore__ inline void InitLoopParams();
    // 各个层级的Process函数
    __aicore__ inline void ProcessInALoop(uint64_t aUbStart, uint64_t aUbLength); // 第一重循环体
    __aicore__ inline void ProcessInANextALoop(uint64_t aUbStart, uint64_t aUbLength, uint64_t nextAUbStart,
                                               uint64_t nextAUbLength); // 第二重循环体
    // 拷入拷出函数
    __aicore__ inline void CopyInX(uint64_t aUbStart, uint64_t nextAUbStart, uint64_t rUbStart, uint64_t aUbLength,
                                   uint64_t nextAUbLength, uint64_t rUbLength, uint64_t nextAAlign);
    __aicore__ inline void CopyInXFlat(uint64_t aUbStart, uint64_t aUbLength);
    __aicore__ inline void ProcessFlat(uint64_t aUbStart, uint64_t aUbLength);
    __aicore__ inline void CopyOutValueAndIndices(uint64_t aUbStart, uint64_t nextAUbStart, uint64_t aUbLength,
                                                  uint64_t nextAUbLength, uint64_t nextAAlign);
    // 计算函数
    __aicore__ inline void Compute(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, LocalTensor<T2>& tempIndiceUb,
                                   LocalTensor<T1>& tempValueUb, uint64_t aUbStart, uint64_t aUbLength,
                                   uint64_t nextAUbStart, uint64_t nextAUbLength, uint64_t nextAAlign);
    __aicore__ inline void ComputeVF(LocalTensor<T1>& srcUb, LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb,
                                     uint32_t aUbLength, uint32_t nextAUbLength, uint32_t srcDimA1, uint32_t dstDimA1,
                                     uint32_t rUbLength, uint32_t startR);

private:
    constexpr static uint64_t MIN_DTYPE_SIZE = sizeof(T1) > sizeof(T3) ? sizeof(T3) : sizeof(T1);

    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECOUT, 1> outQueueIndice_;
    TQue<QuePosition::VECOUT, 1> outQueueValues_;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T1> valuesGm_;
    GlobalTensor<T3> indiceGm_;
    TBuf<QuePosition::VECCALC> tempValue_;
    TBuf<QuePosition::VECCALC> tempIndice_;
    TBuf<QuePosition::VECCALC> castIndice_;

    uint64_t blockIdx_ = 0;
    uint64_t aBlockStart_ = 0;
    uint64_t aBlockLength_ = 0;
    uint64_t nextABlockStart_ = 0;
    uint64_t nextABlockLength_ = 0;
    uint64_t ubFactorA_ = 0;
    uint64_t ubFactorNextA_ = 0;
    uint64_t ubFactorR_ = 0;
    uint64_t ubFactorNextAAlign_ = 0;
    uint8_t useFlatPath_ = 0;

    // tiling params
    const ArgMaxWithValueTilingData* tilingData_;
    TPipe* pipe_;
};

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::Init(
    GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, const ArgMaxWithValueTilingData* tilingData,
    TPipe* pipe)
{
    pipe_ = pipe;
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();

    InitAllGlobalBuffer(x, indice, values);
    InitAllBuffer();
    InitLoopParams();
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::InitAllGlobalBuffer(GM_ADDR x,
                                                                                                         GM_ADDR indice,
                                                                                                         GM_ADDR values)
{
    indiceGm_.SetGlobalBuffer((__gm__ T3*)indice);
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    valuesGm_.SetGlobalBuffer((__gm__ T1*)values);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::InitLoopParams()
{
    uint64_t blockNumA = Ops::Base::CeilDiv<int64_t>(tilingData_->aSize, tilingData_->blkFactor);
    uint64_t blockNumNextA = Ops::Base::CeilDiv<int64_t>(tilingData_->nextASize, tilingData_->blkFactor2nd);
    uint64_t aBlockIdx = blockIdx_ % blockNumA;
    uint64_t nextABlockIdx = blockIdx_ / blockNumA;
    aBlockLength_ = tilingData_->blkFactor;
    nextABlockLength_ = tilingData_->blkFactor2nd;
    if (aBlockIdx == blockNumA - 1) {
        aBlockLength_ = tilingData_->blkTailFactor;
    }
    if (nextABlockIdx == blockNumNextA - 1) {
        nextABlockLength_ = tilingData_->blkTailFactor2nd;
    }
    this->aBlockStart_ = aBlockIdx * tilingData_->blkFactor;
    this->nextABlockStart_ = nextABlockIdx * tilingData_->blkFactor2nd;
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::InitAllBuffer()
{
    ubFactorA_ = tilingData_->cutASize;
    ubFactorNextA_ = tilingData_->cutNextASize;
    ubFactorR_ = tilingData_->cutRSize;
    useFlatPath_ = tilingData_->useFlatPath;
    ubFactorNextAAlign_ = Ops::Base::CeilAlign<int64_t>(ubFactorNextA_, BLOCK_SIZE / MIN_DTYPE_SIZE);
    if (useFlatPath_) {
        ubFactorR_ = tilingData_->rSize;
    }

    // 初始化输入输出Buffer
    pipe_->InitBuffer(inQueueX_, VALUE_TWO, ubFactorA_ * ubFactorR_ * ubFactorNextAAlign_ * sizeof(T1));
    pipe_->InitBuffer(outQueueValues_, VALUE_TWO, ubFactorA_ * ubFactorNextAAlign_ * sizeof(T1));
    pipe_->InitBuffer(outQueueIndice_, VALUE_TWO, ubFactorA_ * ubFactorNextAAlign_ * sizeof(T3));
    // 初始化临时空间
    pipe_->InitBuffer(tempValue_, ubFactorA_ * ubFactorNextAAlign_ * sizeof(T1));
    pipe_->InitBuffer(tempIndice_, ubFactorA_ * ubFactorNextAAlign_ * sizeof(T2));
    if constexpr (!IsSameType<T2, T3>::value) {
        pipe_->InitBuffer(castIndice_, ubFactorA_ * ubFactorNextAAlign_ * sizeof(T2));
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::Process()
{
    if (blockIdx_ >= tilingData_->realCoreNum) {
        return;
    }

    uint64_t aLoopCount = Ops::Base::CeilDiv(aBlockLength_, ubFactorA_);
    for (uint64_t aLoopIdx = 0; aLoopIdx < aLoopCount; aLoopIdx++) {
        uint64_t aUbStart = aBlockStart_ + aLoopIdx * ubFactorA_;
        uint64_t aUbLength = (aLoopIdx == aLoopCount - 1) ? (aBlockLength_ - aLoopIdx * ubFactorA_) : ubFactorA_;
        if (useFlatPath_) {
            ProcessFlat(aUbStart, aUbLength);
        } else {
            ProcessInALoop(aUbStart, aUbLength);
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::ProcessInALoop(uint64_t aUbStart,
                                                                                                    uint64_t aUbLength)
{
    uint64_t aNextALoopCount = Ops::Base::CeilDiv(nextABlockLength_, ubFactorNextA_);
    for (uint64_t nextALoopIdx = 0; nextALoopIdx < aNextALoopCount; nextALoopIdx++) {
        uint64_t nextAUbStart = nextABlockStart_ + nextALoopIdx * ubFactorNextA_;
        uint64_t nextAUbLength = (nextALoopIdx == aNextALoopCount - 1) ?
                                     (nextABlockLength_ - nextALoopIdx * ubFactorNextA_) :
                                     ubFactorNextA_;
        ProcessInANextALoop(aUbStart, aUbLength, nextAUbStart, nextAUbLength);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::ProcessInANextALoop(
    uint64_t aUbStart, uint64_t aUbLength, uint64_t nextAUbStart, uint64_t nextAUbLength)
{
    // 初始化输出
    LocalTensor<T3> indiceUb = outQueueIndice_.AllocTensor<T3>();
    LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
    LocalTensor<T2> tempIndiceUb = tempIndice_.Get<T2>();
    LocalTensor<T1> tempValueUb = tempValue_.Get<T1>();
    // 在R轴进行循环
    uint64_t nextAAlign = Ops::Base::CeilAlign<uint64_t>(nextAUbLength, BLOCK_SIZE / MIN_DTYPE_SIZE);
    if constexpr (IsSameType<T2, T3>::value) {
        Compute(indiceUb, valuesUb, tempIndiceUb, tempValueUb, aUbStart, aUbLength, nextAUbStart, nextAUbLength,
                nextAAlign);
    } else {
        LocalTensor<T2> castUb = castIndice_.Get<T2>();
        Compute(castUb, valuesUb, tempIndiceUb, tempValueUb, aUbStart, aUbLength, nextAUbStart, nextAUbLength,
                nextAAlign);
        Cast(indiceUb, castUb, RoundMode::CAST_NONE, aUbLength * nextAAlign);
    }
    outQueueIndice_.EnQue(indiceUb);
    outQueueValues_.EnQue(valuesUb);
    CopyOutValueAndIndices(aUbStart, nextAUbStart, aUbLength, nextAUbLength, nextAAlign);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::CopyInXFlat(uint64_t aUbStart,
                                                                                                 uint64_t aUbLength)
{
    LocalTensor<T1> xLocal = inQueueX_.AllocTensor<T1>();
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = 1;
    copyExtParams.blockLen = aUbLength * tilingData_->rSize * tilingData_->nextASize * sizeof(T1);
    copyExtParams.srcStride = 0;
    copyExtParams.dstStride = 0;
    DataCopyPadExtParams<T1> padParams{false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[aUbStart * tilingData_->rSize * tilingData_->nextASize], copyExtParams, padParams);
    inQueueX_.EnQue(xLocal);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::ProcessFlat(uint64_t aUbStart,
                                                                                                 uint64_t aUbLength)
{
    uint64_t nextAUbStart = nextABlockStart_;
    uint64_t nextAUbLength = nextABlockLength_;
    uint64_t nextAAlign = ubFactorNextAAlign_;
    LocalTensor<T3> indiceUb = outQueueIndice_.AllocTensor<T3>();
    LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
    if constexpr (IsSameType<T2, T3>::value) {
        CopyInXFlat(aUbStart, aUbLength);
        LocalTensor<T1> xLocal = inQueueX_.DeQue<T1>();
        ComputeVF(xLocal, indiceUb, valuesUb, aUbLength, nextAUbLength, nextAUbLength, nextAAlign, tilingData_->rSize,
                  0);
        inQueueX_.FreeTensor(xLocal);
    } else {
        CopyInXFlat(aUbStart, aUbLength);
        LocalTensor<T1> xLocal = inQueueX_.DeQue<T1>();
        LocalTensor<T2> castUb = castIndice_.Get<T2>();
        ComputeVF(xLocal, castUb, valuesUb, aUbLength, nextAUbLength, nextAUbLength, nextAAlign, tilingData_->rSize, 0);
        inQueueX_.FreeTensor(xLocal);
        Cast(indiceUb, castUb, RoundMode::CAST_NONE, aUbLength * nextAAlign);
    }
    outQueueIndice_.EnQue(indiceUb);
    outQueueValues_.EnQue(valuesUb);
    CopyOutValueAndIndices(aUbStart, nextAUbStart, aUbLength, nextAUbLength, nextAAlign);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::CopyInX(
    uint64_t aUbStart, uint64_t aUbLength, uint64_t nextAUbStart, uint64_t nextAUbLength, uint64_t rUbStart,
    uint64_t rUbLength, uint64_t nextAAlign)
{
    LocalTensor<T1> xLocal = inQueueX_.AllocTensor<T1>();
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1Size = aUbLength;
    loopParams.loop1SrcStride = tilingData_->rSize * tilingData_->nextASize * sizeof(T1);
    loopParams.loop1DstStride = rUbLength * nextAAlign * sizeof(T1);
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyExtParams copyExtParams;
    copyExtParams.blockCount = rUbLength;
    copyExtParams.blockLen = nextAUbLength * sizeof(T1);
    copyExtParams.srcStride = (tilingData_->nextASize - nextAUbLength) * sizeof(T1);
    copyExtParams.dstStride = (nextAAlign - nextAUbLength) * sizeof(T1) / BLOCK_SIZE;
    DataCopyPadExtParams<T1> copyPadExtParams;
    copyPadExtParams.isPad = false;
    copyPadExtParams.leftPadding = 0;
    copyPadExtParams.rightPadding = 0;
    copyPadExtParams.paddingValue = 0;
    DataCopyPad(
        xLocal,
        xGm_[aUbStart * tilingData_->rSize * tilingData_->nextASize + rUbStart * tilingData_->nextASize + nextAUbStart],
        copyExtParams, copyPadExtParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inQueueX_.template EnQue(xLocal);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::CopyOutValueAndIndices(
    uint64_t aUbStart, uint64_t nextAUbStart, uint64_t aUbLength, uint64_t nextAUbLength, uint64_t nextAAlign)
{
    LocalTensor<T3> outputIndice = outQueueIndice_.template DeQue<T3>();
    DataCopyExtParams copyIndicesExtParams;
    copyIndicesExtParams.blockCount = aUbLength;
    copyIndicesExtParams.blockLen = nextAUbLength * sizeof(T3);
    copyIndicesExtParams.srcStride = (nextAAlign - nextAUbLength) * sizeof(T3) / BLOCK_SIZE;
    copyIndicesExtParams.dstStride = (tilingData_->nextASize - nextAUbLength) * sizeof(T3);
    DataCopyPad(indiceGm_[aUbStart * tilingData_->nextASize + nextAUbStart], outputIndice, copyIndicesExtParams);
    outQueueIndice_.FreeTensor(outputIndice);
    LocalTensor<T1> outputValues = outQueueValues_.template DeQue<T1>();
    if constexpr (withValue) {
        DataCopyExtParams copyXExtParams;
        copyXExtParams.blockCount = aUbLength;
        copyXExtParams.blockLen = nextAUbLength * sizeof(T1);
        copyXExtParams.srcStride = (nextAAlign - nextAUbLength) * sizeof(T1) / BLOCK_SIZE;
        copyXExtParams.dstStride = (tilingData_->nextASize - nextAUbLength) * sizeof(T1);
        DataCopyPad(valuesGm_[aUbStart * tilingData_->nextASize + nextAUbStart], outputValues, copyXExtParams);
    }
    outQueueValues_.FreeTensor(outputValues);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::Compute(
    LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, LocalTensor<T2>& tempIndiceUb, LocalTensor<T1>& tempValueUb,
    uint64_t aUbStart, uint64_t aUbLength, uint64_t nextAUbStart, uint64_t nextAUbLength, uint64_t nextAAlign)
{
    uint64_t rLoopCount = Ops::Base::CeilDiv(tilingData_->rSize, ubFactorR_);
    uint64_t rUbStart = 0;
    uint64_t rUbLength = rLoopCount != 1 ? ubFactorR_ : tilingData_->rSize;
    CopyInX(aUbStart, aUbLength, nextAUbStart, nextAUbLength, rUbStart, rUbLength, nextAAlign);
    LocalTensor<T1> xLocal = inQueueX_.template DeQue<T1>();
    ComputeVF(xLocal, indiceUb, valuesUb, aUbLength, nextAUbLength, nextAAlign, nextAAlign, rUbLength, 0);
    inQueueX_.FreeTensor(xLocal);
    for (uint64_t rLoopIdx = 1; rLoopIdx < rLoopCount; rLoopIdx++) {
        rUbStart = rLoopIdx * ubFactorR_;
        rUbLength = (rLoopIdx != rLoopCount - 1) ? ubFactorR_ : tilingData_->rSize - rLoopIdx * ubFactorR_;
        CopyInX(aUbStart, aUbLength, nextAUbStart, nextAUbLength, rUbStart, rUbLength, nextAAlign);
        xLocal = inQueueX_.template DeQue<T1>();
        ComputeVF(xLocal, tempIndiceUb, tempValueUb, aUbLength, nextAUbLength, nextAAlign, nextAAlign, rUbLength,
                  rLoopIdx * ubFactorR_);
        inQueueX_.FreeTensor(xLocal);
        this->template UpdateResult<T1, false>(
            (__ubuf__ T2*)tempIndiceUb.GetPhyAddr(), (__ubuf__ T1*)tempValueUb.GetPhyAddr(),
            (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)valuesUb.GetPhyAddr(), aUbLength * nextAAlign);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueArACutAAndNextA<T1, T2, T3, withValue, isMin>::ComputeVF(
    LocalTensor<T1>& srcUb, LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint32_t aUbLength,
    uint32_t nextAUbLength, uint32_t srcDimA1, uint32_t dstDimA1, uint32_t rUbLength, uint32_t startR)
{
    if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxAraUnAlignInt64<int64_t, uint64_t>(
            (__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)srcUb.GetPhyAddr(),
            aUbLength, rUbLength, nextAUbLength, srcDimA1, dstDimA1, startR);
    } else if constexpr (IsSameType<T1, half>::value || IsSameType<T1, bfloat16_t>::value) {
        this->template ArgMaxAraUnAlign<T1, uint16_t>(
            (__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)srcUb.GetPhyAddr(),
            aUbLength, rUbLength, nextAUbLength, srcDimA1, dstDimA1, startR);
    } else {
        this->template ArgMaxAraUnAlign<T1, uint32_t>(
            (__ubuf__ T1*)valuesUb.GetPhyAddr(), (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)srcUb.GetPhyAddr(),
            aUbLength, rUbLength, nextAUbLength, srcDimA1, dstDimA1, startR);
    }
}

} // namespace ArgMaxWithValue

#endif // ARG_MAX_WITH_VALUE_ARA_CUT_A_AND_NEXT_A_H
