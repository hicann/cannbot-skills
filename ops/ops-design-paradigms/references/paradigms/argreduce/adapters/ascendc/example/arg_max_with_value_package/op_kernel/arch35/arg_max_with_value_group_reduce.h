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
 * \file arg_max_with_value_group_reduce.h
 * \brief arg_max_with_value_group_reduce
 */

#ifndef ARG_MAX_WITH_VALUE_GROUP_REDUCE_H
#define ARG_MAX_WITH_VALUE_GROUP_REDUCE_H

#include "arg_max_with_value_base.h"
#include "op_kernel/platform_util.h"

namespace ArgMaxWithValue {
using namespace AscendC;

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
class ArgMaxWithValueGroupReduce : public ArgMaxWithValueBase<T1, T2, T3, isMin> {
public:
    __aicore__ inline ArgMaxWithValueGroupReduce(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace,
                                const ArgMaxWithValueTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

private:
    template <bool isAra>
    __aicore__ inline void ProcessTemplate();
    template <bool isAra>
    __aicore__ inline void ProcessMiddleLoop(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb);
    template <bool isAra>
    __aicore__ inline void ProcessTailLoop(LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb);
    __aicore__ inline void ProcessMultiCoreResult();
    __aicore__ inline void CopyInX(uint16_t copyNum);
    template <bool isAra>
    __aicore__ inline void Compute(uint64_t dimR, LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb,
                                   uint64_t offset);
    __aicore__ inline void CopyOut(uint64_t offset, uint64_t copyNum);
    __aicore__ inline void CopyInMultiCoreResult();
    template <typename T4>
    __aicore__ inline void ComputeResult();
    template <typename T4>
    __aicore__ inline void ComputeResultInt64();
    template <typename T4>
    __aicore__ inline void ComputeResultInt64AllCore(uint64_t actualNextA);
    __aicore__ inline void CopyOutWorkSpace(uint64_t offset, uint64_t copyNum);
    __aicore__ inline void ProcessRaSplit();
    __aicore__ inline void CopyInXSingleA(uint16_t copyNum, uint64_t aIdx);
    __aicore__ inline void ComputeRa(uint64_t dimR, LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb,
                                     uint64_t offset, uint64_t aIdx);
    template <typename T4>
    __aicore__ inline void ComputeResultAllCore(uint64_t actualNextA);
    __aicore__ inline void ProcessMultiCoreResultAllCore();
    __aicore__ inline void CopyInMultiCoreResultAllCore(uint64_t aIdx, uint64_t nextAOff, uint64_t actualNextA);
    __aicore__ inline void CopyOutWorkSpaceRow(uint64_t aIdx, uint64_t coreIdx);

    constexpr static uint32_t BUFFER_NUM = 2;
    constexpr static uint16_t ELEMENT_PER_BLOCK = 32 / sizeof(T1);
    constexpr static uint64_t ALIGN_128 = 128;
    constexpr static uint64_t ALIGN_32 = 32;

private:
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueIndice_;
    TQue<QuePosition::VECOUT, 1> outQueueIndice_;
    TQue<QuePosition::VECOUT, 1> outQueueValues_;
    TBuf<QuePosition::VECCALC> xBuf32;
    TBuf<QuePosition::VECCALC> outBuf32;
    TBuf<QuePosition::VECCALC> castIndice;

    GlobalTensor<T1> xGm_;
    GlobalTensor<T3> indiceGm_;
    GlobalTensor<T1> valuesGm_;
    GlobalTensor<T1> valueWorkspace_;
    GlobalTensor<float> valueWorkspace32_;
    GlobalTensor<T2> indiceWorkspace_;

    uint32_t blockIdx_ = 0;
    uint64_t blockOffset_ = 0;
    uint64_t loopOffset_ = 0;
    uint64_t blkProcessNum_ = 0;

    uint64_t loopR_ = 0;
    uint64_t tailR_ = 0;
    uint64_t outASize_ = 1;
    uint64_t outAAlign_ = 1;
    uint64_t nextAAlign_ = 1;
    uint64_t nextAAlign128_ = 1;
    uint64_t nextAFloatAlign128_ = 1;
    uint64_t indiceAlign128_ = 1;
    // tiling params
    const ArgMaxWithValueTilingData* tiling_;
    // datacopy params
    DataCopyPadExtParams<T1> padParams_{false, 0, 0, 0};
    DataCopyExtParams copyInParams_{1, 0, 0, 0, 0};
};

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::Init(
    GM_ADDR x, GM_ADDR indice, GM_ADDR values, GM_ADDR workspace, const ArgMaxWithValueTilingData* tilingData,
    TPipe* pipe)
{
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ T1*)x);
    indiceGm_.SetGlobalBuffer((__gm__ T3*)indice);
    valuesGm_.SetGlobalBuffer((__gm__ T1*)values);
    tiling_ = tilingData;
    valueWorkspace_.SetGlobalBuffer((__gm__ T1*)workspace, tiling_->workSpaceSize);
    valueWorkspace32_.SetGlobalBuffer((__gm__ float*)workspace, tiling_->workSpaceSize);
    indiceWorkspace_.SetGlobalBuffer((__gm__ T2*)workspace, tiling_->workSpaceSize);

    blkProcessNum_ = tiling_->blkFactor;
    blockOffset_ = blockIdx_ * tiling_->blkFactor;
    if (blockIdx_ < tiling_->blkTailFactor) {
        blkProcessNum_ += 1;
        blockOffset_ += blockIdx_;
    } else {
        blockOffset_ += tiling_->blkTailFactor;
    }

    outASize_ = tiling_->aSize * tiling_->nextASize;
    outAAlign_ = CeilDivision(outASize_, ELEMENT_PER_BLOCK) * ELEMENT_PER_BLOCK;
    nextAAlign_ = CeilDivision(tiling_->nextASize, ELEMENT_PER_BLOCK) * ELEMENT_PER_BLOCK;
    nextAAlign128_ = CeilDivision(tiling_->nextASize, ALIGN_128 / sizeof(T1)) * (ALIGN_128 / sizeof(T1));
    nextAFloatAlign128_ = CeilDivision(tiling_->nextASize, ALIGN_128 / sizeof(float)) * (ALIGN_128 / sizeof(float));
    indiceAlign128_ = CeilDivision(tiling_->nextASize, ALIGN_128 / sizeof(T2)) * (ALIGN_128 / sizeof(T2));
    uint64_t raSize = CeilDivision(tiling_->cutRSize * tiling_->nextASize, ELEMENT_PER_BLOCK) * ELEMENT_PER_BLOCK;
    uint64_t arAlign = tiling_->aSize * raSize;

    if (tiling_->isRaSplit) {
        uint64_t xBufSize = tiling_->realCoreNum * tiling_->gatherBlockSize;
        raSize = raSize >= xBufSize ? raSize : xBufSize;
        pipe->InitBuffer(inQueueX_, BUFFER_NUM, raSize * sizeof(T1));
        pipe->InitBuffer(inQueueIndice_, 1, xBufSize * sizeof(T2));
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            pipe->InitBuffer(xBuf32, raSize * sizeof(float));
            pipe->InitBuffer(outBuf32, nextAAlign_ * BUFFER_NUM * sizeof(float));
        }
        if constexpr (!IsSameType<T2, T3>::value) {
            pipe->InitBuffer(castIndice, nextAAlign_ * sizeof(T3));
        }
        pipe->InitBuffer(outQueueValues_, 1, nextAAlign_ * BUFFER_NUM * sizeof(T1));
        pipe->InitBuffer(outQueueIndice_, 1, nextAAlign_ * BUFFER_NUM * sizeof(T2));
    } else {
        uint64_t xBufSize = tiling_->realCoreNum * outAAlign_;
        arAlign = arAlign >= xBufSize ? arAlign : xBufSize;
        pipe->InitBuffer(inQueueX_, BUFFER_NUM, arAlign * sizeof(T1) + this->GetVLSize());
        pipe->InitBuffer(inQueueIndice_, 1, xBufSize * sizeof(T2));

        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            pipe->InitBuffer(xBuf32, arAlign * sizeof(float));
            pipe->InitBuffer(outBuf32, outAAlign_ * BUFFER_NUM * sizeof(float));
        }

        if constexpr (!IsSameType<T2, T3>::value) {
            pipe->InitBuffer(castIndice, outAAlign_ * sizeof(T3));
        }

        pipe->InitBuffer(outQueueValues_, 1, outAAlign_ * BUFFER_NUM * sizeof(T1));
        pipe->InitBuffer(outQueueIndice_, 1, outAAlign_ * BUFFER_NUM * sizeof(T2));
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::Process()
{
    if (blockIdx_ >= tiling_->realCoreNum) {
        return;
    }
    if (tiling_->isRaSplit) {
        ProcessRaSplit();
    } else {
        if (tiling_->cutNextASize == 1) {
            ProcessTemplate<false>();
        } else {
            ProcessTemplate<true>();
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::ProcessRaSplit()
{
    loopR_ = blkProcessNum_ / tiling_->cutRSize;
    tailR_ = blkProcessNum_ % tiling_->cutRSize;

    for (uint64_t a = 0; a < tiling_->aSize; a++) {
        LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
        LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
        loopOffset_ = 0;
        // process first r loop for this A row
        if (loopR_ > 0) {
            CopyInXSingleA(tiling_->cutRSize, a);
            ComputeRa(tiling_->cutRSize, indiceUb, valuesUb, 0, a);
        }
        // middle r loops
        for (uint64_t index = 1; index < loopR_; index++) {
            loopOffset_ = index * tiling_->cutRSize;
            CopyInXSingleA(tiling_->cutRSize, a);
            ComputeRa(tiling_->cutRSize, indiceUb, valuesUb, nextAAlign_, a);
            if constexpr (IsSameType<T1, bfloat16_t>::value) {
                LocalTensor<float> outUb32 = outBuf32.Get<float>();
                this->template UpdateResult<float>((__ubuf__ T2*)indiceUb[nextAAlign_].GetPhyAddr(),
                                                   (__ubuf__ float*)outUb32[nextAAlign_].GetPhyAddr(),
                                                   (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                                   (__ubuf__ float*)outUb32.GetPhyAddr(), tiling_->nextASize);
            } else {
                this->template UpdateResult<T1>(
                    (__ubuf__ T2*)indiceUb[nextAAlign_].GetPhyAddr(), (__ubuf__ T1*)valuesUb[nextAAlign_].GetPhyAddr(),
                    (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)valuesUb.GetPhyAddr(), tiling_->nextASize);
            }
        }
        // tail r loop
        if (tailR_ > 0) {
            loopOffset_ = loopR_ * tiling_->cutRSize;
            CopyInXSingleA(tailR_, a);
            if (loopR_ > 0) {
                ComputeRa(tailR_, indiceUb, valuesUb, nextAAlign_, a);
                if constexpr (!IsSameType<T1, bfloat16_t>::value) {
                    this->template UpdateResult<T1>((__ubuf__ T2*)indiceUb[nextAAlign_].GetPhyAddr(),
                                                    (__ubuf__ T1*)valuesUb[nextAAlign_].GetPhyAddr(),
                                                    (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                                    (__ubuf__ T1*)valuesUb.GetPhyAddr(), tiling_->nextASize);
                } else {
                    LocalTensor<float> outUb32 = outBuf32.Get<float>();
                    this->template UpdateResult<float>((__ubuf__ T2*)indiceUb[nextAAlign_].GetPhyAddr(),
                                                       (__ubuf__ float*)outUb32[nextAAlign_].GetPhyAddr(),
                                                       (__ubuf__ T2*)indiceUb.GetPhyAddr(),
                                                       (__ubuf__ float*)outUb32.GetPhyAddr(), tiling_->nextASize);
                }
            } else {
                ComputeRa(tailR_, indiceUb, valuesUb, 0, a);
            }
        }

        outQueueIndice_.EnQue(indiceUb);
        outQueueValues_.EnQue(valuesUb);
        CopyOutWorkSpaceRow(a, blockIdx_);
        outQueueIndice_.FreeTensor(indiceUb);
        outQueueValues_.FreeTensor(valuesUb);
    }

    SyncAll();
    ProcessMultiCoreResultAllCore();
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
template <bool isAra>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::ProcessTemplate()
{
    loopR_ = blkProcessNum_ / tiling_->cutRSize;
    tailR_ = blkProcessNum_ % tiling_->cutRSize;
    LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
    LocalTensor<T1> valuesUb = outQueueValues_.AllocTensor<T1>();
    // process first r loop
    if (loopR_ > 0) {
        CopyInX(tiling_->cutRSize);
        Compute<isAra>(tiling_->cutRSize, indiceUb, valuesUb, 0);
    }
    // process middle r loop
    ProcessMiddleLoop<isAra>(indiceUb, valuesUb);
    // process last r loop
    ProcessTailLoop<isAra>(indiceUb, valuesUb);
    outQueueIndice_.EnQue(indiceUb);
    outQueueValues_.EnQue(valuesUb);
    // move to workspace
    CopyOutWorkSpace(blockIdx_ * outAAlign_, outASize_);
    SyncAll();
    // process multi-core result
    ProcessMultiCoreResult();
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::CopyInXSingleA(uint16_t copyNum,
                                                                                                uint64_t aIdx)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
    copyInParams_.blockCount = 1;
    copyInParams_.blockLen = copyNum * tiling_->nextASize * sizeof(T1);
    copyInParams_.srcStride = 0;
    copyInParams_.dstStride = 0;

    uint64_t gmOffset = (aIdx * tiling_->rSize + blockOffset_ + loopOffset_) * tiling_->nextASize;
    DataCopyPad(xUb, xGm_[gmOffset], copyInParams_, padParams_);
    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
template <bool isAra>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::ProcessMiddleLoop(
    LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb)
{
    for (uint64_t index = 1; index < loopR_; index++) {
        loopOffset_ = index * tiling_->cutRSize;
        CopyInX(tiling_->cutRSize);
        Compute<isAra>(tiling_->cutRSize, indiceUb, valuesUb, outAAlign_);
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            LocalTensor<float> outUb32 = outBuf32.Get<float>();
            this->template UpdateResult<float>(
                (__ubuf__ T2*)indiceUb[outAAlign_].GetPhyAddr(), (__ubuf__ float*)outUb32[outAAlign_].GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ float*)outUb32.GetPhyAddr(), outASize_);
        } else {
            this->template UpdateResult<T1>(
                (__ubuf__ T2*)indiceUb[outAAlign_].GetPhyAddr(), (__ubuf__ T1*)valuesUb[outAAlign_].GetPhyAddr(),
                (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)valuesUb.GetPhyAddr(), outASize_);
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
template <bool isAra>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::ProcessTailLoop(
    LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb)
{
    if (tailR_ > 0) {
        loopOffset_ = loopR_ * tiling_->cutRSize;
        CopyInX(tailR_);
        if (loopR_ > 0) {
            Compute<isAra>(tailR_, indiceUb, valuesUb, outAAlign_);
            if constexpr (!IsSameType<T1, bfloat16_t>::value) {
                this->template UpdateResult<T1>(
                    (__ubuf__ T2*)indiceUb[outAAlign_].GetPhyAddr(), (__ubuf__ T1*)valuesUb[outAAlign_].GetPhyAddr(),
                    (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ T1*)valuesUb.GetPhyAddr(), outASize_);
            } else {
                LocalTensor<float> outUb32 = outBuf32.Get<float>();
                this->template UpdateResult<float>(
                    (__ubuf__ T2*)indiceUb[outAAlign_].GetPhyAddr(), (__ubuf__ float*)outUb32[outAAlign_].GetPhyAddr(),
                    (__ubuf__ T2*)indiceUb.GetPhyAddr(), (__ubuf__ float*)outUb32.GetPhyAddr(), outASize_);
            }
        } else {
            Compute<isAra>(tailR_, indiceUb, valuesUb, 0);
        }
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::ProcessMultiCoreResult()
{
    // process by core0
    if (blockIdx_ == 0) {
        CopyInMultiCoreResult();
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            ComputeResult<float>();
        } else if constexpr (IsSameType<T1, int64_t>::value) {
            ComputeResultInt64<int64_t>();
        } else {
            ComputeResult<T1>();
        }
        CopyOut(0, outASize_);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::CopyInMultiCoreResult()
{
    copyInParams_.blockCount = 1;
    copyInParams_.blockLen = tiling_->realCoreNum * outAAlign_ * sizeof(T1);
    copyInParams_.srcStride = 0;
    copyInParams_.dstStride = 0;
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        DataCopyPadExtParams<float> padParams32{false, 0, 0, 0};
        copyInParams_.blockLen = tiling_->realCoreNum * outAAlign_ * sizeof(float);
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        DataCopyPad(xUb32, valueWorkspace32_, copyInParams_, padParams32);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    } else {
        DataCopyPad(xUb, valueWorkspace_, copyInParams_, padParams_);
    }
    inQueueX_.EnQue(xUb);
    LocalTensor<T2> indicesUb = inQueueIndice_.AllocTensor<T2>();
    DataCopyPadExtParams<T2> padParams{false, 0, 0, 0};
    copyInParams_.blockLen = tiling_->realCoreNum * outAAlign_ * sizeof(T2);
    DataCopyPad(indicesUb, indiceWorkspace_[tiling_->realCoreNum * outAAlign_], copyInParams_, padParams);
    inQueueIndice_.EnQue(indicesUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::CopyInX(uint16_t copyNum)
{
    LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
    // data copy params
    copyInParams_.blockCount = tiling_->aSize;
    copyInParams_.blockLen = copyNum * tiling_->nextASize * sizeof(T1);
    copyInParams_.srcStride = (tiling_->rSize - copyNum) * tiling_->nextASize * sizeof(T1);
    copyInParams_.dstStride = 0;

    DataCopyPad(xUb, xGm_[(blockOffset_ + loopOffset_) * tiling_->nextASize], copyInParams_, padParams_);
    inQueueX_.EnQue(xUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::ComputeRa(
    uint64_t dimR, LocalTensor<T2>& indiceUb, LocalTensor<T1>& valuesUb, uint64_t offset, uint64_t aIdx)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        LocalTensor<float> outUb32 = outBuf32.Get<float>();
        uint64_t alignAR = CeilDivision(dimR * tiling_->nextASize, ELEMENT_PER_BLOCK) * ELEMENT_PER_BLOCK;
        Cast(xUb32, srcUb, RoundMode::CAST_NONE, alignAR);
        inQueueX_.FreeTensor(srcUb);
        this->template ArgMaxRa<float, uint32_t>((__ubuf__ float*)outUb32[offset].GetPhyAddr(),
                                                 (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
                                                 (__ubuf__ float*)xUb32.GetPhyAddr(), tiling_->nextASize,
                                                 tiling_->nextASize, dimR, blockOffset_ + loopOffset_);
    } else if constexpr (IsSameType<T1, half>::value) {
        this->template ArgMaxRa<T1, uint16_t>(
            (__ubuf__ T1*)valuesUb[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
            (__ubuf__ T1*)srcUb.GetPhyAddr(), tiling_->nextASize, tiling_->nextASize, dimR, blockOffset_ + loopOffset_);
    } else if constexpr (!IsSameType<T1, int64_t>::value) {
        this->template ArgMaxRa<T1, uint32_t>(
            (__ubuf__ T1*)valuesUb[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
            (__ubuf__ T1*)srcUb.GetPhyAddr(), tiling_->nextASize, tiling_->nextASize, dimR, blockOffset_ + loopOffset_);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        this->template ArgMaxAraUnAlignInt64<int64_t, uint64_t>(
            (__ubuf__ T1*)valuesUb[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
            (__ubuf__ T1*)srcUb.GetPhyAddr(), 1, dimR, tiling_->nextASize, tiling_->nextASize, tiling_->nextASize,
            blockOffset_ + loopOffset_);
    }
    if constexpr (!IsSameType<T1, bfloat16_t>::value) {
        inQueueX_.FreeTensor(srcUb);
    }
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
template <bool isAra>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::Compute(uint64_t dimR,
                                                                                         LocalTensor<T2>& indiceUb,
                                                                                         LocalTensor<T1>& valuesUb,
                                                                                         uint64_t offset)
{
    LocalTensor<T1> srcUb = inQueueX_.DeQue<T1>();
    if constexpr (isAra) {
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            LocalTensor<float> xUb32 = xBuf32.Get<float>();
            LocalTensor<float> outUb32 = outBuf32.Get<float>();
            uint64_t alignR = CeilDivision(dimR * tiling_->nextASize, ELEMENT_PER_BLOCK) * ELEMENT_PER_BLOCK;
            Cast(xUb32, srcUb, RoundMode::CAST_NONE, tiling_->aSize * alignR);
            this->template ArgMaxGatherV1<float, uint32_t, int32_t, true>(
                (__ubuf__ float*)outUb32[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
                (__ubuf__ float*)xUb32.GetPhyAddr(), tiling_->aSize, dimR, tiling_->nextASize,
                blockOffset_ + loopOffset_);
        } else if constexpr (IsSameType<T1, half>::value) {
            this->template ArgMaxGatherV1<T1, uint16_t, int16_t, true>(
                (__ubuf__ T1*)valuesUb[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
                (__ubuf__ T1*)srcUb.GetPhyAddr(), tiling_->aSize, dimR, tiling_->nextASize, blockOffset_ + loopOffset_);
        } else if constexpr (!IsSameType<T1, int64_t>::value) {
            this->template ArgMaxGatherV1<T1, uint32_t, int32_t, true>(
                (__ubuf__ T1*)valuesUb[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
                (__ubuf__ T1*)srcUb.GetPhyAddr(), tiling_->aSize, dimR, tiling_->nextASize, blockOffset_ + loopOffset_);
        } else if constexpr (IsSameType<T1, int64_t>::value) {
            this->template ArgMaxGatherV1Int64<T1, uint64_t, int64_t, true>(
                (__ubuf__ T1*)valuesUb[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
                (__ubuf__ T1*)srcUb.GetPhyAddr(), tiling_->aSize, dimR, tiling_->nextASize, blockOffset_ + loopOffset_);
        }
    } else {
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            LocalTensor<float> xUb32 = xBuf32.Get<float>();
            LocalTensor<float> outUb32 = outBuf32.Get<float>();
            uint64_t alignR = CeilDivision(dimR, ELEMENT_PER_BLOCK) * ELEMENT_PER_BLOCK;
            Cast(xUb32, srcUb, RoundMode::CAST_NONE, tiling_->aSize * alignR);
            this->template ArgMaxV1<float, uint32_t, true>(
                (__ubuf__ float*)outUb32[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
                (__ubuf__ float*)xUb32.GetPhyAddr(), outASize_, dimR, blockOffset_ + loopOffset_);
        } else if constexpr (IsSameType<T1, half>::value) {
            this->template ArgMaxV1<T1, uint16_t, true>(
                (__ubuf__ T1*)valuesUb[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
                (__ubuf__ T1*)srcUb.GetPhyAddr(), outASize_, dimR, blockOffset_ + loopOffset_);
        } else if constexpr (!IsSameType<T1, int64_t>::value) {
            this->template ArgMaxV1<T1, uint32_t, true>(
                (__ubuf__ T1*)valuesUb[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
                (__ubuf__ T1*)srcUb.GetPhyAddr(), outASize_, dimR, blockOffset_ + loopOffset_);
        } else if constexpr (IsSameType<T1, int64_t>::value) {
            this->template ArgMaxV1Int64<true>(
                (__ubuf__ T1*)valuesUb[offset].GetPhyAddr(), (__ubuf__ T2*)indiceUb[offset].GetPhyAddr(),
                (__ubuf__ T1*)srcUb.GetPhyAddr(), outASize_, dimR, blockOffset_ + loopOffset_);
        }
    }

    inQueueX_.FreeTensor(srcUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
template <typename T4>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::ComputeResult()
{
    LocalTensor<T4> srcValueUb = inQueueX_.DeQue<T4>();
    LocalTensor<T2> srcIndiceUb = inQueueIndice_.DeQue<T2>();
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        srcValueUb = xBuf32.Get<float>();
    }

    LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
    LocalTensor<T4> valuesUb = outQueueValues_.AllocTensor<T4>();

    auto srcIndicePtr = (__ubuf__ T2*)srcIndiceUb.GetPhyAddr();
    auto srcValuePtr = (__ubuf__ T4*)srcValueUb.GetPhyAddr();
    auto dstIndicePtr = (__ubuf__ T2*)indiceUb.GetPhyAddr();
    auto dstValuePtr = (__ubuf__ T4*)valuesUb.GetPhyAddr();

    uint16_t vl = this->GetVLSize();
    uint16_t loopA = CeilDivision(outASize_, vl);
    uint32_t processA = outASize_;
    uint32_t sregMask = outASize_;
    uint32_t dimA = outAAlign_;

    uint16_t loopR = tiling_->realCoreNum > 1 ? (tiling_->realCoreNum - 1) : 0;
    uint16_t offsetA = outASize_ >= vl ? vl : outASize_;
    uint32_t repeatElmB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T4> vregValue0, vregValue1;
        AscendC::Reg::RegTensor<T2> vregIndices0, vregIndices1;
        AscendC::Reg::RegTensor<T2> indicesHigher0, indicesHigher1;
        AscendC::Reg::MaskReg mask, maskLower, maskHigher, cmpMask, nanMask, pregLower, pregHigher;
        for (uint16_t i = 0; i < loopA; i++) {
            auto tmpValuePtr = srcValuePtr + dimA;
            auto tmpIndicePtr = srcIndicePtr + dimA;
            mask = AscendC::Reg::UpdateMask<T4>(processA);
            AscendC::Reg::AddrReg valueOffset = AscendC::Reg::CreateAddrReg<T4>(i, vl);
            MicroAPI::LoadAlign(vregValue0, srcValuePtr, valueOffset);
            AscendC::Reg::AddrReg indicesOffset = AscendC::Reg::CreateAddrReg<uint32_t>(i, vl);
            MicroAPI::LoadAlign(vregIndices0, srcIndicePtr, indicesOffset);
            if constexpr (IsSameType<T1, half>::value) {
                maskLower = AscendC::Reg::UpdateMask<T2>(sregMask);
                maskHigher = AscendC::Reg::UpdateMask<T2>(sregMask);
                MicroAPI::LoadAlign(indicesHigher0, srcIndicePtr + repeatElmB32, indicesOffset);
            }
            for (uint16_t j = 0; j < loopR; j++) {
                AscendC::Reg::AddrReg valueAreg = AscendC::Reg::CreateAddrReg<T4>(i, vl, j, dimA);
                AscendC::Reg::AddrReg indicesAreg = AscendC::Reg::CreateAddrReg<uint32_t>(i, vl, j, dimA);
                MicroAPI::LoadAlign(vregValue1, tmpValuePtr, valueAreg);
                if constexpr (isMin) {
                    Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                } else {
                    Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                }
                Compare<T4, CMPMODE::NE>(nanMask, vregValue0, vregValue0, mask);
                AscendC::Reg::Or(cmpMask, cmpMask, nanMask, mask);
                Select(vregValue0, vregValue0, vregValue1, cmpMask);
                if constexpr (IsSameType<T1, half>::value) {
                    AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(pregLower, cmpMask);
                    AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(pregHigher, cmpMask);
                    MicroAPI::LoadAlign(vregIndices1, tmpIndicePtr, indicesAreg);
                    MicroAPI::LoadAlign(indicesHigher1, tmpIndicePtr + repeatElmB32, indicesAreg);
                    Select(vregIndices0, vregIndices0, vregIndices1, pregLower);
                    Select(indicesHigher0, indicesHigher0, indicesHigher1, pregHigher);
                } else {
                    MicroAPI::LoadAlign(vregIndices1, tmpIndicePtr, indicesAreg);
                    Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
                }
            }
            MicroAPI::StoreAlign(dstValuePtr, vregValue0, valueOffset, mask);
            if constexpr (IsSameType<T1, half>::value) {
                MicroAPI::StoreAlign(dstIndicePtr, vregIndices0, indicesOffset, maskLower);
                MicroAPI::StoreAlign(dstIndicePtr + repeatElmB32, indicesHigher0, indicesOffset, maskHigher);
            } else {
                MicroAPI::StoreAlign(dstIndicePtr, vregIndices0, indicesOffset, mask);
            }
        }
    }
    outQueueIndice_.EnQue(indiceUb);
    outQueueValues_.EnQue(valuesUb);
    inQueueX_.FreeTensor(srcValueUb);
    inQueueIndice_.FreeTensor(srcIndiceUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
template <typename T4>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::ComputeResultInt64()
{
    LocalTensor<T4> srcValueUb = inQueueX_.DeQue<T4>();
    LocalTensor<T2> srcIndiceUb = inQueueIndice_.DeQue<T2>();

    LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
    LocalTensor<T4> valuesUb = outQueueValues_.AllocTensor<T4>();

    auto srcIndicePtr = (__ubuf__ T2*)srcIndiceUb.GetPhyAddr();
    auto srcValuePtr = (__ubuf__ T4*)srcValueUb.GetPhyAddr();
    auto dstIndicePtr = (__ubuf__ T2*)indiceUb.GetPhyAddr();
    auto dstValuePtr = (__ubuf__ T4*)valuesUb.GetPhyAddr();

    uint16_t vl = this->GetVLSize();
    uint16_t loopA = CeilDivision(outASize_, vl);
    uint32_t processA = outASize_;
    uint32_t dimA = outAAlign_;

    uint16_t loopR = tiling_->realCoreNum > 1 ? (tiling_->realCoreNum - 1) : 0;
    uint16_t offsetA = outASize_ >= vl ? vl : outASize_;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T4, AscendC::Reg::RegTraitNumTwo> vregValue0, vregValue1;
        AscendC::Reg::RegTensor<T2, AscendC::Reg::RegTraitNumTwo> vregIndices0, vregIndices1;
        AscendC::Reg::MaskReg mask, cmpMask;
        for (uint16_t i = 0; i < loopA; i++) {
            auto tmpValuePtr = srcValuePtr + dimA;
            auto tmpIndicePtr = srcIndicePtr + dimA;
            mask = AscendC::Reg::UpdateMask<uint32_t>(processA);
            MicroAPI::LoadAlign(vregValue0, (__ubuf__ T4*)srcValuePtr + i * vl);
            MicroAPI::LoadAlign(vregIndices0, (__ubuf__ T2*)srcIndicePtr + i * vl);
            for (uint16_t j = 0; j < loopR; j++) {
                MicroAPI::LoadAlign(vregValue1, (__ubuf__ T4*)tmpValuePtr + i * vl + j * dimA);
                if constexpr (isMin) {
                    Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                } else {
                    Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                }
                Select(vregValue0, vregValue0, vregValue1, cmpMask);
                MicroAPI::LoadAlign(vregIndices1, (__ubuf__ T2*)tmpIndicePtr + i * vl + j * dimA);
                Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
            }
            MicroAPI::StoreAlign((__ubuf__ T4*)dstValuePtr + i * vl, vregValue0, mask);
            MicroAPI::StoreAlign((__ubuf__ T2*)dstIndicePtr + i * vl, vregIndices0, mask);
        }
    }
    outQueueIndice_.EnQue(indiceUb);
    outQueueValues_.EnQue(valuesUb);
    inQueueX_.FreeTensor(srcValueUb);
    inQueueIndice_.FreeTensor(srcIndiceUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
template <typename T4>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::ComputeResultInt64AllCore(
    uint64_t actualNextA)
{
    LocalTensor<T4> srcValueUb;
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        srcValueUb = xBuf32.Get<T4>();
    } else {
        srcValueUb = inQueueX_.DeQue<T4>();
    }
    LocalTensor<T2> srcIndiceUb = inQueueIndice_.DeQue<T2>();

    LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
    LocalTensor<T4> valuesUb = outQueueValues_.AllocTensor<T4>();

    auto srcIndicePtr = (__ubuf__ T2*)srcIndiceUb.GetPhyAddr();
    auto srcValuePtr = (__ubuf__ T4*)srcValueUb.GetPhyAddr();
    auto dstIndicePtr = (__ubuf__ T2*)indiceUb.GetPhyAddr();
    auto dstValuePtr = (__ubuf__ T4*)valuesUb.GetPhyAddr();

    uint16_t vl = this->GetVLSize();
    uint16_t loopA = CeilDivision(actualNextA, vl);
    uint32_t processA = actualNextA;
    uint64_t actualNextAAlign = (actualNextA * sizeof(T4) + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32 / sizeof(T4);
    uint32_t dimA = actualNextAAlign;
    uint32_t dimIndicates = (actualNextA * sizeof(T2) + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32 / sizeof(T2);

    uint16_t loopR = tiling_->realCoreNum > 1 ? (tiling_->realCoreNum - 1) : 0;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T4, AscendC::Reg::RegTraitNumTwo> vregValue0, vregValue1;
        AscendC::Reg::RegTensor<T2, AscendC::Reg::RegTraitNumTwo> vregIndices0, vregIndices1;
        AscendC::Reg::MaskReg mask, cmpMask;
        for (uint16_t i = 0; i < loopA; i++) {
            auto tmpValuePtr = srcValuePtr + dimA;
            auto tmpIndicePtr = srcIndicePtr + dimIndicates;
            mask = AscendC::Reg::UpdateMask<uint32_t>(processA);
            MicroAPI::LoadAlign(vregValue0, (__ubuf__ T4*)srcValuePtr + i * vl);
            MicroAPI::LoadAlign(vregIndices0, (__ubuf__ T2*)srcIndicePtr + i * vl);
            for (uint16_t j = 0; j < loopR; j++) {
                MicroAPI::LoadAlign(vregValue1, (__ubuf__ T4*)tmpValuePtr + i * vl + j * dimA);
                if constexpr (isMin) {
                    Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                } else {
                    Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                }
                Select(vregValue0, vregValue0, vregValue1, cmpMask);
                MicroAPI::LoadAlign(vregIndices1, (__ubuf__ T2*)tmpIndicePtr + i * vl + j * dimIndicates);
                Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
            }
            MicroAPI::StoreAlign((__ubuf__ T4*)dstValuePtr + i * vl, vregValue0, mask);
            MicroAPI::StoreAlign((__ubuf__ T2*)dstIndicePtr + i * vl, vregIndices0, mask);
        }
    }
    outQueueIndice_.EnQue(indiceUb);
    outQueueValues_.EnQue(valuesUb);
    if constexpr (!IsSameType<T1, bfloat16_t>::value) {
        inQueueX_.FreeTensor(srcValueUb);
    }
    inQueueIndice_.FreeTensor(srcIndiceUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
template <typename T4>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::ComputeResultAllCore(
    uint64_t actualNextA)
{
    LocalTensor<T4> srcValueUb;
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        srcValueUb = xBuf32.Get<T4>();
    } else {
        srcValueUb = inQueueX_.DeQue<T4>();
    }
    LocalTensor<T2> srcIndiceUb = inQueueIndice_.DeQue<T2>();

    LocalTensor<T2> indiceUb = outQueueIndice_.AllocTensor<T2>();
    LocalTensor<T4> valuesUb = outQueueValues_.AllocTensor<T4>();

    auto srcIndicePtr = (__ubuf__ T2*)srcIndiceUb.GetPhyAddr();
    auto srcValuePtr = (__ubuf__ T4*)srcValueUb.GetPhyAddr();
    auto dstIndicePtr = (__ubuf__ T2*)indiceUb.GetPhyAddr();
    auto dstValuePtr = (__ubuf__ T4*)valuesUb.GetPhyAddr();

    uint16_t vl = this->GetVLSize();
    uint16_t loopA = CeilDivision(actualNextA, vl);
    uint32_t processA = actualNextA;
    uint32_t sregMask = actualNextA;
    uint64_t actualNextAAlign = (actualNextA * sizeof(T4) + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32 / sizeof(T4);
    uint32_t dimA = actualNextAAlign;
    uint32_t dimIndicates = (actualNextA * sizeof(T2) + ALIGN_32 - 1) / ALIGN_32 * ALIGN_32 / sizeof(T2);

    uint16_t loopR = tiling_->realCoreNum > 1 ? (tiling_->realCoreNum - 1) : 0;
    uint16_t offsetA = actualNextA >= vl ? vl : actualNextA;
    uint32_t repeatElmB32 = Ops::Base::GetVRegSize() / sizeof(int32_t);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T4> vregValue0, vregValue1;
        AscendC::Reg::RegTensor<T2> vregIndices0, vregIndices1;
        AscendC::Reg::RegTensor<T2> indicesHigher0, indicesHigher1;
        AscendC::Reg::MaskReg mask, maskLower, maskHigher, cmpMask, nanMask, pregLower, pregHigher;
        for (uint16_t i = 0; i < loopA; i++) {
            auto tmpValuePtr = srcValuePtr + dimA;
            auto tmpIndicePtr = srcIndicePtr + dimIndicates;
            mask = AscendC::Reg::UpdateMask<T4>(processA);
            AscendC::Reg::AddrReg valueOffset = AscendC::Reg::CreateAddrReg<T4>(i, vl);
            MicroAPI::LoadAlign(vregValue0, srcValuePtr, valueOffset);
            AscendC::Reg::AddrReg indicesOffset = AscendC::Reg::CreateAddrReg<uint32_t>(i, vl);
            MicroAPI::LoadAlign(vregIndices0, srcIndicePtr, indicesOffset);
            if constexpr (IsSameType<T1, half>::value) {
                maskLower = AscendC::Reg::UpdateMask<T2>(sregMask);
                maskHigher = AscendC::Reg::UpdateMask<T2>(sregMask);
                MicroAPI::LoadAlign(indicesHigher0, srcIndicePtr + repeatElmB32, indicesOffset);
            }
            for (uint16_t j = 0; j < loopR; j++) {
                AscendC::Reg::AddrReg valueAreg = AscendC::Reg::CreateAddrReg<T4>(i, vl, j, dimA);
                AscendC::Reg::AddrReg indicesAreg = AscendC::Reg::CreateAddrReg<uint32_t>(i, vl, j, dimIndicates);
                MicroAPI::LoadAlign(vregValue1, tmpValuePtr, valueAreg);
                if constexpr (isMin) {
                    Compare<T4, CMPMODE::LE>(cmpMask, vregValue0, vregValue1, mask);
                } else {
                    Compare<T4, CMPMODE::GE>(cmpMask, vregValue0, vregValue1, mask);
                }
                Compare<T4, CMPMODE::NE>(nanMask, vregValue0, vregValue0, mask);
                AscendC::Reg::Or(cmpMask, cmpMask, nanMask, mask);
                Select(vregValue0, vregValue0, vregValue1, cmpMask);
                if constexpr (IsSameType<T1, half>::value) {
                    AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::LOWEST>(pregLower, cmpMask);
                    AscendC::Reg::UnPack<AscendC::Reg::HighLowPart::HIGHEST>(pregHigher, cmpMask);
                    MicroAPI::LoadAlign(vregIndices1, tmpIndicePtr, indicesAreg);
                    MicroAPI::LoadAlign(indicesHigher1, tmpIndicePtr + repeatElmB32, indicesAreg);
                    Select(vregIndices0, vregIndices0, vregIndices1, pregLower);
                    Select(indicesHigher0, indicesHigher0, indicesHigher1, pregHigher);
                } else {
                    MicroAPI::LoadAlign(vregIndices1, tmpIndicePtr, indicesAreg);
                    Select(vregIndices0, vregIndices0, vregIndices1, cmpMask);
                }
            }
            MicroAPI::StoreAlign(dstValuePtr, vregValue0, valueOffset, mask);
            if constexpr (IsSameType<T1, half>::value) {
                MicroAPI::StoreAlign(dstIndicePtr, vregIndices0, indicesOffset, maskLower);
                MicroAPI::StoreAlign(dstIndicePtr + repeatElmB32, indicesHigher0, indicesOffset, maskHigher);
            } else {
                MicroAPI::StoreAlign(dstIndicePtr, vregIndices0, indicesOffset, mask);
            }
        }
    }
    outQueueIndice_.EnQue(indiceUb);
    outQueueValues_.EnQue(valuesUb);
    if constexpr (!IsSameType<T1, bfloat16_t>::value) {
        inQueueX_.FreeTensor(srcValueUb);
    }
    inQueueIndice_.FreeTensor(srcIndiceUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::ProcessMultiCoreResultAllCore()
{
    uint64_t chunkSize = tiling_->gatherBlockSize;
    uint64_t chunksPerA = CeilDivision(tiling_->nextASize, chunkSize);
    uint64_t totalChunks = tiling_->aSize * chunksPerA;

    if (blockIdx_ >= totalChunks) {
        return;
    }

    uint64_t chunkIdx = blockIdx_;
    uint64_t a = chunkIdx / chunksPerA;
    uint64_t chunkInA = chunkIdx % chunksPerA;
    uint64_t nextAOff = chunkInA * chunkSize;
    uint64_t actualNextA = chunkSize;
    if (nextAOff + actualNextA > tiling_->nextASize) {
        actualNextA = tiling_->nextASize - nextAOff;
    }

    CopyInMultiCoreResultAllCore(a, nextAOff, actualNextA);
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        ComputeResultAllCore<float>(actualNextA);
    } else if constexpr (IsSameType<T1, int64_t>::value) {
        ComputeResultInt64AllCore<int64_t>(actualNextA);
    } else {
        ComputeResultAllCore<T1>(actualNextA);
    }

    CopyOut(a * tiling_->nextASize + nextAOff, actualNextA);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::CopyInMultiCoreResultAllCore(
    uint64_t aIdx, uint64_t nextAOff, uint64_t actualNextA)
{
    uint64_t valueTotalBytes = tiling_->aSize * tiling_->realCoreNum * nextAAlign128_ * sizeof(T1);
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        valueTotalBytes = tiling_->aSize * tiling_->realCoreNum * nextAFloatAlign128_ * sizeof(float);
    }
    uint64_t indiceBaseOffset = valueTotalBytes / sizeof(T2);

    uint64_t valueBaseOffset = aIdx * tiling_->realCoreNum * nextAAlign128_ + nextAOff;
    uint64_t indiceReadOffset = indiceBaseOffset + aIdx * tiling_->realCoreNum * indiceAlign128_ + nextAOff;

    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        LocalTensor<float> xUb32 = xBuf32.Get<float>();
        copyInParams_.blockCount = tiling_->realCoreNum;
        copyInParams_.blockLen = actualNextA * sizeof(float);
        copyInParams_.srcStride = (nextAFloatAlign128_ - actualNextA) * sizeof(float);
        copyInParams_.dstStride = 0;
        DataCopyPadExtParams<float> padParams32{false, 0, 0, 0};
        uint64_t floatBaseOffset = aIdx * tiling_->realCoreNum * nextAFloatAlign128_ + nextAOff;
        DataCopyPad(xUb32, valueWorkspace32_[floatBaseOffset], copyInParams_, padParams32);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    } else {
        LocalTensor<T1> xUb = inQueueX_.AllocTensor<T1>();
        copyInParams_.blockCount = tiling_->realCoreNum;
        copyInParams_.blockLen = actualNextA * sizeof(T1);
        copyInParams_.srcStride = (nextAAlign128_ - actualNextA) * sizeof(T1);
        copyInParams_.dstStride = 0;
        DataCopyPad(xUb, valueWorkspace_[valueBaseOffset], copyInParams_, padParams_);
        inQueueX_.EnQue(xUb);
    }

    LocalTensor<T2> indicesUb = inQueueIndice_.AllocTensor<T2>();
    copyInParams_.blockLen = actualNextA * sizeof(T2);
    copyInParams_.srcStride = (indiceAlign128_ - actualNextA) * sizeof(T2);
    DataCopyPadExtParams<T2> padParams{false, 0, 0, 0};
    DataCopyPad(indicesUb, indiceWorkspace_[indiceReadOffset], copyInParams_, padParams);
    inQueueIndice_.EnQue(indicesUb);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::CopyOutWorkSpace(uint64_t offset,
                                                                                                  uint64_t copyNum)
{
    LocalTensor<T1> outValues = outQueueValues_.DeQue<T1>();
    LocalTensor<T2> outIndice = outQueueIndice_.DeQue<T2>();

    DataCopyExtParams copyOutValuesParams{1, 0, 0, 0, 0};
    copyOutValuesParams.blockCount = 1;
    copyOutValuesParams.blockLen = copyNum * sizeof(T1);
    copyOutValuesParams.srcStride = 0;
    copyOutValuesParams.dstStride = 0;
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        copyOutValuesParams.blockLen = copyNum * sizeof(float);
        DataCopyPad(valueWorkspace32_[offset], outBuf32.Get<float>(), copyOutValuesParams);
    } else {
        DataCopyPad(valueWorkspace_[offset], outValues, copyOutValuesParams);
    }
    copyOutValuesParams.blockLen = copyNum * sizeof(T2);
    DataCopyPad(indiceWorkspace_[tiling_->realCoreNum * outAAlign_ + offset], outIndice, copyOutValuesParams);

    outQueueValues_.FreeTensor(outValues);
    outQueueIndice_.FreeTensor(outIndice);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::CopyOutWorkSpaceRow(uint64_t aIdx,
                                                                                                     uint64_t coreIdx)
{
    LocalTensor<T1> outValues = outQueueValues_.DeQue<T1>();
    LocalTensor<T2> outIndice = outQueueIndice_.DeQue<T2>();

    uint64_t valueTotalBytes = tiling_->aSize * tiling_->realCoreNum * nextAAlign128_ * sizeof(T1);
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        valueTotalBytes = tiling_->aSize * tiling_->realCoreNum * nextAFloatAlign128_ * sizeof(float);
    }
    uint64_t indiceBaseOffset = valueTotalBytes / sizeof(T2);

    uint64_t valueOffset = aIdx * tiling_->realCoreNum * nextAAlign128_ + coreIdx * nextAAlign128_;
    uint64_t indiceOffset = indiceBaseOffset + aIdx * tiling_->realCoreNum * indiceAlign128_ +
                            coreIdx * indiceAlign128_;

    DataCopyExtParams copyParams{1, 0, 0, 0, 0};
    copyParams.blockCount = 1;
    copyParams.blockLen = tiling_->nextASize * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    if constexpr (IsSameType<T1, bfloat16_t>::value) {
        copyParams.blockLen = tiling_->nextASize * sizeof(float);
        uint64_t floatOffset = aIdx * tiling_->realCoreNum * nextAFloatAlign128_ + coreIdx * nextAFloatAlign128_;
        DataCopyPad(valueWorkspace32_[floatOffset], outBuf32.Get<float>(), copyParams);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    } else {
        DataCopyPad(valueWorkspace_[valueOffset], outValues, copyParams);
    }
    copyParams.blockLen = tiling_->nextASize * sizeof(T2);
    DataCopyPad(indiceWorkspace_[indiceOffset], outIndice, copyParams);

    outQueueValues_.FreeTensor(outValues);
    outQueueIndice_.FreeTensor(outIndice);
}

template <typename T1, typename T2, typename T3, bool withValue, bool isMin>
__aicore__ inline void ArgMaxWithValueGroupReduce<T1, T2, T3, withValue, isMin>::CopyOut(uint64_t offset,
                                                                                         uint64_t copyNum)
{
    LocalTensor<T1> outValues = outQueueValues_.DeQue<T1>();
    if constexpr (withValue) {
        if constexpr (IsSameType<T1, bfloat16_t>::value) {
            LocalTensor<float> outValuesFloat = outValues.template ReinterpretCast<float>();
            this->CastToBF16(outValues, outValuesFloat, copyNum);
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

#endif // ARG_MAX_WITH_VALUE_GROUP_REDUCE_H
