/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DROP_OUT_V3_IMPL_H
#define DROP_OUT_V3_IMPL_H

#include "../../random_common/arch35/random_kernel_base.h"

namespace DropOutV3 {
using namespace AscendC;
using namespace RandomKernelBase;
constexpr static uint32_t STEP = 4;
constexpr static uint32_t NUM_2 = 2;
constexpr static uint32_t NUM_4 = 4;
constexpr static uint32_t NUM_8 = 8;
constexpr static uint32_t NUM_78 = 78;
constexpr static uint32_t NUM_256 = 256;
constexpr static uint32_t NUM_2048 = 2048;

constexpr int64_t CORE_ALIGN_SIZE = 512;
constexpr static int32_t UNROLL = 4;
constexpr static int32_t CUTHREADS = 256;
constexpr uint16_t CORE_THREAD_NUM = 512;
constexpr float double_epsilon = 2.22045e-16f; // std::numeric_limits<double>::epsilon()

template <typename T, typename U>
class DropOutV3Impl {
public:
    __aicore__ inline DropOutV3Impl(){};
    __aicore__ inline void Init(GM_ADDR p, GM_ADDR mask, GM_ADDR workspace, const RandomUnifiedSimtTilingDataStruct *tilingData, TPipe *pipe);
    __aicore__ inline void Process(GM_ADDR x, GM_ADDR y, GM_ADDR mask, const RandomUnifiedSimtTilingDataStruct *tilingData);
    __aicore__ inline void CopyInMask(const int64_t offset, const uint32_t count);
    __aicore__ inline void CompareMask(uint32_t count);
    __aicore__ inline void CopyOutMask(const int64_t offset, const uint32_t count);
    __aicore__ inline void UpdateMask(const RandomUnifiedSimtTilingDataStruct *tilingData);
    __aicore__ inline uint64_t GetVectorSize(uint64_t eleCount);
    __aicore__ inline bool IsProbEqual(float a, float b);

private:
    TPipe *pipe_;
    GlobalTensor<uint8_t> maskGM_;
    GlobalTensor<uint8_t> maskWorkspace_;
    TQue<QuePosition::VECIN, NUM_2> maskInQueue_;
    TQue<QuePosition::VECOUT, NUM_2> maskOutQueue_;

    float prob_ = 0.0f;
    uint32_t blockIdx_ = 0;
    int64_t queSize_ = 0;
};

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::Init(GM_ADDR p, GM_ADDR mask, GM_ADDR workspace, const RandomUnifiedSimtTilingDataStruct *tilingData, TPipe *pipe)
{
    pipe_ = pipe;
    prob_ = tilingData->prob;
    maskGM_.SetGlobalBuffer((__gm__ uint8_t*)mask);
    maskWorkspace_.SetGlobalBuffer((__gm__ uint8_t*)workspace);

    queSize_ = Ops::Base::FloorAlign((tilingData->ubSize / NUM_4) , static_cast<int64_t>(NUM_256));
    pipe_->InitBuffer(maskInQueue_, NUM_2, queSize_);
    pipe_->InitBuffer(maskOutQueue_, NUM_2, queSize_);
}

template <typename T, int32_t VEC>
__simt_vf__ __aicore__ LAUNCH_BOUND(CORE_THREAD_NUM) inline void SimtDropOutVec(
    __gm__ volatile T *inputGM, __gm__ volatile T *outputGM, __gm__ volatile uint8_t *maskGM, uint64_t totalThreads, uint64_t magic, uint64_t shift,
    int64_t elementNum, int64_t seed, int64_t offset, float p)
{
    uint32_t key[ALG_KEY_SIZE] = {0, 0};
    uint32_t counter[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
    // todo 除0
    float scale = 1.0f / p;
    PhiloxAlgParsInit(key, counter, seed, offset);
    int64_t idx = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
    int32_t randSize = (VEC + NUM_4 - 1) / NUM_4;
    for (int64_t linearIndex = idx * VEC; linearIndex < elementNum;
         linearIndex += Simt::GetThreadNum() * Simt::GetBlockNum() * VEC) {
        float resultsAll[VEC_16] = {0.0};
        for (uint8_t randIdx = 0; randIdx < randSize; randIdx++) {
            uint32_t counterTmp[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
            CopyArray<ALG_COUNTER_SIZE>(counterTmp, counter);
            ThreadMappingAndSkip<VEC, CONTINUOUS_USE>(linearIndex + randIdx * NUM_4, counterTmp, magic, shift, totalThreads);
            float results[ALG_COUNTER_SIZE];
            PhiloxRandomSimt(key, counterTmp, results);
            for (uint8_t i = 0; i < NUM_4; i++) {
                resultsAll[randIdx * NUM_4 + i] = results[i];
            }
        }
        for (uint8_t iVec = 0; iVec < VEC; iVec++) {
            uint8_t maskBit = (resultsAll[iVec] < p) ? 1 : 0;
            float fMaskBit = (resultsAll[iVec] < p) ? 1.0f : 0.0f;
            outputGM[linearIndex + iVec] = inputGM[linearIndex + iVec] * fMaskBit * scale;
            maskGM[linearIndex + iVec] = maskBit;
        }
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(CORE_THREAD_NUM) inline void ProcessZero(__gm__ volatile T *outputGM,
    __gm__ volatile uint8_t *maskGM, int64_t elementNum)
{
    for (int64_t linearIndex = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); linearIndex < elementNum;
        linearIndex += Simt::GetThreadNum() * Simt::GetBlockNum()) {
            outputGM[linearIndex] = 0;
            maskGM[linearIndex] = 0;
    }
}

template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(CORE_THREAD_NUM) inline void SimtDropOut(
    __gm__ volatile T *inputGM, __gm__ volatile T *outputGM, __gm__ volatile uint8_t *maskGM, uint64_t totalThreads, uint64_t magic, uint64_t shift,
    int64_t elementNum, int64_t seed, int64_t offset, float p)
{
    uint32_t key[ALG_KEY_SIZE] = {0, 0};
    uint32_t counter[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
    float scale = 1.0f / p;
    PhiloxAlgParsInit(key, counter, seed, offset);
    int64_t idx = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx();
    int64_t repeatTime = (elementNum + totalThreads * UNROLL - 1) / (totalThreads * UNROLL);
    for (int64_t loopIdx = 0; loopIdx < repeatTime; loopIdx++ ) {
        for (int64_t linearIndex = idx; linearIndex < totalThreads;
            linearIndex += Simt::GetThreadNum() * Simt::GetBlockNum()) {
            uint32_t counterTmp[ALG_COUNTER_SIZE] = {0, 0, 0, 0};
            CopyArray<ALG_COUNTER_SIZE>(counterTmp, counter);
            ThreadMappingAndSkip<STEP, DIS_CONTINUOUS_USE>(linearIndex + totalThreads * UNROLL * loopIdx, counterTmp, magic, shift, totalThreads);
            float results[ALG_COUNTER_SIZE];
            PhiloxRandomSimt(key, counterTmp, results);
            for (uint8_t iStep = 0; iStep < STEP; iStep++) {
                int64_t li = linearIndex + totalThreads * iStep + loopIdx * totalThreads * UNROLL;
                if (li < elementNum) {
                    uint8_t maskBit = (results[iStep] < p) ? 1 : 0;
                    float fMaskBit = (results[iStep] < p) ? 1.0f : 0.0f;
                    outputGM[li] = inputGM[li] * fMaskBit * scale;
                    maskGM[li] = maskBit;
                }
            }
        }
    }
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::CopyInMask(const int64_t offset, const uint32_t count)
{
    LocalTensor<uint8_t> maskUb = maskInQueue_.AllocTensor<uint8_t>();
    DataCopyExtParams copyParams {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(count * sizeof(uint8_t)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };
    DataCopyPadExtParams<uint8_t> padParams {
        true,
        static_cast<uint8_t>(0),
        static_cast<uint8_t>(Ops::Base::CeilAlign(count, NUM_8) - count),
        static_cast<uint8_t>(0)
    };

    DataCopyPad(maskUb, maskWorkspace_[offset], copyParams, padParams);
    maskInQueue_.EnQue(maskUb);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::CompareMask(uint32_t count)
{
    LocalTensor<uint8_t> maskIn = maskInQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> maskOut = maskOutQueue_.AllocTensor<uint8_t>();
    uint32_t countAlign256 = Ops::Base::CeilAlign(count, NUM_256);
    uint8_t value = 1;
    AscendC::CompareScalar(maskOut, maskIn, value, CMPMODE::EQ, countAlign256);

    maskOutQueue_.EnQue<uint8_t>(maskOut);
    maskInQueue_.FreeTensor(maskIn);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::CopyOutMask(const int64_t offset, const uint32_t count)
{
    LocalTensor<uint8_t> maskOutputUb = maskOutQueue_.DeQue<uint8_t>();
    DataCopyExtParams copyParamsMaskOut {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(Ops::Base::CeilDiv(count, NUM_8) * sizeof(uint8_t)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };
    DataCopyPad(maskGM_[offset / NUM_8], maskOutputUb, copyParamsMaskOut);
    maskOutQueue_.FreeTensor(maskOutputUb);
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::UpdateMask(const RandomUnifiedSimtTilingDataStruct *tilingData)
{
    int64_t valueSize = sizeof(uint8_t);
    int64_t perCoreHandleMaskAlign =
        Ops::Base::CeilAlign(Ops::Base::CeilDiv(tilingData->outputSize, tilingData->usedCoreNum), CORE_ALIGN_SIZE);
    int64_t simdBlockNum = Ops::Base::CeilDiv(tilingData->outputSize, perCoreHandleMaskAlign);
    if (blockIdx_ >= simdBlockNum) {
        return;
    }

    int64_t alignFactor = Ops::Base::GetUbBlockSize() / valueSize;
    int64_t ubFactor = Ops::Base::FloorAlign(queSize_ / valueSize, alignFactor);
    int64_t blockFactor = Ops::Base::CeilDiv(perCoreHandleMaskAlign, ubFactor);
    int64_t tailUbFactor = perCoreHandleMaskAlign - (blockFactor - 1) * ubFactor;
    int64_t tailCoreHandleRandom = tilingData->outputSize - (simdBlockNum - 1) * perCoreHandleMaskAlign;
    if (blockIdx_ == simdBlockNum - 1) {
        blockFactor = Ops::Base::CeilDiv(tailCoreHandleRandom, ubFactor);
        tailUbFactor = tailCoreHandleRandom - (blockFactor - 1) * ubFactor;
    }
    int64_t blockOffSet = blockIdx_ * perCoreHandleMaskAlign;

    for (auto idx = 0; idx < blockFactor; idx++) {
        int64_t currUbTilingSize = ubFactor;
        if (idx == blockFactor - 1) {
            currUbTilingSize = tailUbFactor;
        }
        CopyInMask(blockOffSet + idx * ubFactor, currUbTilingSize);
        CompareMask(currUbTilingSize);
        CopyOutMask(blockOffSet + idx * ubFactor, currUbTilingSize);
    }
}

template <typename T, typename U>
__aicore__ inline uint64_t DropOutV3Impl<T, U>::GetVectorSize(uint64_t eleCount)
{
    uint64_t vecSize = VEC_8;
    if (eleCount % VEC_2 != 0) {
        return 1;
    }

    uint64_t optimalVecSize = VEC_16 / static_cast<uint64_t>(sizeof(T));
    vecSize = (vecSize < optimalVecSize) ? vecSize : optimalVecSize;

    bool canVectorize = true;
    do {
        canVectorize = (eleCount % vecSize) == 0;
        if (!canVectorize) {
            vecSize /= NUM_2;
        }
    } while (vecSize > 1 && !canVectorize);
    return vecSize;
}

template <typename T, typename U>
__aicore__ inline bool DropOutV3Impl<T, U>::IsProbEqual(float a, float b)
{
    return std::abs(a - b) <= double_epsilon;
}

template <typename T, typename U>
__aicore__ inline void DropOutV3Impl<T, U>::Process(
    GM_ADDR x, GM_ADDR y, GM_ADDR mask, const RandomUnifiedSimtTilingDataStruct *tilingData)
{
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= tilingData->usedCoreNum) {
        return;
    }

    if (IsProbEqual(prob_, 0.0f)) {
        AscendC::Simt::VF_CALL<ProcessZero<T>>(AscendC::Simt::Dim3(CORE_THREAD_NUM),
            (__gm__ volatile T*)y, (__gm__ volatile uint8_t*)(maskWorkspace_.GetPhyAddr()), tilingData->outputSize);
        SyncAll();
        UpdateMask(tilingData);
        return;
    }
    // 2048 / 256 * 78 * 256 = 159744
    int64_t blockSize = NUM_256;
    int64_t maxThreadsPerMultiProcessor = NUM_2048;
    int64_t blocksPerSM = maxThreadsPerMultiProcessor / blockSize;
    int64_t multiProcessorCount = NUM_78;
    int64_t grid = (tilingData->outputSize + blockSize - 1) / blockSize;
    grid = (multiProcessorCount * blocksPerSM < grid) ? multiProcessorCount * blocksPerSM : grid;

    uint64_t vecSize = GetVectorSize(tilingData->outputSize);
    uint64_t totalThreads = grid * CUTHREADS;
    uint64_t magic, shift;
    GetUintDivMagicAndShift(magic, shift, totalThreads);

    switch (vecSize) {
        case VEC_16:
            AscendC::Simt::VF_CALL<SimtDropOutVec<T,VEC_16>>(AscendC::Simt::Dim3(CORE_THREAD_NUM),
                (__gm__ volatile T*)x, (__gm__ volatile T*)y, (__gm__ volatile uint8_t*)(maskWorkspace_.GetPhyAddr()),
                totalThreads, magic, shift, tilingData->outputSize, tilingData->seed, tilingData->offset, prob_);
            break;
        case VEC_8:
            AscendC::Simt::VF_CALL<SimtDropOutVec<T,VEC_8>>(AscendC::Simt::Dim3(CORE_THREAD_NUM),
                (__gm__ volatile T*)x, (__gm__ volatile T*)y, (__gm__ volatile uint8_t*)(maskWorkspace_.GetPhyAddr()),
                totalThreads, magic, shift, tilingData->outputSize, tilingData->seed, tilingData->offset, prob_);
            break;
        case VEC_4:
            AscendC::Simt::VF_CALL<SimtDropOutVec<T,VEC_4>>(AscendC::Simt::Dim3(CORE_THREAD_NUM),
                (__gm__ volatile T*)x, (__gm__ volatile T*)y, (__gm__ volatile uint8_t*)(maskWorkspace_.GetPhyAddr()),
                totalThreads, magic, shift, tilingData->outputSize, tilingData->seed, tilingData->offset, prob_);
            break;
        case VEC_2:
            AscendC::Simt::VF_CALL<SimtDropOutVec<T,VEC_2>>(AscendC::Simt::Dim3(CORE_THREAD_NUM),
                (__gm__ volatile T*)x, (__gm__ volatile T*)y, (__gm__ volatile uint8_t*)(maskWorkspace_.GetPhyAddr()),
                totalThreads, magic, shift, tilingData->outputSize, tilingData->seed, tilingData->offset, prob_);
            break;
        default:
            AscendC::Simt::VF_CALL<SimtDropOut<T>>(AscendC::Simt::Dim3(CORE_THREAD_NUM),
                (__gm__ volatile T*)x, (__gm__ volatile T*)y, (__gm__ volatile uint8_t*)(maskWorkspace_.GetPhyAddr()),
                totalThreads, magic, shift, tilingData->outputSize, tilingData->seed, tilingData->offset, prob_);
            break;
    }
    SyncAll();
    UpdateMask(tilingData);
}
}  // namespace DropOutV3
#endif  // DROP_OUT_V3_IMPL_H
