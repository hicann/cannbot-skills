/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file euclidean_norm_base.h
 * \brief EuclideanNorm 算子 Base kernel 类实现（新范式）。
 *
 * 数学：y = sqrt( sum( x^2 ) along axes )
 *
 * 新范式架构（reduction 二分归约范式）：
 *   - 统一使用 TBuf + SetFlag/WaitFlag（不使用 TQue）
 *   - Reduce 直写 cacheBuf[cacheID × levelStride]，消除 tmpBuf 中转
 *   - DoCaching 在 cacheBuf 内部吸收低层后覆写
 *   - 不需要 ClearCacheTree（DoCaching 覆盖写保证正确性）
 *   - Buffer 体系：preInBuf(DT) + preReduceResult(fp32) + preReduceResultTail(fp32) + cacheBuf + outBuf(DT)
 *
 * Reducer = sum：
 *   pad_value = 0（PAD_CLEAR_VALUE）
 *   PreElewise = Cast(b16→fp32) + Square（VF 融合，链长 2 ≤ 7）
 *   PostElewise = Sqrt + Cast(fp32→DT)（VF 融合，链长 2 ≤ 7）
 */
#ifndef OPS_NORM_EUCLIDEAN_NORM_BASE_H_
#define OPS_NORM_EUCLIDEAN_NORM_BASE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "adv_api/reduce/reduce.h"
#include "euclidean_norm_tiling_data.h"
#include "euclidean_norm_tiling_key.h"

namespace NsEuclideanNorm {

using namespace AscendC;

constexpr uint32_t VL_BYTES = Ops::Base::GetVRegSize();
constexpr uint32_t REP_F32 = VL_BYTES / sizeof(float);
constexpr uint16_t REP_F32_U16 = static_cast<uint16_t>(REP_F32);
constexpr uint32_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();
constexpr uint32_t UB_BLOCK_F32 = UB_BLOCK_BYTES / sizeof(float);
constexpr uint64_t UINT64_BITS = 64;
constexpr uint64_t UINT64_TOP_BIT_IDX = UINT64_BITS - 1;

// A/R 轴模式化后偶位 A、奇位 R，相邻同类型轴（同为 A 或同为 R）的固定间距。
constexpr int32_t AXIS_INTERVAL = 2;
// b16（fp16/bf16）单元素字节数。
constexpr size_t BYTES_PER_B16_ELEM = 2;
// ReduceSum srcShape 的 2D 维度（AR/RA 两 pattern 均为二维）。
constexpr size_t REDUCE_SHAPE_DIM = 2;
// FindNearestPower2 数学边界：v ≤ 2 时最近二次幂为 1。
constexpr uint64_t NEAREST_POW2_SMALL_BOUND = 2;

// DataCopyPad 轴层级索引：[0]最内块长 / [1]blockCount / [2]loop1 / [3]loop2 / [4..]外层软循环。
constexpr int32_t BLOCK_COUNT_AXIS_IDX = 1;
constexpr int32_t LOOP1_AXIS_IDX = 2;
constexpr int32_t LOOP2_AXIS_IDX = 3;
constexpr int32_t OUTER_LOOP_AXIS_BASE = 4;

constexpr AscendC::Reg::CastTrait CAST_TRAIT_TO_FP32{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::Reg::RoundMode::CAST_NONE};

constexpr AscendC::Reg::CastTrait CAST_TRAIT_FROM_FP32_FP16{
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

constexpr AscendC::Reg::CastTrait CAST_TRAIT_FROM_FP32_INT32{
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC};

constexpr float PAD_CLEAR_VALUE = 0.0f; // pad_value（sum reducer）

template <typename DType>
__simd_vf__ inline void CastSquareVfImpl(__ubuf__ DType* src, __ubuf__ float* dst, uint32_t totalElems,
                                         uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<DType, float>;
    constexpr bool IsB16 = (sizeof(DType) == BYTES_PER_B16_ELEM);
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        if constexpr (IsFp32) {
            AscendC::Reg::LoadAlign(f32Reg, src + off);
        } else if constexpr (IsB16) {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + off);
            AscendC::Reg::Cast<float, DType, CAST_TRAIT_TO_FP32>(f32Reg, b16Reg, mask);
        } else {
            AscendC::Reg::RegTensor<int32_t> iReg;
            AscendC::Reg::LoadAlign(iReg, src + off);
            AscendC::Reg::Cast<float, int32_t, CAST_TRAIT_TO_FP32>(f32Reg, iReg, mask);
        }

        AscendC::Reg::Mul(f32Reg, f32Reg, f32Reg, mask);
        AscendC::Reg::StoreAlign(dst + off, f32Reg, mask);
    }
}

__simd_vf__ inline void ClearChunkExtTailRVfImpl(__ubuf__ float* base, uint32_t extStart, uint32_t aStride,
                                                 uint32_t extLanes, uint16_t aU16, uint16_t repPerA)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);

    for (uint16_t aIdx = 0; aIdx < aU16; ++aIdx) {
        int32_t aOff = static_cast<int32_t>(aIdx) * static_cast<int32_t>(aStride);
        uint32_t remaining = extLanes;
        for (uint16_t r = 0; r < repPerA; ++r) {
            int32_t off =
                aOff + static_cast<int32_t>(extStart) + static_cast<int32_t>(r) * static_cast<int32_t>(REP_F32);
            auto mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::StoreAlign(base + off, idReg, mask);
        }
    }
}

__simd_vf__ inline void ClearChunkExtTailAVfImpl(__ubuf__ float* base, uint32_t startElem, uint32_t totalClear,
                                                 uint16_t repCount)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalClear;
    for (uint16_t i = 0; i < repCount; ++i) {
        int32_t off = static_cast<int32_t>(startElem) + static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::StoreAlign(base + off, idReg, mask);
    }
}

__simd_vf__ inline void ClearInnerBurstTailPadVfImpl(__ubuf__ float* base, uint16_t rowCntU16, int32_t rowStrideI,
                                                     int32_t windowOff, uint32_t padEnd, uint32_t partialStartInBlock)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);

    uint32_t cntEnd = padEnd;
    uint32_t cntStart = partialStartInBlock;
    auto maskEnd = AscendC::Reg::UpdateMask<float>(cntEnd);
    auto maskStart = AscendC::Reg::UpdateMask<float>(cntStart);
    auto allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg notStart, padMask;
    AscendC::Reg::Not(notStart, maskStart, allMask);
    AscendC::Reg::And(padMask, maskEnd, notStart, allMask);

    for (uint16_t row = 0; row < rowCntU16; ++row) {
        int32_t rowOff = static_cast<int32_t>(row) * rowStrideI;
        AscendC::Reg::StoreAlign(base + rowOff + windowOff, idReg, padMask);
    }
}

__simd_vf__ inline void MergeTmpBufVfImpl(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf, uint32_t totalElems,
                                          uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> aReg, bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(aReg, mainBuf + off);
        AscendC::Reg::LoadAlign(bReg, tailBuf + off);
        AscendC::Reg::Add(aReg, aReg, bReg, mask);
        AscendC::Reg::StoreAlign(mainBuf + off, aReg, mask);
    }
}

__simd_vf__ inline void DoCachingVfImpl(__ubuf__ float* cacheBuf, uint32_t laneN, uint32_t levelStride,
                                        int32_t levelOff, uint16_t repeatTime, uint16_t cacheLevelCnt)
{
    AscendC::Reg::RegTensor<float> aReg, bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneN;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(aReg, cacheBuf + levelOff + off);

        for (uint16_t j = 0; j < cacheLevelCnt; ++j) {
            int32_t lowerLevelOff = static_cast<int32_t>(j) * static_cast<int32_t>(levelStride) + off;
            AscendC::Reg::LoadAlign(bReg, cacheBuf + lowerLevelOff);
            AscendC::Reg::Add(aReg, aReg, bReg, mask);
        }
        AscendC::Reg::StoreAlign(cacheBuf + levelOff + off, aReg, mask);
    }
}

template <typename DType>
__simd_vf__ inline void PostElewiseVfImpl(__ubuf__ float* rootPtr, __ubuf__ DType* outPtr, uint32_t laneN,
                                          uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<DType, float>;
    constexpr bool IsB16 = (sizeof(DType) == BYTES_PER_B16_ELEM);
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneN;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(f32Reg, rootPtr + off);
        AscendC::Reg::Sqrt(f32Reg, f32Reg, mask);

        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(outPtr + off, f32Reg, mask);
        } else if constexpr (IsB16) {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::Cast<DType, float, CAST_TRAIT_FROM_FP32_FP16>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<DType, AscendC::Reg::StoreDist::DIST_PACK_B32>(outPtr + off, b16Reg, mask);
        } else {
            AscendC::Reg::RegTensor<int32_t> iReg;
            AscendC::Reg::Cast<int32_t, float, CAST_TRAIT_FROM_FP32_INT32>(iReg, f32Reg, mask);
            AscendC::Reg::StoreAlign(outPtr + off, iReg, mask);
        }
    }
}

struct UBAxisDesc {
    int32_t gmIdx;
    int64_t actualNum;
    int64_t paddedNum;
    int64_t gmStride;
};

template <typename DType>
class EuclideanNormBaseKernel {
public:
    using DT = DType;

    __aicore__ inline EuclideanNormBaseKernel() {}

    // Base 初始化：缓存 TilingData 指针、现算二分树/输出步长派生量、绑定 GM、
    // 分配 5 个 UB buffer（preIn/preRes/preResTail/cache/out）、取 4 类 eventID。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const EuclideanNormTilingData* td, TPipe* pipe);
    // Base 主流程：多核均分 A 迭代（大核+1 轮），每轮 DoOneAChunk→PostElewise→CopyOut，
    // 跨迭代靠 V_MTE2/MTE3_V 事件反向配对实现流水重叠。
    __aicore__ inline void Process();

protected:
    __aicore__ inline void UnravelBlockLoop(int64_t& aLoopStart, int64_t& aLoopEnd);
    __aicore__ inline void UnravelALoop(int64_t aLoopIdx, int64_t aIdx[], int64_t& aSplitChunkIdx);
    __aicore__ inline int64_t UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[], int64_t& rChunkIdx, int64_t& rLen);

    __aicore__ inline void DoOneAChunk(int64_t outerGmOff, int64_t aLen);
    // 后处理：cacheBuf 树根逐元素 Sqrt + 缩位 Cast 回 DT 写 outBuf（VF 融合，链长 2）。
    __aicore__ inline void PostElewise(int64_t aLen);
    // 输出搬运：outBuf 有效段（aLen×innerAProd）DataCopyPad 直写 GM y（tail-R/tail-A 两路径）。
    __aicore__ inline void CopyOut(int64_t outerOutOff, int64_t aLen);

    // 构造 GM→UB 的轴映射表（actual/padded 双值 + GM 步长），供 DoCopyInTile 组装 DataCopyPad 参数。
    __aicore__ inline int32_t BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[]);
    // 输入搬运：按轴映射表组装 extParams/loopParams（K≥3 开 Loop 模式），外层 for 覆盖 >4 维。
    __aicore__ inline void DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen, __ubuf__ DT* preIn);

    __aicore__ inline void CastSquareVf(__ubuf__ DT* src, __ubuf__ float* dst);
    __aicore__ inline void ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen);
    __aicore__ inline void ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen);
    __aicore__ inline void MergeTmpBufVf(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf);
    __aicore__ inline void DoCachingVf(uint16_t cacheID);

    __aicore__ inline int32_t LastAAxis() const;
    __aicore__ inline int32_t LastRAxis() const;
    __aicore__ inline uint16_t GetCacheID(int64_t idx) const;
    __aicore__ inline uint64_t FindNearestPower2(uint64_t v) const;
    __aicore__ inline uint64_t CalLog2(uint64_t v) const;
    __aicore__ inline int64_t RLenOfChunk(int64_t rChunkIdx) const;

    const EuclideanNormTilingData* td_ = nullptr;
    bool isTailR_ = false; // tail 类型：Init 由 axisNum 奇偶现算
    int64_t rSplitChunkCnt_ = 0;
    int64_t bisectionPos_ = 0;
    int64_t bisectionTail_ = 0;
    int64_t cacheCount_ = 0;
    int64_t outStride_[MAX_PATTERN_RANK] = {0};

    GlobalTensor<DT> xGm_;
    GlobalTensor<DT> yGm_;
    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> preInBuf_;
    TBuf<QuePosition::VECCALC> preReduceResult_;
    TBuf<QuePosition::VECCALC> preReduceResultTail_;
    TBuf<QuePosition::VECCALC> cacheBuf_;
    TBuf<QuePosition::VECCALC> outBuf_;
    event_t evMTE2toV_ = 0;
    event_t evVtoMTE2_ = 0;
    event_t evVtoMTE3_ = 0;
    event_t evMte3toV_ = 0;
};

template <typename DType>
__aicore__ inline int32_t EuclideanNormBaseKernel<DType>::LastAAxis() const
{
    for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 0) {
            return i;
        }
    }
    return 0;
}

template <typename DType>
__aicore__ inline int32_t EuclideanNormBaseKernel<DType>::LastRAxis() const
{
    for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 1) {
            return i;
        }
    }
    return 1;
}

template <typename DType>
__aicore__ inline uint64_t EuclideanNormBaseKernel<DType>::FindNearestPower2(uint64_t v) const
{
    if (v == 0) {
        return 0;
    }
    if (v <= NEAREST_POW2_SMALL_BOUND) {
        return 1;
    }
    const uint64_t num = v - 1;
    const uint64_t pow = UINT64_TOP_BIT_IDX - AscendC::ScalarCountLeadingZero(num);
    return static_cast<uint64_t>(1) << pow;
}

template <typename DType>
__aicore__ inline uint64_t EuclideanNormBaseKernel<DType>::CalLog2(uint64_t v) const
{
    uint64_t res = 0;
    while (v > 1) {
        v >>= 1;
        ++res;
    }
    return res;
}

template <typename DType>
__aicore__ inline uint16_t EuclideanNormBaseKernel<DType>::GetCacheID(int64_t idx) const
{
    const uint64_t v = static_cast<uint64_t>(idx);
    return static_cast<uint16_t>(AscendC::ScalarGetCountOfValue<1>(v ^ (v + 1)) - 1);
}

template <typename DType>
__aicore__ inline int64_t EuclideanNormBaseKernel<DType>::RLenOfChunk(int64_t rChunkIdx) const
{
    const int64_t rAxisSize = td_->axisShape[td_->rSplitIdx];
    const int64_t start = rChunkIdx * td_->rUbFactor;
    return (start + td_->rUbFactor > rAxisSize) ? (rAxisSize - start) : td_->rUbFactor;
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::Init(GM_ADDR x, GM_ADDR y, const EuclideanNormTilingData* td,
                                                            TPipe* pipe)
{
    td_ = td;
    isTailR_ = (td_->axisNum % AXIS_INTERVAL == 0); // 偶数轴→tail-R，奇数轴→tail-A

    rSplitChunkCnt_ = Ops::Base::CeilDiv(td->axisShape[td->rSplitIdx], td->rUbFactor);
    bisectionPos_ = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(td->rLoopCntTotal)));
    bisectionTail_ = td->rLoopCntTotal - bisectionPos_;
    cacheCount_ = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisectionPos_))) + 1;

    {
        int64_t outStrideAcc = 1;
        for (int32_t i = td->axisNum - 1; i >= 0; --i) {
            if (i % AXIS_INTERVAL == 0) {
                outStride_[i] = outStrideAcc;
                outStrideAcc *= td->axisShape[i];
            }
        }
    }

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(y));

    pipe_ = pipe;
    pipe_->InitBuffer(preInBuf_, td->preBufSize);
    pipe_->InitBuffer(preReduceResult_, td->preBufSize);
    pipe_->InitBuffer(preReduceResultTail_, td->preBufSize);
    pipe_->InitBuffer(cacheBuf_, td->cacheBufUbSize);
    pipe_->InitBuffer(outBuf_, td->postBufSize);

    evMTE2toV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    evVtoMTE2_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    evVtoMTE3_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    evMte3toV_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::UnravelBlockLoop(int64_t& aLoopStart, int64_t& aLoopEnd)
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx < static_cast<int64_t>(td_->aBigCoreCnt)) {
        aLoopStart = blockIdx * td_->aBigCoreLoopCnt;
        aLoopEnd = aLoopStart + td_->aBigCoreLoopCnt;
    } else {
        aLoopStart = static_cast<int64_t>(td_->aBigCoreCnt) * td_->aBigCoreLoopCnt +
                     (blockIdx - static_cast<int64_t>(td_->aBigCoreCnt)) * td_->aSmallCoreLoopCnt;
        aLoopEnd = aLoopStart + td_->aSmallCoreLoopCnt;
    }
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::UnravelALoop(int64_t aLoopIdx, int64_t aIdx[],
                                                                    int64_t& aSplitChunkIdx)
{
    int64_t aLoopRem = aLoopIdx;
    aSplitChunkIdx = aLoopRem % td_->aSplitChunkCnt;
    aLoopRem /= td_->aSplitChunkCnt;
    for (int32_t k = td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
        aIdx[k] = aLoopRem % td_->axisShape[k];
        aLoopRem /= td_->axisShape[k];
    }
}

template <typename DType>
__aicore__ inline int64_t EuclideanNormBaseKernel<DType>::UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[],
                                                                       int64_t& rChunkIdx, int64_t& rLen)
{
    rChunkIdx = rIdx % rSplitChunkCnt_;
    int64_t rLoopRem = rIdx / rSplitChunkCnt_;
    int64_t gmOff = 0;
    for (int32_t k = td_->rSplitIdx - AXIS_INTERVAL; k >= 1; k -= AXIS_INTERVAL) {
        rOuterIdx[k] = rLoopRem % td_->axisShape[k];
        rLoopRem /= td_->axisShape[k];
        gmOff += rOuterIdx[k] * td_->axisStride[k];
    }
    rLen = RLenOfChunk(rChunkIdx);
    return gmOff + rChunkIdx * td_->rUbFactor * td_->axisStride[td_->rSplitIdx];
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(td_->usedCoreNum)) {
        return;
    }

    int64_t aLoopStart = 0;
    int64_t aLoopEnd = 0;
    UnravelBlockLoop(aLoopStart, aLoopEnd);

    const int64_t aSplitAxisSize = td_->axisShape[td_->aSplitIdx];
    const int64_t aSplitStride = td_->axisStride[td_->aSplitIdx];
    const int64_t aSplitOutStr = outStride_[td_->aSplitIdx];

    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        int64_t aIdx[MAX_PATTERN_RANK] = {0};
        int64_t aSplitChunkIdx = 0;
        UnravelALoop(aLoopIdx, aIdx, aSplitChunkIdx);

        int64_t chunkGmOff = 0;
        int64_t chunkOutOff = 0;
        for (int32_t k = td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
            chunkGmOff += aIdx[k] * td_->axisStride[k];
            chunkOutOff += aIdx[k] * outStride_[k];
        }

        const int64_t aChunkStart = aSplitChunkIdx * td_->aUbFactor;
        const int64_t aEnd = aChunkStart + td_->aUbFactor;
        const int64_t aLen = (aEnd > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) : td_->aUbFactor;

        if (aLoopIdx != aLoopStart) {
            WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_);
            WaitFlag<HardEvent::MTE3_V>(evMte3toV_); // 上轮 CopyOut(MTE3) 读完 outBuf，V 才可写
        }

        chunkGmOff += aChunkStart * aSplitStride;
        chunkOutOff += aChunkStart * aSplitOutStr;

        DoOneAChunk(chunkGmOff, aLen);
        PostElewise(aLen);
        CopyOut(chunkOutOff, aLen);

        if (aLoopIdx != aLoopEnd - 1) {
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2_);
            SetFlag<HardEvent::MTE3_V>(evMte3toV_); // MTE3 流标记读完 outBuf
        }
    }
}

// 单个 A chunk 的完整 R 二分归约：主段 tile 装载→Cast+Square→（tail 配对 Merge）→ReduceSum 直写
// cacheBuf[cacheID×levelStride]→DoCaching 就地吸收低层。跨 rIdx 迭代以 V_MTE2 事件防 preIn/preRes WAR。
template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::DoOneAChunk(int64_t outerGmOff, int64_t aLen)
{
    for (int64_t rIdx = 0; rIdx < bisectionPos_; ++rIdx) {
        if (rIdx != 0) {
            WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_);
        }

        int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
        int64_t rChunkIdxMain = 0;
        int64_t rLenMain = 0;
        const int64_t rOffMain = UnravelRLoop(rIdx, rOuterIdx, rChunkIdxMain, rLenMain);

        __ubuf__ DT* preIn = reinterpret_cast<__ubuf__ DT*>(preInBuf_.Get<DT>().GetPhyAddr());
        __ubuf__ float* preRes = reinterpret_cast<__ubuf__ float*>(preReduceResult_.Get<float>().GetPhyAddr());

        DoCopyInTile(outerGmOff + rOffMain, aLen, rLenMain, preIn);

        SetFlag<HardEvent::MTE2_V>(evMTE2toV_);
        WaitFlag<HardEvent::MTE2_V>(evMTE2toV_);

        CastSquareVf(preIn, preRes);
        if (rLenMain < td_->rUbFactor) {
            ClearChunkExtensionVf(preRes, rLenMain);
        }
        if (isTailR_) {
            ClearInnerBurstTailPadVf(preRes, rLenMain);
        }

        if (rIdx < bisectionTail_) {
            int64_t rOuterIdxTail[MAX_PATTERN_RANK] = {0};
            int64_t rChunkIdxTail = 0;
            int64_t rLenTail = 0;
            const int64_t rOffTail = UnravelRLoop(rIdx + bisectionPos_, rOuterIdxTail, rChunkIdxTail, rLenTail);

            __ubuf__ float* preResTail =
                reinterpret_cast<__ubuf__ float*>(preReduceResultTail_.Get<float>().GetPhyAddr());

            SetFlag<HardEvent::V_MTE2>(evVtoMTE2_);
            WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_);

            DoCopyInTile(outerGmOff + rOffTail, aLen, rLenTail, preIn);

            SetFlag<HardEvent::MTE2_V>(evMTE2toV_);
            WaitFlag<HardEvent::MTE2_V>(evMTE2toV_);

            CastSquareVf(preIn, preResTail);
            if (rLenTail < td_->rUbFactor) {
                ClearChunkExtensionVf(preResTail, rLenTail);
            }
            if (isTailR_) {
                ClearInnerBurstTailPadVf(preResTail, rLenTail);
            }
            MergeTmpBufVf(preRes, preResTail);
        }

        const uint16_t cacheID = GetCacheID(rIdx);
        const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t levelStride = Ops::Base::CeilAlign(laneA, UB_BLOCK_F32);
        const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);

        if (isTailR_) {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {laneA,
                                                   static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(
                cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(),
                srcShape, /*srcInnerPad=*/true);
        } else {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign),
                                                   laneA};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
                cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(),
                srcShape, /*srcInnerPad=*/true);
        }
        DoCachingVf(cacheID);

        if (rIdx != bisectionPos_ - 1) {
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2_);
        }
    }
}

template <typename DType>
__aicore__ inline int32_t EuclideanNormBaseKernel<DType>::BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[])
{
    int32_t k = 0;
    const int32_t lastA = LastAAxis();
    const int32_t lastR = LastRAxis();
    const int64_t bsElem = static_cast<int64_t>(UB_BLOCK_BYTES) / static_cast<int64_t>(sizeof(DT));

    if (isTailR_) {
        for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 1) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->rSplitIdx) {
                actual = rLen;
                padded = td_->rUbFactorAlign;
            } else if (i == lastR) {
                actual = td_->axisShape[i];
                padded = Ops::Base::CeilAlign(actual, bsElem);
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
        for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 0) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->aSplitIdx) {
                actual = aLen;
                padded = td_->aUbFactor;
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
    } else {
        for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 0) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->aSplitIdx) {
                actual = aLen;
                padded = td_->aUbFactor;
            } else if (i == lastA) {
                actual = td_->axisShape[i];
                padded = Ops::Base::CeilAlign(actual, bsElem);
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
        for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 1) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->rSplitIdx) {
                actual = rLen;
                padded = td_->rUbFactorAlign;
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
    }
    return k;
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen,
                                                                    __ubuf__ DT* preIn)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t axisCnt = BuildUBAxes(aLen, rLen, ubAxes);

    DataCopyExtParams extParams;
    LoopModeParams loopParams;
    loopParams.loop1Size = 0;
    loopParams.loop1SrcStride = 0;
    loopParams.loop1DstStride = 0;
    loopParams.loop2Size = 0;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;

    const int64_t dtBytes = static_cast<int64_t>(sizeof(DT));
    extParams.blockLen = static_cast<uint32_t>(ubAxes[0].actualNum * dtBytes);

    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};

    const int64_t copyPadBytes =
        Ops::Base::CeilAlign(static_cast<int64_t>(extParams.blockLen), static_cast<int64_t>(UB_BLOCK_BYTES));
    const int64_t target0Bytes = ubAxes[0].paddedNum * dtBytes;
    extParams.dstStride = (target0Bytes - copyPadBytes) / static_cast<int64_t>(UB_BLOCK_BYTES);

    if (axisCnt > BLOCK_COUNT_AXIS_IDX) {
        extParams.blockCount = static_cast<uint16_t>(ubAxes[BLOCK_COUNT_AXIS_IDX].actualNum);
        extParams.srcStride =
            ubAxes[BLOCK_COUNT_AXIS_IDX].gmStride * dtBytes - static_cast<int64_t>(extParams.blockLen);
    } else {
        extParams.blockCount = 1;
        extParams.srcStride = 0;
    }

    int64_t ubStride[MAX_PATTERN_RANK] = {0};
    ubStride[0] = dtBytes;
    for (int32_t i = 1; i < axisCnt; ++i) {
        ubStride[i] = ubStride[i - 1] * ubAxes[i - 1].paddedNum;
    }

    if (axisCnt > LOOP1_AXIS_IDX) {
        loopParams.loop1Size = static_cast<uint32_t>(ubAxes[LOOP1_AXIS_IDX].actualNum);
        loopParams.loop1SrcStride =
            static_cast<uint64_t>(ubAxes[LOOP1_AXIS_IDX].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop1DstStride = static_cast<uint64_t>(ubStride[LOOP1_AXIS_IDX]);
        loopParams.loop2Size = 1;
    }
    if (axisCnt > LOOP2_AXIS_IDX) {
        loopParams.loop2Size = static_cast<uint32_t>(ubAxes[LOOP2_AXIS_IDX].actualNum);
        loopParams.loop2SrcStride =
            static_cast<uint64_t>(ubAxes[LOOP2_AXIS_IDX].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop2DstStride = static_cast<uint64_t>(ubStride[LOOP2_AXIS_IDX]);
    }

    const bool useLoopMode = (axisCnt > LOOP1_AXIS_IDX);
    if (useLoopMode) {
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    }

    int64_t outerProd = 1;
    for (int32_t k = OUTER_LOOP_AXIS_BASE; k < axisCnt; ++k) {
        outerProd *= ubAxes[k].actualNum;
    }

    auto preInLocal = preInBuf_.Get<DT>();
    for (int64_t outerFlat = 0; outerFlat < outerProd; ++outerFlat) {
        int64_t addGmOffElem = 0;
        int64_t addUbOffBytes = 0;
        int64_t outerRem = outerFlat;
        for (int32_t k = OUTER_LOOP_AXIS_BASE; k < axisCnt; ++k) {
            const int64_t axisSize = ubAxes[k].actualNum;
            const int64_t axisIdx = outerRem % axisSize;
            outerRem /= axisSize;
            addGmOffElem += axisIdx * ubAxes[k].gmStride;
            addUbOffBytes += axisIdx * ubStride[k];
        }
        const int64_t ubOffElems = addUbOffBytes / dtBytes;
        DataCopyPad(preInLocal[ubOffElems], xGm_[baseGmOff + addGmOffElem], extParams, padParams);
    }

    if (useLoopMode) {
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::CastSquareVf(__ubuf__ DT* src, __ubuf__ float* dst)
{
    const uint32_t totalElems =
        static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign * td_->innerRProdAlign);
    const uint16_t repeatTime =
        static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, static_cast<uint32_t>(REP_F32_U16)));
    asc_vf_call<CastSquareVfImpl<DT>>(src, dst, totalElems, repeatTime);
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen)
{
    if (rLen >= td_->rUbFactor) {
        return;
    }

    if (isTailR_) {
        const uint32_t aBundleEntries = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t innerRPA = static_cast<uint32_t>(td_->innerRProdAlign);
        const uint32_t rLenInner = static_cast<uint32_t>(rLen) * innerRPA;
        const uint32_t extStart = Ops::Base::CeilAlign(rLenInner, UB_BLOCK_F32);
        const uint32_t aStride = static_cast<uint32_t>(td_->rUbFactorAlign) * innerRPA;
        if (extStart >= aStride) {
            return;
        }
        const uint32_t extLanes = aStride - extStart;
        const uint32_t repPerA = Ops::Base::CeilDiv(extLanes, REP_F32);
        const uint16_t aU16 = static_cast<uint16_t>(aBundleEntries);

        asc_vf_call<ClearChunkExtTailRVfImpl>(base, extStart, aStride, extLanes, aU16, static_cast<uint16_t>(repPerA));
    } else {
        const uint32_t cellElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->innerRProdAlign);
        const uint32_t startElem = static_cast<uint32_t>(rLen) * cellElems;
        const uint32_t totalClear = (static_cast<uint32_t>(td_->rUbFactor) - static_cast<uint32_t>(rLen)) * cellElems;
        const uint32_t repCount = Ops::Base::CeilDiv(totalClear, REP_F32);

        asc_vf_call<ClearChunkExtTailAVfImpl>(base, startElem, totalClear, static_cast<uint16_t>(repCount));
    }
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen)
{
    const uint32_t bsInput = UB_BLOCK_BYTES / static_cast<uint32_t>(sizeof(DT));
    const int32_t lastR = LastRAxis();
    const uint32_t validR =
        (td_->rSplitIdx == lastR) ? static_cast<uint32_t>(rLen) : static_cast<uint32_t>(td_->axisShape[lastR]);
    if (validR % bsInput == 0) {
        return;
    }
    const uint32_t rowStride = (td_->rSplitIdx == lastR) ?
                                   static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign) :
                                   Ops::Base::CeilAlign(static_cast<uint32_t>(td_->axisShape[lastR]), bsInput);
    // rSplit!=LastR：按全量行清（partial chunk 需要的行在 UB 中不连续——A entry 在最外层，
    // 行号含 aStride 跳跃），多清的 stale 行其 BurstPad 位置同为脏数据，多清无害
    const uint32_t rowCnt =
        (td_->rSplitIdx == lastR) ?
            static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign) :
            static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign * td_->innerRProdAlign) /
                rowStride;

    const uint32_t padEndInRow = Ops::Base::CeilAlign(validR, bsInput);
    const uint32_t partialBlockIdx = validR / UB_BLOCK_F32;
    const uint32_t partialStartInBlock = validR % UB_BLOCK_F32;
    const uint32_t padEnd = padEndInRow - partialBlockIdx * UB_BLOCK_F32;

    asc_vf_call<ClearInnerBurstTailPadVfImpl>(base, static_cast<uint16_t>(rowCnt), static_cast<int32_t>(rowStride),
                                              static_cast<int32_t>(partialBlockIdx * UB_BLOCK_F32), padEnd,
                                              partialStartInBlock);
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::MergeTmpBufVf(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf)
{
    const uint32_t totalElems =
        static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign * td_->innerRProdAlign);
    const uint16_t repeatTime =
        static_cast<uint16_t>(Ops::Base::CeilDiv(totalElems, static_cast<uint32_t>(REP_F32_U16)));
    asc_vf_call<MergeTmpBufVfImpl>(mainBuf, tailBuf, totalElems, repeatTime);
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::DoCachingVf(uint16_t cacheID)
{
    const uint32_t laneN = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t levelStride = Ops::Base::CeilAlign(laneN, UB_BLOCK_F32);
    const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);
    const uint16_t repeatTime = static_cast<uint16_t>(Ops::Base::CeilDiv(laneN, static_cast<uint32_t>(REP_F32_U16)));

    __ubuf__ float* cachePtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());
    asc_vf_call<DoCachingVfImpl>(cachePtr, laneN, levelStride, levelOff, repeatTime, cacheID);
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::PostElewise(int64_t aLen)
{
    const uint32_t laneN = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t levelStride = Ops::Base::CeilAlign(laneN, UB_BLOCK_F32);
    const int32_t rootOff = static_cast<int32_t>(cacheCount_ - 1) * static_cast<int32_t>(levelStride);

    __ubuf__ float* rootPtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr()) + rootOff;
    __ubuf__ DT* outPtr = reinterpret_cast<__ubuf__ DT*>(outBuf_.Get<DT>().GetPhyAddr());

    const uint16_t repeatTime = static_cast<uint16_t>(Ops::Base::CeilDiv(laneN, static_cast<uint32_t>(REP_F32_U16)));

    asc_vf_call<PostElewiseVfImpl<DT>>(rootPtr, outPtr, laneN, repeatTime);

    SetFlag<HardEvent::V_MTE3>(evVtoMTE3_);
    WaitFlag<HardEvent::V_MTE3>(evVtoMTE3_);
}

template <typename DType>
__aicore__ inline void EuclideanNormBaseKernel<DType>::CopyOut(int64_t outerOutOff, int64_t aLen)
{
    auto outLocal = outBuf_.Get<DT>();

    DataCopyExtParams outParams;
    if (isTailR_) {
        int64_t innerAProd = 1;
        for (int32_t k = td_->aSplitIdx + AXIS_INTERVAL; k <= LastAAxis(); k += AXIS_INTERVAL) {
            innerAProd *= td_->axisShape[k];
        }
        outParams.blockLen = static_cast<uint32_t>(aLen * innerAProd * static_cast<int64_t>(sizeof(DT)));
        outParams.blockCount = 1;
    } else {
        const int32_t lastA = LastAAxis();
        const int64_t lastASize = td_->axisShape[lastA];
        if (td_->aSplitIdx == lastA) {
            outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(DT)));
            outParams.blockCount = 1;
        } else {
            int64_t innerAProd = 1;
            for (int32_t k = td_->aSplitIdx + AXIS_INTERVAL; k <= lastA; k += AXIS_INTERVAL) {
                innerAProd *= td_->axisShape[k];
            }
            outParams.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(DT)));
            outParams.blockCount = static_cast<uint16_t>(aLen * innerAProd / lastASize);
        }
    }
    outParams.srcStride = 0;
    outParams.dstStride = 0;
    DataCopyPad(yGm_[outerOutOff], outLocal, outParams);
}

} // namespace NsEuclideanNorm

#endif // OPS_NORM_EUCLIDEAN_NORM_BASE_H_
