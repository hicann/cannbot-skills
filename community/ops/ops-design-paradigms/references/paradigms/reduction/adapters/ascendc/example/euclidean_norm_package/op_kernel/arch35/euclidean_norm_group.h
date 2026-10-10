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
 * \file euclidean_norm_group.h
 * \brief EuclideanNorm 算子 Group 模板 kernel 类实现（新范式）。
 *
 * Group 模板：A×R 2D 分核 Phase 1 → SyncAll → Phase 2 RA mini-kernel。
 */
#ifndef OPS_NORM_EUCLIDEAN_NORM_GROUP_H_
#define OPS_NORM_EUCLIDEAN_NORM_GROUP_H_

#include "euclidean_norm_base.h"

namespace NsEuclideanNorm {

// Phase2 后处理 VF 复用 Base 的 PostElewiseVfImpl（fp32 源 → Sqrt → 缩位 Cast → DT 目的，两者逐 token 等价）。

template <typename DType>
class EuclideanNormGroupKernel : public EuclideanNormBaseKernel<DType> {
public:
    using DT = DType;
    using Base = EuclideanNormBaseKernel<DType>;

    __aicore__ inline EuclideanNormGroupKernel() {}

    // Group 初始化：复用 Base::Init（GM/Buffer/event，rGroupCnt 经 Base::td_ 直读），
    // 再绑定 workspace 与派生量 aTotal（A 轴总乘积 = 输出元素数 = workspace 列数）。
    __aicore__ inline void InitGroup(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const EuclideanNormTilingData* td,
                                     TPipe* pipe)
    {
        Base::Init(x, y, td, pipe);
        wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        aTotal_ = 1;
        for (int32_t i = 0; i < Base::td_->axisNum; i += AXIS_INTERVAL) {
            aTotal_ *= Base::td_->axisShape[i];
        }
    }

    // Group 主流程：Phase1 各核算 A×R 部分平方和写 workspace → SyncAll 全核同步 → Phase2 跨行 RA 归约输出。
    __aicore__ inline void ProcessGroup()
    {
        Phase1Process();
        SyncAll();
        Phase2Process();
    }

private:
    __aicore__ inline void Phase1Process();
    __aicore__ inline void Phase2Process();
    __aicore__ inline void DoOneAChunkGroup(int64_t outerGmOff, int64_t aLen, int64_t rStart, int64_t rEnd);
    __aicore__ inline void Phase1OutputToWorkspace(int64_t wsColOff, int64_t aLen, int64_t rChunkIdx, int64_t rCount);

    GlobalTensor<float> wsGm_;
    int64_t aTotal_ = 0;
};

// Phase1：本核负责 1 个 A chunk × 1 段 R（核号 = aChunkIdx×rGroupCnt + rChunkIdx），
// 局部 rCount 上跑二分树归约，树根（部分平方和）经 Phase1OutputToWorkspace 落 workspace 对应行。
template <typename DType>
__aicore__ inline void EuclideanNormGroupKernel<DType>::Phase1Process()
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(Base::td_->usedCoreNum)) {
        return;
    }

    const int64_t rOuter = Base::td_->rLoopCntTotal;

    int64_t aChunkIdx = blockIdx / Base::td_->rGroupCnt;
    int64_t rChunkIdx = blockIdx % Base::td_->rGroupCnt;

    // R 方向大小核式均匀分配：rGroupCnt ≤ rOuter 恒成立（tiling 保证），每组 ≥1 chunk，无空组
    int64_t rSmallGroupLoopCnt = rOuter / Base::td_->rGroupCnt;
    int64_t rBigGroupCnt = rOuter % Base::td_->rGroupCnt;
    int64_t rBigGroupLoopCnt = rSmallGroupLoopCnt + (rBigGroupCnt > 0 ? 1 : 0);
    int64_t rStart = 0;
    int64_t rCount = 0;
    if (rChunkIdx < rBigGroupCnt) {
        rStart = rChunkIdx * rBigGroupLoopCnt;
        rCount = rBigGroupLoopCnt;
    } else {
        rStart = rBigGroupCnt * rBigGroupLoopCnt + (rChunkIdx - rBigGroupCnt) * rSmallGroupLoopCnt;
        rCount = rSmallGroupLoopCnt;
    }
    int64_t rEnd = rStart + rCount;
    if (rStart >= rOuter) {
        return; // 防御性早退（理论上不可达）
    }

    int64_t aLoopIdx = aChunkIdx;
    int64_t aIdx[MAX_PATTERN_RANK] = {0};
    int64_t aSplitChunkIdx = 0;
    Base::UnravelALoop(aLoopIdx, aIdx, aSplitChunkIdx);

    const int64_t aSplitAxisSize = Base::td_->axisShape[Base::td_->aSplitIdx];
    const int64_t aSplitStride = Base::td_->axisStride[Base::td_->aSplitIdx];
    const int64_t aSplitOutStr = Base::outStride_[Base::td_->aSplitIdx];
    const int64_t aChunkStart = aSplitChunkIdx * Base::td_->aUbFactor;
    const int64_t aEndVal = aChunkStart + Base::td_->aUbFactor;
    const int64_t aLen = (aEndVal > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) : Base::td_->aUbFactor;
    if (aLen <= 0) {
        return;
    }

    int64_t chunkGmOff = 0;
    int64_t chunkOutOff = 0;
    for (int32_t k = Base::td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
        chunkGmOff += aIdx[k] * Base::td_->axisStride[k];
        chunkOutOff += aIdx[k] * Base::outStride_[k];
    }
    chunkGmOff += aChunkStart * aSplitStride;
    chunkOutOff += aChunkStart * aSplitOutStr;

    DoOneAChunkGroup(chunkGmOff, aLen, rStart, rEnd);
    Phase1OutputToWorkspace(chunkOutOff, aLen, rChunkIdx, rCount);
}

// Phase1 单 A chunk 的 R 段归约：复用 Base 的 CopyIn/CastSquare/清零/Merge/Reduce/DoCaching 子步骤，
// 与 Base::DoOneAChunk 的区别是 R 迭代区间限定在 [rStart, rEnd)（本核分到的 R 段）。
template <typename DType>
__aicore__ inline void EuclideanNormGroupKernel<DType>::DoOneAChunkGroup(int64_t outerGmOff, int64_t aLen,
                                                                         int64_t rStart, int64_t rEnd)
{
    int64_t rCount = rEnd - rStart;
    if (rCount <= 0) {
        return;
    }

    int64_t bisectionPos = static_cast<int64_t>(Base::FindNearestPower2(static_cast<uint64_t>(rCount)));
    int64_t bisectionTail = rCount - bisectionPos;

    for (int64_t rIdx = 0; rIdx < bisectionPos; ++rIdx) {
        if (rIdx != 0) {
            WaitFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_);
        }

        int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
        int64_t rChunkMain = 0;
        int64_t rLenMain = 0;
        const int64_t rOffMain = Base::UnravelRLoop(rStart + rIdx, rOuterIdx, rChunkMain, rLenMain);

        __ubuf__ DT* preIn = reinterpret_cast<__ubuf__ DT*>(Base::preInBuf_.template Get<DT>().GetPhyAddr());
        __ubuf__ float* preRes =
            reinterpret_cast<__ubuf__ float*>(Base::preReduceResult_.template Get<float>().GetPhyAddr());

        Base::DoCopyInTile(outerGmOff + rOffMain, aLen, rLenMain, preIn);

        SetFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);
        WaitFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);

        Base::CastSquareVf(preIn, preRes);
        if (rLenMain < Base::td_->rUbFactor) {
            Base::ClearChunkExtensionVf(preRes, rLenMain);
        }
        if (Base::isTailR_) {
            Base::ClearInnerBurstTailPadVf(preRes, rLenMain);
        }

        if (rIdx < bisectionTail) {
            int64_t rOuterIdxTail[MAX_PATTERN_RANK] = {0};
            int64_t rChunkTail = 0;
            int64_t rLenTail = 0;
            const int64_t rOffTail =
                Base::UnravelRLoop(rStart + rIdx + bisectionPos, rOuterIdxTail, rChunkTail, rLenTail);

            __ubuf__ float* preResTail =
                reinterpret_cast<__ubuf__ float*>(Base::preReduceResultTail_.template Get<float>().GetPhyAddr());

            SetFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_);
            WaitFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_);

            Base::DoCopyInTile(outerGmOff + rOffTail, aLen, rLenTail, preIn);

            SetFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);
            WaitFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);

            Base::CastSquareVf(preIn, preResTail);
            if (rLenTail < Base::td_->rUbFactor) {
                Base::ClearChunkExtensionVf(preResTail, rLenTail);
            }
            if (Base::isTailR_) {
                Base::ClearInnerBurstTailPadVf(preResTail, rLenTail);
            }
            Base::MergeTmpBufVf(preRes, preResTail);
        }

        const uint16_t cacheID = Base::GetCacheID(rIdx);
        const uint32_t laneA = static_cast<uint32_t>(Base::td_->aUbFactor * Base::td_->innerAProdAlign);
        const uint32_t levelStride = Ops::Base::CeilAlign(laneA, UB_BLOCK_F32);
        const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);

        if (Base::isTailR_) {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {
                laneA, static_cast<uint32_t>(Base::td_->rUbFactorAlign * Base::td_->innerRProdAlign)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
                Base::cacheBuf_.template Get<float>()[levelOff], Base::preReduceResult_.template Get<float>(),
                Base::preReduceResultTail_.template Get<uint8_t>(), srcShape, true);
        } else {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {
                static_cast<uint32_t>(Base::td_->rUbFactorAlign * Base::td_->innerRProdAlign), laneA};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                Base::cacheBuf_.template Get<float>()[levelOff], Base::preReduceResult_.template Get<float>(),
                Base::preReduceResultTail_.template Get<uint8_t>(), srcShape, true);
        }
        Base::DoCachingVf(cacheID);

        if (i != bisectionPos - 1) {
            SetFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_);
        }
    }
}

// Phase1 输出：按本核局部 rCount 现算二分树根偏移，把 cacheBuf 树根（部分平方和）拷到
// workspace 第 rChunkIdx 行 [wsColOff, wsColOff+aLen) 列（fp32，rGroupCnt 行 × aTotal 列矩阵）。
template <typename DType>
__aicore__ inline void EuclideanNormGroupKernel<DType>::Phase1OutputToWorkspace(int64_t wsColOff, int64_t aLen,
                                                                                int64_t rChunkIdx, int64_t rCount)
{
    int64_t bisectionPos = static_cast<int64_t>(Base::FindNearestPower2(static_cast<uint64_t>(rCount)));
    int64_t cacheCount = static_cast<int64_t>(Base::CalLog2(static_cast<uint64_t>(bisectionPos))) + 1;
    int64_t laneN = Base::td_->aUbFactor * Base::td_->innerAProdAlign;
    int64_t levelStride = Ops::Base::CeilAlign(laneN, static_cast<int64_t>(UB_BLOCK_F32));
    int64_t rootOff = (cacheCount - 1) * levelStride;

    auto cacheLocal = Base::cacheBuf_.template Get<float>();

    SetFlag<HardEvent::V_MTE3>(Base::evVtoMTE3_);
    WaitFlag<HardEvent::V_MTE3>(Base::evVtoMTE3_);

    DataCopyExtParams ext;
    if (Base::isTailR_) {
        int64_t innerAProd = 1;
        for (int32_t k = Base::td_->aSplitIdx + AXIS_INTERVAL; k <= Base::LastAAxis(); k += AXIS_INTERVAL) {
            innerAProd *= Base::td_->axisShape[k];
        }
        ext.blockLen = static_cast<uint32_t>(aLen * innerAProd * static_cast<int64_t>(sizeof(float)));
        ext.blockCount = 1;
        ext.srcStride = 0;
    } else {
        const int32_t lastA = Base::LastAAxis();
        const int64_t lastASize = Base::td_->axisShape[lastA];
        if (Base::td_->aSplitIdx == lastA) {
            ext.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
            ext.blockCount = 1;
            ext.srcStride = 0;
        } else {
            int64_t innerAProd = 1;
            for (int32_t k = Base::td_->aSplitIdx + AXIS_INTERVAL; k <= lastA; k += AXIS_INTERVAL) {
                innerAProd *= Base::td_->axisShape[k];
            }
            ext.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(float)));
            ext.blockCount = static_cast<uint16_t>(aLen * innerAProd / lastASize);
            int64_t bsElem = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(DType));
            int64_t lastASizeAlign = Ops::Base::CeilAlign(lastASize, bsElem);
            ext.srcStride = (lastASizeAlign - lastASize) * static_cast<int64_t>(sizeof(float)) / UB_BLOCK_BYTES;
        }
    }
    ext.dstStride = 0;

    int64_t wsOff = rChunkIdx * aTotal_ + wsColOff;
    DataCopyPad(wsGm_[wsOff], cacheLocal[rootOff], ext);
}

// Phase2：workspace（rGroupCnt 行 × aTotal 列）按行方向 RA 归约得完整 Σx²，Sqrt+Cast 后写 GM y。
// aUbFactorP2 在 kernel 内现算（preBufSize/postBufSize/aTotal 三重约束取 min），A chunk 重切分多核。
template <typename DType>
__aicore__ inline void EuclideanNormGroupKernel<DType>::Phase2Process()
{
    int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());

    const int64_t preInElems = Base::td_->preBufSize / static_cast<int64_t>(sizeof(float));
    constexpr int64_t bsFp32 = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(float));

    int64_t aUbFactorP2 = preInElems / Base::td_->rGroupCnt;
    if (aUbFactorP2 >= bsFp32) {
        aUbFactorP2 = (aUbFactorP2 / bsFp32) * bsFp32;
    }
    {
        int64_t limit1 = Base::td_->postBufSize / static_cast<int64_t>(sizeof(DT));
        aUbFactorP2 = (aUbFactorP2 < limit1) ? aUbFactorP2 : limit1;
        aUbFactorP2 = (aUbFactorP2 < aTotal_) ? aUbFactorP2 : aTotal_;
    }

    const int64_t aSplitChunkCntP2 = Ops::Base::CeilDiv(aTotal_, aUbFactorP2);
    const int64_t aLoopCntTotalP2 = aSplitChunkCntP2;

    const int64_t aSmallCoreLoopCntP2 = aLoopCntTotalP2 / Base::td_->usedCoreNum;
    const int64_t aBigCoreCntP2 = aLoopCntTotalP2 % Base::td_->usedCoreNum;
    const int64_t aBigCoreLoopCntP2 = aSmallCoreLoopCntP2 + (aBigCoreCntP2 > 0 ? 1 : 0);
    const int64_t usedCoreNumP2 = (aSmallCoreLoopCntP2 > 0) ? Base::td_->usedCoreNum : aBigCoreCntP2;
    if (blockIdx >= usedCoreNumP2) {
        return;
    }

    int64_t aLoopStart = 0;
    int64_t aLoopEnd = 0;
    if (blockIdx < aBigCoreCntP2) {
        aLoopStart = blockIdx * aBigCoreLoopCntP2;
        aLoopEnd = aLoopStart + aBigCoreLoopCntP2;
    } else {
        aLoopStart = aBigCoreCntP2 * aBigCoreLoopCntP2 + (blockIdx - aBigCoreCntP2) * aSmallCoreLoopCntP2;
        aLoopEnd = aLoopStart + aSmallCoreLoopCntP2;
    }

    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        int64_t aSplitChunkIdx = aLoopIdx;
        int64_t aOff = aSplitChunkIdx * aUbFactorP2;
        int64_t aLen = aUbFactorP2;
        if (aOff + aLen > aTotal_) {
            aLen = aTotal_ - aOff;
        }
        int64_t aLenUb = Ops::Base::CeilAlign(aLen, bsFp32);

        // ── 跨迭代反向：上轮 ReduceSum(V) 读 preReduceResult → 本轮 CopyIn(MTE2) 覆写同一 buffer（V→MTE2 WAR）
        // 首轮跳过 WaitFlag（没有上一轮）
        if (aLoopIdx != aLoopStart) {
            WaitFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_);
            WaitFlag<HardEvent::MTE3_V>(Base::evMte3toV_); // 上轮 CopyOut(MTE3) 读完 outBuf，V 才可写
        }

        auto preReduceLocal = Base::preReduceResult_.template Get<float>();
        {
            DataCopyExtParams ext;
            ext.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
            ext.blockCount = static_cast<uint16_t>(Base::td_->rGroupCnt);
            ext.srcStride = aTotal_ * static_cast<int64_t>(sizeof(float)) - static_cast<int64_t>(ext.blockLen);
            ext.dstStride = 0;
            DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
            DataCopyPad(preReduceLocal, wsGm_[aOff], ext, padParams);
        }

        {
            SetFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);
            WaitFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);

            auto cacheLocal = Base::cacheBuf_.template Get<float>();
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {static_cast<uint32_t>(Base::td_->rGroupCnt),
                                                   static_cast<uint32_t>(aLenUb)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
                cacheLocal, preReduceLocal, Base::preReduceResultTail_.template Get<uint8_t>(), srcShape, true);

            __ubuf__ float* srcPtr = reinterpret_cast<__ubuf__ float*>(cacheLocal.GetPhyAddr());
            __ubuf__ DT* dstPtr = reinterpret_cast<__ubuf__ DT*>(Base::outBuf_.template Get<DT>().GetPhyAddr());
            const uint16_t repeatTime = static_cast<uint16_t>(
                Ops::Base::CeilDiv(static_cast<uint32_t>(aLen), static_cast<uint32_t>(REP_F32_U16)));
            asc_vf_call<PostElewiseVfImpl<DT>>(srcPtr, dstPtr, static_cast<uint32_t>(aLen), repeatTime);

            SetFlag<HardEvent::V_MTE3>(Base::evVtoMTE3_);
            WaitFlag<HardEvent::V_MTE3>(Base::evVtoMTE3_);
        }

        {
            auto outLocal = Base::outBuf_.template Get<DT>();
            DataCopyExtParams outParams;
            outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(DT)));
            outParams.blockCount = 1;
            outParams.srcStride = 0;
            outParams.dstStride = 0;
            DataCopyPad(Base::yGm_[aOff], outLocal, outParams);
        }

        // ── 跨迭代反向：本轮 V 读 preReduceResult → 下轮 CopyIn(MTE2) 覆写同一 buffer（V→MTE2 WAR）
        // 末轮跳过 SetFlag（没有下一轮）
        if (aLoopIdx != aLoopEnd - 1) {
            SetFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_);
            SetFlag<HardEvent::MTE3_V>(Base::evMte3toV_); // MTE3 流标记读完 outBuf
        }
    }
}

} // namespace NsEuclideanNorm

#endif // OPS_NORM_EUCLIDEAN_NORM_GROUP_H_
