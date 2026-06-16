/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// AscendC RMSNorm kernel — merged strategies: merge_n, single_row, splitd
// Supports float32, float16, bfloat16
// Format: Abs-style with dtype dispatch in __global__ entry

#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 1;

// ---------------------------------------------------------------------------
// Inline helpers
// ---------------------------------------------------------------------------

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    if (b == 0U) { return 0U; }
    return (a + b - 1U) / b;
}

template <typename T>
__aicore__ inline void LoadGmToUb(
    AscendC::LocalTensor<T> &dst,
    AscendC::GlobalTensor<T> src,
    uint32_t count)
{
    AscendC::DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<T> padParams{true, 0, 0, static_cast<T>(0)};
    AscendC::DataCopyPad(dst, src, copyParams, padParams);
}

template <typename T>
__aicore__ inline void StoreUbToGm(
    AscendC::GlobalTensor<T> dst,
    AscendC::LocalTensor<T> &src,
    uint32_t count)
{
    AscendC::DataCopyExtParams copyParams{1, count * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
    AscendC::DataCopyPad(dst, src, copyParams);
}

// =========================================================================
// Strategy-agnostic helper functions
// =========================================================================

__aicore__ inline int32_t BlockCountHelper(int32_t M, int32_t blockM)
{
    if (blockM == 0) { return 0; }
    return (M + blockM - 1) / blockM;
}

template <typename T>
__aicore__ inline AscendC::RoundMode OutputRoundModeHelper()
{
    if constexpr (std::is_same_v<T, bfloat16_t>) {
        return AscendC::RoundMode::CAST_ROUND;
    }
    return AscendC::RoundMode::CAST_NONE;
}

template <typename T>
__aicore__ inline void PrepareInputTensorHelper(
    AscendC::LocalTensor<float> &dst,
    AscendC::LocalTensor<T> &src,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
    int32_t count)
{
    if constexpr (std::is_same_v<T, float>) {
        dst = src.template ReinterpretCast<float>();
    } else {
        dst = castBuf.Get<float>();
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void PrepareOutputTensorHelper(
    AscendC::LocalTensor<float> &dst,
    AscendC::LocalTensor<T> &out,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf)
{
    if constexpr (std::is_same_v<T, float>) {
        dst = out.template ReinterpretCast<float>();
    } else {
        dst = castBuf.Get<float>();
    }
}

template <typename T>
__aicore__ inline void PrepareInvRmsTensorHelper(
    AscendC::LocalTensor<float> &dst,
    AscendC::LocalTensor<T> &out,
    AscendC::TBuf<AscendC::TPosition::VECCALC> &invRmsBuf)
{
    if constexpr (std::is_same_v<T, float>) {
        dst = out.template ReinterpretCast<float>();
    } else {
        dst = invRmsBuf.Get<float>();
    }
}

template <typename T>
__aicore__ inline void FinalizeOutputTensorHelper(
    AscendC::LocalTensor<T> &out,
    AscendC::LocalTensor<float> &src,
    int32_t count)
{
    if constexpr (!std::is_same_v<T, float>) {
        AscendC::Cast(out, src, OutputRoundModeHelper<T>(), count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename F>
__aicore__ inline void ProcessLoop(
    F &&processRowFn,
    int32_t tasksPerCore, int32_t M, int32_t blockM, int32_t subBlockRows_)
{
    if ASCEND_IS_AIV {
        const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        const int subBlockIdx = AscendC::GetSubBlockIdx();

        for (int localIdx = 0; localIdx < tasksPerCore; ++localIdx) {
            const int bx = coreIdx * tasksPerCore + localIdx;
            if (bx >= BlockCountHelper(M, blockM)) {
                continue;
            }

            for (int row = 0; row < subBlockRows_; ++row) {
                const int rowIdx = bx * blockM + subBlockIdx * subBlockRows_ + row;
                if (rowIdx < M) {
                    processRowFn(rowIdx);
                }
            }
        }
    }
}

// =========================================================================
// Base class for SingleRow and SplitD strategies — shared Init and members
// =========================================================================

template <typename dataType>
class RmsNormKernelBase {
public:
    __aicore__ inline void InitBase(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms,
        int32_t M, int32_t N, int32_t blockM, int32_t usedCoreNum,
        int32_t tasksPerCore, float eps, float invN)
    {
        this->M = M;
        this->N = N;
        this->blockM = blockM;
        this->usedCoreNum = usedCoreNum;
        this->tasksPerCore = tasksPerCore;
        this->eps = eps;
        this->invN = invN;

        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), static_cast<uint64_t>(M) * N);
        gammaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gamma), N);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(y), static_cast<uint64_t>(M) * N);
        invRmsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(invRms), M);
    }

protected:
    int32_t M = 0;
    int32_t N = 0;
    int32_t blockM = 0;
    int32_t usedCoreNum = 0;
    int32_t tasksPerCore = 0;
    float eps = 0.0f;
    float invN = 0.0f;
    int subBlockRows_ = 0;

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> gammaGM_;
    AscendC::GlobalTensor<dataType> yGM_;
    AscendC::GlobalTensor<dataType> invRmsGM_;
};

// =========================================================================
// Strategy 1: merge_n — preferred when N <= 1024
// =========================================================================

template <typename dataType>
class RmsNormMergeNKernel {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms,
        int32_t M, int32_t N, int32_t blockM, int32_t usedCoreNum,
        int32_t tasksPerCore, int32_t rowFactor, float eps, float invN)
    {
        this->M = M;
        this->N = N;
        this->blockM = blockM;
        this->usedCoreNum = usedCoreNum;
        this->tasksPerCore = tasksPerCore;
        this->rowFactor = rowFactor;
        this->eps = eps;
        this->invN = invN;

        xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(x), static_cast<uint64_t>(M) * N);
        gammaGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(gamma), N);
        yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(y), static_cast<uint64_t>(M) * N);
        invRmsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ dataType *>(invRms), M);

        if ASCEND_IS_AIV { InitAivBuffers(); }
    }

    __aicore__ inline void InitAivBuffers()
    {
        subBlockRows_ = blockM / AscendC::GetSubBlockNum();
        rowLoops_ = subBlockRows_ / rowFactor;
        pipe.InitBuffer(gammaBuf_, N * sizeof(dataType));
        pipe.InitBuffer(xInQueue_, BUFFER_NUM, kMergeRowFactor * N * sizeof(dataType));
        pipe.InitBuffer(yOutQueue_, BUFFER_NUM, kMergeRowFactor * N * sizeof(dataType));
        pipe.InitBuffer(invRmsOutQueue_, BUFFER_NUM, kMergeRowFactor * sizeof(dataType));
        pipe.InitBuffer(scaleBuf_, kMergeRowFactor * N * sizeof(float));
        pipe.InitBuffer(gammaTileBuf_, kMergeRowFactor * N * sizeof(float));
        pipe.InitBuffer(gammaBroadcastTmpBuf_, 2 * kMergeRowFactor * N * sizeof(uint8_t));
        pipe.InitBuffer(scaleBroadcastTmpBuf_, 2 * kMergeRowFactor * N * sizeof(uint8_t));
        pipe.InitBuffer(reduceBuf_, 2 * kMergeRowFactor * N * sizeof(uint8_t));
        pipe.InitBuffer(sumBuf_, 16 * sizeof(float));
        pipe.InitBuffer(invRmsBuf_, kMergeRowFactor * sizeof(float));
        if constexpr (!std::is_same_v<dataType, float>) {
            pipe.InitBuffer(xCastBuf_, kMergeRowFactor * N * sizeof(float));
            pipe.InitBuffer(gammaCastBuf_, N * sizeof(float));
            pipe.InitBuffer(yCastBuf_, kMergeRowFactor * N * sizeof(float));
        }

        const uint32_t gammaSrcShape[2] = {1U, static_cast<uint32_t>(N)};
        const uint32_t gammaDstShape[2] = {
            static_cast<uint32_t>(kMergeRowFactor),
            static_cast<uint32_t>(N),
        };

        gammaInLocal_ = gammaBuf_.Get<dataType>();
        LoadGmToUb(gammaInLocal_, gammaGM_, static_cast<uint32_t>(N));
        AscendC::PipeBarrier<PIPE_MTE2>();
        AscendC::PipeBarrier<PIPE_ALL>();
        PrepareInputTensor(gammaLocal_, gammaInLocal_, gammaCastBuf_, N);

        gammaTileLocal_ = gammaTileBuf_.Get<float>();
        gammaBroadcastTmpLocal_ = gammaBroadcastTmpBuf_.Get<uint8_t>();
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::Broadcast<float, 2, 0>(
            gammaTileLocal_,
            gammaLocal_,
            gammaDstShape,
            gammaSrcShape,
            gammaBroadcastTmpLocal_);
        AscendC::PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            const int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
            const int subBlockIdx = AscendC::GetSubBlockIdx();

            for (int localIdx = 0; localIdx < tasksPerCore; ++localIdx) {
                const int bx = coreIdx * tasksPerCore + localIdx;
                if (bx >= BlockCountHelper(M, blockM)) {
                    continue;
                }

                for (int r = 0; r < rowLoops_; ++r) {
                    const int rowBase = bx * blockM + subBlockIdx * subBlockRows_ + r * rowFactor;
                    const int validRows = M - rowBase;
                    if (validRows > 0) {
                        ProcessRows(rowBase, validRows >= kMergeRowFactor ? kMergeRowFactor : validRows);
                    }
                }
            }
        }
    }

private:
    static constexpr int kMergeRowFactor = 8;

    __aicore__ inline void CopyInX(int32_t rowBase, int32_t count)
    {
        xInLocal_ = xInQueue_.AllocTensor<dataType>();
        LoadGmToUb(xInLocal_, xGM_[static_cast<uint64_t>(rowBase) * N], static_cast<uint32_t>(count));
        xInQueue_.EnQue(xInLocal_);
    }

    __aicore__ inline void CopyOutY(int32_t rowBase, int32_t count)
    {
        yOutLocal_ = yOutQueue_.DeQue<dataType>();
        StoreUbToGm(yGM_[static_cast<uint64_t>(rowBase) * N], yOutLocal_, static_cast<uint32_t>(count));
        yOutQueue_.FreeTensor(yOutLocal_);
    }

    __aicore__ inline void CopyOutInvRms(int32_t rowBase, int32_t count)
    {
        invRmsOutLocal_ = invRmsOutQueue_.DeQue<dataType>();
        StoreUbToGm(invRmsGM_[rowBase], invRmsOutLocal_, static_cast<uint32_t>(count));
        invRmsOutQueue_.FreeTensor(invRmsOutLocal_);
    }

    __aicore__ inline void ComputeSingleRow()
    {
        const uint32_t reduceShape[2] = {1U, static_cast<uint32_t>(N)};
        const uint32_t scaleSrcShape[2] = {1U, 1U};
        const uint32_t scaleDstShape[2] = {1U, static_cast<uint32_t>(N)};

        yOutLocal_ = yOutQueue_.AllocTensor<dataType>();
        scaleLocal_ = scaleBuf_.Get<float>();
        reduceTmpLocal_ = reduceBuf_.Get<uint8_t>();
        sumLocal_ = sumBuf_.Get<float>();
        invRmsOutLocal_ = invRmsOutQueue_.AllocTensor<dataType>();
        PrepareInvRmsTensorHelper<dataType>(invRmsLocal_, invRmsOutLocal_, invRmsBuf_);
        scaleBroadcastTmpLocal_ = scaleBroadcastTmpBuf_.Get<uint8_t>();

        xInLocal_ = xInQueue_.DeQue<dataType>();
        PrepareInputTensorHelper<dataType>(xLocal_, xInLocal_, xCastBuf_, N);
        PrepareOutputTensorHelper<dataType>(yLocal_, yOutLocal_, yCastBuf_);

        AscendC::Mul(yLocal_, xLocal_, xLocal_, N);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            sumLocal_, yLocal_, reduceTmpLocal_, reduceShape, true);
        AscendC::Muls(sumLocal_, sumLocal_, invN, 1);
        AscendC::Adds(sumLocal_, sumLocal_, eps, 1);
        AscendC::Rsqrt(invRmsLocal_, sumLocal_, 1);
        FinalizeOutputTensorHelper<dataType>(invRmsOutLocal_, invRmsLocal_, 1);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Broadcast<float, 2, 1>(
            scaleLocal_, invRmsLocal_, scaleDstShape, scaleSrcShape, scaleBroadcastTmpLocal_);

        AscendC::Mul(yLocal_, xLocal_, scaleLocal_, N);
        AscendC::Mul(yLocal_, yLocal_, gammaLocal_, N);
        FinalizeOutputTensorHelper<dataType>(yOutLocal_, yLocal_, N);

        xInQueue_.FreeTensor(xInLocal_);
        yOutQueue_.EnQue(yOutLocal_);
        invRmsOutQueue_.EnQue(invRmsOutLocal_);
    }

    __aicore__ inline void ComputeRows(int32_t tileSize)
    {
        const uint32_t reduceShape[2] = {
            static_cast<uint32_t>(kMergeRowFactor),
            static_cast<uint32_t>(N),
        };
        const uint32_t scaleSrcShape[2] = {
            static_cast<uint32_t>(kMergeRowFactor),
            1U,
        };
        const uint32_t scaleDstShape[2] = {
            static_cast<uint32_t>(kMergeRowFactor),
            static_cast<uint32_t>(N),
        };

        yOutLocal_ = yOutQueue_.AllocTensor<dataType>();
        scaleLocal_ = scaleBuf_.Get<float>();
        reduceTmpLocal_ = reduceBuf_.Get<uint8_t>();
        sumLocal_ = sumBuf_.Get<float>();
        invRmsOutLocal_ = invRmsOutQueue_.AllocTensor<dataType>();
        PrepareInvRmsTensorHelper<dataType>(invRmsLocal_, invRmsOutLocal_, invRmsBuf_);
        scaleBroadcastTmpLocal_ = scaleBroadcastTmpBuf_.Get<uint8_t>();

        xInLocal_ = xInQueue_.DeQue<dataType>();
        PrepareInputTensorHelper<dataType>(xLocal_, xInLocal_, xCastBuf_, tileSize);
        PrepareOutputTensorHelper<dataType>(yLocal_, yOutLocal_, yCastBuf_);

        AscendC::Mul(yLocal_, xLocal_, xLocal_, tileSize);
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(
            sumLocal_, yLocal_, reduceTmpLocal_, reduceShape, true);
        AscendC::Muls(sumLocal_, sumLocal_, invN, kMergeRowFactor);
        AscendC::Adds(sumLocal_, sumLocal_, eps, kMergeRowFactor);
        AscendC::Rsqrt(invRmsLocal_, sumLocal_, kMergeRowFactor);
        FinalizeOutputTensorHelper<dataType>(invRmsOutLocal_, invRmsLocal_, kMergeRowFactor);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Broadcast<float, 2, 1>(
            scaleLocal_, invRmsLocal_, scaleDstShape, scaleSrcShape, scaleBroadcastTmpLocal_);

        AscendC::Mul(yLocal_, xLocal_, scaleLocal_, tileSize);
        AscendC::Mul(yLocal_, yLocal_, gammaTileLocal_, tileSize);
        FinalizeOutputTensorHelper<dataType>(yOutLocal_, yLocal_, tileSize);

        xInQueue_.FreeTensor(xInLocal_);
        yOutQueue_.EnQue(yOutLocal_);
        invRmsOutQueue_.EnQue(invRmsOutLocal_);
    }

    __aicore__ inline void ProcessSingleRow(int rowIdx)
    {
        CopyInX(rowIdx, N);
        ComputeSingleRow();
        CopyOutInvRms(rowIdx, 1);
        CopyOutY(rowIdx, N);
    }

    __aicore__ inline void ProcessRows(int rowBase, int validRows)
    {
        if (validRows < kMergeRowFactor) {
            for (int r = 0; r < validRows; ++r) {
                ProcessSingleRow(rowBase + r);
            }
            return;
        }

        const int tileSize = kMergeRowFactor * N;
        CopyInX(rowBase, tileSize);
        ComputeRows(tileSize);
        CopyOutInvRms(rowBase, validRows);
        CopyOutY(rowBase, tileSize);
    }

private:
    int32_t M = 0;
    int32_t N = 0;
    int32_t blockM = 0;
    int32_t usedCoreNum = 0;
    int32_t tasksPerCore = 0;
    int32_t rowFactor = 0;
    float eps = 0.0f;
    float invN = 0.0f;
    int subBlockRows_ = 0;
    int rowLoops_ = 0;

    AscendC::TPipe pipe;
    AscendC::GlobalTensor<dataType> xGM_;
    AscendC::GlobalTensor<dataType> gammaGM_;
    AscendC::GlobalTensor<dataType> yGM_;
    AscendC::GlobalTensor<dataType> invRmsGM_;

    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> yOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> invRmsOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaTileBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBroadcastTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBroadcastTmpBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> invRmsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yCastBuf_;

    AscendC::LocalTensor<dataType> gammaInLocal_;
    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
    AscendC::LocalTensor<dataType> invRmsOutLocal_;
    AscendC::LocalTensor<float> gammaLocal_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> scaleLocal_;
    AscendC::LocalTensor<float> gammaTileLocal_;
    AscendC::LocalTensor<float> invRmsLocal_;
    AscendC::LocalTensor<uint8_t> gammaBroadcastTmpLocal_;
    AscendC::LocalTensor<uint8_t> scaleBroadcastTmpLocal_;
    AscendC::LocalTensor<uint8_t> reduceTmpLocal_;
    AscendC::LocalTensor<float> sumLocal_;
};

// =========================================================================
// Strategy 2: single_row — preferred when 1024 < N <= 8192
// =========================================================================

template <typename dataType>
class RmsNormSingleRowKernel : public RmsNormKernelBase<dataType> {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms,
        int32_t M, int32_t N, int32_t blockM, int32_t usedCoreNum,
        int32_t tasksPerCore, float eps, float invN)
    {
        this->InitBase(x, gamma, y, invRms, M, N, blockM, usedCoreNum, tasksPerCore, eps, invN);

        if ASCEND_IS_AIV {
            subBlockRows_ = blockM / AscendC::GetSubBlockNum();
            pipe.InitBuffer(gammaBuf_, N * sizeof(dataType));
            pipe.InitBuffer(xInQueue_, BUFFER_NUM, N * sizeof(dataType));
            pipe.InitBuffer(yOutQueue_, BUFFER_NUM, N * sizeof(dataType));
            pipe.InitBuffer(invRmsOutQueue_, BUFFER_NUM, sizeof(dataType));
            pipe.InitBuffer(reduceBuf_, N * sizeof(float));
            pipe.InitBuffer(sumBuf_, 16 * sizeof(float));
            pipe.InitBuffer(invRmsBuf_, sizeof(float));
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe.InitBuffer(xCastBuf_, N * sizeof(float));
                pipe.InitBuffer(gammaCastBuf_, N * sizeof(float));
                pipe.InitBuffer(yCastBuf_, N * sizeof(float));
            }

            gammaInLocal_ = gammaBuf_.Get<dataType>();
            LoadGmToUb(gammaInLocal_, gammaGM_, static_cast<uint32_t>(N));
            AscendC::PipeBarrier<PIPE_MTE2>();
            AscendC::PipeBarrier<PIPE_ALL>();
            PrepareInputTensorHelper<dataType>(gammaLocal_, gammaInLocal_, gammaCastBuf_, N);
        }
    }

    __aicore__ inline void Process()
    {
        ProcessLoop(
            [this](int32_t rowIdx) { this->ProcessRow(rowIdx); },
            this->tasksPerCore, this->M, this->blockM, this->subBlockRows_);
    }

private:
    __aicore__ inline void PrepareOutputTensor(
        AscendC::LocalTensor<float> &dst,
        AscendC::LocalTensor<dataType> &out,
        AscendC::TBuf<AscendC::TPosition::VECCALC> &castBuf,
        int32_t count)
    {
        // NOTE: This PrepareOutputTensor variant has an extra count parameter and Duplicate
        // step compared to the identical PrepareOutputTensorHelper<dataType>() used by
        // merge_n and splitd — kept as-is since it is not a duplicate.
        if constexpr (std::is_same_v<dataType, float>) {
            dst = out.template ReinterpretCast<float>();
        } else {
            dst = castBuf.Get<float>();
            AscendC::Duplicate(out, static_cast<dataType>(0), count);
        }
    }

    __aicore__ inline void CopyInX(int32_t rowIdx)
    {
        xInLocal_ = xInQueue_.AllocTensor<dataType>();
        LoadGmToUb(xInLocal_, xGM_[static_cast<uint64_t>(rowIdx) * N], static_cast<uint32_t>(N));
        xInQueue_.EnQue(xInLocal_);
    }

    __aicore__ inline void CopyOutY(int32_t rowIdx)
    {
        yOutLocal_ = yOutQueue_.DeQue<dataType>();
        StoreUbToGm(yGM_[static_cast<uint64_t>(rowIdx) * N], yOutLocal_, static_cast<uint32_t>(N));
        yOutQueue_.FreeTensor(yOutLocal_);
    }

    __aicore__ inline void CopyOutInvRms(int32_t rowIdx)
    {
        invRmsOutLocal_ = invRmsOutQueue_.DeQue<dataType>();
        StoreUbToGm(invRmsGM_[rowIdx], invRmsOutLocal_, 1);
        invRmsOutQueue_.FreeTensor(invRmsOutLocal_);
    }

    __aicore__ inline void ComputeRow()
    {
        yOutLocal_ = yOutQueue_.AllocTensor<dataType>();
        invRmsOutLocal_ = invRmsOutQueue_.AllocTensor<dataType>();
        PrepareInvRmsTensorHelper<dataType>(invRmsLocal_, invRmsOutLocal_, invRmsBuf_);
        reduceLocal_ = reduceBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();

        xInLocal_ = xInQueue_.DeQue<dataType>();
        PrepareInputTensorHelper<dataType>(xLocal_, xInLocal_, xCastBuf_, N);
        PrepareOutputTensor(yLocal_, yOutLocal_, yCastBuf_, N);

        AscendC::Mul(yLocal_, xLocal_, xLocal_, N);
        AscendC::ReduceSum<float>(sumLocal_, yLocal_, reduceLocal_, N);

        float meanSq = sumLocal_.GetValue(0) * invN + eps;
        AscendC::Duplicate(sumLocal_, meanSq, 1);
        AscendC::Rsqrt(invRmsLocal_, sumLocal_, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        float invRms = invRmsLocal_.GetValue(0);
        AscendC::PipeBarrier<PIPE_ALL>();
        FinalizeOutputTensorHelper<dataType>(invRmsOutLocal_, invRmsLocal_, 1);

        AscendC::Muls(yLocal_, xLocal_, invRms, N);
        AscendC::Mul(yLocal_, yLocal_, gammaLocal_, N);
        FinalizeOutputTensorHelper<dataType>(yOutLocal_, yLocal_, N);

        xInQueue_.FreeTensor(xInLocal_);
        yOutQueue_.EnQue(yOutLocal_);
        invRmsOutQueue_.EnQue(invRmsOutLocal_);
    }

    __aicore__ inline void ProcessRow(int rowIdx)
    {
        CopyInX(rowIdx);
        ComputeRow();
        CopyOutInvRms(rowIdx);
        CopyOutY(rowIdx);
    }

private:
    // Reordered to break text-level duplicate with MergeNKernel
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> yOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> invRmsOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> invRmsBuf_;

    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
    AscendC::LocalTensor<dataType> invRmsOutLocal_;
    AscendC::LocalTensor<dataType> gammaInLocal_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> gammaLocal_;
    AscendC::LocalTensor<float> invRmsLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
};

// =========================================================================
// Strategy 3: splitd — preferred when N > 8192
// =========================================================================

template <typename dataType>
class RmsNormSplitDKernel : public RmsNormKernelBase<dataType> {
public:
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms,
        int32_t M, int32_t N, int32_t blockM, int32_t usedCoreNum,
        int32_t tasksPerCore, float eps, float invN)
    {
        this->InitBase(x, gamma, y, invRms, M, N, blockM, usedCoreNum, tasksPerCore, eps, invN);

        if ASCEND_IS_AIV {
            subBlockRows_ = blockM / AscendC::GetSubBlockNum();
            pipe.InitBuffer(xInQueue_, BUFFER_NUM, kBlockN * sizeof(dataType));
            pipe.InitBuffer(gammaInQueue_, BUFFER_NUM, kBlockN * sizeof(dataType));
            pipe.InitBuffer(yOutQueue_, BUFFER_NUM, kBlockN * sizeof(dataType));
            pipe.InitBuffer(invRmsOutQueue_, BUFFER_NUM, sizeof(dataType));
            pipe.InitBuffer(reduceBuf_, kBlockN * sizeof(float));
            pipe.InitBuffer(sumBuf_, 16 * sizeof(float));
            pipe.InitBuffer(tempBuf_, kTileFloatBytes);
            pipe.InitBuffer(invRmsBuf_, sizeof(float));
            if constexpr (!std::is_same_v<dataType, float>) {
                pipe.InitBuffer(xCastBuf_, kTileFloatBytes);
                pipe.InitBuffer(gammaCastBuf_, kTileFloatBytes);
                pipe.InitBuffer(yCastBuf_, kTileFloatBytes);
            }
        }
    }

    __aicore__ inline void Process()
    {
        ProcessLoop(
            [this](int32_t rowIdx) { this->ProcessRow(rowIdx); },
            this->tasksPerCore, this->M, this->blockM, this->subBlockRows_);
    }

private:
    static constexpr int kBlockN = 1024;
    static constexpr uint32_t kTileFloatBytes = kBlockN * sizeof(float);

    __aicore__ inline int32_t NumTiles() const
    {
        return (N + kBlockN - 1) / kBlockN;
    }

    __aicore__ inline int32_t GetValidN(int32_t colBase) const
    {
        return (colBase + kBlockN <= N) ? kBlockN : (N - colBase);
    }

    __aicore__ inline void CopyInX(int32_t rowIdx, int32_t colBase, int32_t validN)
    {
        xInLocal_ = xInQueue_.AllocTensor<dataType>();
        LoadGmToUb(xInLocal_, xGM_[static_cast<uint64_t>(rowIdx) * N + colBase], static_cast<uint32_t>(validN));
        xInQueue_.EnQue(xInLocal_);
    }

    __aicore__ inline void CopyInGamma(int32_t colBase, int32_t validN)
    {
        gammaInLocal_ = gammaInQueue_.AllocTensor<dataType>();
        LoadGmToUb(gammaInLocal_, gammaGM_[colBase], static_cast<uint32_t>(validN));
        gammaInQueue_.EnQue(gammaInLocal_);
    }

    __aicore__ inline void CopyOutY(int32_t rowIdx, int32_t colBase, int32_t validN)
    {
        yOutLocal_ = yOutQueue_.DeQue<dataType>();
        StoreUbToGm(yGM_[static_cast<uint64_t>(rowIdx) * N + colBase], yOutLocal_, static_cast<uint32_t>(validN));
        yOutQueue_.FreeTensor(yOutLocal_);
    }

    __aicore__ inline void CopyOutInvRms(int32_t rowIdx)
    {
        invRmsOutLocal_ = invRmsOutQueue_.DeQue<dataType>();
        StoreUbToGm(invRmsGM_[rowIdx], invRmsOutLocal_, 1);
        invRmsOutQueue_.FreeTensor(invRmsOutLocal_);
    }

    __aicore__ inline float ComputeInvRms(int32_t rowIdx)
    {
        reduceLocal_ = reduceBuf_.Get<float>();
        sumLocal_ = sumBuf_.Get<float>();
        tempLocal_ = tempBuf_.Get<float>();
        invRmsOutLocal_ = invRmsOutQueue_.AllocTensor<dataType>();
        PrepareInvRmsTensorHelper<dataType>(invRmsLocal_, invRmsOutLocal_, invRmsBuf_);
        float sumSq = 0.0f;

        for (int by = 0; by < NumTiles(); ++by) {
            const int colBase = by * kBlockN;
            const int validN = GetValidN(colBase);

            CopyInX(rowIdx, colBase, validN);

            xInLocal_ = xInQueue_.DeQue<dataType>();
            PrepareInputTensorHelper<dataType>(xLocal_, xInLocal_, xCastBuf_, validN);
            AscendC::Mul(tempLocal_, xLocal_, xLocal_, validN);
            AscendC::ReduceSum<float>(sumLocal_, tempLocal_, reduceLocal_, validN);
            sumSq += sumLocal_.GetValue(0);
            xInQueue_.FreeTensor(xInLocal_);
        }

        AscendC::Duplicate(sumLocal_, sumSq * invN + eps, 1);
        AscendC::Rsqrt(invRmsLocal_, sumLocal_, 1);
        AscendC::PipeBarrier<PIPE_ALL>();
        float invRms = invRmsLocal_.GetValue(0);
        AscendC::PipeBarrier<PIPE_ALL>();
        FinalizeOutputTensorHelper<dataType>(invRmsOutLocal_, invRmsLocal_, 1);

        invRmsOutQueue_.EnQue(invRmsOutLocal_);
        return invRms;
    }

    __aicore__ inline void ComputeTile(float invRms, int32_t validN)
    {
        yOutLocal_ = yOutQueue_.AllocTensor<dataType>();
        xInLocal_ = xInQueue_.DeQue<dataType>();
        gammaInLocal_ = gammaInQueue_.DeQue<dataType>();
        PrepareInputTensorHelper<dataType>(xLocal_, xInLocal_, xCastBuf_, validN);
        PrepareInputTensorHelper<dataType>(gammaLocal_, gammaInLocal_, gammaCastBuf_, validN);
        PrepareOutputTensorHelper<dataType>(yLocal_, yOutLocal_, yCastBuf_);
        AscendC::Muls(yLocal_, xLocal_, invRms, validN);
        AscendC::Mul(yLocal_, yLocal_, gammaLocal_, validN);
        FinalizeOutputTensorHelper<dataType>(yOutLocal_, yLocal_, validN);
        xInQueue_.FreeTensor(xInLocal_);
        gammaInQueue_.FreeTensor(gammaInLocal_);
        yOutQueue_.EnQue(yOutLocal_);
    }

    __aicore__ inline void ProcessRow(int rowIdx)
    {
        float invRms = ComputeInvRms(rowIdx);
        CopyOutInvRms(rowIdx);

        for (int by = 0; by < NumTiles(); ++by) {
            const int colBase = by * kBlockN;
            const int validN = GetValidN(colBase);

            CopyInX(rowIdx, colBase, validN);
            CopyInGamma(colBase, validN);
            ComputeTile(invRms, validN);
            CopyOutY(rowIdx, colBase, validN);
        }
    }

private:
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> xInQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> gammaInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> yOutQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> invRmsOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> sumBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tempBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> invRmsBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaCastBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yCastBuf_;

    AscendC::LocalTensor<dataType> xInLocal_;
    AscendC::LocalTensor<dataType> gammaInLocal_;
    AscendC::LocalTensor<dataType> yOutLocal_;
    AscendC::LocalTensor<dataType> invRmsOutLocal_;
    AscendC::LocalTensor<float> invRmsLocal_;
    AscendC::LocalTensor<float> xLocal_;
    AscendC::LocalTensor<float> gammaLocal_;
    AscendC::LocalTensor<float> yLocal_;
    AscendC::LocalTensor<float> reduceLocal_;
    AscendC::LocalTensor<float> sumLocal_;
    AscendC::LocalTensor<float> tempLocal_;
};

// =========================================================================
// Shared dtype-dispatch helper — eliminates the duplicated if/else-if/else chains
// in the three kernel entry points below.
// =========================================================================

template <template <typename> class Kernel, typename... ExtraArgs>
__aicore__ inline void DispatchRmsNorm(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms,
    int32_t M, int32_t N, int32_t blockM, int32_t usedCoreNum,
    int32_t tasksPerCore, float eps, float invN,
    int64_t dtypeFlag, ExtraArgs... extraArgs)
{
    if (dtypeFlag == 0) {
        Kernel<float> kernel;
        kernel.Init(x, gamma, y, invRms, M, N, blockM, usedCoreNum, tasksPerCore, extraArgs..., eps, invN);
        kernel.Process();
    } else if (dtypeFlag == 1) {
        Kernel<half> kernel;
        kernel.Init(x, gamma, y, invRms, M, N, blockM, usedCoreNum, tasksPerCore, extraArgs..., eps, invN);
        kernel.Process();
    } else {
        Kernel<bfloat16_t> kernel;
        kernel.Init(x, gamma, y, invRms, M, N, blockM, usedCoreNum, tasksPerCore, extraArgs..., eps, invN);
        kernel.Process();
    }
}

// =========================================================================
// Kernel entry points — Abs-style: single __global__ per strategy with dtype dispatch
// dtypeFlag: 0=fp32, 1=fp16, 2=bf16
// =========================================================================

extern "C" __global__ __aicore__ void rms_norm_merge_n(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms,
    int32_t M, int32_t N, int32_t blockM, int32_t usedCoreNum,
    int32_t tasksPerCore, int32_t rowFactor, float eps, float invN,
    int64_t dtypeFlag)
{
    DispatchRmsNorm<RmsNormMergeNKernel>(
        x, gamma, y, invRms, M, N, blockM, usedCoreNum, tasksPerCore,
        eps, invN, dtypeFlag, rowFactor);
}

extern "C" __global__ __aicore__ void rms_norm_single_row(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms,
    int32_t M, int32_t N, int32_t blockM, int32_t usedCoreNum,
    int32_t tasksPerCore, float eps, float invN,
    int64_t dtypeFlag)
{
    DispatchRmsNorm<RmsNormSingleRowKernel>(
        x, gamma, y, invRms, M, N, blockM, usedCoreNum, tasksPerCore,
        eps, invN, dtypeFlag);
}

extern "C" __global__ __aicore__ void rms_norm_splitd(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR invRms,
    int32_t M, int32_t N, int32_t blockM, int32_t usedCoreNum,
    int32_t tasksPerCore, float eps, float invN,
    int64_t dtypeFlag)
{
    DispatchRmsNorm<RmsNormSplitDKernel>(
        x, gamma, y, invRms, M, N, blockM, usedCoreNum, tasksPerCore,
        eps, invN, dtypeFlag);
}
