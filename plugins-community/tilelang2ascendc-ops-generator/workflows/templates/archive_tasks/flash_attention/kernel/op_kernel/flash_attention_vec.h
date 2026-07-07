/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_VEC_H
#define FLASH_ATTENTION_VEC_H

#include "kernel_operator.h"
#include "utils/flash_attention_tiling.h"
#include "kernel_common.h"
#include "workspace_queue.h"

using namespace AscendC;

constexpr SoftmaxConfig FA_SOFTMAX_CFG = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
constexpr uint32_t SOFTMAX_TMP_BUF_SIZE = 2048;
constexpr float SOFTMAX_NEG_INF = -1073741824.0f;

template <typename QType>
class FlashAttentionVec {
    static constexpr uint32_t C0 = 32 / sizeof(QType);
    static constexpr uint16_t BRCB_NUM = 32 / sizeof(float);
    static constexpr uint32_t VEC2_M_CHUNK = 8;

public:
    __aicore__ inline FlashAttentionVec() {}

    __aicore__ inline void Init(const FlashAttentionTiling &tiling,
                                GlobalTensor<float> &wsSGm, GlobalTensor<QType> &wsPGm,
                                GlobalTensor<float> &wsOGm, GlobalTensor<float> &wsMetaGm,
                                GlobalTensor<float> &wsAccOGm,
                                GlobalTensor<QType> &outGm)
    {
        tiling_ = tiling;
        wsSGm_ = wsSGm;
        wsPGm_ = wsPGm;
        wsOGm_ = wsOGm;
        wsMetaGm_ = wsMetaGm;
        wsAccOGm_ = wsAccOGm;
        outGm_ = outGm;
        dimAlign_ = AlignUp(tiling_.dim, C0);
        sQueue_.Init(wsSGm_, BLOCK_M * BLOCK_N, SIG_S_READY, SIG_S_FREE);
        pQueue_.Init(wsPGm_, BLOCK_M * BLOCK_N, SIG_P_READY, SIG_P_FREE);
        oQueue_.Init(wsOGm_, BLOCK_M * dimAlign_, SIG_O_READY, SIG_O_FREE);
        subBlockNum_ = GetSubBlockNum();
        subBlockIdx_ = GetSubBlockIdx();
        subBlockRows_ = (subBlockNum_ > 0) ? BLOCK_M / subBlockNum_ : BLOCK_M;
        rowStart_ = subBlockIdx_ * subBlockRows_;
    }

    __aicore__ inline void InitBuffers(TPipe *pipe)
    {
        uint32_t dim = dimAlign_;
        uint32_t inputBufSize = BLOCK_M * BLOCK_N * sizeof(float);
        uint32_t vec2ChunkSize = VEC2_M_CHUNK * dim * sizeof(float);
        if (vec2ChunkSize > inputBufSize) {
            inputBufSize = vec2ChunkSize;
        }
        pipe->InitBuffer(inputQue1_, 2, inputBufSize);

        uint32_t outputBufSize = BLOCK_M * BLOCK_N * sizeof(QType);
        uint32_t vec2OutHalf = VEC2_M_CHUNK * dim * sizeof(QType);
        uint32_t vec2OutFloat = VEC2_M_CHUNK * dim * sizeof(float);
        if (vec2OutHalf > outputBufSize) outputBufSize = vec2OutHalf;
        if (vec2OutFloat > outputBufSize) outputBufSize = vec2OutFloat;
        pipe->InitBuffer(outputQue1_, 1, outputBufSize);

        pipe->InitBuffer(tmpBuf_, 16 * 1024);
        pipe->InitBuffer(softmaxMaxBuf_, SOFTMAX_TMP_BUF_SIZE);
        pipe->InitBuffer(softmaxSumBuf_, SOFTMAX_TMP_BUF_SIZE);
        pipe->InitBuffer(softmaxExpBuf_, SOFTMAX_TMP_BUF_SIZE);
        pipe->InitBuffer(softmaxMaxDefaultBuf_, SOFTMAX_TMP_BUF_SIZE);
        pipe->InitBuffer(softmaxSumDefaultBuf_, SOFTMAX_TMP_BUF_SIZE);

        uint32_t brcbRowsAlign = ((BLOCK_M + BRCB_NUM - 1) / BRCB_NUM) * BRCB_NUM;
        uint32_t brcbSize = brcbRowsAlign * BRCB_NUM * sizeof(float);
        pipe->InitBuffer(brcbBuf_, brcbSize);

        pipe->InitBuffer(maskBuf_, BLOCK_N * sizeof(float));

        softmaxMaxUb_ = softmaxMaxBuf_.Get<float>();
        softmaxSumUb_ = softmaxSumBuf_.Get<float>();
        softmaxExpUb_ = softmaxExpBuf_.Get<float>();
        softmaxMaxDefaultUb_ = softmaxMaxDefaultBuf_.Get<float>();
        softmaxSumDefaultUb_ = softmaxSumDefaultBuf_.Get<float>();

        Duplicate(softmaxMaxDefaultUb_, SOFTMAX_NEG_INF, SOFTMAX_TMP_BUF_SIZE / sizeof(float));
        Duplicate(softmaxSumDefaultUb_, 0.0f, SOFTMAX_TMP_BUF_SIZE / sizeof(float));

        sQueue_.InitFreeSlotsMte2();
        oQueue_.InitFreeSlotsMte2();
    }

    __aicore__ inline void InitState() {}

    __aicore__ inline void ComputeVec1(int slot, bool isFirst, bool isTailKV)
    {
        auto sSlot = sQueue_.ConsumerAcquire();

        uint32_t tileSize = subBlockRows_ * BLOCK_N;

        LocalTensor<float> sUb = inputQue1_.AllocTensor<float>();
        uint64_t sOffset = (uint64_t)rowStart_ * BLOCK_N;
        SetWaitFlag<HardEvent::MTE3_MTE2>();
        DataCopy(sUb, sSlot[sOffset], tileSize);
        inputQue1_.EnQue(sUb);
        sUb = inputQue1_.DeQue<float>();
        SetWaitFlag<HardEvent::MTE2_V>();
        sQueue_.ConsumerReleaseMte2();

        Muls(sUb, sUb, tiling_.smScale, tileSize);
        PipeBarrier<PIPE_V>();

        if (isTailKV) {
            ApplyMaskOnTailKV(sUb);
        }

        ComputeSoftmaxAndStoreMeta(sUb, slot, isFirst);

        LocalTensor<QType> pHalf = outputQue1_.AllocTensor<QType>();
        Cast(pHalf, sUb, RoundMode::CAST_ROUND, tileSize);
        outputQue1_.EnQue(pHalf);
        pHalf = outputQue1_.DeQue<QType>();

        auto pSlot = pQueue_.ProducerAcquire();
        uint64_t pOffset = (uint64_t)rowStart_ * BLOCK_N;
        DataCopy(pSlot[pOffset], pHalf, tileSize);
        outputQue1_.FreeTensor(pHalf);
        inputQue1_.FreeTensor(sUb);
        SetWaitFlag<HardEvent::MTE3_MTE2>();

        pQueue_.ProducerReleaseMte3();
    }

    __aicore__ inline void ComputeVec2(int slot, bool isFirst, bool isLast)
    {
        auto oSlot = oQueue_.ConsumerAcquire();

        uint32_t dim = dimAlign_;
        uint32_t mChunk = VEC2_M_CHUNK;
        uint32_t numChunks = subBlockRows_ / mChunk;
        uint32_t tailChunk = subBlockRows_ % mChunk;

        SetWaitFlag<HardEvent::MTE3_MTE2>();
        for (uint32_t ci = 0; ci < numChunks + (tailChunk > 0 ? 1 : 0); ci++) {
            ProcessVec2Chunk(oSlot, ci, slot, isFirst, isLast,
                             mChunk, numChunks, tailChunk, dim);
        }
        oQueue_.ConsumerReleaseMte2();
    }

private:
    // Load softmax running state (max / sum) from metadata workspace or use defaults.
    __aicore__ inline void LoadSoftmaxState(LocalTensor<float> &inMaxTensor,
                                             LocalTensor<float> &inSumTensor,
                                             int slot, bool isFirst)
    {
        if (isFirst) {
            inMaxTensor = softmaxMaxDefaultUb_;
            inSumTensor = softmaxSumDefaultUb_;
        } else {
            uint32_t prevSlot = (slot + RING_SLOTS - 1) % RING_SLOTS;
            uint32_t prevStateBase = prevSlot * BLOCK_M + rowStart_;
            LocalTensor<float> inStateUb = inputQue1_.AllocTensor<float>();
            DataCopy(inStateUb, wsMetaGm_[prevStateBase], subBlockRows_);
            inputQue1_.EnQue(inStateUb);
            inStateUb = inputQue1_.DeQue<float>();
            LocalTensor<float> inSumUb = outputQue1_.AllocTensor<float>();
            DataCopy(inSumUb, wsMetaGm_[RING_SLOTS * BLOCK_M + prevStateBase], subBlockRows_);
            outputQue1_.EnQue(inSumUb);
            inSumUb = outputQue1_.DeQue<float>();
            SetWaitFlag<HardEvent::MTE2_V>();
            inMaxTensor = inStateUb;
            inSumTensor = inSumUb;
        }
    }

    // Compute softmax on sUb, then store max/sum/exp metadata to workspace.
    __aicore__ inline void ComputeSoftmaxAndStoreMeta(LocalTensor<float> &sUb,
                                                       int slot, bool isFirst)
    {
        uint32_t stateBase = slot * BLOCK_M + rowStart_;

        LocalTensor<uint8_t> softmaxTmpUb = tmpBuf_.Get<uint8_t>();
        SoftMaxShapeInfo srcShape;
        srcShape.srcM = subBlockRows_;
        srcShape.srcK = BLOCK_N;
        srcShape.oriSrcM = subBlockRows_;
        srcShape.oriSrcK = BLOCK_N;
        SoftMaxTiling smTiling = SoftMaxFlashV2TilingFunc(
            srcShape, sizeof(float), sizeof(float), softmaxTmpUb.GetSize(), true, false);

        LocalTensor<float> inMaxTensor;
        LocalTensor<float> inSumTensor;
        LoadSoftmaxState(inMaxTensor, inSumTensor, slot, isFirst);

        SoftmaxFlashV2<float, true, true, false, false, FA_SOFTMAX_CFG>(
            sUb, softmaxSumUb_, softmaxMaxUb_, sUb, softmaxExpUb_,
            inSumTensor, inMaxTensor, softmaxTmpUb, smTiling, srcShape);
        PipeBarrier<PIPE_V>();

        SetWaitFlag<HardEvent::V_MTE3>();
        DataCopy(wsMetaGm_[stateBase], softmaxMaxUb_, subBlockRows_);
        DataCopy(wsMetaGm_[RING_SLOTS * BLOCK_M + stateBase], softmaxSumUb_, subBlockRows_);
        DataCopy(wsMetaGm_[2 * RING_SLOTS * BLOCK_M + stateBase], softmaxExpUb_, subBlockRows_);
        if (!isFirst) {
            inputQue1_.FreeTensor(inMaxTensor);
            outputQue1_.FreeTensor(inSumTensor);
        }

        SetWaitFlag<HardEvent::MTE3_V>();
    }

    // Mask invalid KV columns in the tail block with -inf.
    __aicore__ inline void ApplyMaskOnTailKV(LocalTensor<float> &sUb)
    {
        LocalTensor<float> maskUb = maskBuf_.Get<float>();
        Duplicate(maskUb, SOFTMAX_NEG_INF, BLOCK_N);
        Duplicate(maskUb, 0.0f, tiling_.tailValid);
        PipeBarrier<PIPE_V>();
        for (uint32_t row = 0; row < subBlockRows_; row++) {
            Add(sUb[row * BLOCK_N], sUb[row * BLOCK_N], maskUb, BLOCK_N);
        }
        PipeBarrier<PIPE_V>();
    }

    // Process one chunk in Vec2: load O, merge prev accumulator, save or finalize.
    __aicore__ inline void ProcessVec2Chunk(LocalTensor<float> &oSlot, uint32_t ci,
                                             uint32_t slot, bool isFirst, bool isLast,
                                             uint32_t mChunk, uint32_t numChunks,
                                             uint32_t tailChunk, uint32_t dim)
    {
        uint32_t startRow = ci * mChunk;
        uint32_t dealRows = (ci < numChunks) ? mChunk : tailChunk;
        uint32_t chunkSize = dealRows * dim;
        uint32_t rowOffset = rowStart_ + startRow;
        uint64_t stateRowBase = (uint64_t)slot * BLOCK_M + rowOffset;

        LocalTensor<float> oNewUb = inputQue1_.AllocTensor<float>();
        uint64_t oOffset = (uint64_t)rowOffset * dim;
        DataCopy(oNewUb, oSlot[oOffset], chunkSize);
        inputQue1_.EnQue(oNewUb);
        oNewUb = inputQue1_.DeQue<float>();
        SetWaitFlag<HardEvent::MTE2_V>();

        DataCopy(softmaxExpUb_, wsMetaGm_[2 * RING_SLOTS * BLOCK_M + stateRowBase], dealRows);
        DataCopy(softmaxSumUb_, wsMetaGm_[RING_SLOTS * BLOCK_M + stateRowBase], dealRows);
        SetWaitFlag<HardEvent::MTE2_S>();

        if (!isFirst) {
            MergePreviousAccumulator(oNewUb, slot, rowOffset, dealRows, dim, chunkSize);
        }

        if (isLast) {
            RowDivsImpl(oNewUb, oNewUb, softmaxSumUb_, dealRows, dim);
            PipeBarrier<PIPE_V>();
            FinalizeOutputChunk(oNewUb, rowOffset, dealRows);
        } else {
            StoreAccumulatorChunk(oNewUb, slot, rowOffset, chunkSize, dim);
        }
        inputQue1_.FreeTensor(oNewUb);
    }

    // Merge previous accumulator: load, rescale by exp, add to current.
    __aicore__ inline void MergePreviousAccumulator(LocalTensor<float> &oNewUb,
                                                      uint32_t slot, uint32_t rowOffset,
                                                      uint32_t dealRows, uint32_t dim,
                                                      uint32_t chunkSize)
    {
        LocalTensor<float> oPrevUb = inputQue1_.AllocTensor<float>();
        uint32_t prevSlot = (slot + RING_SLOTS - 1) % RING_SLOTS;
        uint64_t accOffset = ((uint64_t)prevSlot * BLOCK_M + rowOffset) * dim;
        DataCopy(oPrevUb, wsAccOGm_[accOffset], chunkSize);
        inputQue1_.EnQue(oPrevUb);
        oPrevUb = inputQue1_.DeQue<float>();
        SetWaitFlag<HardEvent::MTE2_V>();

        RowMulsImpl(oPrevUb, oPrevUb, softmaxExpUb_, dealRows, dim);
        PipeBarrier<PIPE_V>();

        Add(oNewUb, oNewUb, oPrevUb, chunkSize);
        PipeBarrier<PIPE_V>();

        inputQue1_.FreeTensor(oPrevUb);
    }

    // Store current accumulator chunk to workspace for next iteration.
    __aicore__ inline void StoreAccumulatorChunk(LocalTensor<float> &oNewUb,
                                                   uint32_t slot, uint32_t rowOffset,
                                                   uint32_t chunkSize, uint32_t dim)
    {
        PipeBarrier<PIPE_V>();
        LocalTensor<float> oOutUb = outputQue1_.AllocTensor<float>();
        DataCopy(oOutUb, oNewUb, chunkSize);
        outputQue1_.EnQue(oOutUb);
        oOutUb = outputQue1_.DeQue<float>();

        uint64_t accOutOffset = ((uint64_t)slot * BLOCK_M + rowOffset) * dim;
        DataCopy(wsAccOGm_[accOutOffset], oOutUb, chunkSize);
        SetWaitFlag<HardEvent::MTE3_V>();
        outputQue1_.FreeTensor(oOutUb);
    }

    __aicore__ inline void RowMulsImpl(LocalTensor<float> &dst, LocalTensor<float> &src,
                                       LocalTensor<float> &scale, uint32_t rows, uint32_t cols)
    {
        for (uint32_t row = 0; row < rows; row++) {
            float alpha = scale.GetValue(row);
            Muls(dst[row * cols], src[row * cols], alpha, cols);
        }
    }

    __aicore__ inline void RowDivsImpl(LocalTensor<float> &dst, LocalTensor<float> &src,
                                       LocalTensor<float> &scale, uint32_t rows, uint32_t cols)
    {
        for (uint32_t row = 0; row < rows; row++) {
            float inv = 1.0f / scale.GetValue(row);
            Muls(dst[row * cols], src[row * cols], inv, cols);
        }
    }

    __aicore__ inline void FinalizeOutputChunk(LocalTensor<float> &oUb, uint32_t startRow, uint32_t dealRows)
    {
        uint32_t actualDim = tiling_.dim;
        uint32_t dim = dimAlign_;
        uint32_t qSeqLen = tiling_.qSeqLen;

        // Clamp output rows to [0, qSeqLen)
        uint32_t globalRowStart = curBx_ * BLOCK_M + startRow;
        if (globalRowStart >= (uint32_t)qSeqLen) return;
        uint32_t maxValidRows = qSeqLen - globalRowStart;
        if (dealRows > maxValidRows) dealRows = maxValidRows;

        LocalTensor<QType> outHalf = outputQue1_.AllocTensor<QType>();

        if (dim == actualDim) {
            Cast(outHalf, oUb, RoundMode::CAST_ROUND, dealRows * dim);
            PipeBarrier<PIPE_V>();

            uint64_t outBase = ((uint64_t)curBz_ * tiling_.heads * qSeqLen +
                                (uint64_t)curBy_ * qSeqLen +
                                (uint64_t)curBx_ * BLOCK_M +
                                (uint64_t)startRow) * actualDim;
            DataCopy(outGm_[outBase], outHalf, dealRows * actualDim);
        } else {
            for (uint32_t i = 0; i < dealRows; i++) {
                Cast(outHalf[i * actualDim], oUb[i * dim], RoundMode::CAST_ROUND, actualDim);
            }
            PipeBarrier<PIPE_V>();

            uint64_t outBase = ((uint64_t)curBz_ * tiling_.heads * qSeqLen +
                                (uint64_t)curBy_ * qSeqLen +
                                (uint64_t)curBx_ * BLOCK_M +
                                (uint64_t)startRow) * actualDim;
            for (uint32_t i = 0; i < dealRows; i++) {
                DataCopy(outGm_[outBase + i * actualDim], outHalf[i * actualDim], actualDim);
            }
        }

        SetWaitFlag<HardEvent::MTE3_V>();

        outputQue1_.FreeTensor(outHalf);
    }

public:
    int curBz_, curBy_, curBx_;

private:
    FlashAttentionTiling tiling_;
    uint32_t dimAlign_;
    uint32_t subBlockNum_;
    uint32_t subBlockIdx_;
    uint32_t subBlockRows_;
    uint32_t rowStart_;

    GlobalTensor<float> wsSGm_;
    GlobalTensor<QType> wsPGm_;
    GlobalTensor<float> wsOGm_;
    GlobalTensor<float> wsMetaGm_;
    GlobalTensor<float> wsAccOGm_;
    GlobalTensor<QType> outGm_;
    WorkspaceQueue<float, RING_SLOTS> sQueue_;
    WorkspaceQueue<QType, RING_SLOTS> pQueue_;
    WorkspaceQueue<float, RING_SLOTS> oQueue_;

    TQue<TPosition::VECIN, 2> inputQue1_;
    TQue<TPosition::VECOUT, 1> outputQue1_;

    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> softmaxMaxBuf_;
    TBuf<TPosition::VECCALC> softmaxSumBuf_;
    TBuf<TPosition::VECCALC> softmaxExpBuf_;
    LocalTensor<float> softmaxMaxUb_;
    LocalTensor<float> softmaxSumUb_;
    LocalTensor<float> softmaxExpUb_;

    TBuf<TPosition::VECCALC> softmaxMaxDefaultBuf_;
    TBuf<TPosition::VECCALC> softmaxSumDefaultBuf_;
    LocalTensor<float> softmaxMaxDefaultUb_;
    LocalTensor<float> softmaxSumDefaultUb_;

    TBuf<TPosition::VECCALC> brcbBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
};

#endif // FLASH_ATTENTION_VEC_H
