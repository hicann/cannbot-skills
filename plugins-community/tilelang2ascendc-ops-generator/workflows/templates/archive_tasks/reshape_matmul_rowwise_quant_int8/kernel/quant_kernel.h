/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @file quant_kernel.h
 *
 * Vector kernel for row-wise dynamic int8 quantization.
 * Two-pass processing:
 *   Pass 1: Scan all n_tiles → compute per-row absmax → row_scale = absmax / 127
 *   Pass 2: For each n_tile → per-row mul by 1/scale → Cast(fp32→fp16→int8) → write output
 */
#ifndef QUANT_KERNEL_H
#define QUANT_KERNEL_H

#include "kernel_operator.h"

using namespace AscendC;

template <typename outType>
class QuantKernel {
public:
    __aicore__ inline QuantKernel() {}
    __aicore__ inline void Init(int subBlockM, int blockN, int nTiles, AscendC::TPipe *pipe);
    __aicore__ inline void Process(GlobalTensor<float> &wsBase,
                                   GlobalTensor<outType> &yBase,
                                   int wsRowStride, int yRowStride);

private:
    __aicore__ inline void LoadTileFromWS(GlobalTensor<float> &wsBase, int tileIdx);
    __aicore__ inline void StoreTileToY(GlobalTensor<outType> &yBase, int tileIdx);

    int subBlockM_;
    int blockN_;
    int nTiles_;
    int tileSize_;

    // Data flow queues (depth=0, use reference-form API)
    AscendC::LocalTensor<float> inLocal_;
    AscendC::LocalTensor<outType> outLocal_;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueue_;

    // Intermediate buffers
    AscendC::TBuf<AscendC::TPosition::VECCALC> absBuf_;       // |x| for absmax
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowAbsmaxBuf_; // per-row running max
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowMaxTileBuf_;// per-row max of current tile
    AscendC::TBuf<AscendC::TPosition::VECCALC> rowScaleBuf_;  // per-row scale = absmax/127
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16Buf_;      // fp16 intermediate
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf_;    // work buffer for ReduceMax

    int wsRowStride_;
    int yRowStride_;
};

template <typename outType>
__aicore__ inline void QuantKernel<outType>::Init(int subBlockM, int blockN, int nTiles,
                                                   AscendC::TPipe *pipe)
{
    subBlockM_ = subBlockM;
    blockN_ = blockN;
    nTiles_ = nTiles;
    tileSize_ = subBlockM * blockN;

    pipe->InitBuffer(inQueue_, 1, tileSize_ * sizeof(float));
    pipe->InitBuffer(outQueue_, 1, tileSize_ * sizeof(outType));
    pipe->InitBuffer(absBuf_, tileSize_ * sizeof(float));
    // Align to 32 bytes: subBlockM floats, minimum 8 floats (32 bytes)
    int scalarBufSize = ((subBlockM * sizeof(float) + 31) / 32) * 32;
    pipe->InitBuffer(rowAbsmaxBuf_, scalarBufSize);
    pipe->InitBuffer(rowMaxTileBuf_, scalarBufSize);
    pipe->InitBuffer(rowScaleBuf_, scalarBufSize);
    pipe->InitBuffer(fp16Buf_, tileSize_ * sizeof(half));
    // ReduceMax work buffer: needs at least blockN * sizeof(float) bytes
    pipe->InitBuffer(reduceBuf_, blockN * sizeof(float));
}

template <typename outType>
__aicore__ inline void QuantKernel<outType>::LoadTileFromWS(
    GlobalTensor<float> &wsBase, int tileIdx)
{
    inQueue_.AllocTensor<float>(inLocal_);
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = (uint16_t)subBlockM_;
    copyParams.blockLen = (uint16_t)(blockN_ * sizeof(float) / AscendC::DEFAULT_C0_SIZE);
    copyParams.srcStride = (uint16_t)((wsRowStride_ - blockN_) * sizeof(float) / AscendC::DEFAULT_C0_SIZE);
    copyParams.dstStride = 0;
    AscendC::DataCopy(inLocal_, wsBase[tileIdx * blockN_], copyParams);
    inQueue_.EnQue(inLocal_);
}

template <typename outType>
__aicore__ inline void QuantKernel<outType>::StoreTileToY(
    GlobalTensor<outType> &yBase, int tileIdx)
{
    outQueue_.DeQue<outType>(outLocal_);
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = (uint16_t)subBlockM_;
    copyParams.blockLen = (uint16_t)(blockN_ * sizeof(outType) / AscendC::DEFAULT_C0_SIZE);
    copyParams.srcStride = 0;
    copyParams.dstStride = (uint16_t)((yRowStride_ - blockN_) * sizeof(outType) / AscendC::DEFAULT_C0_SIZE);
    AscendC::DataCopy(yBase[tileIdx * blockN_], outLocal_, copyParams);
    outQueue_.FreeTensor(outLocal_);
}

template <typename outType>
__aicore__ inline void QuantKernel<outType>::Process(
    GlobalTensor<float> &wsBase,
    GlobalTensor<outType> &yBase,
    int wsRowStride, int yRowStride)
{
    wsRowStride_ = wsRowStride;
    yRowStride_ = yRowStride;

    AscendC::LocalTensor<float> absLocal = absBuf_.Get<float>();
    AscendC::LocalTensor<float> rowAbsmax = rowAbsmaxBuf_.Get<float>();
    AscendC::LocalTensor<float> rowMaxTile = rowMaxTileBuf_.Get<float>();
    AscendC::LocalTensor<float> rowScale = rowScaleBuf_.Get<float>();
    AscendC::LocalTensor<float> reduceWork = reduceBuf_.Get<float>();

    // =========== Pass 1: compute per-row absmax across all n_tiles ===========
    AscendC::Duplicate(rowAbsmax, 0.0f, subBlockM_);
    PipeBarrier<PIPE_V>();

    for (int by = 0; by < nTiles_; by++) {
        LoadTileFromWS(wsBase, by);
        inQueue_.DeQue<float>(inLocal_);

        // Abs
        AscendC::Abs(absLocal, inLocal_, tileSize_);
        PipeBarrier<PIPE_V>();

        // Per-row ReduceMax: find max of each row (blockN_ elements)
        for (int i = 0; i < subBlockM_; i++) {
            AscendC::ReduceMax(rowMaxTile, absLocal[i * blockN_], reduceWork, blockN_);
            PipeBarrier<PIPE_V>();
            // Update running max
            float tileMax = rowMaxTile.GetValue(0);
            float curMax = rowAbsmax.GetValue(i);
            if (tileMax > curMax) {
                rowAbsmax.SetValue(i, tileMax);
            }
        }

        inQueue_.FreeTensor(inLocal_);
    }

    // row_scale = row_absmax * (1/127)
    float inv127 = 1.0f / 127.0f;
    AscendC::Muls(rowScale, rowAbsmax, inv127, subBlockM_);
    PipeBarrier<PIPE_V>();

    // =========== Pass 2: quantize each tile ===========
    AscendC::LocalTensor<half> fp16Local = fp16Buf_.Get<half>();

    for (int by = 0; by < nTiles_; by++) {
        LoadTileFromWS(wsBase, by);
        inQueue_.DeQue<float>(inLocal_);

        // Per-row division: multiply by 1/scale
        for (int i = 0; i < subBlockM_; i++) {
            float scaleVal = rowScale.GetValue(i);
            if (scaleVal > 0.0f) {
                float invScale = 1.0f / scaleVal;
                AscendC::Muls(inLocal_[i * blockN_], inLocal_[i * blockN_], invScale, blockN_);
            }
        }
        PipeBarrier<PIPE_V>();

        // Cast fp32 -> fp16
        AscendC::Cast(fp16Local, inLocal_, AscendC::RoundMode::CAST_NONE, tileSize_);
        PipeBarrier<PIPE_V>();

        // Cast fp16 -> int8
        outQueue_.AllocTensor<outType>(outLocal_);
        AscendC::Cast(outLocal_, fp16Local, AscendC::RoundMode::CAST_NONE, tileSize_);

        inQueue_.FreeTensor(inLocal_);
        outQueue_.EnQue(outLocal_);

        // Write to output
        StoreTileToY(yBase, by);
    }
}

#endif // QUANT_KERNEL_H
