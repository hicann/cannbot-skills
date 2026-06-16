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
 * @file scale_kernel.h
 *
 * Vector kernel for int8 matmul scale operator.
 * Replaces LeakyReLU with: Cast(int32→fp32) → per-row Mul(scale) → Cast(fp32→fp16)
 *
 * Translated from TileLang design:
 *   T.copy(workspace[...], acc_i32_ub)           → CopyIn: DataCopy GM→VECIN
 *   T.copy(Scale[...], scale_row_ub)              → DataCopy GM→VECCALC (scaleBuf)
 *   T.tile.cast(acc_fp32_ub, acc_i32_ub)          → Cast(int32→float32) in castBuf
 *   for i: T.tile.mul(row, row, scale)            → per-row Mul in castBuf
 *   T.tile.cast(out_ub, acc_fp32_ub)              → Cast(float32→float16) to VECOUT
 *   T.copy(out_ub, C[...])                        → CopyOut: DataCopy VECOUT→GM
 */
#ifndef SCALE_KERNEL_H
#define SCALE_KERNEL_H

#include "kernel_operator.h"

using namespace AscendC;

template <typename accType, typename outType>
class ScaleKernel {
public:
    __aicore__ inline ScaleKernel() {}
    __aicore__ inline void Init(int subBlockM, int blockN, AscendC::TPipe *pipe);
    __aicore__ inline void Process(GlobalTensor<accType> &slotGM,
                                   GlobalTensor<float> &scaleGM,
                                   GlobalTensor<outType> &cGlobal, int rowStride);
private:
    int subBlockM_;
    int blockN_;
    int tileSize_;

    AscendC::LocalTensor<accType> inLocal_;
    AscendC::LocalTensor<outType> outLocal_;

    AscendC::TQue<AscendC::TPosition::VECIN, 0> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> outQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castBuf_;   // float32 intermediate
    AscendC::TBuf<AscendC::TPosition::VECCALC> scaleBuf_;  // float32 scale vector
};

template <typename accType, typename outType>
__aicore__ inline void ScaleKernel<accType, outType>::Init(int subBlockM, int blockN, AscendC::TPipe *pipe)
{
    subBlockM_ = subBlockM;
    blockN_ = blockN;
    tileSize_ = subBlockM * blockN;

    pipe->InitBuffer(inQueue_, 1, tileSize_ * sizeof(accType));
    pipe->InitBuffer(outQueue_, 1, tileSize_ * sizeof(outType));
    pipe->InitBuffer(castBuf_, tileSize_ * sizeof(float));
    pipe->InitBuffer(scaleBuf_, blockN * sizeof(float));
}

template <typename accType, typename outType>
__aicore__ inline void ScaleKernel<accType, outType>::Process(
    GlobalTensor<accType> &slotGM,
    GlobalTensor<float> &scaleGM,
    GlobalTensor<outType> &cGlobal, int rowStride)
{
    // CopyIn: workspace slot → inQueue
    inQueue_.AllocTensor<accType>(inLocal_);
    AscendC::DataCopy(inLocal_, slotGM, tileSize_);
    inQueue_.EnQue(inLocal_);

    // Load scale vector → scaleBuf (TBuf, manual sync)
    AscendC::LocalTensor<float> scaleLocal = scaleBuf_.Get<float>();
    AscendC::DataCopy(scaleLocal, scaleGM, blockN_);
    PipeBarrier<PIPE_MTE2>();

    // Compute: Cast(int32→fp32) → per-row Mul(scale) → Cast(fp32→fp16)
    inQueue_.DeQue<accType>(inLocal_);
    AscendC::LocalTensor<float> fp32Local = castBuf_.Get<float>();

    // int32 → float32
    AscendC::Cast(fp32Local, inLocal_, AscendC::RoundMode::CAST_NONE, tileSize_);
    PipeBarrier<PIPE_V>();

    // Per-row multiply by scale vector
    for (int i = 0; i < subBlockM_; i++) {
        AscendC::Mul(fp32Local[i * blockN_], fp32Local[i * blockN_], scaleLocal, blockN_);
    }
    PipeBarrier<PIPE_V>();

    // float32 → float16
    outQueue_.AllocTensor<outType>(outLocal_);
    AscendC::Cast(outLocal_, fp32Local, AscendC::RoundMode::CAST_NONE, tileSize_);

    inQueue_.FreeTensor(inLocal_);
    outQueue_.EnQue(outLocal_);

    // CopyOut
    outQueue_.DeQue<outType>(outLocal_);
    AscendC::DataCopyParams copyParam = {
        (uint16_t)subBlockM_,
        (uint16_t)(blockN_ * sizeof(outType) / AscendC::DEFAULT_C0_SIZE),
        0,
        (uint16_t)((rowStride - blockN_) * sizeof(outType) / AscendC::DEFAULT_C0_SIZE)
    };
    AscendC::DataCopy(cGlobal, outLocal_, copyParam);
    outQueue_.FreeTensor(outLocal_);
}

#endif // SCALE_KERNEL_H
