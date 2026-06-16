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
 * @file leakyrelu.h
 *
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */

#ifndef LEAK_RELU_H
#define LEAK_RELU_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"

using namespace AscendC;

template <typename inType, typename outType>
class LeakyKernel {
public:
    __aicore__ inline LeakyKernel() {}
    __aicore__ inline void Init(int tileM, int tileN, AscendC::TPipe *pipe);
    __aicore__ inline void Process(GlobalTensor<inType> &blockGM, GlobalTensor<outType> &cGlobal, int rowStride);
private:
    int tileM;
    int tileN;
    int tileSize;
    AscendC::LocalTensor<inType> reluInLocal;
    AscendC::LocalTensor<outType> reluOutLocal;
    AscendC::LocalTensor<outType> reluCastLocal;
    AscendC::TQue<AscendC::TPosition::VECIN, 0> reluInQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> reluOutQueue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> castBuf_;
};

/**
  * @brief  Init LeakyKernel buffer and tile params.
  * @param  tileM: Tile row size.
  * @param  tileN: Tile column size.
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
template <typename inType, typename outType>
__aicore__ inline void LeakyKernel<inType, outType>::Init(int tileM, int tileN, AscendC::TPipe *pipe)
{
    this->tileM = tileM;
    this->tileN = tileN;
    this->tileSize = tileM * tileN;
    pipe->InitBuffer(reluInQueue_, 1, tileSize * sizeof(inType)); // Init output buffer.
    pipe->InitBuffer(reluOutQueue_, 1, tileSize * sizeof(outType)); // Init output buffer.
    if constexpr (!std::is_same_v<inType, outType>) {
        pipe->InitBuffer(castBuf_, tileSize * sizeof(outType)); // Init cast buffer.
    }
}

/**
  * @brief  Read input from workspace GM, compute leakyRelu, write result to output GM.
  * @param  blockGM: Global tensor to read input data from (workspace slot).
  * @param  cGlobal: Global tensor to write output data to (with offset already applied).
  * @param  rowStride: Row stride of the destination matrix in elements.
  * @retval None
  */
template <typename inType, typename outType>
__aicore__ inline void LeakyKernel<inType, outType>::Process(GlobalTensor<inType> &blockGM,
                                                             GlobalTensor<outType> &cGlobal, int rowStride)
{
    // CopyIn
    reluInQueue_.AllocTensor<inType>(reluInLocal);
    DataCopy(reluInLocal, blockGM, tileSize);
    reluInQueue_.EnQue(reluInLocal);

    // Compute
    reluOutQueue_.AllocTensor<outType>(reluOutLocal);
    reluInQueue_.DeQue<inType>(reluInLocal);
    if constexpr (!std::is_same_v<inType, outType>) {
        Cast(reluCastLocal, reluInLocal, RoundMode::CAST_ROUND, tileSize);
        PipeBarrier<PIPE_V>();
    } else {
        reluCastLocal = reluInLocal.template ReinterpretCast<outType>();
    }
    LeakyRelu(reluOutLocal, reluCastLocal, (outType)0.001, tileSize);
    reluInQueue_.FreeTensor(reluInLocal);
    reluOutQueue_.EnQue(reluOutLocal);

    // CopyOut
    reluOutQueue_.DeQue<outType>(reluOutLocal);
    AscendC::DataCopyParams copyParam = {
        (uint16_t)tileM,
        (uint16_t)(tileN * sizeof(outType) / AscendC::DEFAULT_C0_SIZE),
        0,
        (uint16_t)((rowStride - tileN) * sizeof(outType) / AscendC::DEFAULT_C0_SIZE)
    };
    DataCopy(cGlobal, reluOutLocal, copyParam);
    reluOutQueue_.FreeTensor(reluOutLocal);
}


#endif // #ifndef LEAK_RELU_H