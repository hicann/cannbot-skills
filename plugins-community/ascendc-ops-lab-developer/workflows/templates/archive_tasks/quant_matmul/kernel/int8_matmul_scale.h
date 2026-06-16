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
 * @file int8_matmul_scale.h
 *
 * Top-level kernel orchestrator for Int8 MatMul + Scale.
 * AIC (Cube): int8 matmul → workspace (int32)
 * AIV (Vector): Cast(int32→fp32) → Mul(scale) → Cast(fp32→fp16) → output
 */
#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "workspace_queue.h"
#include "kernel_common.h"
#include "int8_matmul_scale_tiling.h"
#include "scale_kernel.h"
#include "matmul.h"

#define CUBE_NOTIFY_VECTOR_ID 0x8
#define VECTOR_NOTIFY_CUBE_ID 0x9

using namespace AscendC;

class Int8MatmulScaleKernel {
public:
    __aicore__ inline Int8MatmulScaleKernel() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR scale, GM_ADDR c,
                                GM_ADDR workspace, GM_ADDR tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

    MatmulKernel<int8_t, int8_t, int32_t> mm_;
    AscendC::GlobalTensor<int8_t> aGM_;
    AscendC::GlobalTensor<int8_t> bGM_;
    AscendC::GlobalTensor<float> scaleGM_;
    AscendC::GlobalTensor<half> cGM_;
    WorkspaceQueue<int32_t, WORKSPACE_DEPTH> wsQueue_;
    Int8MatmulScaleTiling tiling;
    ScaleKernel<int32_t, half> scaleKernel_;
    BlockScheduler sched_;
    int subTileM_;
};

__aicore__ inline void Int8MatmulScaleKernel::Init(GM_ADDR a, GM_ADDR b, GM_ADDR scale, GM_ADDR c,
                                                    GM_ADDR workspace, GM_ADDR tilingGM,
                                                    AscendC::TPipe *pipe)
{
    CopyTiling(&this->tiling, tilingGM);

    aGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(a), tiling.M * tiling.K);
    bGM_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(b), tiling.K * tiling.N);
    scaleGM_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(scale), tiling.N);
    cGM_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c), tiling.M * tiling.N);

    int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    sched_.Init(tiling.M, tiling.N, tiling.baseM, tiling.baseN,
                AscendC::GetBlockNum(), coreIdx);

    uint32_t wsOffset = coreIdx * WORKSPACE_DEPTH * tiling.baseM * tiling.baseN;
    wsQueue_.Init(workspace + wsOffset * sizeof(int32_t),
                  tiling.baseM * tiling.baseN,
                  CUBE_NOTIFY_VECTOR_ID, VECTOR_NOTIFY_CUBE_ID);

    if ASCEND_IS_AIC {
        mm_.Init(tiling.K, tiling.K, tiling.N, *pipe);
    }

    if ASCEND_IS_AIV {
        subTileM_ = tiling.baseM / AscendC::GetSubBlockNum();
        scaleKernel_.Init(subTileM_, tiling.baseN, pipe);
        wsQueue_.InitFreeSlots();
    }
}

__aicore__ inline void Int8MatmulScaleKernel::Process()
{
    int mIdx = 0, nIdx = 0;
    while (sched_.HasNext()) {
        sched_.Next(mIdx, nIdx);
        if ASCEND_IS_AIC {
            auto slot = wsQueue_.ProducerAcquire();
            auto aBlock = aGM_[mIdx * tiling.baseM * tiling.K];
            auto bBlock = bGM_[nIdx * tiling.baseN];
            mm_.ComputeBlock(aBlock, bBlock, slot);
            wsQueue_.ProducerRelease();
        }
        if ASCEND_IS_AIV) {
            auto slot = wsQueue_.ConsumerAcquire();
            int rowOffset = AscendC::GetSubBlockIdx() * subTileM_;
            auto subSlot = slot[rowOffset * tiling.baseN];
            auto scaleBlock = scaleGM_[nIdx * tiling.baseN];
            auto cBlock = cGM_[(mIdx * tiling.baseM + rowOffset) * tiling.N + nIdx * tiling.baseN];
            scaleKernel_.Process(subSlot, scaleBlock, cBlock, tiling.N);
            wsQueue_.ConsumerRelease();
        }
    }
}
