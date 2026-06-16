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
 * @file matmul_leakyrelu.h
 *
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "workspace_queue.h"
#include "kernel_common.h"
#include "matmul_leakyrelu_tiling.h"
#include "leakyrelu.h"
#include "matmul.h"

#define CUBE_NOTIFY_VECTOR_ID 0x8
#define VECTOR_NOTIFY_CUBE_ID 0x9

using namespace AscendC;

template <typename aType, typename bType, typename accType, typename outType>
class MatmulLeakyKernel {
public:
    __aicore__ inline MatmulLeakyKernel() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace,
                                GM_ADDR tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

    MatmulKernel<aType, bType, accType> mm_;
    AscendC::GlobalTensor<aType> aGM_;
    AscendC::GlobalTensor<bType> bGM_;
    AscendC::GlobalTensor<outType> cGM_;
    WorkspaceQueue<accType, WORKSPACE_DEPTH> wsQueue_;
    MatmulLeakyReluTiling tiling;
    LeakyKernel<accType, outType> leakyKernel_;
    BlockScheduler sched_;
    int subTileM_;
};

/**
  * @brief  Set matmulLeaky input and output gm addr of current core.
  * @param  a: A matrix gm addr.
  * @param  b: B matrix gm addr.
  * @param  c: C matrix gm addr.
  * @param  workspace: Temporary gm space addr required by matmul calc.
  * @param  tiling: matmul tiling data.
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
template <typename aType, typename bType, typename accType, typename outType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, accType, outType>::Init(GM_ADDR a, GM_ADDR b,
                                                                    GM_ADDR c, GM_ADDR workspace,
                                                                    GM_ADDR tilingGM, AscendC::TPipe *pipe)
{
    CopyTiling(&this->tiling, tilingGM);

    aGM_.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.K);
    bGM_.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.K * tiling.N);
    cGM_.SetGlobalBuffer(reinterpret_cast<__gm__ outType *>(c), tiling.M * tiling.N);

    int coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
    sched_.Init(tiling.M, tiling.N, tiling.baseM, tiling.baseN,
                AscendC::GetBlockNum(), coreIdx);

    // Setup workspace ring buffer for cube->vector data transfer
    uint32_t wsOffset = coreIdx * WORKSPACE_DEPTH * tiling.baseM * tiling.baseN;
    wsQueue_.Init(workspace + wsOffset * sizeof(accType),
                  tiling.baseM * tiling.baseN,
                  CUBE_NOTIFY_VECTOR_ID, VECTOR_NOTIFY_CUBE_ID);

    if ASCEND_IS_AIC {
        mm_.Init(tiling.K, tiling.K, tiling.N,
                 tiling.baseM, tiling.baseN, tiling.baseK, tiling.l1Prefetch,
                 *pipe);
    }

    if ASCEND_IS_AIV {
        subTileM_ = tiling.baseM / AscendC::GetSubBlockNum();
        leakyKernel_.Init(subTileM_, tiling.baseN, pipe);
        wsQueue_.InitFreeSlots();
    }
}

/**
  * @brief  Main process of matmul calculation
  * @retval None
  */
template <typename aType, typename bType, typename accType, typename outType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, accType, outType>::Process()
{
    int mIdx, nIdx;
    while (sched_.HasNext()) {
        sched_.Next(mIdx, nIdx);
        if ASCEND_IS_AIC {
            auto slot = wsQueue_.ProducerAcquire();
            auto aBlock = aGM_[mIdx * tiling.baseM * tiling.K];
            auto bBlock = bGM_[nIdx * tiling.baseN];
            mm_.ComputeBlock(aBlock, bBlock, slot);
            wsQueue_.ProducerRelease();
        }
        if ASCEND_IS_AIV {
            auto slot = wsQueue_.ConsumerAcquire();
            int rowOffset = AscendC::GetSubBlockIdx() * subTileM_;
            auto subSlot = slot[rowOffset * tiling.baseN];
            auto cBlock = cGM_[(mIdx * tiling.baseM + rowOffset) * tiling.N + nIdx * tiling.baseN];
            leakyKernel_.Process(subSlot, cBlock, tiling.N);
            wsQueue_.ConsumerRelease();
        }
    }
}
