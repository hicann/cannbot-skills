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
#include "matmul_leakyrelu_common.h"
#include "matmul_leakyrelu_tiling.h"
#include "leakyrelu.h"
#include "matmul_tile.h"

#define CUBE_NOTIFY_VECTOR_ID 0x8
using namespace AscendC;

template <typename aType, typename bType, typename accType, typename outType>
class MatmulLeakyKernel {
public:
    __aicore__ inline MatmulLeakyKernel() {}
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, GM_ADDR tGM);
    __aicore__ inline void Process();

private:
    GlobalTensor<aType> aGM_;
    GlobalTensor<bType> bGM_;
    GlobalTensor<outType> cGM_;
    GlobalTensor<accType> wsGM_;
    MatmulLeakyReluTiling tiling;
    LeakyKernel<accType, outType> leakyKernel_;
    TPipe aivPipe_;
    BlockScheduler sched_;
    int groupIdx_;
    int subTileM_;
};

/**
  * @brief  Set matmulLeaky input and output gm addr of current core.
  * @param  a: A matrix gm addr.
  * @param  b: B matrix gm addr.
  * @param  c: C matrix gm addr.
  * @param  ws: Temporary gm space addr required by matmul calc.
  * @param  tGM: matmul tiling data.
  * @retval None
  */
template <typename aType, typename bType, typename accType, typename outType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, accType, outType>::Init(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR ws, GM_ADDR tGM)
{
    CopyTiling(&tiling, tGM);

    aGM_.SetGlobalBuffer((__gm__ aType *)a, tiling.M * tiling.K);
    bGM_.SetGlobalBuffer((__gm__ bType *)b, tiling.K * tiling.N);
    cGM_.SetGlobalBuffer((__gm__ outType *)c, tiling.M * tiling.N);
    if constexpr (!std::is_same_v<accType, outType>) {
        wsGM_.SetGlobalBuffer((__gm__ accType *)ws, tiling.M * tiling.N);
    }

    groupIdx_ = GetBlockIdx() / GetSubBlockNum();
    sched_.Init(tiling.M, tiling.N, tiling.baseM, tiling.baseN, GetBlockNum(), groupIdx_);

    if ASCEND_IS_AIV {
        subTileM_ = tiling.baseM / GetSubBlockNum();
        leakyKernel_.Init(subTileM_, tiling.baseN, &aivPipe_);
    }
}

/**
  * @brief  Main process of matmul calculation.
  *         AIC: GM→L1→L0A/L0B, Mmad, Fixpipe L0C→GM.
  *         AIV: Read workspace, Cast, LeakyRelu, Write output.
  * @retval None
  */
template <typename aType, typename bType, typename accType, typename outType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, accType, outType>::Process()
{
    int mi, ni;

    if ASCEND_IS_AIC {
        uint32_t a1s = (uint32_t)(tiling.baseM * tiling.baseK);
        uint32_t b1s = (uint32_t)(tiling.baseK * tiling.baseN);
        uint32_t mns = (uint32_t)(tiling.baseM * tiling.baseN);

        LocalMemAllocator<Hardware::L1> L1;
        LocalMemAllocator<Hardware::L0A> L0A;
        LocalMemAllocator<Hardware::L0B> L0B;
        LocalMemAllocator<Hardware::L0C> L0C;

        LocalTensor<aType> a1 = L1.Alloc<TPosition::A1, aType>(a1s);
        LocalTensor<bType> b1 = L1.Alloc<TPosition::B1, bType>(b1s);
        LocalTensor<aType> a2 = L0A.Alloc<TPosition::A2, aType>(a1s);
        LocalTensor<bType> b2 = L0B.Alloc<TPosition::B2, bType>(b1s);
        LocalTensor<accType> cL = L0C.Alloc<TPosition::CO1, accType>(mns);

        uint32_t kT = tiling.K / tiling.baseK;

        while (sched_.HasNext()) {
            sched_.Next(mi, ni);

            auto aB = aGM_[mi * tiling.baseM * tiling.K];
            auto bB = bGM_[ni * tiling.baseN];

            for (uint32_t kt = 0; kt < kT; kt++) {
                LoadNdGmToNzL1(a1, aB[kt * tiling.baseK], tiling.baseM, tiling.baseK, tiling.K);
                LoadNdGmToNzL1(b1, bB[kt * tiling.baseK * tiling.N], tiling.baseK, tiling.baseN, tiling.N);
                PipeBarrier<PIPE_ALL>();

                LoadNzL1ToZzL0A(a2, a1, tiling.baseM, tiling.baseK, tiling.baseM);
                LoadNzL1ToZnL0B(b2, b1, tiling.baseK, tiling.baseN, tiling.baseK);
                PipeBarrier<PIPE_ALL>();

                MmadParams mp;
                mp.m = tiling.baseM;
                mp.n = tiling.baseN;
                mp.k = tiling.baseK;
                mp.cmatrixInitVal = (kt == 0);
                Mmad(cL, a2, b2, mp);
                PipeBarrier<PIPE_ALL>();
            }
            PipeBarrier<PIPE_ALL>();

            if constexpr (std::is_same_v<accType, outType>) {
                auto cB = cGM_[mi * tiling.baseM * tiling.N + ni * tiling.baseN];
                FixpipeNzL0cToNdGmStride(cB, cL, tiling.baseM, tiling.baseN, tiling.N);
            } else {
                auto wB = wsGM_[mi * tiling.baseM * tiling.N + ni * tiling.baseN];
                FixpipeNzL0cToNdGmStride(wB, cL, tiling.baseM, tiling.baseN, tiling.N);
            }
            PipeBarrier<PIPE_ALL>();
        }

        CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_NOTIFY_VECTOR_ID);
    }

    if ASCEND_IS_AIV {
        CrossCoreWaitFlag<0x2>(CUBE_NOTIFY_VECTOR_ID);

        while (sched_.HasNext()) {
            sched_.Next(mi, ni);

            int ro = GetSubBlockIdx() * subTileM_;
            auto cB = cGM_[(mi * tiling.baseM + ro) * tiling.N + ni * tiling.baseN];

            if constexpr (std::is_same_v<accType, outType>) {
                leakyKernel_.Process(cB, cB, tiling.N);
            } else {
                auto wB = wsGM_[(mi * tiling.baseM + ro) * tiling.N + ni * tiling.baseN];
                leakyKernel_.Process(wB, cB, tiling.N);
            }
        }
    }
}
