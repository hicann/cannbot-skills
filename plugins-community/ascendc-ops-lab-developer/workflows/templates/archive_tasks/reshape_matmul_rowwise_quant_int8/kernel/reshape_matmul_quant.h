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
 * @file reshape_matmul_quant.h
 *
 * Top-level kernel for reshape + matmul + row-wise dynamic int8 quantization.
 *
 * Architecture:
 *   - T.Kernel(m_num): each core handles one m-block, processes ALL n_tiles
 *   - AIC (Cube): loops over n_tiles, computes matmul with reshape addressing,
 *     writes ALL results to workspace, then signals ONCE via CrossCoreSetFlag
 *   - AIV (Vector): waits for signal, then two-pass quantization:
 *     Pass 1: scan all tiles for per-row absmax
 *     Pass 2: divide by scale, cast fp32→fp16→int8, write output
 *
 * Key difference from matmul_leakyrelu: bulk sync (single signal after ALL tiles)
 * instead of per-tile ring buffer.
 */
#pragma once

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "../../matmul_leakyrelu/kernel/kernel_common.h"
#include "reshape_matmul_quant_tiling.h"
#include "quant_kernel.h"
#include "matmul.h"

#define CUBE_NOTIFY_VECTOR_ID 0x8

using namespace AscendC;

template <typename aType, typename accType, typename outType>
class ReshapeMatmulQuantKernel {
public:
    __aicore__ inline ReshapeMatmulQuantKernel() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR h, GM_ADDR y, GM_ADDR workspace,
                                GM_ADDR tilingGM, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

private:
    MatmulKernel<aType, accType> mm_;
    QuantKernel<outType> quantKernel_;
    ReshapeMatmulQuantTiling tiling;
    AscendC::TPipe *pipe_;

    AscendC::GlobalTensor<aType> xGM_;
    AscendC::GlobalTensor<aType> hGM_;
    AscendC::GlobalTensor<outType> yGM_;
    AscendC::GlobalTensor<accType> wsGM_;

    int coreIdx_;
    int subTileM_;
};

template <typename aType, typename accType, typename outType>
__aicore__ inline void ReshapeMatmulQuantKernel<aType, accType, outType>::Init(
    GM_ADDR x, GM_ADDR h, GM_ADDR y, GM_ADDR workspace,
    GM_ADDR tilingGM, AscendC::TPipe *pipe)
{
    pipe_ = pipe;
    CopyTiling(&this->tiling, tilingGM);

    xGM_.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(x), tiling.M * tiling.N);
    hGM_.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(h), tiling.H_K * tiling.H_K);
    yGM_.SetGlobalBuffer(reinterpret_cast<__gm__ outType *>(y), tiling.M * tiling.N);
    wsGM_.SetGlobalBuffer(reinterpret_cast<__gm__ accType *>(workspace), tiling.M * tiling.N);

    // Each physical core handles one m-block
    // GetBlockIdx() returns physical block idx; divide by SubBlockNum for core idx
    coreIdx_ = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();

    if ASCEND_IS_AIC {
        // ldA = N (X is M×N), ldB = H_K (H is H_K×H_K)
        mm_.Init(tiling.N, tiling.H_K, *pipe);
    }

    if ASCEND_IS_AIV {
        subTileM_ = tiling.baseM / AscendC::GetSubBlockNum();
        quantKernel_.Init(subTileM_, tiling.baseN, tiling.nTiles, pipe);
    }
}

template <typename aType, typename accType, typename outType>
__aicore__ inline void ReshapeMatmulQuantKernel<aType, accType, outType>::Process()
{
    int bx = coreIdx_;

    if ASCEND_IS_AIC {
        // Cube: loop over all n_tiles, compute matmul with reshape addressing
        for (int by = 0; by < tiling.nTiles; by++) {
            int groupId = by / tiling.nTilesPerH;
            int colInGroup = by % tiling.nTilesPerH;

            // A block: X[bx*baseM, groupId*H_K] with K_total=H_K, ldA=N
            auto aBlock = xGM_[bx * tiling.baseM * tiling.N + groupId * tiling.H_K];
            // B block: H[0, colInGroup*baseN] with K_total=H_K, ldB=H_K
            auto bBlock = hGM_[colInGroup * tiling.baseN];
            // Workspace: ws[bx*baseM, by*baseN] with dstStride=N
            auto wsBlock = wsGM_[bx * tiling.baseM * tiling.N + by * tiling.baseN];

            mm_.ComputeBlock(aBlock, bBlock, wsBlock, tiling.H_K, tiling.N);
        }

        // Signal vector: all tiles done
        CrossCoreSetFlag<0x2, PIPE_FIX>(CUBE_NOTIFY_VECTOR_ID);
    }

    if ASCEND_IS_AIV {
        // Wait for cube to finish all tiles
        CrossCoreWaitFlag<0x2>(CUBE_NOTIFY_VECTOR_ID);

        int rowOffset = AscendC::GetSubBlockIdx() * subTileM_;

        // Workspace base for this sub-block: ws[bx*baseM + rowOffset, 0]
        auto wsBase = wsGM_[(bx * tiling.baseM + rowOffset) * tiling.N];
        // Output base: Y[bx*baseM + rowOffset, 0]
        auto yBase = yGM_[(bx * tiling.baseM + rowOffset) * tiling.N];

        quantKernel_.Process(wsBase, yBase, tiling.N, tiling.N);
    }
}
