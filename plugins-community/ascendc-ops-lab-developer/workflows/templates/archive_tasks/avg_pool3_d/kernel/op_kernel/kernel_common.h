/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AVG_POOL3_D_KERNEL_COMMON_H
#define AVG_POOL3_D_KERNEL_COMMON_H

#include <cstddef>
#include <cstdint>

#include "kernel_operator.h"
#include "avg_pool3_d_tiling.h"

__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    if (b == 0U) { return 0U; }
    return (a + b - 1U) / b;
}

template <typename T>
__aicore__ inline void CopyTiling(T *tiling, GM_ADDR tilingGM)
{
    int32_t *dst = reinterpret_cast<int32_t *>(tiling);
    auto *src = reinterpret_cast<__gm__ int32_t *>(tilingGM);
    for (size_t i = 0; i < sizeof(T) / sizeof(int32_t); ++i) {
        dst[i] = src[i];
    }
}

// Shared across all AvgPool3D kernel variants (split_c, split_w, reduce_d,
// generic, multi_w). Computes the effective pooling divisor for the given
// output position, accounting for padding and countIncludePad/divisorOverride.
__aicore__ inline int ComputeDivisorHelper(int od, int oh, int ow,
                                            const AvgPool3DKernelTiling &tiling)
{
    if (tiling.divisorOverride > 0) {
        return tiling.divisorOverride;
    }
    if (tiling.countIncludePad > 0) {
        return tiling.kD * tiling.kH * tiling.kW;
    }

    const int dStart = od * tiling.sD - tiling.pD;
    const int hStart = oh * tiling.sH - tiling.pH;
    const int wStart = ow * tiling.sW - tiling.pW;

    const int dBegin = dStart < 0 ? 0 : dStart;
    const int hBegin = hStart < 0 ? 0 : hStart;
    const int wBegin = wStart < 0 ? 0 : wStart;

    int dEnd = dStart + tiling.kD;
    int hEnd = hStart + tiling.kH;
    int wEnd = wStart + tiling.kW;

    if (dEnd > tiling.D) { dEnd = tiling.D; }
    if (hEnd > tiling.H) { hEnd = tiling.H; }
    if (wEnd > tiling.W) { wEnd = tiling.W; }

    const int validD = dEnd > dBegin ? (dEnd - dBegin) : 0;
    const int validH = hEnd > hBegin ? (hEnd - hBegin) : 0;
    const int validW = wEnd > wBegin ? (wEnd - wBegin) : 0;
    return validD * validH * validW;
}

// Shared worker-loop helper — replaces the identical Process() body in AvgPool3DSplitCKernel
// and AvgPool3DSplitWKernel (see "duplicate code" notes in those files).
template <typename F>
__aicore__ inline void AvgPool3DProcessLoop(F &&processRowFn, int launchBlocks, int N, int outSpatial)
{
    if ASCEND_IS_AIV {
        const int workerId = static_cast<int>(AscendC::GetBlockIdx());
        int workerCount = launchBlocks;
        if (workerCount <= 0) {
            workerCount = 1;
        }
        const int totalRows = N * outSpatial;
        for (int outRow = workerId; outRow < totalRows; outRow += workerCount) {
            processRowFn(outRow);
        }
    }
}

// Shared output-store helper — replaces the identical accumulator-to-GM output section in
// AvgPool3DSplitCKernel::ProcessOneRowChunk and AvgPool3DSplitWKernel::ProcessOneRowChunk.
__aicore__ inline void AvgPool3DStoreOutput(
    AscendC::TQue<AscendC::TPosition::VECOUT, 0> &outQueue,
    AscendC::LocalTensor<float> &outLocal,
    AscendC::LocalTensor<float> &accLocal,
    const AscendC::GlobalTensor<float> &yGM,
    int outRow, int cBase, int cLen,
    int od, int oh, int ow,
    const AvgPool3DKernelTiling &tiling)
{
    outQueue.AllocTensor<float>(outLocal);
    const int divisor = ComputeDivisorHelper(od, oh, ow, tiling);
    if (divisor <= 0) {
        AscendC::Duplicate(outLocal, 0.0f, cLen);
    } else {
        const float invDiv = 1.0f / static_cast<float>(divisor);
        AscendC::Muls(outLocal, accLocal, invDiv, cLen);
    }
    outQueue.EnQue(outLocal);
    outQueue.DeQue<float>(outLocal);
    AscendC::DataCopy(yGM[static_cast<uint64_t>(outRow) * tiling.C + cBase], outLocal, cLen);
    outQueue.FreeTensor(outLocal);
}

#endif
