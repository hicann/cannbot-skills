/*
 * Copyright (c) 2026 Huawei Technologies, Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "avg_pool3_d_kernel_base.h"

class AvgPool3DSplitCKernel : public AvgPool3DKernelBase<AvgPool3DSplitCKernel> {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGM, AscendC::TPipe *pipe)
    {
        CopyTiling(&tiling_, tilingGM);
        InitCore(x, y, pipe, tiling_.blockC);
    }

    __aicore__ inline void Process()
    {
        AvgPool3DProcessLoop([this](int outRow) { ProcessOneRow(outRow); },
                              tiling_.launchBlocks, tiling_.N, tiling_.outSpatial);
    }

    __aicore__ inline void ProcessOneRow(int outRow)
    {
        const int nIdx = outRow / tiling_.outSpatial;
        const int outRem = outRow - nIdx * tiling_.outSpatial;
        const int od = outRem / (tiling_.OH * tiling_.OW);
        const int odRem = outRem - od * (tiling_.OH * tiling_.OW);
        const int oh = odRem / tiling_.OW;
        const int ow = odRem - oh * tiling_.OW;

        for (int cBase = 0; cBase < tiling_.C; cBase += tiling_.blockC) {
            ProcessOneRowChunk(nIdx, od, oh, ow, outRow, cBase, tiling_.blockC);
        }
    }

private:
    friend class AvgPool3DKernelBase<AvgPool3DSplitCKernel>;
};
