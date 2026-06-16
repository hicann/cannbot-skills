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

class AvgPool3DSplitWKernel : public AvgPool3DKernelBase<AvgPool3DSplitWKernel> {
public:
    __aicore__ inline void Process()
    {
        AvgPool3DProcessLoop([this](int outRow) { ProcessOneRow(outRow); },
                              tiling_.launchBlocks, tiling_.N, tiling_.outSpatial);
    }

protected:
    friend class AvgPool3DKernelBase<AvgPool3DSplitWKernel>;

    __aicore__ inline void ProcessOneRowChunk(int nIdx, int od, int oh, int ow, int outRow, int cBase, int cLen)
    {
        InitAccumulator(cLen);

        int wTile = tiling_.kW;
        if (tiling_.splitWTileKw > 0 && tiling_.splitWTileKw < wTile) {
            wTile = tiling_.splitWTileKw;
        }
        const int kwGroups = static_cast<int>(CeilDivU32(static_cast<uint32_t>(tiling_.kW), static_cast<uint32_t>(wTile)));

        PoolKdKhLoop(nIdx, od, oh, ow, cBase, cLen, wTile, kwGroups);

        AvgPool3DStoreOutput(outQueue_, outLocal_, accLocal_, yGM_,
                              outRow, cBase, cLen, od, oh, ow, tiling_);
    }
};
