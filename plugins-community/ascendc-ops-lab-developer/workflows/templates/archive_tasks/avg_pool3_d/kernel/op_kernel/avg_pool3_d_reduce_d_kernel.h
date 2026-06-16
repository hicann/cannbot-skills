/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "avg_pool3_d_kernel_base.h"

class AvgPool3DReduceDKernel : public AvgPool3DKernelBase<AvgPool3DReduceDKernel> {
protected:
    friend class AvgPool3DKernelBase<AvgPool3DReduceDKernel>;

    __aicore__ inline void ProcessOneRowChunk(int nIdx, int od, int oh, int ow, int outRow, int cBase, int cLen)
    {
        InitAccumulator(cLen);

        const int ihVal = oh * tiling_.sH - tiling_.pH;
        const int iwVal = ow * tiling_.sW - tiling_.pW;
        if (ihVal >= 0 && ihVal < tiling_.H && iwVal >= 0 && iwVal < tiling_.W) {
            for (int kd = 0; kd < tiling_.kD; ++kd) {
                const int idVal = od * tiling_.sD - tiling_.pD + kd;
                if (idVal < 0 || idVal >= tiling_.D) {
                    continue;
                }
                const int inRow = nIdx * tiling_.inSpatial + idVal * tiling_.hw + ihVal * tiling_.W + iwVal;
                AccumulateOne(inRow, cBase, cLen);
            }
        }

        AvgPool3DStoreOutput(outQueue_, outLocal_, accLocal_, yGM_,
                              outRow, cBase, cLen, od, oh, ow, tiling_);
    }
};
