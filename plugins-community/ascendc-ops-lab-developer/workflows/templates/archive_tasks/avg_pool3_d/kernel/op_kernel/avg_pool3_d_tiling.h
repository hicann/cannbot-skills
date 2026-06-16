/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AVG_POOL3_D_TILING_H
#define AVG_POOL3_D_TILING_H

#include <cstdint>

constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr int32_t DEFAULT_VEC_NUM = 2;

struct AvgPool3DKernelTiling {
    int32_t N;
    int32_t C;
    int32_t D;
    int32_t H;
    int32_t W;

    int32_t OD;
    int32_t OH;
    int32_t OW;

    int32_t kD;
    int32_t kH;
    int32_t kW;

    int32_t sD;
    int32_t sH;
    int32_t sW;

    int32_t pD;
    int32_t pH;
    int32_t pW;

    int32_t countIncludePad;
    int32_t divisorOverride;

    int32_t splitMode;
    int32_t blockC;
    int32_t splitWTileKw;
    int32_t multiWWindow;

    int32_t blockM;
    int32_t subBlockM;
    int32_t mNum;
    int32_t outSpatial;
    int32_t inSpatial;
    int32_t hw;

    int32_t cNum;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t launchBlocks;

    int32_t vectorLen;
    int32_t reserved0;
    int32_t reserved1;
};

#endif
