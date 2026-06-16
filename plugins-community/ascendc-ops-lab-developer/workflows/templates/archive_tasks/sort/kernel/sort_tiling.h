/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_TILING_H
#define SORT_TILING_H

#include <cstdint>

constexpr int32_t SORT_TILING_MODE_FULLLOAD = 0;
constexpr int32_t SORT_TILING_MODE_SINGLECORE = 1;
constexpr int32_t SORT_TILING_MODE_MULTICORE = 2;

constexpr int32_t SORT_MAX_CORES = 20;

struct SortKernelTiling {
    int32_t tilingMode;
    int32_t totalLength;
    int32_t sortNum;
    int32_t coreNum;

    int32_t perCoreElements;
    int32_t lastCoreElements;
    int32_t perCoreLoops;
    int32_t lastCoreLoops;
    int32_t perCorePerLoopElements;
    int32_t lastCorePerLoopElements;
    int32_t perCoreLastLoopElements;
    int32_t lastCoreLastLoopElements;

    int32_t oneLoopMaxElements;

    int32_t needCoreNum;
};

#endif
