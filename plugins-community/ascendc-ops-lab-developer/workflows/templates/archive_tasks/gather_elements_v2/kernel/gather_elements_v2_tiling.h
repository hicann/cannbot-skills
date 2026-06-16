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

#include <cstdint>

constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr int32_t DEFAULT_VEC_NUM = 2;

constexpr int32_t GATHER_MODE_LAST_DIM = 0;
constexpr int32_t GATHER_MODE_TRANSPOSE = 1;
constexpr int32_t GATHER_MODE_SCALAR = 2;

struct GatherElementsV2KernelTiling {
    int32_t M;
    int32_t XRows;
    int32_t XG;
    int32_t IG;
    int32_t XStride;
    int32_t YStride;
    int32_t blockM;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
    int32_t subBlockM;
    int32_t useRowMap;
    int32_t mode;
};
