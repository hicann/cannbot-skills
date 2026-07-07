/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CURRENT_TASK_CONCAT_DIM0_TILING_H
#define CURRENT_TASK_CONCAT_DIM0_TILING_H

#include <cstdint>

struct ConcatDim0Tiling {
    int32_t M0;
    int32_t M1;
    int32_t M2;
    int32_t M3;
    int32_t inputCount;
    int32_t N;
    int32_t totalM;
    int32_t blockM;
    int32_t subBlockM;
    int32_t blockNum;
    int32_t usedCoreNum;
    int32_t tasksPerCore;
};

#endif
