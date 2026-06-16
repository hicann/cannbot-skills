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
 * @file matmul_leakyrelu_tiling.h
 *
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef MATMUL_LEAKYRELU_TILING_H
#define MATMUL_LEAKYRELU_TILING_H

#include <cstdint>

constexpr int32_t DEFAULT_BASE_M = 128;
constexpr int32_t DEFAULT_BASE_N = 128;
constexpr int32_t DEFAULT_BASE_K = 128;
constexpr int32_t DEFAULT_L1_PREFETCH = 4;
constexpr int32_t WORKSPACE_DEPTH = 4;

#pragma pack(push, 8)
struct MatmulLeakyReluTiling {
    int32_t M;
    int32_t N;
    int32_t K;
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    int32_t l1Prefetch;
};
#pragma pack(pop)

#endif // MATMUL_LEAKYRELU_TILING_H
