/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RESHAPE_MATMUL_QUANT_TILING_H
#define RESHAPE_MATMUL_QUANT_TILING_H

#include <cstdint>

constexpr int32_t DEFAULT_BASE_M = 128;
constexpr int32_t DEFAULT_BASE_N = 256;
constexpr int32_t DEFAULT_BASE_K = 64;
constexpr int32_t DEFAULT_K_L1 = 256;

#pragma pack(push, 8)
struct ReshapeMatmulQuantTiling {
    int32_t M;
    int32_t N;
    int32_t H_K;       // square matrix dimension (H is H_K x H_K)
    int32_t baseM;
    int32_t baseN;
    int32_t baseK;
    int32_t K_L1;      // L1 prefetch depth
    int32_t nTiles;     // N / baseN
    int32_t nTilesPerH; // H_K / baseN
};
#pragma pack(pop)

#endif // RESHAPE_MATMUL_QUANT_TILING_H
