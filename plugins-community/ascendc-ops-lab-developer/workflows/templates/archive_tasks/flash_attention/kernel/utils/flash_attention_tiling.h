/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_TILING_H
#define FLASH_ATTENTION_TILING_H

#include <cstdint>

constexpr uint32_t BLOCK_M = 64;
constexpr uint32_t BLOCK_N = 64;
constexpr uint32_t BASE_K = 128;
constexpr uint32_t PRELAUNCH = 2;
constexpr uint32_t RING_SLOTS = PRELAUNCH + 1;  // 3
constexpr uint32_t MAX_CORES = 20;

// AIC<->AIV workspace queue sync signals.
constexpr uint32_t SIG_S_READY = 0;  // AIC MM1 done -> AIV Vec1
constexpr uint32_t SIG_S_FREE  = 1;  // AIV Vec1 done reading S -> AIC MM1
constexpr uint32_t SIG_P_READY = 2;  // AIV Vec1 done -> AIC MM2
constexpr uint32_t SIG_P_FREE  = 3;  // AIC MM2 done reading P -> AIV Vec1
constexpr uint32_t SIG_O_READY = 4;  // AIC MM2 done -> AIV Vec2
constexpr uint32_t SIG_O_FREE  = 5;  // AIV Vec2 done reading O -> AIC MM2

#pragma pack(push, 8)
struct FlashAttentionTiling {
    int32_t batch;
    int32_t heads;
    int32_t qSeqLen;       // original Q sequence length
    int32_t kvSeqLen;      // original KV sequence length
    int32_t dim;
    int32_t blockM;        // 64
    int32_t blockN;        // 64
    float   smScale;       // 1/sqrt(dim)
    int32_t usedCoreNum;
    int32_t tailValid;     // kvSeqLen % BLOCK_N, valid columns in last KV tile
    int32_t qSeqLenAlign;  // aligned Q seq len (to BLOCK_M)
    int32_t kvSeqLenAlign; // aligned KV seq len (to BLOCK_N)
};
#pragma pack(pop)

#endif // FLASH_ATTENTION_TILING_H
