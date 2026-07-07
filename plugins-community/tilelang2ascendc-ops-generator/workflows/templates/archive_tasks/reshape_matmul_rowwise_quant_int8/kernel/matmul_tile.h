/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_TILE_RESHAPE_H
#define MATMUL_TILE_RESHAPE_H

#include "kernel_operator.h"

// Re-export shared helpers from matmul_leakyrelu.
#include "../../matmul_leakyrelu/kernel/matmul_tile.h"

// L0C(Nz) -> GM(ND) with configurable dstStride.
// Delegates to FixpipeNzL0cToNdGmStride from flash_attention/matmul_tile.h.
template<typename T>
__aicore__ inline void FixpipeNzL0cToNdGm(const AscendC::GlobalTensor<T> &dst,
                                           const AscendC::LocalTensor<T> &src,
                                           uint32_t m, uint32_t n, uint32_t dstStride)
{
    FixpipeNzL0cToNdGmStride(dst, src, m, n, dstStride);
}

#endif // MATMUL_TILE_RESHAPE_H
