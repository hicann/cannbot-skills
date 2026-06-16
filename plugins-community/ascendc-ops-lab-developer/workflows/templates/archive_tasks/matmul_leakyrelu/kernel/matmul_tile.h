/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_TILE_LEAKYRELU_H
#define MATMUL_TILE_LEAKYRELU_H

#include "../../flash_attention/kernel/op_kernel/matmul_tile.h"

// This header extends flash_attention's matmul_tile.h with an int8_t overload
// of LoadNzL1ToZnL0B. All other helpers (LoadNdGmToNzL1, LoadNzL1ToZzL0A,
// LoadNzL1ToZnL0B<T>, FixpipeNzL0cToNdGm, FixpipeNzL0cToNdGmStride) are
// inherited from flash_attention/kernel/op_kernel/matmul_tile.h.

// L1(Nz) -> L0B(Zn) for int8_t
__aicore__ inline void LoadNzL1ToZnL0B(const AscendC::LocalTensor<int8_t> &dst,
                                       const AscendC::LocalTensor<int8_t> &src,
                                       uint32_t k, uint32_t n, uint32_t colC0Stride)
{
    static constexpr uint32_t FRAC_M  = 16;
    static constexpr uint32_t FRAC_N  = 16;
    static constexpr uint32_t C0_SIZE = 32;
    static constexpr uint32_t FRAC_K  = C0_SIZE;
    uint32_t dstOffset = FRAC_K * n;
    uint32_t srcOffset = FRAC_K * C0_SIZE;
    // Nz -> Zn
    AscendC::LoadData2dTransposeParams loadDataParams;
    loadDataParams.repeatTimes = n / C0_SIZE;
    loadDataParams.srcStride = colC0Stride / FRAC_K;
    loadDataParams.dstGap = 1;
    loadDataParams.dstFracGap = 0;
    for (int i = 0; i < k / FRAC_K; ++i) {
        AscendC::LoadDataWithTranspose(dst[i * dstOffset], src[i * srcOffset], loadDataParams);
    }
}

#endif // MATMUL_TILE_LEAKYRELU_H
