/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_H
#define MATMUL_H

#include "kernel_operator.h"
#include "matmul_tile.h"
#include "int8_matmul_scale_tiling.h"
#include "../../_matmul_kernel_common.h"

constexpr uint32_t baseM = DEFAULT_BASE_M;
constexpr uint32_t baseN = DEFAULT_BASE_N;
constexpr uint32_t baseK = DEFAULT_BASE_K;

constexpr uint32_t baseMK = baseM * baseK;
constexpr uint32_t baseKN = baseK * baseN;
constexpr uint32_t baseMN = baseM * baseN;

constexpr uint32_t L1_PREFETCH = 3;

template <typename aType, typename bType, typename cType>
class MatmulKernel : public MatmulKernelBase<MatmulKernel<aType, bType, cType>, aType, bType, cType> {
    static constexpr uint32_t base_m() { return baseM; }
    static constexpr uint32_t base_n() { return baseN; }
    static constexpr uint32_t base_k() { return baseK; }
    static constexpr uint32_t base_mk() { return baseMK; }
    static constexpr uint32_t base_kn() { return baseKN; }
    static constexpr uint32_t base_mn() { return baseMN; }
    static constexpr uint32_t l1_prefetch() { return L1_PREFETCH; }

    friend class MatmulKernelBase<MatmulKernel<aType, bType, cType>, aType, bType, cType>;

public:
    __aicore__ inline MatmulKernel() {}
    __aicore__ inline void Init(uint32_t k, uint32_t lda, uint32_t ldb, AscendC::TPipe &pipe);
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::Init(
    uint32_t k, uint32_t lda, uint32_t ldb, AscendC::TPipe &pipe)
{
    ASSERT(k % baseK == 0);
    this->k_ = k;
    this->a_dvalue_ = lda;
    this->b_dvalue_ = ldb;

    pipe.InitBuffer(this->inQueueA1, 2, baseMK * L1_PREFETCH * sizeof(aType));
    pipe.InitBuffer(this->inQueueA2, 2, baseMK * sizeof(aType));
    pipe.InitBuffer(this->inQueueB1, 2, baseKN * L1_PREFETCH * sizeof(bType));
    pipe.InitBuffer(this->inQueueB2, 2, baseKN * sizeof(bType));
    pipe.InitBuffer(this->outQueueCO1, 1, baseMN * sizeof(cType));
}

#endif // MATMUL_H
