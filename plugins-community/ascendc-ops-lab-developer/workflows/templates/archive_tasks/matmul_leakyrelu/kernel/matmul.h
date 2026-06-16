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
 * @file matmul.h
 *
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef MATMUL_CUSTOM_H
#define MATMUL_CUSTOM_H

#include "kernel_operator.h"
#include "matmul_tile.h"
#include "matmul_leakyrelu_tiling.h"
#include "../../_matmul_kernel_common.h"

template <typename aType, typename bType, typename cType>
class MatmulKernel : public MatmulKernelBase<MatmulKernel<aType, bType, cType>, aType, bType, cType> {
    uint32_t base_m() const { return baseM_; }
    uint32_t base_n() const { return baseN_; }
    uint32_t base_k() const { return baseK_; }
    uint32_t base_mk() const { return baseMK_; }
    uint32_t base_kn() const { return baseKN_; }
    uint32_t base_mn() const { return baseMN_; }
    uint32_t l1_prefetch() const { return l1Prefetch_; }

    friend class MatmulKernelBase<MatmulKernel<aType, bType, cType>, aType, bType, cType>;

    uint32_t l1Prefetch_;
    uint32_t baseM_, baseN_, baseK_;
    uint32_t baseMK_, baseKN_, baseMN_;

public:
    __aicore__ inline MatmulKernel() {}
    __aicore__ inline void Init(uint32_t k, uint32_t lda, uint32_t ldb,
                                uint32_t baseM, uint32_t baseN, uint32_t baseK,
                                uint32_t l1Prefetch,
                                AscendC::TPipe &pipe);
};

template <typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernel<aType, bType, cType>::Init(
    uint32_t k, uint32_t lda, uint32_t ldb,
    uint32_t baseM, uint32_t baseN, uint32_t baseK,
    uint32_t l1Prefetch,
    AscendC::TPipe &pipe)
{
    ASSERT(k % baseK == 0);

    l1Prefetch_ = l1Prefetch;
    baseM_ = baseM;
    baseN_ = baseN;
    baseK_ = baseK;
    baseMK_ = baseM_ * baseK_;
    baseKN_ = baseK_ * baseN_;
    baseMN_ = baseM_ * baseN_;
    this->k_ = k;
    this->a_dvalue_ = lda;
    this->b_dvalue_ = ldb;

    pipe.InitBuffer(this->inQueueA1, 2, baseMK_ * l1Prefetch_ * sizeof(aType));
    pipe.InitBuffer(this->inQueueA2, 2, baseMK_ * sizeof(aType));
    pipe.InitBuffer(this->inQueueB1, 2, baseKN_ * l1Prefetch_ * sizeof(bType));
    pipe.InitBuffer(this->inQueueB2, 2, baseKN_ * sizeof(bType));
    pipe.InitBuffer(this->outQueueCO1, 1, baseMN_ * sizeof(cType));
}

#endif // MATMUL_CUSTOM_H
