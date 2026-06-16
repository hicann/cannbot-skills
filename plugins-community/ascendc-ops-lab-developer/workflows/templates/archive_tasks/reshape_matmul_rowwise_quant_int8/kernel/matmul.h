/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_CUSTOM_H
#define MATMUL_CUSTOM_H

#include "kernel_operator.h"
#include "matmul_tile.h"
#include "reshape_matmul_quant_tiling.h"

constexpr uint32_t baseM = DEFAULT_BASE_M;
constexpr uint32_t baseN = DEFAULT_BASE_N;
constexpr uint32_t baseK = DEFAULT_BASE_K;
constexpr uint32_t kL1   = DEFAULT_K_L1;

constexpr uint32_t baseMK = baseM * baseK;
constexpr uint32_t baseKN = baseK * baseN;
constexpr uint32_t baseMN = baseM * baseN;

// Number of k-tiles to batch into one L1 DMA transfer.
constexpr uint32_t L1_PREFETCH = kL1 / baseK;  // 256/64 = 4

template <typename aType, typename cType>
class MatmulKernel {
    static constexpr uint32_t C0 = 32 / sizeof(aType);

public:
    __aicore__ inline MatmulKernel() {}
    __aicore__ inline void Init(uint32_t ldA, uint32_t ldB, AscendC::TPipe &pipe);

    // Compute one baseM x baseN tile, write to workspace GM with dstStride
    __aicore__ inline void ComputeBlock(const AscendC::GlobalTensor<aType> &aBlock,
                                        const AscendC::GlobalTensor<aType> &bBlock,
                                        const AscendC::GlobalTensor<cType> &wsBlock,
                                        uint32_t K_total, uint32_t dstStride);

private:
    __aicore__ inline void CopyA(const AscendC::GlobalTensor<aType> &A, uint32_t kLen);
    __aicore__ inline void CopyB(const AscendC::GlobalTensor<aType> &B, uint32_t kLen);
    __aicore__ inline void SplitA(const AscendC::LocalTensor<aType> &a1Local,
                                  uint32_t offset, uint32_t colC0Stride);
    __aicore__ inline void SplitB(const AscendC::LocalTensor<aType> &b1Local,
                                  uint32_t offset, uint32_t colC0Stride);
    __aicore__ inline void Compute(const AscendC::LocalTensor<cType> &c1Local, bool cmatrixInitVal);
    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<cType> &C, uint32_t dstStride);

private:
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    uint32_t aDValue, bDValue;
};

template <typename aType, typename cType>
__aicore__ inline void MatmulKernel<aType, cType>::Init(uint32_t ldA, uint32_t ldB,
                                                         AscendC::TPipe &pipe)
{
    aDValue = ldA;
    bDValue = ldB;

    pipe.InitBuffer(inQueueA1, 2, baseMK * L1_PREFETCH * sizeof(aType));
    pipe.InitBuffer(inQueueA2, 2, baseMK * sizeof(aType));
    pipe.InitBuffer(inQueueB1, 2, baseKN * L1_PREFETCH * sizeof(aType));
    pipe.InitBuffer(inQueueB2, 2, baseKN * sizeof(aType));
    pipe.InitBuffer(outQueueCO1, 1, baseMN * sizeof(cType));
}

template <typename aType, typename cType>
__aicore__ inline void MatmulKernel<aType, cType>::ComputeBlock(
    const AscendC::GlobalTensor<aType> &aBlock,
    const AscendC::GlobalTensor<aType> &bBlock,
    const AscendC::GlobalTensor<cType> &wsBlock,
    uint32_t K_total, uint32_t dstStride)
{
    AscendC::LocalTensor<cType> c1Local = outQueueCO1.AllocTensor<cType>();
    uint32_t kTiles = K_total / baseK;

    for (uint32_t outer = 0; outer < kTiles; outer += L1_PREFETCH) {
        uint32_t count = (kTiles - outer < L1_PREFETCH) ? (kTiles - outer) : L1_PREFETCH;
        uint32_t kLen = count * baseK;

        CopyA(aBlock[outer * baseK], kLen);
        CopyB(bBlock[outer * baseK * bDValue], kLen);

        AscendC::LocalTensor<aType> a1Local = inQueueA1.DeQue<aType>();
        AscendC::LocalTensor<aType> b1Local = inQueueB1.DeQue<aType>();

        for (uint32_t i = 0; i < count; i++) {
            SplitA(a1Local, i * baseMK, baseM);
            SplitB(b1Local, i * baseK * C0, kLen);
            Compute(c1Local, (outer + i == 0));
        }

        inQueueA1.FreeTensor(a1Local);
        inQueueB1.FreeTensor(b1Local);
    }

    outQueueCO1.EnQue(c1Local);
    CopyOut(wsBlock, dstStride);
}

template <typename aType, typename cType>
__aicore__ inline void MatmulKernel<aType, cType>::CopyA(
    const AscendC::GlobalTensor<aType> &A, uint32_t kLen)
{
    AscendC::LocalTensor<aType> a1Local = inQueueA1.AllocTensor<aType>();
    LoadNdGmToNzL1(a1Local, A, baseM, kLen, aDValue);
    inQueueA1.EnQue(a1Local);
}

template <typename aType, typename cType>
__aicore__ inline void MatmulKernel<aType, cType>::CopyB(
    const AscendC::GlobalTensor<aType> &B, uint32_t kLen)
{
    AscendC::LocalTensor<aType> b1Local = inQueueB1.AllocTensor<aType>();
    LoadNdGmToNzL1(b1Local, B, kLen, baseN, bDValue);
    inQueueB1.EnQue(b1Local);
}

template <typename aType, typename cType>
__aicore__ inline void MatmulKernel<aType, cType>::SplitA(
    const AscendC::LocalTensor<aType> &a1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.AllocTensor<aType>();
    LoadNzL1ToZzL0A(a2Local, a1Local[offset], baseM, baseK, colC0Stride);
    inQueueA2.EnQue(a2Local);
}

template <typename aType, typename cType>
__aicore__ inline void MatmulKernel<aType, cType>::SplitB(
    const AscendC::LocalTensor<aType> &b1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<aType> b2Local = inQueueB2.AllocTensor<aType>();
    LoadNzL1ToZnL0B(b2Local, b1Local[offset], baseK, baseN, colC0Stride);
    inQueueB2.EnQue(b2Local);
}

template <typename aType, typename cType>
__aicore__ inline void MatmulKernel<aType, cType>::Compute(
    const AscendC::LocalTensor<cType> &c1Local, bool cmatrixInitVal)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.DeQue<aType>();
    AscendC::LocalTensor<aType> b2Local = inQueueB2.DeQue<aType>();
    AscendC::MmadParams mmadParams;
    mmadParams.m = baseM;
    mmadParams.n = baseN;
    mmadParams.k = baseK;
    mmadParams.cmatrixInitVal = cmatrixInitVal;
    AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
    inQueueA2.FreeTensor(a2Local);
    inQueueB2.FreeTensor(b2Local);
}

template <typename aType, typename cType>
__aicore__ inline void MatmulKernel<aType, cType>::CopyOut(
    const AscendC::GlobalTensor<cType> &C, uint32_t dstStride)
{
    auto c1Local = outQueueCO1.DeQue<cType>();
    FixpipeNzL0cToNdGm(C, c1Local, baseM, baseN, dstStride);
    outQueueCO1.FreeTensor(c1Local);
}

#endif // MATMUL_CUSTOM_H
