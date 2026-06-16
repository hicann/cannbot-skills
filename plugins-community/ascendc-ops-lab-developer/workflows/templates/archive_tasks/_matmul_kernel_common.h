/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MATMUL_KERNEL_COMMON_H
#define MATMUL_KERNEL_COMMON_H

#include "kernel_operator.h"

// CRTP base for MatmulKernel templates shared between quant_matmul and matmul_leakyrelu.
// Derived must provide accessors: base_m(), base_n(), base_k(), base_mk(), base_kn(), base_mn(),
// l1_prefetch().
template <typename Derived, typename aType, typename bType, typename cType>
class MatmulKernelBase {
    static_assert(std::is_same<aType, bType>::value, "aType and bType must be the same type");
    static constexpr uint32_t C0 = 32 / sizeof(aType);

public:
    __aicore__ inline MatmulKernelBase() {}
    __aicore__ inline void ComputeBlock(const AscendC::GlobalTensor<aType> &aBlock,
                                        const AscendC::GlobalTensor<bType> &bBlock,
                                        const AscendC::GlobalTensor<cType> &cBlock);

protected:
    AscendC::TQue<AscendC::TPosition::A1, 1> inQueueA1;
    AscendC::TQue<AscendC::TPosition::A2, 1> inQueueA2;
    AscendC::TQue<AscendC::TPosition::B1, 1> inQueueB1;
    AscendC::TQue<AscendC::TPosition::B2, 1> inQueueB2;
    AscendC::TQue<AscendC::TPosition::CO1, 1> outQueueCO1;

    uint32_t k_;
    uint32_t a_dvalue_, b_dvalue_;

    __aicore__ inline void CopyA(const AscendC::GlobalTensor<aType> &A, uint32_t kLen);
    __aicore__ inline void CopyB(const AscendC::GlobalTensor<bType> &B, uint32_t kLen);
    __aicore__ inline void SplitA(const AscendC::LocalTensor<aType> &a1Local,
                                  uint32_t offset, uint32_t colC0Stride);
    __aicore__ inline void SplitB(const AscendC::LocalTensor<bType> &b1Local,
                                  uint32_t offset, uint32_t colC0Stride);
    __aicore__ inline void Compute(const AscendC::LocalTensor<cType> &c1Local, bool cmatrixInitVal);
    __aicore__ inline void CopyOut(const AscendC::GlobalTensor<cType> &C);

private:
    __aicore__ inline Derived &self() { return static_cast<Derived &>(*this); }
    __aicore__ inline const Derived &self() const { return static_cast<const Derived &>(*this); }
};

// ---------------------------------------------------------------------------
// ComputeBlock
// ---------------------------------------------------------------------------
template <typename Derived, typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelBase<Derived, aType, bType, cType>::ComputeBlock(
    const AscendC::GlobalTensor<aType> &aBlock,
    const AscendC::GlobalTensor<bType> &bBlock,
    const AscendC::GlobalTensor<cType> &cBlock)
{
    AscendC::LocalTensor<cType> c1Local = outQueueCO1.AllocTensor<cType>();
    uint32_t kTiles = k_ / self().base_k();

    for (uint32_t outer = 0; outer < kTiles; outer += self().l1_prefetch()) {
        uint32_t count = (kTiles - outer < self().l1_prefetch())
                             ? (kTiles - outer) : self().l1_prefetch();
        uint32_t kLen = count * self().base_k();

        CopyA(aBlock[outer * self().base_k()], kLen);
        CopyB(bBlock[outer * self().base_k() * b_dvalue_], kLen);

        AscendC::LocalTensor<aType> a1Local = inQueueA1.DeQue<aType>();
        AscendC::LocalTensor<bType> b1Local = inQueueB1.DeQue<bType>();

        for (uint32_t i = 0; i < count; i++) {
            SplitA(a1Local, i * self().base_mk(), self().base_m());
            SplitB(b1Local, i * self().base_k() * C0, kLen);
            Compute(c1Local, (outer + i == 0));
        }

        inQueueA1.FreeTensor(a1Local);
        inQueueB1.FreeTensor(b1Local);
    }

    outQueueCO1.EnQue(c1Local);
    CopyOut(cBlock);
}

// ---------------------------------------------------------------------------
// CopyA
// ---------------------------------------------------------------------------
template <typename Derived, typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelBase<Derived, aType, bType, cType>::CopyA(
    const AscendC::GlobalTensor<aType> &A, uint32_t kLen)
{
    auto a1Local = inQueueA1.AllocTensor<aType>();
    LoadNdGmToNzL1(a1Local, A, self().base_m(), kLen, a_dvalue_);
    inQueueA1.EnQue(a1Local);
}

// ---------------------------------------------------------------------------
// CopyB
// ---------------------------------------------------------------------------
template <typename Derived, typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelBase<Derived, aType, bType, cType>::CopyB(
    const AscendC::GlobalTensor<bType> &B, uint32_t kLen)
{
    AscendC::LocalTensor<bType> b1Local = inQueueB1.AllocTensor<bType>();
    LoadNdGmToNzL1(b1Local, B, kLen, self().base_n(), b_dvalue_);
    inQueueB1.EnQue(b1Local);
}

// ---------------------------------------------------------------------------
// SplitA
// ---------------------------------------------------------------------------
template <typename Derived, typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelBase<Derived, aType, bType, cType>::SplitA(
    const AscendC::LocalTensor<aType> &a1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.AllocTensor<aType>();
    LoadNzL1ToZzL0A(a2Local, a1Local[offset], self().base_m(), self().base_k(), colC0Stride);
    inQueueA2.EnQue(a2Local);
}

// ---------------------------------------------------------------------------
// SplitB
// ---------------------------------------------------------------------------
template <typename Derived, typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelBase<Derived, aType, bType, cType>::SplitB(
    const AscendC::LocalTensor<bType> &b1Local, uint32_t offset, uint32_t colC0Stride)
{
    AscendC::LocalTensor<bType> b2Local = inQueueB2.AllocTensor<bType>();
    LoadNzL1ToZnL0B(b2Local, b1Local[offset], self().base_k(), self().base_n(), colC0Stride);
    inQueueB2.EnQue(b2Local);
}

// ---------------------------------------------------------------------------
// Compute
// ---------------------------------------------------------------------------
template <typename Derived, typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelBase<Derived, aType, bType, cType>::Compute(
    const AscendC::LocalTensor<cType> &c1Local, bool cmatrixInitVal)
{
    AscendC::LocalTensor<aType> a2Local = inQueueA2.DeQue<aType>();
    AscendC::LocalTensor<bType> b2Local = inQueueB2.DeQue<bType>();
    AscendC::MmadParams mmadParams;
    mmadParams.m = self().base_m();
    mmadParams.n = self().base_n();
    mmadParams.k = self().base_k();
    mmadParams.cmatrixInitVal = cmatrixInitVal;
    AscendC::Mmad(c1Local, a2Local, b2Local, mmadParams);
    inQueueA2.FreeTensor(a2Local);
    inQueueB2.FreeTensor(b2Local);
}

// ---------------------------------------------------------------------------
// CopyOut
// ---------------------------------------------------------------------------
template <typename Derived, typename aType, typename bType, typename cType>
__aicore__ inline void MatmulKernelBase<Derived, aType, bType, cType>::CopyOut(
    const AscendC::GlobalTensor<cType> &C)
{
    auto c1Local = outQueueCO1.DeQue<cType>();
    FixpipeNzL0cToNdGm(C, c1Local, self().base_m(), self().base_n());
    outQueueCO1.FreeTensor(c1Local);
}

#endif // MATMUL_KERNEL_COMMON_H
