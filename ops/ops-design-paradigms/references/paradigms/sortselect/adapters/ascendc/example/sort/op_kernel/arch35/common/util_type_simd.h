/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UTIL_TYPE_SIMD_H
#define UTIL_TYPE_SIMD_H
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"

constexpr uint32_t UB_BLOCK_SIZE = Ops::Base::GetUbBlockSize(); // UB 块字节数。

__aicore__ inline uint32_t RoundUpAlign(uint32_t x)
{
    if (UB_BLOCK_SIZE == 0) {
        return x;
    }
    return (x + UB_BLOCK_SIZE - 1U) / UB_BLOCK_SIZE * UB_BLOCK_SIZE;
}

__aicore__ inline uint64_t RoundUpAlignUint64(uint64_t x)
{
    if (UB_BLOCK_SIZE == 0) {
        return x;
    }
    return (x + UB_BLOCK_SIZE - 1ULL) / UB_BLOCK_SIZE * UB_BLOCK_SIZE;
}

template <typename T>
struct DoubleBufferSimd {
    AscendC::GlobalTensor<T> doubleBuffer_[2];
    int selector_ = 0;
    __aicore__ inline DoubleBufferSimd() {}
    __aicore__ inline void SetDoubleBuffer(AscendC::GlobalTensor<T> currentBuffer,
                                           AscendC::GlobalTensor<T> alternateBuffer)
    {
        selector_ = 0;
        doubleBuffer_[0] = currentBuffer;
        doubleBuffer_[1] = alternateBuffer;
    }
    __aicore__ inline AscendC::GlobalTensor<T> Current() const
    {
        return doubleBuffer_[selector_];
    }
    __aicore__ inline AscendC::GlobalTensor<T> Alternate() const
    {
        return doubleBuffer_[selector_ ^ 1];
    }
    __aicore__ inline void UpdateSelect()
    {
        selector_ = selector_ ^ 1;
    }
};
#endif
