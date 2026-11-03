/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sort_lib_util.h
 * \brief Internal utilities: CopyGmToUb, CopyUbToGm, DoubleBufferSimd, SyncEvent.
 *
 * \internal  Do not include directly — used internally by SortLib.
 */

#ifndef SORT_LIB_UTIL_H
#define SORT_LIB_UTIL_H

#include "kernel_operator.h"

namespace SortLib::detail {

constexpr uint32_t UB_ALIGN_BYTES = 32U;  // UB/workspace 32 字节对齐单位
constexpr uint32_t FILL_MIN_BYTES = 256U; // FillZeros 每次 Fill 的最小字节粒度

__aicore__ inline uint32_t RoundUpAlign(uint32_t x, uint32_t block)
{
    if (block == 0) {
        return x;
    }
    return (x + block - 1U) / block * block;
}

__aicore__ inline uint32_t RoundUpAlign(uint32_t x)
{
    return RoundUpAlign(x, UB_ALIGN_BYTES);
}

template <typename T>
struct DoubleBufferSimd {
    AscendC::GlobalTensor<T> doubleBuffer_[2];
    int selector_ = 0;

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
        selector_ ^= 1;
    }
    __aicore__ inline void Swap()
    {
        selector_ ^= 1;
    }
};

template <typename T>
__aicore__ inline void CopyGmToUb(AscendC::GlobalTensor<T> src, AscendC::LocalTensor<T>& dst, uint64_t srcOffset,
                                  uint32_t count)
{
    AscendC::DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = count * sizeof(T);
    params.srcStride = 0;
    params.dstStride = 0;
    AscendC::DataCopyPad(dst, src[srcOffset], params, padParams);
}

template <typename T>
__aicore__ inline void CopyUbToGm(AscendC::GlobalTensor<T> dst, AscendC::LocalTensor<T>& src, uint64_t dstOffset,
                                  uint32_t count)
{
    AscendC::DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = count * sizeof(T);
    params.srcStride = 0;
    params.dstStride = 0;
    AscendC::DataCopyPad(dst[dstOffset], src, params);
}

template <AscendC::HardEvent E>
__aicore__ inline void SyncEvent()
{
    auto id = static_cast<event_t>(GetTPipePtr()->FetchEventID(E));
    AscendC::SetFlag<E>(id);
    AscendC::WaitFlag<E>(id);
}

template <typename T>
__aicore__ inline void FillZeros(AscendC::GlobalTensor<T>& gm, uint64_t totalCount, uint32_t realCoreNum)
{
    uint64_t minElem = FILL_MIN_BYTES / sizeof(T);

    uint64_t perCore = (totalCount + realCoreNum - 1) / realCoreNum;
    perCore = (perCore + minElem - 1) / minElem * minElem;

    uint64_t coresNeed = (totalCount + perCore - 1) / perCore;
    uint64_t lastCount = totalCount - (coresNeed - 1) * perCore;

    if (static_cast<uint32_t>(AscendC::GetBlockIdx()) < coresNeed) {
        uint64_t myCount = (static_cast<uint32_t>(AscendC::GetBlockIdx()) == coresNeed - 1) ? lastCount : perCore;
        AscendC::GlobalTensor<T> sub = gm[static_cast<uint64_t>(AscendC::GetBlockIdx()) * perCore];
        AscendC::Fill(sub, myCount, static_cast<T>(0));
    }
}

} // namespace SortLib::detail

#endif
