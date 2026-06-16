/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_COMMON_H
#define KERNEL_COMMON_H
// Template copy for: matmul_leakyrelu

#include "../../common/tiling_copy.h"

__aicore__ inline uint32_t CeilDiv32(uint32_t a, uint32_t b)
{
    if (b == 0) { return 0; }
    return (a + b - 1) / b;
}

class BlockScheduler {
public:
    __aicore__ inline void Init(int M, int N, int baseM, int baseN, int numBlocks, int blockIdx)
    {
        if (baseM == 0 || baseN == 0 || numBlocks == 0) {
            mBlocks_ = 0;
            nBlocks_ = 0;
            startBlock_ = 0;
            endBlock_ = 0;
            current_ = 0;
            return;
        }
        mBlocks_ = M / baseM;
        nBlocks_ = N / baseN;
        int totalBlocks = mBlocks_ * nBlocks_;
        int blocksPerCore = totalBlocks / numBlocks;
        int remainder = totalBlocks % numBlocks;
        startBlock_ = blockIdx * blocksPerCore + (blockIdx < remainder ? blockIdx : remainder);
        endBlock_ = startBlock_ + blocksPerCore + (blockIdx < remainder ? 1 : 0);
        current_ = startBlock_;
    }

    __aicore__ inline bool HasNext()
    {
        return current_ < endBlock_;
    }

    __aicore__ inline void Next(int &mIdx, int &nIdx)
    {
        mIdx = current_ / nBlocks_;
        nIdx = current_ % nBlocks_;
        current_++;
    }

private:
    int mBlocks_;
    int nBlocks_;
    int startBlock_;
    int endBlock_;
    int current_;
};

#endif // KERNEL_COMMON_H
