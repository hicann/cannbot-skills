/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef KERNEL_COMMON_H
#define KERNEL_COMMON_H

#include "../../../common/tiling_copy.h"

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    if (b == 0) { return 0; }
    return (a + b - 1) / b;
}

template <AscendC::HardEvent evt>
__aicore__ inline void SetWaitFlag() {
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    AscendC::SetFlag<evt>(eventId);
    AscendC::WaitFlag<evt>(eventId);
}

// 1D block scheduler: distributes totalBlocks evenly across cores
class BlockScheduler1D {
public:
    __aicore__ inline void Init(int totalBlocks, int numCores, int coreIdx)
    {
        if (numCores > 0) {
            int blocksPerCore = totalBlocks / numCores;
            int remainder = totalBlocks % numCores;
            startBlock_ = coreIdx * blocksPerCore + (coreIdx < remainder ? coreIdx : remainder);
            endBlock_ = startBlock_ + blocksPerCore + (coreIdx < remainder ? 1 : 0);
            current_ = startBlock_;
        } else {
            startBlock_ = 0;
            endBlock_ = 0;
            current_ = 0;
        }
    }

    __aicore__ inline bool HasNext() { return current_ < endBlock_; }

    __aicore__ inline int Next()
    {
        return current_++;
    }

private:
    int startBlock_;
    int endBlock_;
    int current_;
};

#endif // KERNEL_COMMON_H
