/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_WORKSPACE_QUEUE_H
#define FLASH_ATTENTION_WORKSPACE_QUEUE_H

#include "kernel_operator.h"

template <typename T, uint32_t DEPTH>
class WorkspaceQueue {
public:
    __aicore__ inline WorkspaceQueue() {}

    __aicore__ inline void Init(const AscendC::GlobalTensor<T> &workspace, uint32_t slotSize,
                                uint16_t producerNotifyConsumerId,
                                uint16_t consumerNotifyProducerId)
    {
        workspace_ = workspace;
        slotSize_ = slotSize;
        head_ = 0;
        tail_ = 0;
        producerNotifyConsumerId_ = producerNotifyConsumerId;
        consumerNotifyProducerId_ = consumerNotifyProducerId;
    }

    __aicore__ inline void InitFreeSlotsMte2()
    {
        for (uint32_t i = 0; i < DEPTH; ++i) {
            AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(consumerNotifyProducerId_);
        }
    }

    __aicore__ inline AscendC::GlobalTensor<T> ProducerAcquire()
    {
        AscendC::CrossCoreWaitFlag<0x2>(consumerNotifyProducerId_);
        return workspace_[head_ % DEPTH * slotSize_];
    }

    __aicore__ inline void ProducerReleaseFix()
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(producerNotifyConsumerId_);
        head_++;
    }

    __aicore__ inline void ProducerReleaseMte3()
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE3>(producerNotifyConsumerId_);
        head_++;
    }

    __aicore__ inline AscendC::GlobalTensor<T> ConsumerAcquire()
    {
        AscendC::CrossCoreWaitFlag<0x2>(producerNotifyConsumerId_);
        return workspace_[tail_ % DEPTH * slotSize_];
    }

    __aicore__ inline void ConsumerReleaseMte2()
    {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(consumerNotifyProducerId_);
        tail_++;
    }

private:
    AscendC::GlobalTensor<T> workspace_;
    uint32_t slotSize_;
    uint32_t head_;
    uint32_t tail_;
    uint16_t producerNotifyConsumerId_;
    uint16_t consumerNotifyProducerId_;
};

#endif // FLASH_ATTENTION_WORKSPACE_QUEUE_H
