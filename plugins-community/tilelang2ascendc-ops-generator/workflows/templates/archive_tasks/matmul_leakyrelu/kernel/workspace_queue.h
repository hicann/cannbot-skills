/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef WORKSPACE_QUEUE_H
#define WORKSPACE_QUEUE_H

#include "kernel_operator.h"

// Canonical implementation — also duplicated in
// quant_matmul/kernel/workspace_queue.h. Keep both in sync.

/**
 * Ring buffer over workspace GM for AIC (cube) -> AIV (vector) data transfer.
 *
 * Producer (AIC) writes slots via Fixpipe, consumer (AIV) reads via MTE2.
 * Synchronization uses CrossCoreSetFlag/WaitFlag mode 0x2 (AIC <-> AIV).
 *
 * Usage:
 *   WorkspaceQueue<float, 4> queue;
 *   queue.Init(workspace, slotSize, CUBE_FLAG_ID, VECTOR_FLAG_ID);
 *
 *   // AIV init: mark all slots free
 *   if ASCEND_IS_AIV { queue.InitFreeSlots(); }
 *
 *   // AIC producer loop:
 *   auto slot = queue.ProducerAcquire();   // wait free, get slot
 *   Fixpipe(slot, ...);                    // write data
 *   queue.ProducerRelease();               // signal data ready
 *
 *   // AIV consumer loop:
 *   auto slot = queue.ConsumerAcquire();   // wait data ready, get slot
 *   DataCopy(local, slot, ...);            // read data
 *   queue.ConsumerRelease();               // signal slot free
 */
template <typename T, uint32_t DEPTH>
class WorkspaceQueue {
public:
    __aicore__ inline WorkspaceQueue() {}

    __aicore__ inline void Init(GM_ADDR workspace, uint32_t slotSize,
                                uint16_t cubeNotifyVecId, uint16_t vecNotifyCubeId);

    // AIV calls at init to mark all slots as free
    __aicore__ inline void InitFreeSlots();

    // Producer (AIC): wait for free slot, return its GlobalTensor
    __aicore__ inline AscendC::GlobalTensor<T> ProducerAcquire();
    // Producer (AIC): signal data ready, advance to next slot
    __aicore__ inline void ProducerRelease();

    // Consumer (AIV): wait for data ready, return its GlobalTensor
    __aicore__ inline AscendC::GlobalTensor<T> ConsumerAcquire();
    // Consumer (AIV): signal slot free, advance to next slot
    __aicore__ inline void ConsumerRelease();

private:
    AscendC::GlobalTensor<T> workspace_;
    uint32_t slotSize_;
    uint32_t head_;  // producer write index
    uint32_t tail_;  // consumer read index
    uint16_t cubeNotifyVecId_;
    uint16_t vecNotifyCubeId_;
};

template <typename T, uint32_t DEPTH>
__aicore__ inline void WorkspaceQueue<T, DEPTH>::Init(
    GM_ADDR workspace, uint32_t slotSize,
    uint16_t cubeNotifyVecId, uint16_t vecNotifyCubeId)
{
    slotSize_ = slotSize;
    head_ = 0;
    tail_ = 0;
    cubeNotifyVecId_ = cubeNotifyVecId;
    vecNotifyCubeId_ = vecNotifyCubeId;
    workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(workspace), DEPTH * slotSize);
}

template <typename T, uint32_t DEPTH>
__aicore__ inline void WorkspaceQueue<T, DEPTH>::InitFreeSlots()
{
    for (uint32_t i = 0; i < DEPTH; ++i) {
        AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(vecNotifyCubeId_);
    }
}

template <typename T, uint32_t DEPTH>
__aicore__ inline AscendC::GlobalTensor<T> WorkspaceQueue<T, DEPTH>::ProducerAcquire()
{
    AscendC::CrossCoreWaitFlag<0x2>(vecNotifyCubeId_);
    return workspace_[head_ % DEPTH * slotSize_];
}

template <typename T, uint32_t DEPTH>
__aicore__ inline void WorkspaceQueue<T, DEPTH>::ProducerRelease()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(cubeNotifyVecId_);
    head_++;
}

template <typename T, uint32_t DEPTH>
__aicore__ inline AscendC::GlobalTensor<T> WorkspaceQueue<T, DEPTH>::ConsumerAcquire()
{
    AscendC::CrossCoreWaitFlag<0x2>(cubeNotifyVecId_);
    return workspace_[tail_ % DEPTH * slotSize_];
}

template <typename T, uint32_t DEPTH>
__aicore__ inline void WorkspaceQueue<T, DEPTH>::ConsumerRelease()
{
    AscendC::CrossCoreSetFlag<0x2, PIPE_MTE2>(vecNotifyCubeId_);
    tail_++;
}

#endif // WORKSPACE_QUEUE_H