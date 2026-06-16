/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// flash_attention op_host — tiling calculation + kernel dispatch

#include <cstdint>

#include <torch/extension.h>
#include <torch/library.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

#include "kernels/aclrtlaunch_flash_attention_bf16.h"
#include "kernels/aclrtlaunch_flash_attention_fp16.h"

#include "utils/flash_attention_tiling.h"
#include "utils/flash_attention_common.h"

namespace ascend_kernel {

void PrepareFlashAttention(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v,
                            at::Tensor &output, int32_t &usedCoreNum, at::Tensor &tilingNpu)
{
    TORCH_CHECK(q.dim() == 4, "q must be 4D [batch, heads, qSeqLen, dim]");
    TORCH_CHECK(k.dim() == 4, "k must be 4D [batch, heads, kvSeqLen, dim]");
    TORCH_CHECK(v.dim() == 4, "v must be 4D [batch, heads, kvSeqLen, dim]");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have same dtype");
    TORCH_CHECK(q.scalar_type() == v.scalar_type(), "q and v must have same dtype");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(k.is_contiguous(), "k must be contiguous");
    TORCH_CHECK(v.is_contiguous(), "v must be contiguous");
    TORCH_CHECK(q.device().type() == at::kPrivateUse1, "q must be on NPU/PrivateUse1 device");

    uint32_t batch    = q.sizes()[0];
    uint32_t heads    = q.sizes()[1];
    uint32_t qSeqLen  = q.sizes()[2];
    uint32_t dim      = q.sizes()[3];
    uint32_t kvSeqLen = k.sizes()[2];

    uint32_t qSeqLenAlign  = AlignUpHost(qSeqLen,  BLOCK_M);
    uint32_t kvSeqLenAlign = AlignUpHost(kvSeqLen, BLOCK_N);

    uint32_t tailValid  = kvSeqLen % BLOCK_N;
    int totalBlocks     = CeilDivHost(qSeqLenAlign, BLOCK_M) * heads * batch;
    uint32_t usedCores  = totalBlocks < static_cast<int>(MAX_CORES) ? totalBlocks : MAX_CORES;

    output = at::zeros({(int64_t)batch, (int64_t)heads,
                        (int64_t)qSeqLen, (int64_t)dim},
                       q.options());

    at::Tensor tilingCpu = at::empty({(int64_t)sizeof(FlashAttentionTiling)},
                                     at::device(at::kCPU).dtype(at::kByte));
    auto *tilingPtr = reinterpret_cast<FlashAttentionTiling *>(tilingCpu.data_ptr());
    tilingPtr->batch         = static_cast<int32_t>(batch);
    tilingPtr->heads         = static_cast<int32_t>(heads);
    tilingPtr->qSeqLen       = static_cast<int32_t>(qSeqLen);
    tilingPtr->kvSeqLen      = static_cast<int32_t>(kvSeqLen);
    tilingPtr->dim           = static_cast<int32_t>(dim);
    tilingPtr->blockM        = BLOCK_M;
    tilingPtr->blockN        = BLOCK_N;
    tilingPtr->smScale       = 1.0f / sqrtf(static_cast<float>(dim));
    tilingPtr->usedCoreNum   = static_cast<int32_t>(usedCores);
    tilingPtr->tailValid     = static_cast<int32_t>(tailValid);
    tilingPtr->qSeqLenAlign  = static_cast<int32_t>(qSeqLenAlign);
    tilingPtr->kvSeqLenAlign = static_cast<int32_t>(kvSeqLenAlign);

    tilingNpu = tilingCpu.to(at::kPrivateUse1);
    usedCoreNum = static_cast<int32_t>(usedCores);
}

// Shared workspace allocation: computes per-core workspace sizes and allocates the tensor.
static at::Tensor AllocateFlashAttentionWorkspace(const at::Tensor &q, int32_t usedCoreNum)
{
    uint32_t dim         = q.sizes()[3];
    uint32_t dimAlign    = AlignUpHost(dim, 16);
    uint64_t qElemSize   = sizeof(uint16_t);
    uint64_t wsSSize     = (uint64_t)RING_SLOTS * BLOCK_M * BLOCK_N * sizeof(float);
    uint64_t wsPSize     = (uint64_t)RING_SLOTS * BLOCK_M * BLOCK_N * qElemSize;
    uint64_t wsOSize     = (uint64_t)RING_SLOTS * BLOCK_M * dimAlign * sizeof(float);
    uint64_t wsMetaSize  = (uint64_t)RING_SLOTS * BLOCK_M * 3 * sizeof(float);
    uint64_t wsAccOSize  = (uint64_t)RING_SLOTS * BLOCK_M * dimAlign * sizeof(float);
    uint64_t perCoreBytes = wsSSize + wsPSize + wsOSize + wsMetaSize + wsAccOSize;
    uint64_t totalWsBytes = perCoreBytes * usedCoreNum;
    return at::zeros({(int64_t)totalWsBytes},
                     at::device(at::kPrivateUse1).dtype(at::kByte));
}

at::Tensor flash_attention_fp16(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v)
{
    int32_t usedCoreNum;
    at::Tensor tilingNpu;
    at::Tensor output;
    PrepareFlashAttention(q, k, v, output, usedCoreNum, tilingNpu);

    at::Tensor workspace = AllocateFlashAttentionWorkspace(q, usedCoreNum);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    aclrtlaunch_flash_attention_fp16(
        static_cast<uint32_t>(usedCoreNum), aclStream,
        const_cast<void*>(q.storage().data()),
        const_cast<void*>(k.storage().data()),
        const_cast<void*>(v.storage().data()),
        const_cast<void*>(output.storage().data()),
        const_cast<void*>(workspace.storage().data()),
        const_cast<void*>(tilingNpu.storage().data()));
    return output;
}

at::Tensor flash_attention_bf16(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v)
{
    int32_t usedCoreNum;
    at::Tensor tilingNpu;
    at::Tensor output;
    PrepareFlashAttention(q, k, v, output, usedCoreNum, tilingNpu);

    at::Tensor workspace = AllocateFlashAttentionWorkspace(q, usedCoreNum);

    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    aclrtlaunch_flash_attention_bf16(
        static_cast<uint32_t>(usedCoreNum), aclStream,
        const_cast<void*>(q.storage().data()),
        const_cast<void*>(k.storage().data()),
        const_cast<void*>(v.storage().data()),
        const_cast<void*>(output.storage().data()),
        const_cast<void*>(workspace.storage().data()),
        const_cast<void*>(tilingNpu.storage().data()));
    return output;
}

} // namespace ascend_kernel
