/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "acl/acl.h"
#include "conv3ddw_kernel.h"

template <typename T>
__global__ __aicore__ __schedmode__(1) void conv3ddw_entry(GM_ADDR x, GM_ADDR filterSize, GM_ADDR dedy, GM_ADDR dedw,
                                                           GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    Conv3DdwTilingData td;
    Conv3Ddw::ReadTiling(tiling, td);
    Conv3Ddw::InitializeOutput(dedw, td);
    if ASCEND_IS_AIC {
        if (AscendC::GetBlockIdx() >= td.usedCoreNum)
            return;
        Conv3Ddw::Kernel<T> op;
        op.Init(x, dedy, dedw, td);
        op.Process();
        op.End();
    }
}

extern "C" void LaunchConv3DdwFp16(uint32_t blocks, void* stream, void* x, void* filterSize, void* dedy, void* dedw,
                                   void* workspace, void* tiling)
{
    conv3ddw_entry<half><<<blocks, nullptr, static_cast<aclrtStream>(stream)>>>(
        (GM_ADDR)x, (GM_ADDR)filterSize, (GM_ADDR)dedy, (GM_ADDR)dedw, (GM_ADDR)workspace, (GM_ADDR)tiling);
}

extern "C" void LaunchConv3DdwBf16(uint32_t blocks, void* stream, void* x, void* filterSize, void* dedy, void* dedw,
                                   void* workspace, void* tiling)
{
    conv3ddw_entry<bfloat16_t><<<blocks, nullptr, static_cast<aclrtStream>(stream)>>>(
        (GM_ADDR)x, (GM_ADDR)filterSize, (GM_ADDR)dedy, (GM_ADDR)dedw, (GM_ADDR)workspace, (GM_ADDR)tiling);
}
