/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_quant_matmul.h"
#include "int8_matmul_scale_tiling.h"

namespace ascend_kernel {

at::Tensor quant_matmul(const at::Tensor &a, const at::Tensor &b, const at::Tensor &scale)
{
    TORCH_CHECK(a.dim() == 2, "a must be 2D");
    TORCH_CHECK(b.dim() == 2, "b must be 2D");
    TORCH_CHECK(scale.dim() == 1, "scale must be 1D");
    TORCH_CHECK(a.sizes()[1] == b.sizes()[0], "k dimension must match");
    TORCH_CHECK(a.scalar_type() == at::kChar, "a must be int8");
    TORCH_CHECK(b.scalar_type() == at::kChar, "b must be int8");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "scale must be float32");
    TORCH_CHECK(scale.sizes()[0] == b.sizes()[1], "scale size must match N");

    uint32_t usedCoreNum = 2;
    uint32_t m = static_cast<uint32_t>(a.sizes()[0]);
    uint32_t n = static_cast<uint32_t>(b.sizes()[1]);
    uint32_t k = static_cast<uint32_t>(a.sizes()[1]);

    at::Tensor c = at::empty({static_cast<int64_t>(m), static_cast<int64_t>(n)},
                             at::device(at::kPrivateUse1).dtype(at::kHalf));

    at::Tensor t = at::empty({static_cast<int64_t>(sizeof(Int8MatmulScaleTiling))},
                             at::device(at::kCPU).dtype(at::kByte));
    auto *tiling_ptr = reinterpret_cast<Int8MatmulScaleTiling *>(t.data_ptr());
    tiling_ptr->M = static_cast<int32_t>(m);
    tiling_ptr->N = static_cast<int32_t>(n);
    tiling_ptr->K = static_cast<int32_t>(k);
    tiling_ptr->baseM = DEFAULT_BASE_M;
    tiling_ptr->baseN = DEFAULT_BASE_N;
    tiling_ptr->baseK = DEFAULT_BASE_K;
    auto tiling_npu = t.to(at::kPrivateUse1);

    int64_t workSpaceSize = static_cast<int64_t>(tiling_ptr->baseM) * tiling_ptr->baseN *
                            WORKSPACE_DEPTH * sizeof(int32_t) * usedCoreNum;
    at::Tensor w = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));

    uint32_t blockDim = usedCoreNum;
    EXEC_KERNEL_CMD(quant_matmul, blockDim, a, b, scale, c, w, tiling_npu);
    return c;
}

} // namespace ascend_kernel
