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
#include "aclrtlaunch_matmul_leakyrelu_fp16.h"
#include "aclrtlaunch_matmul_leakyrelu_int8.h"
#include "matmul_leakyrelu_tiling.h"

namespace ascend_kernel {

static inline int64_t GetWorkspaceSize(const MatmulLeakyReluTiling *tiling, int accElementSize,
                                       uint32_t usedCoreNum)
{
    return tiling->baseM * tiling->baseN * WORKSPACE_DEPTH * accElementSize * usedCoreNum;
}

at::Tensor matmul_leakyrelu(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.dim() == 2, "matmul_leakyrelu a must be two-dimensional.");
    TORCH_CHECK(b.dim() == 2, "matmul_leakyrelu b must be two-dimensional.");
    TORCH_CHECK(a.sizes()[1] == b.sizes()[0], "matmul_leakyrelu k must be same.");
    TORCH_CHECK(a.scalar_type() == b.scalar_type(), "matmul_leakyrelu a and b must have same dtype.");

    uint32_t usedCoreNum = 20;
    uint32_t m = static_cast<uint32_t>(a.sizes()[0]);
    uint32_t n = static_cast<uint32_t>(b.sizes()[1]);
    uint32_t k = static_cast<uint32_t>(a.sizes()[1]);

    at::Tensor c = at::empty({static_cast<int64_t>(m), static_cast<int64_t>(n)},
                             at::device(at::kPrivateUse1).dtype(at::kFloat));

    at::Tensor t = at::empty({static_cast<int64_t>(sizeof(MatmulLeakyReluTiling))},
                             at::device(at::kCPU).dtype(at::kByte));
    auto *tiling_ptr = reinterpret_cast<MatmulLeakyReluTiling *>(t.data_ptr());
    tiling_ptr->M = static_cast<int32_t>(m);
    tiling_ptr->N = static_cast<int32_t>(n);
    tiling_ptr->K = static_cast<int32_t>(k);
    tiling_ptr->baseM = DEFAULT_BASE_M;
    tiling_ptr->baseN = DEFAULT_BASE_N;
    tiling_ptr->baseK = DEFAULT_BASE_K;
    tiling_ptr->l1Prefetch = DEFAULT_L1_PREFETCH;
    auto tiling_npu = t.to(at::kPrivateUse1);

    uint32_t blockDim = usedCoreNum;

    if (a.scalar_type() == at::kHalf) {
        int accElementSize = sizeof(float);
        int64_t workSpaceSize = GetWorkspaceSize(tiling_ptr, accElementSize, usedCoreNum);
        at::Tensor w = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));
        EXEC_KERNEL_CMD(matmul_leakyrelu_fp16, blockDim, a, b, c, w, tiling_npu);
    } else if (a.scalar_type() == at::kChar) {
        int accElementSize = sizeof(int32_t);
        int64_t workSpaceSize = GetWorkspaceSize(tiling_ptr, accElementSize, usedCoreNum);
        at::Tensor w = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));
        EXEC_KERNEL_CMD(matmul_leakyrelu_int8, blockDim, a, b, c, w, tiling_npu);
    } else {
        TORCH_CHECK(false, "matmul_leakyrelu unsupported dtype, expected float16 or int8.");
    }

    return c;
}

} // namespace ascend_kernel
