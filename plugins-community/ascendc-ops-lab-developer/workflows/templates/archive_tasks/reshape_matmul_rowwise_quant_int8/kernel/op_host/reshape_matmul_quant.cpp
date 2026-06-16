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
#include "aclrtlaunch_reshape_matmul_quant.h"
#include "reshape_matmul_quant_tiling.h"

namespace ascend_kernel {

at::Tensor reshape_matmul_quant(const at::Tensor &x, const at::Tensor &h)
{
    TORCH_CHECK(x.dim() == 2, "x must be 2D");
    TORCH_CHECK(h.dim() == 2, "h must be 2D");
    TORCH_CHECK(h.sizes()[0] == h.sizes()[1], "h must be square");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "x must be bfloat16");
    TORCH_CHECK(h.scalar_type() == at::kBFloat16, "h must be bfloat16");

    int32_t M = static_cast<int32_t>(x.sizes()[0]);
    int32_t N = static_cast<int32_t>(x.sizes()[1]);
    int32_t H_K = static_cast<int32_t>(h.sizes()[0]);

    TORCH_CHECK(N % H_K == 0, "N must be divisible by H_K");
    TORCH_CHECK(M % DEFAULT_BASE_M == 0, "M must be divisible by baseM");
    TORCH_CHECK(N % DEFAULT_BASE_N == 0, "N must be divisible by baseN");
    TORCH_CHECK(H_K % DEFAULT_BASE_K == 0, "H_K must be divisible by baseK");

    int32_t nTiles = N / DEFAULT_BASE_N;
    int32_t nTilesPerH = H_K / DEFAULT_BASE_N;
    int32_t mNum = M / DEFAULT_BASE_M;

    uint32_t usedCoreNum = static_cast<uint32_t>(mNum);

    at::Tensor y = at::empty({M, N}, at::device(at::kPrivateUse1).dtype(at::kChar));

    int64_t wsSize = static_cast<int64_t>(M) * N * sizeof(float);
    at::Tensor w = at::empty({wsSize}, at::device(at::kPrivateUse1).dtype(at::kByte));

    at::Tensor t = at::empty({static_cast<int64_t>(sizeof(ReshapeMatmulQuantTiling))},
                             at::device(at::kCPU).dtype(at::kByte));
    auto *tp = reinterpret_cast<ReshapeMatmulQuantTiling *>(t.data_ptr());
    tp->M = M;
    tp->N = N;
    tp->H_K = H_K;
    tp->baseM = DEFAULT_BASE_M;
    tp->baseN = DEFAULT_BASE_N;
    tp->baseK = DEFAULT_BASE_K;
    tp->K_L1 = DEFAULT_K_L1;
    tp->nTiles = nTiles;
    tp->nTilesPerH = nTilesPerH;
    auto tiling_npu = t.to(at::kPrivateUse1);

    uint32_t blockDim = usedCoreNum;
    EXEC_KERNEL_CMD(reshape_matmul_quant, blockDim, x, h, y, w, tiling_npu);
    return y;
}

} // namespace ascend_kernel
