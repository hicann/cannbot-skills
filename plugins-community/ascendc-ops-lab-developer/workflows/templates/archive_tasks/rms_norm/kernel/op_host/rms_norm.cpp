/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// RMSNorm op_host — tiling calculation + strategy dispatch + EXEC_KERNEL_CMD launch
// Format: Abs-style with platform_ascendc and EXEC_KERNEL_CMD

#include <algorithm>
#include <cstdint>

#include <torch/extension.h>
#include <torch/library.h>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"

#include "aclrtlaunch_rms_norm_merge_n.h"
#include "aclrtlaunch_rms_norm_single_row.h"
#include "aclrtlaunch_rms_norm_splitd.h"

// ---------------------------------------------------------------------------
// Tiling constants
// ---------------------------------------------------------------------------

constexpr int32_t DEFAULT_BLOCK_M = 64;
constexpr int32_t DEFAULT_ROW_FACTOR = 8;
constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;

// ---------------------------------------------------------------------------
// Host function
// ---------------------------------------------------------------------------

namespace ascend_kernel {

std::vector<at::Tensor> rms_norm(const at::Tensor &x, const at::Tensor &gamma, double eps)
{
    TORCH_CHECK(x.dim() == 2, "x must be [M, N]");
    TORCH_CHECK(gamma.dim() == 1, "gamma must be [N]");
    TORCH_CHECK(
        x.scalar_type() == at::kFloat || x.scalar_type() == at::kHalf ||
            x.scalar_type() == at::kBFloat16,
        "x must be float16, float32, or bfloat16");
    TORCH_CHECK(gamma.scalar_type() == x.scalar_type(),
        "gamma must have the same dtype as x");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(gamma.is_contiguous(), "gamma must be contiguous");
    TORCH_CHECK(static_cast<int32_t>(x.sizes()[1]) == static_cast<int32_t>(gamma.sizes()[0]),
        "gamma shape mismatch");

    int32_t M = static_cast<int32_t>(x.sizes()[0]);
    int32_t N = static_cast<int32_t>(x.sizes()[1]);

    int32_t mNum = (M + DEFAULT_BLOCK_M - 1) / DEFAULT_BLOCK_M;
    int32_t usedCoreNum = std::min<int32_t>(DEFAULT_NUM_PHYSICAL_CORES, std::max<int32_t>(1, mNum));
    int32_t tasksPerCore = (mNum + usedCoreNum - 1) / usedCoreNum;

    at::Tensor y = at::empty_like(x);
    at::Tensor invRms = at::empty({M}, x.options());

    // dtype flag: 0=fp32, 1=fp16, 2=bf16
    int64_t dtypeFlag = 0;
    if (x.scalar_type() == at::kHalf) {
        dtypeFlag = 1;
    } else if (x.scalar_type() == at::kBFloat16) {
        dtypeFlag = 2;
    }

    // All tiling params as left values (required by EXEC_KERNEL_CMD)
    int32_t blockM = DEFAULT_BLOCK_M;
    int32_t rowFactor = DEFAULT_ROW_FACTOR;
    float epsVal = static_cast<float>(eps);
    float invN = 1.0f / static_cast<float>(N);
    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    // Strategy dispatch: select kernel name and params
    if (N <= 1024) {
        EXEC_KERNEL_CMD(rms_norm_merge_n, blockDim,
                        x, gamma, y, invRms,
                        M, N, blockM, usedCoreNum, tasksPerCore, rowFactor, epsVal, invN,
                        dtypeFlag);
    } else if (N > 8192) {
        EXEC_KERNEL_CMD(rms_norm_splitd, blockDim,
                        x, gamma, y, invRms,
                        M, N, blockM, usedCoreNum, tasksPerCore, epsVal, invN,
                        dtypeFlag);
    } else {
        EXEC_KERNEL_CMD(rms_norm_single_row, blockDim,
                        x, gamma, y, invRms,
                        M, N, blockM, usedCoreNum, tasksPerCore, epsVal, invN,
                        dtypeFlag);
    }

    return {y, invRms};
}

}  // namespace ascend_kernel
