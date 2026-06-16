/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// concat_dv2 op_host — tiling calculation + kernel dispatch + launch via EXEC_KERNEL_CMD

#include <algorithm>
#include <array>
#include <cstdint>

#include <torch/extension.h>
#include <torch/library.h>

#include "torch_kernel_helper.h"
#include "kernels/aclrtlaunch_concat_dim0_1.h"
#include "kernels/aclrtlaunch_concat_dim0_2.h"
#include "kernels/aclrtlaunch_concat_dim0_3.h"
#include "kernels/aclrtlaunch_concat_dim0_4.h"

#include "concat_dim0_tiling.h"

namespace ascend_kernel {

constexpr int32_t DEFAULT_NUM_PHYSICAL_CORES = 20;
constexpr int32_t DEFAULT_VEC_NUM = 2;

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    if (b == 0) { return 0; }
    return (a + b - 1) / b;
}

int32_t ChooseBlockM(int32_t totalM)
{
    for (int32_t candidate : {64, 32, 16, 8, 4, 2}) {
        if (totalM >= candidate) {
            return candidate;
        }
    }
    return 2;
}

void CheckInputTensor(const at::Tensor &tensor, const char *name, const at::Tensor &reference)
{
    TORCH_CHECK(tensor.dim() == 2, name, " must be a 2D tensor");
    TORCH_CHECK(tensor.scalar_type() == at::kFloat, name, " must be float32");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(tensor.device().type() == reference.device().type(), name, " must be on the same device as x0");
    TORCH_CHECK(tensor.sizes()[1] == reference.sizes()[1], name, " must have the same trailing width as x0");
}

struct ConcatDim0LaunchData {
    at::Tensor y;
    at::Tensor tilingNpu;
    int32_t usedCoreNum;
};

ConcatDim0LaunchData PrepareConcatDim0(const std::array<const at::Tensor *, 4> &inputs, int32_t inputCount)
{
    TORCH_CHECK(inputCount >= 1 && inputCount <= 4, "inputCount must be in [1, 4]");

    const at::Tensor &x0 = *inputs[0];
    CheckInputTensor(x0, "x0", x0);
    TORCH_CHECK(x0.device().type() == at::kPrivateUse1, "x0 must be on NPU/PrivateUse1 device");

    int32_t m[4] = {
        static_cast<int32_t>(x0.sizes()[0]),
        0,
        0,
        0,
    };
    for (int32_t index = 1; index < inputCount; ++index) {
        const at::Tensor &tensor = *inputs[index];
        const std::string name = "x" + std::to_string(index);
        CheckInputTensor(tensor, name.c_str(), x0);
        m[index] = static_cast<int32_t>(tensor.sizes()[0]);
    }

    const int32_t n = static_cast<int32_t>(x0.sizes()[1]);
    const int32_t totalM = m[0] + m[1] + m[2] + m[3];
    at::Tensor y = at::empty({static_cast<int64_t>(totalM), static_cast<int64_t>(n)}, x0.options());
    if (totalM == 0 || n == 0) {
        return {y, at::Tensor(), 0};
    }

    const int32_t blockM = ChooseBlockM(totalM);
    const int32_t subBlockM = blockM / DEFAULT_VEC_NUM;
    const int32_t blockNum = CeilDivI32(totalM, blockM);
    const int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, blockNum);
    const int32_t tasksPerCore = CeilDivI32(blockNum, usedCoreNum);

    at::Tensor tilingCpu = at::empty(
        {static_cast<long>(sizeof(ConcatDim0Tiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<ConcatDim0Tiling *>(tilingCpu.data_ptr());
    tiling->M0 = m[0];
    tiling->M1 = m[1];
    tiling->M2 = m[2];
    tiling->M3 = m[3];
    tiling->inputCount = inputCount;
    tiling->N = n;
    tiling->totalM = totalM;
    tiling->blockM = blockM;
    tiling->subBlockM = subBlockM;
    tiling->blockNum = blockNum;
    tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    return {y, tilingNpu, usedCoreNum};
}

at::Tensor concat_dim0_1(const at::Tensor &x0)
{
    auto data = PrepareConcatDim0({&x0, nullptr, nullptr, nullptr}, 1);
    if (data.y.numel() == 0) {
        return data.y;
    }
    uint32_t blockDim = static_cast<uint32_t>(data.usedCoreNum);
    EXEC_KERNEL_CMD(concat_dim0_1, blockDim,
                    x0, nullptr, nullptr, nullptr, data.y, data.tilingNpu);
    return data.y;
}

at::Tensor concat_dim0_2(const at::Tensor &x0, const at::Tensor &x1)
{
    auto data = PrepareConcatDim0({&x0, &x1, nullptr, nullptr}, 2);
    if (data.y.numel() == 0) {
        return data.y;
    }
    uint32_t blockDim = static_cast<uint32_t>(data.usedCoreNum);
    EXEC_KERNEL_CMD(concat_dim0_2, blockDim,
                    x0, x1, nullptr, nullptr, data.y, data.tilingNpu);
    return data.y;
}

at::Tensor concat_dim0_3(const at::Tensor &x0, const at::Tensor &x1, const at::Tensor &x2)
{
    auto data = PrepareConcatDim0({&x0, &x1, &x2, nullptr}, 3);
    if (data.y.numel() == 0) {
        return data.y;
    }
    uint32_t blockDim = static_cast<uint32_t>(data.usedCoreNum);
    EXEC_KERNEL_CMD(concat_dim0_3, blockDim,
                    x0, x1, x2, nullptr, data.y, data.tilingNpu);
    return data.y;
}

at::Tensor concat_dim0_4(const at::Tensor &x0, const at::Tensor &x1, const at::Tensor &x2, const at::Tensor &x3)
{
    auto data = PrepareConcatDim0({&x0, &x1, &x2, &x3}, 4);
    if (data.y.numel() == 0) {
        return data.y;
    }
    uint32_t blockDim = static_cast<uint32_t>(data.usedCoreNum);
    EXEC_KERNEL_CMD(concat_dim0_4, blockDim,
                    x0, x1, x2, x3, data.y, data.tilingNpu);
    return data.y;
}

} // namespace ascend_kernel
