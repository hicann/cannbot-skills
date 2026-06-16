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
#include "aclrtlaunch_gather_elements_v2_last_dim_fp16.h"
#include "aclrtlaunch_gather_elements_v2_last_dim_fp32.h"
#include "aclrtlaunch_gather_elements_v2_transpose_fp16.h"
#include "aclrtlaunch_gather_elements_v2_transpose_fp32.h"
#include "aclrtlaunch_gather_elements_v2_scalar_fp16.h"
#include "aclrtlaunch_gather_elements_v2_scalar_fp32.h"
#include "gather_elements_v2_tiling.h"

namespace ascend_kernel {

constexpr int32_t GATHER_MODE_LAST_DIM = 0;
constexpr int32_t GATHER_MODE_TRANSPOSE = 1;
constexpr int32_t GATHER_MODE_SCALAR = 2;

inline int32_t CeilDivI32(int32_t a, int32_t b)
{
    if (b == 0) { return 0; }
    return (a + b - 1) / b;
}

inline int32_t ChooseBlockM(int32_t m)
{
    int32_t blockM = 32;
    if (m < 32) blockM = 16;
    if (m < 16) blockM = 8;
    if (m < 8) blockM = 4;
    if (m < 4) blockM = 2;
    if (m < 2) blockM = 1;
    return blockM;
}

static inline void LaunchGatherElementsV2Kernel(
    const at::Tensor &x, const at::Tensor &index, const at::Tensor &row_map,
    at::Tensor &y, int32_t xStride, int32_t yStride, int32_t ig32,
    int32_t blockM, int32_t usedCoreNum, int32_t tasksPerCore,
    int32_t subBlockM, int32_t m, int32_t xRows, int32_t mode32)
{
    at::Tensor tilingCpu = at::empty(
        {static_cast<int64_t>(sizeof(GatherElementsV2KernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto *tiling = reinterpret_cast<GatherElementsV2KernelTiling *>(tilingCpu.data_ptr());
    tiling->M = m;                tiling->XRows = xRows;
    tiling->XG = xStride;        tiling->IG = ig32;
    tiling->XStride = xStride;   tiling->YStride = yStride;
    tiling->blockM = blockM;     tiling->usedCoreNum = usedCoreNum;
    tiling->tasksPerCore = tasksPerCore;
    tiling->subBlockM = subBlockM;
    tiling->useRowMap = (mode32 == GATHER_MODE_LAST_DIM) ? 0 : 1;
    tiling->mode = mode32;

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);
    uint32_t blockDim = static_cast<uint32_t>(usedCoreNum);

    bool isFp16 = (x.scalar_type() == at::kHalf);
    if (mode32 == GATHER_MODE_LAST_DIM) {
        if (isFp16) {
            EXEC_KERNEL_CMD(gather_elements_v2_last_dim_fp16, blockDim, x, index, row_map, y, tilingNpu);
        } else {
            EXEC_KERNEL_CMD(gather_elements_v2_last_dim_fp32, blockDim, x, index, row_map, y, tilingNpu);
        }
    } else if (mode32 == GATHER_MODE_TRANSPOSE) {
        if (isFp16) {
            EXEC_KERNEL_CMD(gather_elements_v2_transpose_fp16, blockDim, x, index, row_map, y, tilingNpu);
        } else {
            EXEC_KERNEL_CMD(gather_elements_v2_transpose_fp32, blockDim, x, index, row_map, y, tilingNpu);
        }
    } else {
        if (isFp16) {
            EXEC_KERNEL_CMD(gather_elements_v2_scalar_fp16, blockDim, x, index, row_map, y, tilingNpu);
        } else {
            EXEC_KERNEL_CMD(gather_elements_v2_scalar_fp32, blockDim, x, index, row_map, y, tilingNpu);
        }
    }
}

at::Tensor gather_elements_v2(
    const at::Tensor &x,
    const at::Tensor &index,
    const at::Tensor &row_map,
    int64_t ig,
    int64_t mode)
{
    TORCH_CHECK(x.device().is_privateuseone(), "x must be on NPU");
    TORCH_CHECK(index.device().is_privateuseone(), "index must be on NPU");
    TORCH_CHECK(row_map.device().is_privateuseone(), "row_map must be on NPU");
    TORCH_CHECK(x.dim() == 2, "x must be [XRows, XStride]");
    TORCH_CHECK(index.dim() == 2, "index must be [M, YStride]");
    TORCH_CHECK(row_map.dim() == 1, "row_map must be [M]");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(index.is_contiguous(), "index must be contiguous");
    TORCH_CHECK(row_map.is_contiguous(), "row_map must be contiguous");
    TORCH_CHECK(index.scalar_type() == at::kInt, "index must be int32");
    TORCH_CHECK(row_map.scalar_type() == at::kInt, "row_map must be int32");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kFloat, "x must be float16 or float32");

    const int32_t mode32 = static_cast<int32_t>(mode);
    const int32_t m = static_cast<int32_t>(index.size(0));
    TORCH_CHECK(index.size(0) == row_map.size(0), "index and row_map row count must match");
    const int32_t xRows = static_cast<int32_t>(x.size(0));
    const int32_t xStride = static_cast<int32_t>(x.size(1));
    const int32_t yStride = static_cast<int32_t>(index.size(1));
    const int32_t ig32 = static_cast<int32_t>(ig);

    TORCH_CHECK(ig32 >= 0, "ig must be non-negative");
    TORCH_CHECK(ig32 <= yStride, "ig must not exceed index width");

    const int32_t blockM = ChooseBlockM(std::max(m, 1));
    const int32_t mNum = CeilDivI32(std::max(m, 1), blockM);
    const int32_t usedCoreNum = std::min(DEFAULT_NUM_PHYSICAL_CORES, std::max(mNum, 1));
    const int32_t tasksPerCore = CeilDivI32(mNum, usedCoreNum);
    const int32_t subBlockM = std::max(blockM / DEFAULT_VEC_NUM, 1);

    at::Tensor y = at::zeros({m, yStride}, x.options());
    LaunchGatherElementsV2Kernel(x, index, row_map, y, xStride, yStride, ig32, blockM,
                                  usedCoreNum, tasksPerCore, subBlockM, m, xRows, mode32);

    return y;
}

} // namespace ascend_kernel
