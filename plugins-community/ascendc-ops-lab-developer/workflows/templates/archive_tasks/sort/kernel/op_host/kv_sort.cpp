/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <cstring>

#include "torch_kernel_helper.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_kv_sort.h"
#include "sort_tiling.h"

namespace ascend_kernel {

constexpr int32_t NUM_CORES = 20;
constexpr int64_t UB_SORT_CAPACITY = 2048;
constexpr int64_t MULTI_CORE_PER_LOOP = 512;
constexpr int64_t ONE_LOOP_MAX = 256;

static void computeTiling(SortKernelTiling* t, int32_t totalLength)
{
    *t = SortKernelTiling{};
    t->totalLength = totalLength;

    int64_t aligned = ((int64_t)totalLength * 4 + 31) / 32 * 32 / 4;
    int64_t sortNum = ((aligned + 31) / 32) * 32;
    t->sortNum = static_cast<int32_t>(sortNum);

    if (totalLength <= (int32_t)UB_SORT_CAPACITY) {
        t->tilingMode = SORT_TILING_MODE_FULLLOAD;
        t->coreNum = 1;
        t->needCoreNum = 1;
        return;
    }

    if (totalLength <= (int32_t)(UB_SORT_CAPACITY * 2)) {
        t->tilingMode = SORT_TILING_MODE_SINGLECORE;
        t->coreNum = 1;
        t->needCoreNum = 1;
        return;
    }

    t->tilingMode = SORT_TILING_MODE_MULTICORE;

    int32_t perLoop = static_cast<int32_t>(MULTI_CORE_PER_LOOP);
    int32_t needCores = std::min<int32_t>(NUM_CORES, (totalLength + perLoop - 1) / perLoop);
    if (needCores < 2) needCores = 2;

    t->coreNum = needCores;
    t->needCoreNum = needCores;

    int32_t perCore = (totalLength + needCores - 1) / needCores;
    perCore = ((perCore + 31) / 32) * 32;
    int32_t lastCore = totalLength - perCore * (needCores - 1);
    if (lastCore <= 0) {
        perCore = totalLength / needCores;
        perCore = ((perCore + 31) / 32) * 32;
        lastCore = totalLength - perCore * (needCores - 1);
    }

    t->perCoreElements = perCore;
    t->lastCoreElements = lastCore;

    int32_t loopElem = std::min<int32_t>(perLoop, perCore);
    loopElem = ((loopElem + 31) / 32) * 32;

    t->perCorePerLoopElements = loopElem;
    t->perCoreLoops = (perCore + loopElem - 1) / loopElem;
    t->perCoreLastLoopElements = perCore - loopElem * (t->perCoreLoops - 1);

    t->lastCorePerLoopElements = loopElem;
    t->lastCoreLoops = (lastCore + loopElem - 1) / loopElem;
    t->lastCoreLastLoopElements = lastCore - loopElem * (t->lastCoreLoops - 1);

    t->oneLoopMaxElements = static_cast<int32_t>(ONE_LOOP_MAX);
}

std::tuple<at::Tensor, at::Tensor> kv_sort(const at::Tensor& keys, const at::Tensor& values)
{
    TORCH_CHECK(keys.dim() == 1, "keys must be 1D");
    TORCH_CHECK(values.dim() == 1, "values must be 1D");
    TORCH_CHECK(keys.size(0) == values.size(0), "keys and values must have same length");
    TORCH_CHECK(keys.scalar_type() == at::kInt, "keys must be int32");
    TORCH_CHECK(values.scalar_type() == at::kInt, "values must be int32");
    TORCH_CHECK(keys.is_contiguous() && values.is_contiguous(), "inputs must be contiguous");

    int32_t totalLength = static_cast<int32_t>(keys.size(0));

    at::Tensor sortedKeys = at::empty_like(keys);
    at::Tensor sortedValues = at::empty_like(values);

    at::Tensor tilingCpu = at::empty(
        {static_cast<int64_t>(sizeof(SortKernelTiling))},
        at::device(at::kCPU).dtype(at::kByte));
    auto* tiling = reinterpret_cast<SortKernelTiling*>(tilingCpu.data_ptr());
    computeTiling(tiling, totalLength);

    auto tilingNpu = tilingCpu.to(at::kPrivateUse1);

    int64_t wsBytes = 0;
    if (tiling->tilingMode == SORT_TILING_MODE_MULTICORE) {
        int64_t perCoreSortLen = static_cast<int64_t>(tiling->perCoreElements) * 2;
        int64_t oneBufFloats = static_cast<int64_t>(tiling->needCoreNum) * perCoreSortLen;
        wsBytes = oneBufFloats * sizeof(float) * 2 + 4096;
    }
    at::Tensor workspace = at::empty({wsBytes}, keys.options().dtype(at::kByte));

    uint32_t blockDim = static_cast<uint32_t>(tiling->coreNum);
    EXEC_KERNEL_CMD(kv_sort, blockDim, keys, values, sortedKeys, sortedValues, workspace, tilingNpu);

    return std::make_tuple(sortedKeys, sortedValues);
}

}  // namespace ascend_kernel
