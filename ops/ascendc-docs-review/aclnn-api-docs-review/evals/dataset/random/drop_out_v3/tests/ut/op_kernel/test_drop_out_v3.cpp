/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "random/random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"

using namespace std;


__global__ __aicore__ void drop_out_v3(
    GM_ADDR x, GM_ADDR noise_shape, GM_ADDR p, GM_ADDR seed, GM_ADDR offset,
    GM_ADDR y, GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling);

class drop_out_v3_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "drop_out_v3_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "drop_out_v3_test TearDown\n" << endl;
    }
};

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
    int64_t shapeSize = 1;
    for(auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

TEST_F(drop_out_v3_test, test_case_b32_1)
{
    uint64_t tilingKey = 100;
    uint32_t numBlocks = 1;
    int64_t typeSize = 4;
    std::vector<int64_t> x_shape = {10, 15, 20};
    std::vector<int64_t> noise_shape = {3};
    std::vector<int64_t> p_shape = {1};
    std::vector<int64_t> seed_shape = {1};
    std::vector<int64_t> offset_shape = {4};
    std::vector<int64_t> y_shape = {10, 15, 20};
    std::vector<int64_t> mask_shape = {375};

    size_t param1FileSize = GetShapeSize(x_shape) * sizeof(float);
    size_t param2FileSize = GetShapeSize(noise_shape) * sizeof(int64_t);
    size_t param3FileSize = GetShapeSize(p_shape) * sizeof(float) * typeSize;
    size_t param4FileSize = GetShapeSize(seed_shape) * sizeof(int64_t);
    size_t param5FileSize = GetShapeSize(offset_shape) * sizeof(int64_t);
    size_t param6FileSize = GetShapeSize(y_shape) * sizeof(float);
    size_t param7FileSize = GetShapeSize(mask_shape) * sizeof(uint8_t);

    size_t workspaceFileSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);

    uint8_t *param1 = (uint8_t *)AscendC::GmAlloc((param1FileSize + 31) / 32 * 32);
    uint8_t *param2 = (uint8_t *)AscendC::GmAlloc((param2FileSize + 31) / 32 * 32);
    uint8_t *param3 = (uint8_t *)AscendC::GmAlloc((param3FileSize + 31) / 32 * 32);
    uint8_t *param4 = (uint8_t *)AscendC::GmAlloc((param4FileSize + 31) / 32 * 32);
    uint8_t *param5 = (uint8_t *)AscendC::GmAlloc((param5FileSize + 31) / 32 * 32);
    uint8_t *param6 = (uint8_t *)AscendC::GmAlloc((param6FileSize + 31) / 32 * 32);
    uint8_t *param7 = (uint8_t *)AscendC::GmAlloc((param7FileSize + 31) / 32 * 32);

    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(sizeof(RandomUnifiedSimtTilingDataStruct));
    RandomUnifiedSimtTilingDataStruct *tilingData = reinterpret_cast<RandomUnifiedSimtTilingDataStruct *>(tiling);
    tilingData->usedCoreNum = numBlocks;
    tilingData->outputSize = GetShapeSize(x_shape);
    tilingData->seed = 12345;
    tilingData->offset = 0;
    tilingData->ubSize = 196608;
    tilingData->prob = 0.5f;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(drop_out_v3, numBlocks, param1, param2, param3, param4, param5, param6, param7, workspace, tiling);

    AscendC::GmFree((void *)param1);
    AscendC::GmFree((void *)param2);
    AscendC::GmFree((void *)param3);
    AscendC::GmFree((void *)param4);
    AscendC::GmFree((void *)param5);
    AscendC::GmFree((void *)param6);
    AscendC::GmFree((void *)param7);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
}
