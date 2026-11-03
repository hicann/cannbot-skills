/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <cstring>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../../random_common/op_kernel/arch35/random_unified_tiling_data_arch35.h"

extern "C" __global__ __aicore__ void random_uniform_v2(GM_ADDR shape, GM_ADDR inOffset, GM_ADDR y, GM_ADDR outOffset,
                                                        GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint32_t kNumBlocks = 1;
constexpr uint64_t kTilingKey = 100;
constexpr int64_t kElementCount = 256;
constexpr int64_t kExpectedOffset = kElementCount * 256;

inline size_t Align32(size_t size)
{
    return (size + 31U) / 32U * 32U;
}
} // namespace

class TestRandomUniformV2 : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "TestRandomUniformV2 SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "TestRandomUniformV2 TearDown" << std::endl;
    }
};

TEST_F(TestRandomUniformV2, smoke_float)
{
    auto* shape = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(2 * sizeof(int32_t))));
    auto* inOffset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* y = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(kElementCount * sizeof(float))));
    auto* outOffset = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(int64_t))));
    auto* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(1024 * 1024)));
    auto* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(Align32(sizeof(RandomUnifiedTilingDataStruct))));

    std::memset(y, 0, kElementCount * sizeof(float));
    std::memset(tiling, 0, sizeof(RandomUnifiedTilingDataStruct));
    reinterpret_cast<int32_t*>(shape)[0] = 16;
    reinterpret_cast<int32_t*>(shape)[1] = 16;
    reinterpret_cast<int64_t*>(inOffset)[0] = 0;
    reinterpret_cast<int64_t*>(outOffset)[0] = 0;

    auto* tilingData = reinterpret_cast<RandomUnifiedTilingDataStruct*>(tiling);
    tilingData->usedCoreNum = kNumBlocks;
    tilingData->normalCoreProNum = kElementCount;
    tilingData->tailCoreProNum = kElementCount;
    tilingData->singleBufferSize = kElementCount;
    tilingData->key[0] = 10;
    tilingData->key[1] = 0;
    tilingData->counter[0] = 5;
    tilingData->counter[1] = 0;
    tilingData->counter[2] = 0;
    tilingData->counter[3] = 0;
    tilingData->outputSize = kElementCount;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(kTilingKey);
    ICPU_RUN_KF(random_uniform_v2, kNumBlocks, shape, inOffset, y, outOffset, workspace, tiling);

    EXPECT_EQ(reinterpret_cast<int64_t*>(outOffset)[0], kExpectedOffset);

    AscendC::GmFree(shape);
    AscendC::GmFree(inOffset);
    AscendC::GmFree(y);
    AscendC::GmFree(outOffset);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
