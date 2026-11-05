/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sim_thread_exponential_tiling.cpp
 * \brief
 */

#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include "../../../../op_host/arch35/drop_out_v3_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

class DropOutV3TilingTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "DropOutV3TilingTest  SetUp" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "DropOutV3TilingTest  TearDown" << std::endl;
    }
};

TEST_F(DropOutV3TilingTest, drop_out_v3_tiling_ascendc_float_01)
{
    optiling::RandomOperatorCompileInfo compileInfo = {64, 262144};
    int64_t noise_value = 10;
    float p_value = 0.5;
    int64_t seed_value = 8;
    int64_t offset_value[2] = {0, 4};
    gert::TilingContextPara::TensorDescription noise({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &noise_value);
    gert::TilingContextPara::TensorDescription p({{1}, {1}}, ge::DT_FLOAT, ge::FORMAT_ND, true, &p_value);
    gert::TilingContextPara::TensorDescription seed({{1}, {1}}, ge::DT_INT64, ge::FORMAT_ND, true, &seed_value);
    gert::TilingContextPara::TensorDescription offset({{2}, {2}}, ge::DT_INT64, ge::FORMAT_ND, true, &offset_value);
    gert::TilingContextPara tilingContextPara(
        "DropOutV3", {{{{10, 13, 22, 43}, {10, 13, 22, 43}}, ge::DT_FLOAT, ge::FORMAT_ND}, noise, p, seed, offset},
        {{{{10, 13, 22, 43}, {10, 13, 22, 43}}, ge::DT_FLOAT, ge::FORMAT_ND},
         {{{15376}, {15376}}, ge::DT_INT8, ge::FORMAT_ND}},
        &compileInfo);
    uint64_t expectTilingKey = 100;
    string expectTilingData = "61 122980 8 4 229376 0 1056964608 ";
    std::vector<size_t> expectWorkspaces = {16900224};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
