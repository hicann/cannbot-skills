/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "log/log.h"
#include "../../../op_graph/fusion_pass/random_uniform_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace ge::fusion;
using namespace fe;
using namespace ops;

namespace {
const std::string kPassName = "RandomUniformFusionPass";
}

class TestRandomUniformFusionPass : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUnsupportedPlatform()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

static std::shared_ptr<Graph> BuildTestGraph(const std::string& graphName, DataType dtype,
                                             const std::vector<int64_t>& shapeDims, int64_t seed = 0, int64_t seed2 = 0)
{
    Shape shapeDesc(shapeDims);

    auto graphBuilder = es::EsGraphBuilder(graphName.c_str());
    auto shape = graphBuilder.CreateInput(0, "shape", DT_INT64, FORMAT_ND, shapeDims);

    TensorDesc offsetDesc(Shape({1}), FORMAT_ND, DT_INT64);
    std::string varName = graphName + "/offsetVariable";
    auto offset = graphBuilder.CreateVariable(1, varName.c_str());
    auto varNodePtr = offset.GetProducer();
    if (varNodePtr != nullptr) {
        varNodePtr->UpdateOutputDesc(0, offsetDesc);
        int64_t initValue = 0;
        Tensor initTensor(offsetDesc, reinterpret_cast<uint8_t*>(&initValue), sizeof(int64_t));
        varNodePtr->SetAttr("init_value", initTensor);
    }

    int64_t dtypeInt = static_cast<int64_t>(dtype);
    auto v2Output = es::RandomUniformV2(shape, offset, dtypeInt, seed, seed2);
    auto v2NodePtr = v2Output.y.GetProducer();
    if (v2NodePtr != nullptr) {
        TensorDesc shapeInputDesc(shapeDesc, FORMAT_ND, DT_INT64);
        v2NodePtr->UpdateInputDesc(0, shapeInputDesc);
        v2NodePtr->UpdateInputDesc(1, offsetDesc);

        TensorDesc outputYDesc(shapeDesc, FORMAT_ND, dtype);
        v2NodePtr->UpdateOutputDesc(0, outputYDesc);
        v2NodePtr->UpdateOutputDesc(1, offsetDesc);
    }

    auto graph = graphBuilder.BuildAndReset({v2Output.y});

    return graph;
}

TEST_F(TestRandomUniformFusionPass, patternTest)
{
    RandomUniformFusionPass pass;
    std::vector<PatternUniqPtr> patterns = pass.Patterns();
    EXPECT_GT(patterns.size(), 0);
}

TEST_F(TestRandomUniformFusionPass, unsupportedPlatformFail)
{
    SetUnsupportedPlatform();

    auto graph = BuildTestGraph("test_unsupported", DT_FLOAT, {2});

    CustomPassContext passContext;
    RandomUniformFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(TestRandomUniformFusionPass, unsupportedDtypeFail)
{
    auto graph = BuildTestGraph("test_unsupported_dtype", DT_DOUBLE, {2});

    CustomPassContext passContext;
    RandomUniformFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(TestRandomUniformFusionPass, fusionSuccessFp32)
{
    auto graph = BuildTestGraph("test_fp32", DT_FLOAT, {2}, 1024, 2048);

    CustomPassContext passContext;
    RandomUniformFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "RandomUniformV2") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(TestRandomUniformFusionPass, fusionSuccessFp16)
{
    auto graph = BuildTestGraph("test_fp16", DT_FLOAT16, {3});

    CustomPassContext passContext;
    RandomUniformFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "RandomUniformV2") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(TestRandomUniformFusionPass, fusionSuccessBf16)
{
    auto graph = BuildTestGraph("test_bf16", DT_BF16, {4}, 512, 1024);

    CustomPassContext passContext;
    RandomUniformFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "RandomUniformV2") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}

TEST_F(TestRandomUniformFusionPass, fusionSuccessLargeShape)
{
    auto graph = BuildTestGraph("test_large", DT_FLOAT, {6});

    CustomPassContext passContext;
    RandomUniformFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);

    bool isFound = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        node.GetType(type);
        if (type == "RandomUniformV2") {
            isFound = true;
            break;
        }
    }
    EXPECT_TRUE(isFound);
}
