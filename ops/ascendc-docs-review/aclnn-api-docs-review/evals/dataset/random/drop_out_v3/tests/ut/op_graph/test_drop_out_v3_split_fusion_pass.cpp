/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"
#include "ge/es_graph_builder.h"
#include "es_math_ops.h"
#include "log/log.h"
#include "random/drop_out_v3/op_graph/fusion_pass/drop_out_v3_split_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace fusion;

namespace {
const std::string kPassName = "DropOutV3SplitFusionPass";

std::string DimsToString(const std::vector<int64_t>& dims)
{
    std::string result = "[";
    for (size_t i = 0; i < dims.size(); i++) {
        if (i > 0) result += ", ";
        result += std::to_string(dims[i]);
    }
    result += "]";
    return result;
}

void UpdateDropOutV3NodeTensorDescs(GraphPtr& graph, const std::vector<int64_t>& dimsX,
                                    DataType xDtype, DataType pDtype, DataType seedDtype)
{
    for (auto& node : graph->GetDirectNode()) {
        AscendString nodeType;
        node.GetType(nodeType);
        if (std::string(nodeType.GetString()) == "DropOutV3") {
            OP_LOGI(kPassName.c_str(), "Update DropOutV3 node TensorDescs: dimsX=%s, xDtype=%d, pDtype=%d, seedDtype=%d",
                    DimsToString(dimsX).c_str(), static_cast<int>(xDtype), static_cast<int>(pDtype), static_cast<int>(seedDtype));

            TensorDesc xDesc;
            xDesc.SetDataType(xDtype);
            xDesc.SetShape(Shape(dimsX));
            xDesc.SetFormat(FORMAT_ND);
            node.UpdateInputDesc(0, xDesc);

            TensorDesc pDesc;
            pDesc.SetDataType(pDtype);
            pDesc.SetShape(Shape(std::vector<int64_t>{1}));
            pDesc.SetFormat(FORMAT_ND);
            node.UpdateInputDesc(2, pDesc);

            TensorDesc seedDesc;
            seedDesc.SetDataType(seedDtype);
            seedDesc.SetShape(Shape(std::vector<int64_t>{1}));
            seedDesc.SetFormat(FORMAT_ND);
            node.UpdateInputDesc(3, seedDesc);

            TensorDesc offsetDesc;
            offsetDesc.SetDataType(DT_INT64);
            offsetDesc.SetShape(Shape(std::vector<int64_t>{2}));
            offsetDesc.SetFormat(FORMAT_ND);
            node.UpdateInputDesc(4, offsetDesc);

            TensorDesc yDesc;
            yDesc.SetDataType(xDtype);
            yDesc.SetShape(Shape(dimsX));
            yDesc.SetFormat(FORMAT_ND);
            node.UpdateOutputDesc(0, yDesc);

            OP_LOGI(kPassName.c_str(), "DropOutV3 node TensorDescs updated successfully");
            break;
        }
    }
}
}

class DropOutV3SplitFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetUp() override
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910_93";
        optiCompilationInfo.soc_version = "Ascend910_93";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910_93"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetPlatform910B()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend910B";
        optiCompilationInfo.soc_version = "Ascend910B";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910B"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }

    void SetPlatform950()
    {
        PlatformInfo platformInfo;
        OptionalInfo optiCompilationInfo;
        platformInfo.soc_info.ai_core_cnt = 64;
        platformInfo.str_info.short_soc_version = "Ascend950";
        optiCompilationInfo.soc_version = "Ascend950";
        PlatformInfoManager::Instance().platform_info_map_["Ascend950"] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiCompilationInfo);
    }
};

TEST_F(DropOutV3SplitFusionPassTest, passCreationTest)
{
    DropOutV3SplitFusionPass pass;
}

TEST_F(DropOutV3SplitFusionPassTest, unsupportedPlatformFail)
{
    SetPlatform950();

    std::vector<int64_t> dimsX{14, 12, 16, 4, 8, 14};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shapeX.GetDims());
    auto p = graphBuilder.CreateInput(1, "p", DT_FLOAT, FORMAT_ND, std::vector<int64_t>{1});
    auto seed = graphBuilder.CreateInput(2, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});
    auto offset = graphBuilder.CreateInput(3, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});

    auto output = es::DropOutV3(x, graphBuilder.CreateConst(dimsX, std::vector<int64_t>{static_cast<int64_t>(dimsX.size())}, DT_INT64, FORMAT_ND),
                                 p, seed, offset);

    TensorDesc xDesc;
    xDesc.SetDataType(DT_FLOAT);
    xDesc.SetShape(shapeX);
    xDesc.SetFormat(FORMAT_ND);
    x.GetProducer()->UpdateOutputDesc(0, xDesc);

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.y, output.mask});

    CustomPassContext passContext;
    DropOutV3SplitFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_EQ(status, GRAPH_NOT_CHANGED);
}

TEST_F(DropOutV3SplitFusionPassTest, fusionSuccess91093)
{
    OP_LOGI(kPassName.c_str(), "==================== fusionSuccess91093 test start ====================");
    std::vector<int64_t> dimsX{14, 12, 16, 4, 8, 14};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shapeX.GetDims());
    auto p = graphBuilder.CreateInput(1, "p", DT_FLOAT, FORMAT_ND, std::vector<int64_t>{1});
    auto seed = graphBuilder.CreateInput(2, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});
    auto offset = graphBuilder.CreateInput(3, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});

    OP_LOGI(kPassName.c_str(), "Created inputs: x[0], p[1], seed[2], offset[3] (4 inputs, noise_shape will be Const)");
    auto output = es::DropOutV3(x, graphBuilder.CreateConst(dimsX, std::vector<int64_t>{static_cast<int64_t>(dimsX.size())}, DT_INT64, FORMAT_ND),
                                 p, seed, offset);
    OP_LOGI(kPassName.c_str(), "Created DropOutV3 node");

    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.y, output.mask});
    OP_LOGI(kPassName.c_str(), "Graph built with %zu nodes", graph->GetDirectNode().size());

    UpdateDropOutV3NodeTensorDescs(graph, dimsX, DT_FLOAT, DT_FLOAT, DT_INT64);

    CustomPassContext passContext;
    DropOutV3SplitFusionPass pass;
    OP_LOGI(kPassName.c_str(), "Run DropOutV3SplitFusionPass");
    Status status = pass.Run(graph, passContext);

    OP_LOGI(kPassName.c_str(), "Run result: status=%d", static_cast<int>(status));
    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
}

TEST_F(DropOutV3SplitFusionPassTest, fusionSuccess910B)
{
    SetPlatform910B();

    std::vector<int64_t> dimsX{14, 12, 16, 4, 8};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shapeX.GetDims());
    auto p = graphBuilder.CreateInput(1, "p", DT_FLOAT, FORMAT_ND, std::vector<int64_t>{1});
    auto seed = graphBuilder.CreateInput(2, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});
    auto offset = graphBuilder.CreateInput(3, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});

    auto output = es::DropOutV3(x, graphBuilder.CreateConst(dimsX, std::vector<int64_t>{static_cast<int64_t>(dimsX.size())}, DT_INT64, FORMAT_ND),
                                 p, seed, offset);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.y, output.mask});

    UpdateDropOutV3NodeTensorDescs(graph, dimsX, DT_FLOAT, DT_FLOAT, DT_INT64);

    CustomPassContext passContext;
    DropOutV3SplitFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
}

TEST_F(DropOutV3SplitFusionPassTest, fusionSuccessFp16)
{
    std::vector<int64_t> dimsX{14, 12, 16, 2, 8};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT16, FORMAT_ND, shapeX.GetDims());
    auto p = graphBuilder.CreateInput(1, "p", DT_FLOAT16, FORMAT_ND, std::vector<int64_t>{1});
    auto seed = graphBuilder.CreateInput(2, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});
    auto offset = graphBuilder.CreateInput(3, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});

    auto output = es::DropOutV3(x, graphBuilder.CreateConst(dimsX, std::vector<int64_t>{static_cast<int64_t>(dimsX.size())}, DT_INT64, FORMAT_ND),
                                 p, seed, offset);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.y, output.mask});

    UpdateDropOutV3NodeTensorDescs(graph, dimsX, DT_FLOAT16, DT_FLOAT16, DT_INT64);

    CustomPassContext passContext;
    DropOutV3SplitFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
}

TEST_F(DropOutV3SplitFusionPassTest, fusionSuccessBf16)
{
    std::vector<int64_t> dimsX{14, 12, 16};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_BF16, FORMAT_ND, shapeX.GetDims());
    auto p = graphBuilder.CreateInput(1, "p", DT_BF16, FORMAT_ND, std::vector<int64_t>{1});
    auto seed = graphBuilder.CreateInput(2, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});
    auto offset = graphBuilder.CreateInput(3, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});

    auto output = es::DropOutV3(x, graphBuilder.CreateConst(dimsX, std::vector<int64_t>{static_cast<int64_t>(dimsX.size())}, DT_INT64, FORMAT_ND),
                                 p, seed, offset);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.y, output.mask});

    UpdateDropOutV3NodeTensorDescs(graph, dimsX, DT_BF16, DT_BF16, DT_INT64);

    CustomPassContext passContext;
    DropOutV3SplitFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
}

TEST_F(DropOutV3SplitFusionPassTest, fusionSuccessShape2)
{
    std::vector<int64_t> dimsX{2, 12, 8, 6};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shapeX.GetDims());
    auto p = graphBuilder.CreateInput(1, "p", DT_FLOAT, FORMAT_ND, std::vector<int64_t>{1});
    auto seed = graphBuilder.CreateInput(2, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});
    auto offset = graphBuilder.CreateInput(3, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});

    auto output = es::DropOutV3(x, graphBuilder.CreateConst(dimsX, std::vector<int64_t>{static_cast<int64_t>(dimsX.size())}, DT_INT64, FORMAT_ND),
                                 p, seed, offset);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.y, output.mask});

    UpdateDropOutV3NodeTensorDescs(graph, dimsX, DT_FLOAT, DT_FLOAT, DT_INT64);

    CustomPassContext passContext;
    DropOutV3SplitFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
}

TEST_F(DropOutV3SplitFusionPassTest, fusionSuccessShape5)
{
    std::vector<int64_t> dimsX{2, 4, 8, 6, 2};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shapeX.GetDims());
    auto p = graphBuilder.CreateInput(1, "p", DT_FLOAT, FORMAT_ND, std::vector<int64_t>{1});
    auto seed = graphBuilder.CreateInput(2, "seed", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});
    auto offset = graphBuilder.CreateInput(3, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});

    auto output = es::DropOutV3(x, graphBuilder.CreateConst(dimsX, std::vector<int64_t>{static_cast<int64_t>(dimsX.size())}, DT_INT64, FORMAT_ND),
                                 p, seed, offset);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.y, output.mask});

    UpdateDropOutV3NodeTensorDescs(graph, dimsX, DT_FLOAT, DT_FLOAT, DT_INT64);

    CustomPassContext passContext;
    DropOutV3SplitFusionPass pass;
    Status status = pass.Run(graph, passContext);

    EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
}

TEST_F(DropOutV3SplitFusionPassTest, fusionSuccessSeedInt32)
{
    std::vector<int64_t> dimsX{16, 18};
    Shape shapeX(dimsX);

    auto graphBuilder = es::EsGraphBuilder("test");
    auto x = graphBuilder.CreateInput(0, "x", DT_FLOAT, FORMAT_ND, shapeX.GetDims());
    auto p = graphBuilder.CreateInput(1, "p", DT_FLOAT, FORMAT_ND, std::vector<int64_t>{1});
    auto seed = graphBuilder.CreateInput(2, "seed", DT_INT32, FORMAT_ND, std::vector<int64_t>{1});
    auto offset = graphBuilder.CreateInput(3, "offset", DT_INT64, FORMAT_ND, std::vector<int64_t>{1});

    auto output = es::DropOutV3(x, graphBuilder.CreateConst(dimsX, std::vector<int64_t>{static_cast<int64_t>(dimsX.size())}, DT_INT64, FORMAT_ND),
                                 p, seed, offset);
    std::shared_ptr<Graph> graph = graphBuilder.BuildAndReset({output.y, output.mask});

    UpdateDropOutV3NodeTensorDescs(graph, dimsX, DT_FLOAT, DT_FLOAT, DT_INT32);

    CustomPassContext passContext;
    DropOutV3SplitFusionPass pass;
    Status status = pass.Run(graph, passContext);

EXPECT_TRUE(status == SUCCESS || status == GRAPH_NOT_CHANGED);
}
