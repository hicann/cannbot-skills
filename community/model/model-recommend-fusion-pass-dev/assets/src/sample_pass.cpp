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
#include <string>
#include <vector>
#include "register/register_custom_pass.h"
#include "all_ops.h"
#include "include/pass_utils.h"

using namespace ge;

namespace {

constexpr char kMatMulV2[] = "MatMulV2";
constexpr char kMatMulV3[] = "MatMulV3";
constexpr char kBatchMatMulV2[] = "BatchMatMulV2";
constexpr char kBatchMatMulV3[] = "BatchMatMulV3";
constexpr char kAdd[] = "Add";
constexpr char kConst[] = "Const";
constexpr char kConstant[] = "Constant";
constexpr int32_t kMatMulOutPort = 0;
constexpr int32_t kAddOutPort = 0;
constexpr int32_t kBiasPort = 1;
constexpr int32_t kInputPorts = 2;

struct Match {
    GNode matmul;
    GNode add;
    GNode bias_const;
    int32_t bias_const_port = -1;
};

bool IsConstNode(const GNode& node)
{
    return pass_utils::IsNodeType(node, kConst) || pass_utils::IsNodeType(node, kConstant);
}

// 匹配：MatMul* 的下游唯一 Add，且 Add 的另一个输入是一维常量 bias
bool MatchMatMulAdd(const GNode& matmul, Match& match)
{
    auto out_pairs = matmul.GetOutDataNodesAndPortIndexs(kMatMulOutPort);
    if (out_pairs.size() != 1 || out_pairs[0].first == nullptr) {
        return false;
    }
    GNode add = *out_pairs[0].first;
    if (!pass_utils::IsNodeType(add, kAdd)) {
        return false;
    }

    const std::string matmul_name = pass_utils::GetNodeName(matmul);
    GNode bias_const;
    int32_t bias_const_port = -1;
    for (int32_t port = 0; port < kInputPorts; ++port) {
        auto in_pair = add.GetInDataNodesAndPortIndexs(port);
        if (in_pair.first == nullptr) {
            return false;
        }
        if (pass_utils::GetNodeName(*in_pair.first) == matmul_name) {
            continue;
        }
        if (!IsConstNode(*in_pair.first)) {
            return false;
        }
        TensorDesc const_desc;
        if (in_pair.first->GetOutputDesc(in_pair.second, const_desc) != GRAPH_SUCCESS) {
            return false;
        }
        if (const_desc.GetShape().GetDimNum() != 1) {
            return false;
        }
        bias_const = *in_pair.first;
        bias_const_port = in_pair.second;
    }
    if (bias_const_port < 0) {
        return false;
    }

    match.matmul = matmul;
    match.add = add;
    match.bias_const = bias_const;
    match.bias_const_port = bias_const_port;
    return true;
}

graphStatus RewriteAsBiasAdd(GraphPtr& graph, Match& match)
{
    TensorDesc matmul_desc;
    TensorDesc const_desc;
    TensorDesc add_out_desc;
    if (match.matmul.GetOutputDesc(kMatMulOutPort, matmul_desc) != GRAPH_SUCCESS ||
        match.bias_const.GetOutputDesc(match.bias_const_port, const_desc) != GRAPH_SUCCESS ||
        match.add.GetOutputDesc(kAddOutPort, add_out_desc) != GRAPH_SUCCESS) {
        std::cout << "[SamplePass] get desc failed, skip" << std::endl;
        return GRAPH_FAILED;
    }

    op::BiasAdd bias_add((pass_utils::GetNodeName(match.add) + "_BiasAdd").c_str());
    GNode bias_add_node = graph->AddNodeByOp(bias_add);
    bias_add_node.UpdateInputDesc(0, matmul_desc);
    bias_add_node.UpdateInputDesc(kBiasPort, const_desc);
    if (bias_add.InferShapeAndType() != GRAPH_SUCCESS) {
        std::cout << "[SamplePass] infer shape failed, skip" << std::endl;
        return GRAPH_FAILED;
    }
    bias_add_node.UpdateOutputDesc(0, add_out_desc);

    if (graph->AddDataEdge(match.matmul, kMatMulOutPort, bias_add_node, 0) != GRAPH_SUCCESS ||
        graph->AddDataEdge(match.bias_const, match.bias_const_port, bias_add_node, kBiasPort) != GRAPH_SUCCESS) {
        std::cout << "[SamplePass] add input edge failed" << std::endl;
        return GRAPH_FAILED;
    }

    auto consumers = match.add.GetOutDataNodesAndPortIndexs(kAddOutPort);
    for (auto& [out_node, out_port] : consumers) {
        if (out_node == nullptr) {
            continue;
        }
        if (graph->RemoveEdge(match.add, kAddOutPort, *out_node, out_port) != GRAPH_SUCCESS ||
            graph->AddDataEdge(bias_add_node, kAddOutPort, *out_node, out_port) != GRAPH_SUCCESS) {
            std::cout << "[SamplePass] reconnect consumer failed" << std::endl;
            return GRAPH_FAILED;
        }
    }

    const std::string add_name = pass_utils::GetNodeName(match.add);
    GNode add_copy = match.add;
    if (graph->RemoveNode(add_copy) != GRAPH_SUCCESS) {
        std::cout << "[SamplePass] remove old Add failed" << std::endl;
        return GRAPH_FAILED;
    }

    std::cout << "[SamplePass] fused " << pass_utils::GetNodeName(match.matmul) << " + "
              << add_name << " -> " << pass_utils::GetNodeName(bias_add_node) << std::endl;
    return GRAPH_SUCCESS;
}

graphStatus SamplePass(GraphPtr& graph, CustomPassContext& context)
{
    std::cout << "---- SamplePass begin ----" << std::endl;
    std::vector<GNode> matmul_nodes;
    for (const GNode& node : graph->GetAllNodes()) {
        if (pass_utils::IsAnyNodeType(node, {kMatMulV2, kMatMulV3, kBatchMatMulV2, kBatchMatMulV3})) {
            matmul_nodes.emplace_back(node);
        }
    }

    int64_t fused_count = 0;
    for (GNode& matmul : matmul_nodes) {
        Match match;
        if (!MatchMatMulAdd(matmul, match)) {
            continue;
        }
        if (RewriteAsBiasAdd(graph, match) == GRAPH_SUCCESS) {
            ++fused_count;
        }
    }
    std::cout << "---- SamplePass end, affected count: " << fused_count << " ----" << std::endl;
    return GRAPH_SUCCESS;
}

} // namespace

REGISTER_CUSTOM_PASS("SamplePass").CustomPassFn(SamplePass).Stage(CustomPassStage::kAfterInferShape);
