/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file drop_out_v3_split_fusion_pass.cpp
 * \brief DropOutV3 split fusion pass: DropOutV3 -> StatelessDropOutGenMask + DropOutDoMask
 */

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "es_math_ops.h"
#include "register/register_custom_pass.h"
#include "ge/ge_utils.h"
#include "ge/fusion/graph_rewriter.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "drop_out_v3_split_fusion_pass.h"

namespace ge::fusion {

using namespace ge;
using namespace fe;

namespace {
const std::string kPassName = "DropOutV3SplitFusionPass";

constexpr size_t kIdxX = 0;
constexpr size_t kIdxP = 2;
constexpr size_t kIdxSeed = 3;
constexpr size_t kIdxOffset = 4;

constexpr int32_t kMaxDimBound = 8;

bool CheckDtype(DataType dtype, const std::vector<DataType>& validTypes)
{
    return std::find(validTypes.begin(), validTypes.end(), dtype) != validTypes.end();
}

std::vector<int64_t> GetDimsFromShape(const Shape& shape)
{
    std::vector<int64_t> dims;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        dims.push_back(shape.GetDim(i));
    }
    return dims;
}
}

bool DropOutV3SplitFusionPass::CheckPlatform() const
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get platform_info failed.");
        return false;
    }
    const std::string soc = platformInfo.str_info.short_soc_version;
    if (soc != "Ascend910_93" && soc != "Ascend910B") {
        return false;
    }
    return true;
}

bool DropOutV3SplitFusionPass::CheckDtypes(const GNode &node) const
{
    TensorDesc xDesc;
    node.GetInputDesc(kIdxX, xDesc);
    DataType xDtype = xDesc.GetDataType();
    if (!CheckDtype(xDtype, {DT_FLOAT, DT_FLOAT16, DT_BF16})) {
        OP_LOGE(kPassName.c_str(), "x dtype only support float/float16/bf16, actual: %d", static_cast<int>(xDtype));
        return false;
    }

    TensorDesc pDesc;
    node.GetInputDesc(kIdxP, pDesc);
    DataType pDtype = pDesc.GetDataType();
    if (!CheckDtype(pDtype, {DT_FLOAT, DT_FLOAT16, DT_BF16})) {
        OP_LOGE(kPassName.c_str(), "p dtype only support float/float16/bf16, actual: %d", static_cast<int>(pDtype));
        return false;
    }

    TensorDesc seedDesc;
    node.GetInputDesc(kIdxSeed, seedDesc);
    DataType seedDtype = seedDesc.GetDataType();
    if (!CheckDtype(seedDtype, {DT_INT32, DT_INT64})) {
        OP_LOGE(kPassName.c_str(), "seed dtype only support int32/int64, actual: %d", static_cast<int>(seedDtype));
        return false;
    }

    TensorDesc yDesc;
    node.GetOutputDesc(0, yDesc);
    DataType yDtype = yDesc.GetDataType();
    if (!CheckDtype(yDtype, {DT_FLOAT, DT_FLOAT16, DT_BF16})) {
        OP_LOGE(kPassName.c_str(), "y dtype only support float/float16/bf16, actual: %d", static_cast<int>(yDtype));
        return false;
    }

    if (xDtype != yDtype) {
        OP_LOGE(kPassName.c_str(), "x dtype should same with y dtype, x: %d, y: %d",
                static_cast<int>(xDtype), static_cast<int>(yDtype));
        return false;
    }

    return true;
}

bool DropOutV3SplitFusionPass::CheckNode(const GNode &node) const
{
    AscendString nodeType;
    if (node.GetType(nodeType) != SUCCESS) {
        return false;
    }
    if (std::string(nodeType.GetString()) != "DropOutV3") {
        return false;
    }

    size_t inputSize = node.GetInputsSize();
    if (inputSize != 5) {
        OP_LOGE(kPassName.c_str(), "DropOutV3 input size != 5, actual: %zu", inputSize);
        return false;
    }

    if (!CheckDtypes(node)) {
        return false;
    }

    TensorDesc xDesc;
    node.GetInputDesc(kIdxX, xDesc);
    auto xDimNum = static_cast<int32_t>(xDesc.GetShape().GetDimNum());
    if (xDimNum > kMaxDimBound || xDimNum < 0) {
        OP_LOGE(kPassName.c_str(), "x dim should be [0~8], actual: %d", xDimNum);
        return false;
    }

    return true;
}

InputInfo DropOutV3SplitFusionPass::GetInputInfo(const GNode &node) const
{
    InputInfo info;
    TensorDesc xDesc;
    node.GetInputDesc(kIdxX, xDesc);
    TensorDesc pDesc;
    node.GetInputDesc(kIdxP, pDesc);
    TensorDesc seedDesc;
    node.GetInputDesc(kIdxSeed, seedDesc);
    TensorDesc offsetDesc;
    node.GetInputDesc(kIdxOffset, offsetDesc);

    info.xDims = GetDimsFromShape(xDesc.GetShape());
    info.pDims = GetDimsFromShape(pDesc.GetShape());
    info.seedDims = GetDimsFromShape(seedDesc.GetShape());
    info.offsetDims = GetDimsFromShape(offsetDesc.GetShape());
    info.noiseShapeDims = {static_cast<int64_t>(xDesc.GetShape().GetDimNum())};

    info.xDtype = xDesc.GetDataType();
    info.pDtype = pDesc.GetDataType();
    info.seedDtype = seedDesc.GetDataType();
    info.offsetDtype = offsetDesc.GetDataType();
    info.fmt = xDesc.GetFormat();

    return info;
}

void DropOutV3SplitFusionPass::UpdateTensorDescs(const InputInfo &info,
                                                 const es::EsTensorHolder &rX,
                                                 const es::EsTensorHolder &rProb,
                                                 const es::EsTensorHolder &rSeed,
                                                 const es::EsTensorHolder &rOffset,
                                                 const es::EsTensorHolder &genMask,
                                                 const es::EsTensorHolder &doMask,
                                                 const es::EsTensorHolder &rShapeConst,
                                                 const es::EsTensorHolder &rSeed1) const
{
    TensorDesc shapeConstDesc(Shape(info.noiseShapeDims), FORMAT_ND, DT_INT64);
    rShapeConst.GetProducer()->UpdateOutputDesc(0, shapeConstDesc);

    TensorDesc seed1ConstDesc(Shape({1}), FORMAT_ND, DT_INT64);
    rSeed1.GetProducer()->UpdateOutputDesc(0, seed1ConstDesc);

    TensorDesc inputDesc(Shape(info.xDims), info.fmt, info.xDtype);
    rX.GetProducer()->UpdateOutputDesc(0, inputDesc);

    TensorDesc probDesc(Shape(info.pDims), info.fmt, info.pDtype);
    rProb.GetProducer()->UpdateOutputDesc(0, probDesc);

    TensorDesc seedDescOut(Shape(info.seedDims), FORMAT_ND, info.seedDtype);
    rSeed.GetProducer()->UpdateOutputDesc(0, seedDescOut);

    TensorDesc offsetDescOut(Shape(info.offsetDims), FORMAT_ND, info.offsetDtype);
    rOffset.GetProducer()->UpdateOutputDesc(0, offsetDescOut);

    TensorDesc genMaskDesc(Shape(info.xDims), info.fmt, DT_UINT8);
    genMask.GetProducer()->UpdateOutputDesc(0, genMaskDesc);

    TensorDesc doMaskDesc(Shape(info.xDims), info.fmt, info.xDtype);
    doMask.GetProducer()->UpdateOutputDesc(0, doMaskDesc);

    TensorDesc shapeInputDesc(Shape(info.noiseShapeDims), FORMAT_ND, DT_INT64);
    genMask.GetProducer()->UpdateInputDesc(0, shapeInputDesc);
    genMask.GetProducer()->UpdateInputDesc(1, probDesc);
    genMask.GetProducer()->UpdateInputDesc(2, seedDescOut);
    TensorDesc seed1Desc(Shape({1}), FORMAT_ND, DT_INT64);
    genMask.GetProducer()->UpdateInputDesc(3, seed1Desc);
    genMask.GetProducer()->UpdateInputDesc(4, offsetDescOut);

    doMask.GetProducer()->UpdateInputDesc(0, inputDesc);
    doMask.GetProducer()->UpdateInputDesc(1, genMaskDesc);
    doMask.GetProducer()->UpdateInputDesc(2, probDesc);
}

GraphUniqPtr DropOutV3SplitFusionPass::CreateReplacement(const GNode &node)
{
    InputInfo info = GetInputInfo(node);
    auto builder = es::EsGraphBuilder("replacement");

    auto rX = builder.CreateInput(0, "x", info.xDtype, info.fmt, info.xDims);
    [[maybe_unused]] auto rNoiseShape = builder.CreateInput(1, "noise_shape", DT_INT64, FORMAT_ND, info.noiseShapeDims);
    auto rProb = builder.CreateInput(2, "prob", info.pDtype, info.fmt, info.pDims);
    auto rSeed = builder.CreateInput(3, "seed", info.seedDtype, FORMAT_ND, info.seedDims);
    auto rOffset = builder.CreateInput(4, "offset", info.offsetDtype, FORMAT_ND, info.offsetDims);

    std::vector<int64_t> shapeValue(info.xDims.size(), 0);
    auto rShapeConst = builder.CreateConst(shapeValue, info.noiseShapeDims, DT_INT64, FORMAT_ND);
    std::vector<int64_t> seed1Value = {0};
    std::vector<int64_t> seed1Dims = {1};
    auto rSeed1 = builder.CreateConst(seed1Value, seed1Dims, DT_INT64, FORMAT_ND);

    auto genMask = es::StatelessDropOutGenMask(rShapeConst, rProb, rSeed, rSeed1, rOffset);
    auto doMask = es::DropOutDoMask(rX, genMask, rProb);

    UpdateTensorDescs(info, rX, rProb, rSeed, rOffset, genMask, doMask, rShapeConst, rSeed1);

    std::vector<es::EsTensorHolder> outputs;
    outputs.emplace_back(doMask);
    outputs.emplace_back(genMask);
    return builder.BuildAndReset(outputs);
}

std::unique_ptr<SubgraphBoundary> DropOutV3SplitFusionPass::ConstructBoundary(const GNode &node)
{
    auto boundary = std::make_unique<SubgraphBoundary>();
    for (size_t idx = 0; idx < node.GetInputsSize(); ++idx) {
        SubgraphInput subgraphInput;
        subgraphInput.AddInput({node, static_cast<int64_t>(idx)});
        if (boundary->AddInput(idx, std::move(subgraphInput)) != SUCCESS) {
            OP_LOGE(kPassName.c_str(), "AddInput failed for idx %zu", idx);
            return nullptr;
        }
    }

    SubgraphOutput output0({node, 0});
    if (boundary->AddOutput(0, std::move(output0)) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "AddOutput failed for output 0");
        return nullptr;
    }

    if (node.GetOutputsSize() > 1) {
        SubgraphOutput output1({node, 1});
        if (boundary->AddOutput(1, std::move(output1)) != SUCCESS) {
            OP_LOGE(kPassName.c_str(), "AddOutput failed for output 1");
            return nullptr;
        }
    }
    return boundary;
}

Status DropOutV3SplitFusionPass::Run(GraphPtr &graph, [[maybe_unused]] CustomPassContext &passContext)
{
    OP_LOGI(kPassName.c_str(), "Enter DropOutV3SplitFusionPass");
    if (!CheckPlatform()) {
        return GRAPH_NOT_CHANGED;
    }

    std::vector<GNode> dropOutV3Nodes;
    for (auto &node : graph->GetDirectNode()) {
        if (CheckNode(node)) {
            dropOutV3Nodes.emplace_back(node);
        }
    }
    if (dropOutV3Nodes.empty()) {
        return GRAPH_NOT_CHANGED;
    }

    Graph originGraph = *graph;
    for (auto &node : dropOutV3Nodes) {
        auto replacement = CreateReplacement(node);
        if (!replacement) {
            AscendString nodeName;
            node.GetName(nodeName);
            OP_LOGE(kPassName.c_str(), "CreateReplacement failed for node %s", nodeName.GetString());
            *graph = originGraph;
            return FAILED;
        }

        auto boundary = ConstructBoundary(node);
        if (!boundary) {
            AscendString nodeName;
            node.GetName(nodeName);
            OP_LOGE(kPassName.c_str(), "ConstructBoundary failed for node %s", nodeName.GetString());
            *graph = originGraph;
            return FAILED;
        }

        Status replaceStatus = SubgraphRewriter::Replace(*boundary, *replacement);
        if (replaceStatus != SUCCESS) {
            AscendString nodeName;
            node.GetName(nodeName);
            OP_LOGE(kPassName.c_str(), "SubgraphRewriter::Replace failed for node %s, status=%d",
                    nodeName.GetString(), static_cast<int>(replaceStatus));
            *graph = originGraph;
            return FAILED;
        }
    }

    OP_LOGI(kPassName.c_str(), "DropOutV3SplitFusionPass completed, fused %zu nodes", dropOutV3Nodes.size());
    return SUCCESS;
}

REG_FUSION_PASS(DropOutV3SplitFusionPass).Stage(CustomPassStage::kCompatibleInherited);
}
