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
 * \file drop_out_v3_fusion_pass.cpp
 * \brief drop_out_v3 fusion pass (StatelessDropOutGenMask + DropOutDoMask --> DropOutV3)
 */

#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include "es_math_ops.h"
#include "register/register_custom_pass.h"
#include "ge/ge_utils.h"
#include "platform/platform_info.h"
#include "log/log.h"
#include "drop_out_v3_fusion_pass.h"

namespace ge::fusion {

using namespace ge;
using namespace fe;

namespace {
const std::string kPassName = "DropOutV3FusionPass";
constexpr int64_t kGenMaskCaptureIdx = 0;
constexpr int64_t kDoMaskCaptureIdx = 1;

constexpr size_t kIdxX = 0;
constexpr size_t kIdxShape = 1;
constexpr size_t kIdxProb = 2;
constexpr size_t kIdxSeed = 3;
constexpr size_t kIdxOffset = 4;

constexpr size_t kGenMaskIdxProb = 1;
constexpr size_t kGenMaskIdxSeed = 2;
constexpr size_t kGenMaskIdxOffset = 4;

struct InputParams {
    std::vector<int64_t> xDims;
    std::vector<int64_t> shapeDims;
    std::vector<int64_t> probDims;
    std::vector<int64_t> seedDims;
    std::vector<int64_t> offsetDims;
    Format xFmt;
    Format shapeFmt;
    Format probFmt;
    Format seedFmt;
    Format offsetFmt;
    DataType xDtype;
    DataType probDtype;
    DataType seedDtype;
    DataType offsetDtype;
};

std::vector<int64_t> GetShapeDims(const Shape& shape)
{
    std::vector<int64_t> dims;
    for (size_t i = 0; i < shape.GetDimNum(); i++) {
        dims.push_back(shape.GetDim(i));
    }
    return dims;
}

std::vector<int64_t> GetInputDims(const std::vector<Shape>& inputShapes, size_t idx)
{
    if (idx < inputShapes.size()) {
        return GetShapeDims(inputShapes[idx]);
    }
    return {};
}

Format GetInputFormat(const std::vector<Format>& inputFormats, size_t idx)
{
    return (idx < inputFormats.size()) ? inputFormats[idx] : FORMAT_ND;
}

DataType GetInputDtype(const std::vector<DataType>& inputDtypes, size_t idx)
{
    return (idx < inputDtypes.size()) ? inputDtypes[idx] : DT_FLOAT;
}

void GetInputsInfo(
    const std::vector<SubgraphInput>& subgraphInputs, std::vector<Shape>& inputShapes,
    std::vector<DataType>& inputDtypes, std::vector<Format>& inputFormats)
{
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
        inputDtypes.emplace_back(tensorDesc.GetDataType());
        inputFormats.emplace_back(tensorDesc.GetFormat());
    }
}

Status InferShape(const GraphUniqPtr& replaceGraph, const std::vector<SubgraphInput>& subgraphInputs)
{
    std::vector<Shape> inputShapes;
    for (const auto& subgraphInput : subgraphInputs) {
        auto matchNode = subgraphInput.GetAllInputs().at(0);
        TensorDesc tensorDesc;
        matchNode.node.GetInputDesc(matchNode.index, tensorDesc);
        inputShapes.emplace_back(tensorDesc.GetShape());
    }
    return GeUtils::InferShape(*replaceGraph, inputShapes);
}

void UpdateNodeOutputDesc(es::EsTensorHolder& tensor, DataType dtype, const std::vector<int64_t>& dims, Format fmt)
{
    TensorDesc desc;
    desc.SetDataType(dtype);
    desc.SetShape(Shape(dims));
    desc.SetFormat(fmt);
    tensor.GetProducer()->UpdateOutputDesc(0, desc);
}

bool CheckDtype(DataType dtype, const std::vector<DataType>& validTypes)
{
    return std::find(validTypes.begin(), validTypes.end(), dtype) != validTypes.end();
}

bool GetNodeIo(const std::unique_ptr<MatchResult>& matchResult, int64_t idx, NodeIo& nodeIo)
{
    if (matchResult->GetCapturedTensor(idx, nodeIo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "GetCapturedTensor failed for index %ld", idx);
        return false;
    }
    return true;
}

InputParams GetInputParams(const std::vector<Shape>& inputShapes,
                           const std::vector<DataType>& inputDtypes,
                           const std::vector<Format>& inputFormats)
{
    InputParams params;
    params.xDims = GetInputDims(inputShapes, kIdxX);
    params.shapeDims = GetInputDims(inputShapes, kIdxShape);
    params.probDims = GetInputDims(inputShapes, kIdxProb);
    params.seedDims = GetInputDims(inputShapes, kIdxSeed);
    params.offsetDims = GetInputDims(inputShapes, kIdxOffset);

    params.xFmt = GetInputFormat(inputFormats, kIdxX);
    params.shapeFmt = GetInputFormat(inputFormats, kIdxShape);
    params.probFmt = GetInputFormat(inputFormats, kIdxProb);
    params.seedFmt = GetInputFormat(inputFormats, kIdxSeed);
    params.offsetFmt = GetInputFormat(inputFormats, kIdxOffset);

    params.xDtype = GetInputDtype(inputDtypes, kIdxX);
    params.probDtype = GetInputDtype(inputDtypes, kIdxProb);
    params.seedDtype = GetInputDtype(inputDtypes, kIdxSeed);
    params.offsetDtype = GetInputDtype(inputDtypes, kIdxOffset);

    return params;
}

es::DropOutV3Output CreateDropOutV3Node(es::EsGraphBuilder& builder, const InputParams& params)
{
    auto rX = builder.CreateInput(0, "x", params.xDtype, params.xFmt, params.xDims);
    auto rShape = builder.CreateInput(1, "noise_shape", ge::DT_INT64, params.shapeFmt, params.shapeDims);
    auto rProb = builder.CreateInput(2, "p", params.probDtype, params.probFmt, params.probDims);
    auto rSeed = builder.CreateInput(3, "seed", params.seedDtype, params.seedFmt, params.seedDims);
    auto rOffset = builder.CreateInput(4, "offset", params.offsetDtype, params.offsetFmt, params.offsetDims);

    auto output = es::DropOutV3(rX, rShape, rProb, rSeed, rOffset);

    UpdateNodeOutputDesc(rX, params.xDtype, params.xDims, params.xFmt);
    UpdateNodeOutputDesc(rShape, ge::DT_INT64, params.shapeDims, params.shapeFmt);
    UpdateNodeOutputDesc(rProb, params.probDtype, params.probDims, params.probFmt);
    UpdateNodeOutputDesc(rSeed, params.seedDtype, params.seedDims, params.seedFmt);
    UpdateNodeOutputDesc(rOffset, params.offsetDtype, params.offsetDims, params.offsetFmt);
    UpdateNodeOutputDesc(output.y, params.xDtype, params.xDims, params.xFmt);

    return output;
}
}

std::vector<PatternUniqPtr> DropOutV3FusionPass::Patterns()
{
    OP_LOGI(kPassName.c_str(), "Enter Patterns");
    std::vector<PatternUniqPtr> patternGraphs;

    auto graphBuilder = es::EsGraphBuilder(kPassName.c_str());
    auto shape = graphBuilder.CreateInput(kIdxShape);
    auto prob = graphBuilder.CreateInput(kIdxProb);
    auto seed = graphBuilder.CreateInput(kIdxSeed);
    auto seed1 = graphBuilder.CreateScalar(0);
    auto offset = graphBuilder.CreateInput(kIdxOffset);

    auto mask = es::StatelessDropOutGenMask(shape, prob, seed, seed1, offset);
    auto x = graphBuilder.CreateInput(kIdxX);
    auto y = es::DropOutDoMask(x, mask, prob);

    auto graph = graphBuilder.BuildAndReset({y});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mask.GetProducer(), 0}).CaptureTensor({*y.GetProducer(), 0});
    patternGraphs.emplace_back(std::move(pattern));

    return patternGraphs;
}

bool DropOutV3FusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGI(kPassName.c_str(), "Enter MeetRequirements");
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    if (PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platformInfo, optionalInfo) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Get platform_info failed.");
        return false;
    }
    if (platformInfo.str_info.short_soc_version != "Ascend950") {
        return false;
    }
    return CheckGenMaskNode(matchResult) && CheckDoMaskNode(matchResult);
}

bool DropOutV3FusionPass::CheckGenMaskNode(const std::unique_ptr<MatchResult>& matchResult) const
{
    NodeIo genMaskIo;
    if (!GetNodeIo(matchResult, kGenMaskCaptureIdx, genMaskIo)) {
        return false;
    }

    AscendString nodeTypeStr;
    genMaskIo.node.GetType(nodeTypeStr);
    std::string nodeType(nodeTypeStr.GetString());
    if (nodeType != "StatelessDropOutGenMask") {
        OP_LOGE(kPassName.c_str(), "Expected StatelessDropOutGenMask, got %s", nodeType.c_str());
        return false;
    }

    if (genMaskIo.node.GetInputsSize() != 5) {
        OP_LOGE(kPassName.c_str(), "GenMask input size != 5");
        return false;
    }

    TensorDesc probDesc;
    genMaskIo.node.GetInputDesc(kGenMaskIdxProb, probDesc);
    if (!CheckDtype(probDesc.GetDataType(), {DT_FLOAT, DT_FLOAT16, DT_BF16})) {
        OP_LOGE(kPassName.c_str(), "GenMask prob dtype not supported");
        return false;
    }

    TensorDesc seedDesc;
    genMaskIo.node.GetInputDesc(kGenMaskIdxSeed, seedDesc);
    if (!CheckDtype(seedDesc.GetDataType(), {DT_INT32, DT_INT64})) {
        OP_LOGE(kPassName.c_str(), "GenMask seed dtype not supported");
        return false;
    }

    TensorDesc offsetDesc;
    genMaskIo.node.GetInputDesc(kGenMaskIdxOffset, offsetDesc);
    if (offsetDesc.GetDataType() != DT_INT64) {
        OP_LOGE(kPassName.c_str(), "GenMask offset dtype != DT_INT64");
        return false;
    }

    TensorDesc outputDesc;
    genMaskIo.node.GetOutputDesc(0, outputDesc);
    if (outputDesc.GetDataType() != DT_UINT8) {
        OP_LOGE(kPassName.c_str(), "GenMask output dtype != DT_UINT8");
        return false;
    }
    return true;
}

bool DropOutV3FusionPass::CheckDoMaskNode(const std::unique_ptr<MatchResult>& matchResult) const
{
    NodeIo doMaskIo;
    if (!GetNodeIo(matchResult, kDoMaskCaptureIdx, doMaskIo)) {
        return false;
    }

    AscendString nodeTypeStr;
    doMaskIo.node.GetType(nodeTypeStr);
    std::string nodeType(nodeTypeStr.GetString());
    if (nodeType != "DropOutDoMask") {
        OP_LOGE(kPassName.c_str(), "Expected DropOutDoMask, got %s", nodeType.c_str());
        return false;
    }

    if (doMaskIo.node.GetInputsSize() != 3) {
        OP_LOGE(kPassName.c_str(), "DoMask input size != 3");
        return false;
    }

    TensorDesc inputDesc;
    doMaskIo.node.GetInputDesc(0, inputDesc);
    if (!CheckDtype(inputDesc.GetDataType(), {DT_FLOAT, DT_FLOAT16, DT_BF16})) {
        OP_LOGE(kPassName.c_str(), "DoMask x dtype not supported");
        return false;
    }

    TensorDesc outputDesc;
    doMaskIo.node.GetOutputDesc(0, outputDesc);
    if (!CheckDtype(outputDesc.GetDataType(), {DT_FLOAT, DT_FLOAT16, DT_BF16})) {
        OP_LOGE(kPassName.c_str(), "DoMask y dtype not supported");
        return false;
    }
    return true;
}

GraphUniqPtr DropOutV3FusionPass::Replacement(const std::unique_ptr<MatchResult>& matchResult)
{
    OP_LOGI(kPassName.c_str(), "Enter Replacement");

    std::vector<SubgraphInput> subgraphInputs;
    matchResult->ToSubgraphBoundary()->GetAllInputs(subgraphInputs);

    std::vector<Shape> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<Format> inputFormats;
    GetInputsInfo(subgraphInputs, inputShapes, inputDtypes, inputFormats);

    auto params = GetInputParams(inputShapes, inputDtypes, inputFormats);
    auto builder = es::EsGraphBuilder("replacement");
    auto output = CreateDropOutV3Node(builder, params);

    GraphUniqPtr replaceGraph = builder.BuildAndReset({output.y});
    if (InferShape(replaceGraph, subgraphInputs) != SUCCESS) {
        OP_LOGE(kPassName.c_str(), "Infershape failed.");
        return nullptr;
    }
    return replaceGraph;
}

REG_FUSION_PASS(DropOutV3FusionPass).Stage(CustomPassStage::kCompatibleInherited);
}
