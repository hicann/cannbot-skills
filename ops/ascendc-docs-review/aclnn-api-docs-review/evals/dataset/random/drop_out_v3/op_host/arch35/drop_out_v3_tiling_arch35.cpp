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
 * \file drop_out_v3_tiling_arch35.cpp
 * \brief
 */

#include "drop_out_v3_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_ascendc.h"
#include "op_host/tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/fp16.h"
#include "util/bfloat16.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"

namespace optiling {

static constexpr uint16_t INPUT_IDX_X = 0;
static constexpr uint16_t INPUT_IDX_P = 2;
static constexpr uint16_t INPUT_IDX_SEED = 3;
static constexpr uint16_t INPUT_IDX_OFFSET = 4;
static constexpr uint16_t OUTPUT_IDX_Y = 0;
static constexpr int64_t DCACHE_SIZE = 32768;
static constexpr int64_t CORE_ALIGN_SIZE = 256;
static constexpr int64_t ALIGNMENT_32 = 32;
static constexpr int64_t OFFSET_LIMIT = 4;

OpTilingConfig DropOutV3Tiling::BuildOpConfig()
{
    OpTilingConfig config;

    config.inputCheckRules = {
        {INPUT_IDX_X, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}},
        {INPUT_IDX_P, {{ge::DT_DOUBLE, ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}},
        {INPUT_IDX_SEED, {{ge::DT_INT32, ge::DT_INT64}, 1, {}, nullptr}},
        {INPUT_IDX_OFFSET, {{ge::DT_INT64}, 2, {}, nullptr}}};
    config.outputCheckRules = {
        {OUTPUT_IDX_Y, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {}, nullptr}}};

    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& size) {
        auto inputShape = ctx->GetInputShape(INPUT_IDX_X);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, inputShape);
        auto storageShape = inputShape->GetStorageShape();
        size = storageShape.GetShapeSize();
        size = (size <= 0) ? 1 : size; // scalar case
        return ge::GRAPH_SUCCESS;
    };

    config.getSeedAndOffset = [](gert::TilingContext* ctx, int64_t& seed, int64_t& offset) {
        gert::Shape seedShape;
        auto ret = ExtractTensorValue(ctx, INPUT_IDX_SEED, seedShape);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
            OP_LOGE(ctx->GetNodeName(), "get seed value failed"), return ge::GRAPH_FAILED);
        seed = static_cast<int64_t>(seedShape.GetDim(0));
        gert::Shape offsetShape;
        ret = ExtractTensorValue(ctx, INPUT_IDX_OFFSET, offsetShape);
        OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
            OP_LOGE(ctx->GetNodeName(), "get offset value failed"), return ge::GRAPH_FAILED);
        offset = static_cast<int64_t>(offsetShape.GetDim(1));
        OP_CHECK_IF(offset % OFFSET_LIMIT != 0,
            OP_LOGE(ctx->GetNodeName(), "The offset must be a multiple of 4, but got %ld", offset),
            return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };

    config.kernelMode = RandomKernelMode::SIMT;
    config.DcacheSize = DCACHE_SIZE;
    config.isNeedSyncAll = true;
    config.coreAlignSize = CORE_ALIGN_SIZE;
    return config;
}

template <typename T>
static ge::graphStatus GetFirstValueAsFloat(const gert::TilingContext* context, const gert::Tensor* tensor, float& value)
{
    auto data = tensor->GetData<T>();
    OP_CHECK_NULL_WITH_CONTEXT(context, data);
    value = static_cast<float>(data[0]);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus DropOutV3Tiling::UniqueProcess()
{
    simtTilingData_.ubSize = ubSize_;

    auto pTensor = context_->GetRequiredInputTensor(INPUT_IDX_P);
    OP_CHECK_NULL_WITH_CONTEXT(context_, pTensor);
    OP_CHECK_IF(
        pTensor->GetShapeSize() <= 0,
        OP_LOGE(context_->GetNodeName(), "get const shape of prob failed"), return ge::GRAPH_FAILED);
    auto pDescPtr = context_->GetRequiredInputDesc(INPUT_IDX_P);
    OP_CHECK_NULL_WITH_CONTEXT(context_, pDescPtr);
    float pVal = 0.0f;
    ge::graphStatus ret = ge::GRAPH_SUCCESS;
    switch (pDescPtr->GetDataType()) {
        case ge::DT_DOUBLE:  ret = GetFirstValueAsFloat<double>(context_, pTensor, pVal); break;
        case ge::DT_FLOAT:   ret = GetFirstValueAsFloat<float>(context_, pTensor, pVal); break;
        case ge::DT_FLOAT16: ret = GetFirstValueAsFloat<Ops::Base::fp16_t>(context_, pTensor, pVal); break;
        case ge::DT_BF16:    ret = GetFirstValueAsFloat<Ops::Base::bfloat16>(context_, pTensor, pVal); break;
        default:OP_LOGE(context_->GetNodeName(), "Unsupported p dtype"); return ge::GRAPH_FAILED;
    }
    OP_CHECK_IF(ret != ge::GRAPH_SUCCESS,
        OP_LOGE(context_->GetNodeName(), "get prob value failed"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(pVal < 0.0f || pVal > 1.0f,
        OP_LOGE(context_->GetNodeName(), "The value of p has to be between 0 and 1, but current is %f.", pVal),
        return ge::GRAPH_FAILED);
    simtTilingData_.prob = 1.0f - pVal;

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    workspaceSize_ = Ops::Base::CeilAlign(simtTilingData_.outputSize, ALIGNMENT_32) * sizeof(uint8_t) +
                     ascendcPlatform.GetLibApiWorkSpaceSize();

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus Tiling4DropOutV3(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "Tiling4DropOutV3 running DropOutV3 tiling.");
    DropOutV3Tiling tiling(context);
    return tiling.DoTiling();
}

static ge::graphStatus TilingPrepare4DropOutV3(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "DropOutV3");
}

IMPL_OP_OPTILING(DropOutV3)
    .Tiling(Tiling4DropOutV3)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4DropOutV3)
    .TilingInputsDataDependency({INPUT_IDX_P, INPUT_IDX_SEED, INPUT_IDX_OFFSET});
} // namespace optiling
