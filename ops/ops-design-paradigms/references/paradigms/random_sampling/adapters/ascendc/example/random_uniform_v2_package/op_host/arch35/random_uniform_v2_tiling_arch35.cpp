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
 * \file random_uniform_v2_tiling_arch35.cpp
 * \brief
 */

#include "platform/platform_infos_def.h"
#include "platform/platform_ascendc.h"
#include "op_common/op_host/util/platform_util.h"
#include "../../../random_common/op_host/arch35/random_tiling_base.h"
#include "exe_graph/runtime/shape.h"
#include "op_host/tiling_base_class.h"
#include "random_uniform_v2_tiling_arch35.h"

using namespace ge;

namespace optiling {

// ========== 仅需配置校验规则+特定字段回调函数 ==========
OpTilingConfig RandomUniformV2TilingArch35::BuildOpConfig()
{
    OpTilingConfig config;
    config.inputCheckRules = {
        // 输入索引:  dtype列表，shapeSize，dim_num
        {0, {{ge::DT_INT32, ge::DT_INT64}, -1, {0, 1}, nullptr}}, // shape
        {1, {{ge::DT_INT64}, 1, {}, nullptr}},                    // offset
    };

    config.outputCheckRules = {
        // 输出索引:  dtype列表，shapeSize，dim_num
        {0, {{ge::DT_FLOAT, ge::DT_FLOAT16, ge::DT_BF16}, -1, {0, 1, 2, 3, 4, 5, 6, 7, 8}, nullptr}}}; // y

    // 获取output_size：输入0(shape)的shapeSize
    config.getOutputSize = [](gert::TilingContext* ctx, int64_t& shapeSize) -> ge::graphStatus {
        return RandomUtils::GetAndCheckOutputSize<0, 0, false>(ctx, shapeSize);
    };

    // 获取key[2]：从attr1(seed) counter[4] attr(seed2)
    config.getKeyAndCounter = [](gert::TilingContext* ctx, uint32_t key[2], uint32_t counter[4]) -> ge::graphStatus {
        constexpr int SEED_INDEX = 1;
        constexpr int SEED_INDEX2 = 2;
        return RandomUtils::GetKeyAndCounter<SEED_INDEX, SEED_INDEX2>(ctx, key, counter);
    };

    config.getBufferNum = [](gert::TilingContext* ctx, int64_t& bufNum) -> ge::graphStatus {
        auto outDesc = ctx->GetOutputDesc(0);
        OP_CHECK_NULL_WITH_CONTEXT(ctx, outDesc);
        auto outDtype = outDesc->GetDataType();
        auto outputDtypeSize = ge::GetSizeByDataType(outDtype);
        static constexpr uint32_t BUFFER_NUM = 2;
        bufNum = sizeof(uint32_t) + outputDtypeSize * BUFFER_NUM;
        return ge::GRAPH_SUCCESS;
    };

    config.isNeedSyncAll = true;
    return config;
}

RandomUniformV2TilingArch35::RandomUniformV2TilingArch35(gert::TilingContext* context)
    : RandomTilingArch35(context, BuildOpConfig())
{}

static ge::graphStatus TilingPrepare4RandomUniformV2Tiling(gert::TilingParseContext* context)
{
    return RandomTilingParseArch35(context, "RandomUniformV2");
}

static ge::graphStatus TilingRandomUniformV2(gert::TilingContext* tilingContext)
{
    OP_LOGD(tilingContext, "Entering TilingRandomUniformV2");
    RandomUniformV2TilingArch35 tilingObj(tilingContext);
    return tilingObj.DoTiling();
}

IMPL_OP_OPTILING(RandomUniformV2)
    .Tiling(TilingRandomUniformV2)
    .TilingParse<RandomOperatorCompileInfo>(TilingPrepare4RandomUniformV2Tiling)
    .TilingInputsDataDependency({0});

} // namespace optiling
