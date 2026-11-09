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
 * \file arg_common_base_tiling.cpp
 * \brief
 */
#include "arg_common_base_tiling.h"
#include "arg_common_base_tiling_arch35.h"
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "util/platform_util.h"

namespace optiling {
using namespace ge;
using namespace std;

enum class TilingMode {
    TILING_MODE_INVAID = -1,
    TILING_MODE_LAST_LESS_SEGMENT_LEN = 0,
    TILING_MODE_LAST_OVER_DATA_VECTOR,
    TILING_MODE_LAST_LESS_DATA_VECTOR,
    TILING_MODE_LAST_OVER_SEGMENT_LEN,
    TILING_MODE_LAST_AXIS_VCMP,
    TILING_MODE_LAST_LESS_BLOCK,
    TILING_MODE_NLAST_CUT_FIRST_DIM,
    TILING_MODE_NLAST_CUT_FIRST_DIM_AXIS_LESS,
    TILING_MODE_NLAST_FP_ALIGN,
    TILING_MODE_NLAST_CUT_LAST_DIM,
    TILING_MODE_NLAST_CUT_LAST_DIM_AXIS_LESS,
    TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM,
    TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM_LONG_AXIS,
    TILING_MODE_NLAST_CUT_FIRST_AND_LAST_DIM_SHORT_AXIS,
    TILING_MODE_NO_COMPUTE
};

const std::map<ge::DataType, int64_t> kDtypeSize{
    {ge::DT_FLOAT16, 2}, {ge::DT_FLOAT, 4}, {ge::DT_INT32, 4}, {ge::DT_BF16, 4}, // converted to fp32 when calculated
    {ge::DT_INT64, 8},
};

const int64_t MAX_SEGMENT_LEN = static_cast<int64_t>(2048) * 4;
const int64_t MAX_FIRST_DIM_LEN = 8192;
const int64_t MAX_AXIS_SIZE = 2048;
const int64_t VEC_BLOCK_RATIO = 8;
const int64_t MAX_AXIS_SIZE_ONE_TIME_VALUE = 240;
const int64_t MAX_AXIS_SIZE_ONE_TIME = 248;
const int64_t ONE_TIME_NUM = 32;
const int64_t DATA_2 = 2;
const int64_t DATA_3 = 3;
const int64_t INPUT_IDX_AXIS = 1;
const int64_t AXIS_CUT_BASE = 256;
const int64_t HALF_AXIS_CUT_BASE = 128;
const int64_t RESERVED_SIZE = 1024; // ub_size reserved for scalar args and tensors with fixed size
const int64_t MAX_REPEAT_TIMES = 255;
const int64_t BYTE_BIT_RATIO = 8;
const int64_t NANO_BLOCK_SIZE = 16;

struct Argparams {
    int64_t input_dims;
    int64_t axis;
    int64_t data_calc_each_block;
    int64_t data_calc_each_vector;
    int64_t data_move_each_block;
    int64_t data_move_each_vector;
    int64_t out_move_each_block;
    int64_t out_move_each_vector;
    int64_t segment;
    ge::DataType x_in_dtype;
    ge::DataType index_out_dtype;
};

ge::graphStatus ArgOpsTiling(gert::TilingContext* context)
{
    OP_LOGD(context->GetNodeName(), "ArgOpsTiling running begin");
    auto compile_info = reinterpret_cast<const ArgOpsCompileInfo*>(context->GetCompileInfo());
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    return ArgOpsTilingForAscendC(context, compile_info->coreNum, compile_info->ubSize, compile_info->with_value,
                                  compile_info->vRegSize);
}

ge::graphStatus TilingPrepareForArgOpsAscendC(gert::TilingParseContext* context)
{
    OP_LOGD(context->GetNodeName(), "Enter TilingPrepare for ArgWithValue.");
    auto ci = context->GetCompiledInfo<ArgOpsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, ci);
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    ci->coreNum = ascendcPlatform.GetCoreNumAiv();
    if (ci->coreNum <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "coreNum", std::to_string(ci->coreNum).c_str(),
                                              "The value of coreNum must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    ci->ubSize = ubSize;
    if (ci->ubSize <= 0) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(ci->ubSize).c_str(),
                                              "The value of ubSize must be greater than 0.");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForArgOps(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<ArgOpsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    compile_info->with_value = false;
    compile_info->vRegSize = static_cast<uint64_t>(Ops::Base::GetVRegSize(context));
    TilingPrepareForArgOpsAscendC(context);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForArgWithValueOps(gert::TilingParseContext* context)
{
    auto compile_info = context->GetCompiledInfo<ArgOpsCompileInfo>();
    OP_CHECK_NULL_WITH_CONTEXT(context, compile_info);
    compile_info->with_value = true;
    compile_info->vRegSize = static_cast<uint64_t>(Ops::Base::GetVRegSize(context));
    TilingPrepareForArgOpsAscendC(context);
    return ge::GRAPH_SUCCESS;
}
} // namespace optiling
