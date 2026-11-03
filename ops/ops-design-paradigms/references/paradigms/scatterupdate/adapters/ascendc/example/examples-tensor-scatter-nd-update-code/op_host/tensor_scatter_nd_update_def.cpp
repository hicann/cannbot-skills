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
 * \file tensor_scatter_nd_update_def.cpp
 * \brief operator definition for tensor_scatter_nd_update
 */
#include "register/op_def_registry.h"

namespace ops {

// 数据类型组合（SE §5.3，严格对齐 ascend950 binary_config）：
//   var/updates/var_out 三者同 dtype ∈ {bfloat16, bool, float16, float32, int64,
//                                        int8, uint8, float8_e5m2, float8_e4m3fn, hifloat8}（10 种）
//   indices 独立选择 ∈ {int32, int64}（2 种）
//   组合总数 = 10 数据 dtype × 2 索引 dtype = 20。
//   下列 kDataDtypes/kIdxDtypes 为 20 项的 flat 列表，每 2 项对应同一数据 dtype 的 {int32, int64} 组合。
static const std::vector<ge::DataType> kDataDtypes = {
    ge::DT_BF16,          ge::DT_BF16,          // bfloat16 × int32/int64
    ge::DT_BOOL,          ge::DT_BOOL,          // bool × int32/int64
    ge::DT_FLOAT16,       ge::DT_FLOAT16,       // float16 × int32/int64
    ge::DT_FLOAT,         ge::DT_FLOAT,         // float32 × int32/int64
    ge::DT_INT64,         ge::DT_INT64,         // int64 × int32/int64
    ge::DT_INT8,          ge::DT_INT8,          // int8 × int32/int64
    ge::DT_UINT8,         ge::DT_UINT8,         // uint8 × int32/int64
    ge::DT_FLOAT8_E5M2,   ge::DT_FLOAT8_E5M2,   // float8_e5m2 × int32/int64
    ge::DT_FLOAT8_E4M3FN, ge::DT_FLOAT8_E4M3FN, // float8_e4m3fn × int32/int64
    ge::DT_HIFLOAT8,      ge::DT_HIFLOAT8,      // hifloat8 × int32/int64
};

static const std::vector<ge::DataType> kIdxDtypes = {
    ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32,
    ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
    ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
};

static const std::vector<ge::Format> kFormat = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
};

class TensorScatterNdUpdateDef : public OpDef {
public:
    explicit TensorScatterNdUpdateDef(const char* name)
        : OpDef(name)
    {
        this->Input("var")
            .ParamType(REQUIRED)
            .DataType(kDataDtypes)
            .Format(kFormat)
            .UnknownShapeFormat(kFormat)
            .AutoContiguous();
        this->Input("indices")
            .ParamType(REQUIRED)
            .DataType(kIdxDtypes)
            .Format(kFormat)
            .UnknownShapeFormat(kFormat)
            .AutoContiguous();
        this->Input("updates")
            .ParamType(REQUIRED)
            .DataType(kDataDtypes)
            .Format(kFormat)
            .UnknownShapeFormat(kFormat)
            .AutoContiguous();
        this->Output("var")
            .ParamType(REQUIRED)
            .DataType(kDataDtypes)
            .Format(kFormat)
            .UnknownShapeFormat(kFormat)
            .AutoContiguous();

        this->Attr("use_locking").AttrType(OPTIONAL).Bool(false);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(true)
            .ExtendCfgInfo("opFile.value", "tensor_scatter_nd_update_apt");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

using TensorScatterNdUpdate = TensorScatterNdUpdateDef;

OP_ADD(TensorScatterNdUpdate);
} // namespace ops
