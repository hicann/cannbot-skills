/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_def_registry.h"

namespace ops {
static const std::vector<ge::DataType> DataTypeXY1 = {
    ge::DT_INT32,   ge::DT_INT16,   ge::DT_INT8,   ge::DT_UINT32, ge::DT_UINT16, ge::DT_UINT8,
    ge::DT_BF16,    ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_INT64,  ge::DT_UINT64, ge::DT_INT32,
    ge::DT_INT16,   ge::DT_INT8,    ge::DT_UINT32, ge::DT_UINT16, ge::DT_UINT8,  ge::DT_BF16,
    ge::DT_FLOAT16, ge::DT_FLOAT,   ge::DT_INT64,  ge::DT_UINT64};
static const std::vector<ge::Format> format = {
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
    ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};
static const std::vector<ge::DataType> DataTypeY2 = {
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT32,
    ge::DT_INT32, ge::DT_INT32, ge::DT_INT32, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64,
    ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64, ge::DT_INT64};
class Sort : public OpDef {
public:
    explicit Sort(const char* name)
        : OpDef(name)
    {
        this->Input("x").ParamType(REQUIRED).DataType(DataTypeXY1).Format(format).UnknownShapeFormat(format);
        this->Output("y1").ParamType(REQUIRED).DataType(DataTypeXY1).Format(format).UnknownShapeFormat(format);
        this->Output("y2").ParamType(REQUIRED).DataType(DataTypeY2).Format(format).UnknownShapeFormat(format);
        this->Attr("axis").AttrType(OPTIONAL).Int(-1);
        this->Attr("descending").AttrType(OPTIONAL).Bool(false);
        this->Attr("stable").AttrType(OPTIONAL).Bool(false);
        this->Attr("y2_dtype").AttrType(OPTIONAL).Int(ge::DT_INT32);

        OpAICoreConfig aicoreConfig;
        aicoreConfig.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .ExtendCfgInfo("opFile.value", "sort");
        this->AICore().AddConfig("ascend950", aicoreConfig);
        this->AICore().AddConfig("ascend350", aicoreConfig);
    }
};
OP_ADD(Sort);
} // namespace ops
