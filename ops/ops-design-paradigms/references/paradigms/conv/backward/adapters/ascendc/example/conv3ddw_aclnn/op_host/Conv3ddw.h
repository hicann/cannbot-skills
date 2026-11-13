/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONV3DDW_OP_HOST_DEFINITION_H
#define CONV3DDW_OP_HOST_DEFINITION_H

#include "register/op_def_registry.h"

namespace ops {
class Conv3ddw : public OpDef {
public:
    explicit Conv3ddw(const char* name)
        : OpDef(name)
    {
        /* 2 输入类型 × 2 特征格式 × 3 输出格式；形状张量采用 int32 ND。 */
        const std::vector<ge::DataType> types = {ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                                                 ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16,
                                                 ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT16, ge::DT_BF16};
        const std::vector<ge::Format> features = {
            ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW,
            ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC};
        const std::vector<ge::Format> outputs = {
            ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_NDHWC, ge::FORMAT_NDHWC,
            ge::FORMAT_NDHWC, ge::FORMAT_NDHWC, ge::FORMAT_DHWCN, ge::FORMAT_DHWCN, ge::FORMAT_DHWCN, ge::FORMAT_DHWCN};
        this->Input("x").ParamType(REQUIRED).DataType(types).Format(features).UnknownShapeFormat(features);
        this->Input("filter_size")
            .ParamType(REQUIRED)
            .ValueDepend(OPTIONAL)
            .DataType(std::vector<ge::DataType>(12, ge::DT_INT32))
            .Format(std::vector<ge::Format>(12, ge::FORMAT_ND))
            .UnknownShapeFormat(std::vector<ge::Format>(12, ge::FORMAT_ND));
        this->Input("out_backprop").ParamType(REQUIRED).DataType(types).Format(features).UnknownShapeFormat(features);
        this->Output("y")
            .ParamType(REQUIRED)
            .DataType(std::vector<ge::DataType>(12, ge::DT_FLOAT))
            .Format(outputs)
            .UnknownShapeFormat(outputs);
        this->Attr("strides").AttrType(REQUIRED).ListInt();
        this->Attr("pads").AttrType(REQUIRED).ListInt();
        this->Attr("dilations").AttrType(OPTIONAL).ListInt({1, 1, 1, 1, 1});
        this->Attr("groups").AttrType(OPTIONAL).Int(1);
        this->Attr("data_format").AttrType(OPTIONAL).String("NDHWC");
        this->Attr("enable_hf32").AttrType(OPTIONAL).Bool(false);
        this->Attr("output_padding").AttrType(OPTIONAL).ListInt({0, 0, 0, 0, 0});
        OpAICoreConfig config;
        config.DynamicCompileStaticFlag(true)
            .DynamicFormatFlag(true)
            .DynamicRankSupportFlag(false)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false)
            .ExtendCfgInfo("opFile.value", "conv3ddw")
            .ExtendCfgInfo("opInterface.value", "conv3ddw")
            .ExtendCfgInfo("jitCompile.flag", "false");
        this->AICore().AddConfig("ascend950", config);
    }
};
} // namespace ops

#endif // CONV3DDW_OP_HOST_DEFINITION_H
