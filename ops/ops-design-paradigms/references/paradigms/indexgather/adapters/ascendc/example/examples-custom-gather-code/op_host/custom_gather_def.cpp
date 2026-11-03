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
 * \file custom_gather_def.cpp
 * \brief operator definition for custom_gather
 *
 * dtype 组合（SE 5.7，基于 torch.gather 语义自行设计）：input/output 6 种
 * × index 2 种 = 12 组，各 DataType/Format 数组等长（12 项一一映射，op_def.md）。
 * DynamicCompileStaticFlag(true) 激活 DTYPE_INPUT / DTYPE_INDEX 宏自动实例化
 * （tiling key 不枚举 dtype）。
 */

#include "register/op_def_registry.h"

#include <vector>

namespace ops {

// input/output（SE 5.7）：6 种 dtype，先 int32 index 组再 int64 index 组，共 12 项
static const std::vector<ge::DataType> kInputDtypes = {ge::DT_FLOAT16, ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_FLOAT,
                                                       ge::DT_DOUBLE,  ge::DT_DOUBLE,  ge::DT_BF16,  ge::DT_BF16,
                                                       ge::DT_INT32,   ge::DT_INT32,   ge::DT_INT64, ge::DT_INT64};

// index（SE 5.7）：6 组 int32 + 6 组 int64
static const std::vector<ge::DataType> kIndexDtypes = {ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
                                                       ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64,
                                                       ge::DT_INT32, ge::DT_INT64, ge::DT_INT32, ge::DT_INT64};

static const std::vector<ge::Format> kFormats = {ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND,
                                                 ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND};

class CustomGather : public OpDef {
public:
    explicit CustomGather(const char* name)
        : OpDef(name)
    {
        this->Input("input")
            .ParamType(REQUIRED)
            .DataType(kInputDtypes)
            .Format(kFormats)
            .UnknownShapeFormat(kFormats)
            .AutoContiguous(); // 非连续输入自动连续化（SE 5.3.3）
        this->Input("index")
            .ParamType(REQUIRED)
            .DataType(kIndexDtypes)
            .Format(kFormats)
            .UnknownShapeFormat(kFormats)
            .AutoContiguous();
        // output dtype 与 input 相同（SE 5.5），由 infershape dtype 推导保证
        this->Output("output")
            .ParamType(REQUIRED)
            .DataType(kInputDtypes)
            .Format(kFormats)
            .UnknownShapeFormat(kFormats)
            .AutoContiguous();

        // SE 5.8：无需值依赖（infershape/tiling 仅读 shape 元信息与 dtype）

        // SE 5.2：dim 为 ATTR（对标 torch.gather 的 Python int 形参），ATTR → AttrType(OPTIONAL)
        this->Attr("dim").AttrType(OPTIONAL).Int(0);

        OpAICoreConfig aicoreConfig;
        aicoreConfig
            .DynamicCompileStaticFlag(true) // 必须 true：激活 DTYPE_ 宏自动实例化（dtype_guide §2）
            .DynamicFormatFlag(false)
            .DynamicRankSupportFlag(true)
            .DynamicShapeSupportFlag(true)
            .NeedCheckSupportFlag(false)
            .PrecisionReduceFlag(false) // 纯搬运无归约，禁用精度降低
            .ExtendCfgInfo("opFile.value", "custom_gather");
        this->AICore().AddConfig("ascend950", aicoreConfig);
    }
};

OP_ADD(CustomGather);
} // namespace ops
