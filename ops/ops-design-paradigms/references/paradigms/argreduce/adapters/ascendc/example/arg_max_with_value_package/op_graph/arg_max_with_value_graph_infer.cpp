/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "platform/platform_info.h"

using namespace ge;
namespace ops {
constexpr size_t ATTR_IDX = 2;
static ge::graphStatus InferDataType4ArgMaxWithValue(gert::InferDataTypeContext* context)
{
    OP_LOGD("Begin InferDataType4ArgMaxWithValue");
    auto input_x_dtype = context->GetInputDataType(0);
    ge::DataType indiceDtype = ge::DT_INT32;

    fe::PlatformInfo platform_info;
    fe::OptionalInfo optional_info;
    auto result = fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_info, optional_info);
    if (result == ge::GRAPH_SUCCESS && platform_info.str_info.short_soc_version == "Ascend950") {
        auto attrs = context->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
        auto indice_dtype_ptr = attrs->GetAttrPointer<int64_t>(ATTR_IDX);
        OP_CHECK_NULL_WITH_CONTEXT(context, indice_dtype_ptr); // 仅 Ascend950 要求属性存在
        indiceDtype = static_cast<ge::DataType>(*indice_dtype_ptr);
        OP_LOGI(context->GetNodeName(), "argmaxwithvalueOutputDtypeInfer 1.0, output dtype is %d",
                static_cast<int>(indiceDtype));
    }

    context->SetOutputDataType(0, indiceDtype);
    context->SetOutputDataType(1, input_x_dtype);
    OP_LOGD("End InferDataType4ArgMaxWithValue");
    return ge::GRAPH_SUCCESS;
}

IMPL_OP(ArgMaxWithValue).InferDataType(InferDataType4ArgMaxWithValue);
} // namespace ops
