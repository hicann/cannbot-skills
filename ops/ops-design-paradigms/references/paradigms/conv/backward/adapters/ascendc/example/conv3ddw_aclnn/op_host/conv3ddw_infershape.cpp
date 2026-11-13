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
#include "exe_graph/runtime/infer_shape_context.h"
#include "exe_graph/runtime/infer_datatype_context.h"

namespace {
ge::graphStatus InferShape(gert::InferShapeContext* ctx)
{
    const auto* tensor = ctx->GetInputTensor(1);
    auto* output = ctx->GetOutputShape(0);
    if (tensor == nullptr || output == nullptr || tensor->GetShapeSize() != 5 || tensor->GetDataType() != ge::DT_INT32)
        return ge::GRAPH_FAILED;
    const auto* values = tensor->GetData<int32_t>();
    if (values == nullptr)
        return ge::GRAPH_FAILED;
    output->SetDimNum(5);
    for (size_t i = 0; i < 5; ++i) {
        if (values[i] <= 0)
            return ge::GRAPH_FAILED;
        output->SetDim(i, values[i]);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataType(gert::InferDataTypeContext* ctx)
{
    return ctx->SetOutputDataType(0, ge::DT_FLOAT);
}
} // namespace

IMPL_OP_INFERSHAPE(Conv3ddw).InferShape(InferShape).InferDataType(InferDataType).InputsDataDependency({1});
