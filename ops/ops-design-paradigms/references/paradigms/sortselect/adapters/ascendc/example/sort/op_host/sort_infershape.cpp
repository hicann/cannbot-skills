/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"
#include "register/op_impl_registry.h"

using namespace ge;
namespace ops {
static ge::graphStatus SortInferShapeFunc(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, x1_shape);

    gert::Shape* output_shape_1 = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape_1);
    gert::Shape* output_shape_2 = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, output_shape_2);

    *output_shape_1 = *x1_shape;
    *output_shape_2 = *x1_shape;

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Sort).InferShape(SortInferShapeFunc);
} // namespace ops
