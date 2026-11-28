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
#include "op_common/log/log.h"

using namespace ge;

namespace ops {

// TODO(<op>): 按实际语义推导输出形状
static ge::graphStatus InferShape4SampleOp(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);

    gert::Shape* yShape = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);

    yShape->SetDimNum(xShape->GetDimNum());
    for (size_t i = 0; i < xShape->GetDimNum(); ++i) {
        yShape->SetDim(i, xShape->GetDim(i));
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(SampleOp).InferShape(InferShape4SampleOp);

} // namespace ops
