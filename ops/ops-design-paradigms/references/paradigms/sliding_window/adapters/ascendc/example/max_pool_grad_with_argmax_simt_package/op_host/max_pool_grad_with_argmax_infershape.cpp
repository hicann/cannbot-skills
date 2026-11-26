/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* DRAFT — 范式参考模板，未编译、未上板 */

/*!
 * \file max_pool_grad_with_argmax_infershape.cpp
 * \brief 反向输出 shape 推导：y 与原输入 x 同形（读 shape，无值依赖）
 */

#include "runtime/infer_shape_context.h"
#include "register/op_impl_registry.h"
#include "log/log.h"

using namespace ge;

namespace ops {
static constexpr uint32_t IDX_X = 0;
static constexpr uint32_t IDX_Y = 0;
static constexpr size_t DIM_NUM = 4;

static ge::graphStatus InferShape4MaxPoolGradWithArgmax(gert::InferShapeContext* context)
{
    const gert::Shape* xShape = context->GetInputShape(IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, xShape);
    OP_CHECK_IF(xShape->GetDimNum() != DIM_NUM, OP_LOGE(context, "x must be 4D"), return ge::GRAPH_FAILED);

    gert::Shape* yShape = context->GetOutputShape(IDX_Y);
    OP_CHECK_NULL_WITH_CONTEXT(context, yShape);
    for (size_t i = 0; i < DIM_NUM; ++i) {
        yShape->SetDim(i, xShape->GetDim(i)); // 输出 = 原输入同形
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(MaxPoolGradWithArgmax).InferShape(InferShape4MaxPoolGradWithArgmax);

} // namespace ops
