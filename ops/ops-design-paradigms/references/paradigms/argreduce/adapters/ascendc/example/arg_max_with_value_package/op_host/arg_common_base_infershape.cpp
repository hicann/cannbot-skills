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
 * \file arg_common_base_infershape.cpp
 * \brief
 */

#include "log/log.h"
#include "op_host/infershape_reduce_util.h"
#include "arg_common_base_infershape.h"

using namespace ge;
namespace ops {

static ge::graphStatus ArgMaxInfer(const gert::InferShapeContext* context, const gert::Shape* x_shape,
    gert::Shape* out_shape)
{
    OP_LOGD(context->GetNodeName(), "infershape impl is begin");
    int64_t x_dim = x_shape->GetDimNum();
    if (x_dim <= 1) {
        out_shape->SetDimNum(0);
        OP_LOGD(context->GetNodeName(), "infershape success, out is scalar");
        return GRAPH_SUCCESS;
    }
    auto dtype = context->GetInputTensor(INPUT_IDX_AXIS)->GetDataType();
    if (dtype == ge::DT_INT32) {
        auto* data = context->GetInputTensor(INPUT_IDX_AXIS)->GetData<int32_t>();
        return Ops::Base::ReduceDimsWithoutKeepDims<int32_t>(x_shape, data, 1, out_shape);
    }
    auto* data = context->GetInputTensor(INPUT_IDX_AXIS)->GetData<int64_t>();
    return Ops::Base::ReduceDimsWithoutKeepDims<int64_t>(x_shape, data, 1, out_shape);
}

static ge::graphStatus ArgMaxWithValueInfer(const gert::InferShapeContext* context, const gert::Shape* x_shape,
    gert::Shape* indice_shape, gert::Shape* value_shape)
{
    OP_LOGD(context->GetNodeName(), "ArgMaxWithValueInfer infershape impl is begin");
    if (x_shape->IsScalar()) {
        *indice_shape = *x_shape;
        *value_shape = *x_shape;
        return GRAPH_SUCCESS;
    }
    auto attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const auto* attr_dimension = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_dimension);
    const auto* attr_keep_dims = attrs->GetAttrPointer<bool>(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, attr_keep_dims);
    bool keep_dims = *attr_keep_dims;

    if (keep_dims) {
        *value_shape = *x_shape;
        OP_CHECK_IF(
            Ops::Base::ReduceDimsWithKeepDims<int64_t>(x_shape, attr_dimension, 1, value_shape) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "infershape failed"), return ge::GRAPH_FAILED);
    } else {
        value_shape->SetDimNum(0);
        OP_CHECK_IF(
            Ops::Base::ReduceDimsWithoutKeepDims<int64_t>(x_shape, attr_dimension, 1, value_shape) != GRAPH_SUCCESS,
            OP_LOGE(context->GetNodeName(), "infershape failed"), return ge::GRAPH_FAILED);
    }
    *indice_shape = *value_shape;
    OP_LOGD(context->GetNodeName(), "infershape success");
    return GRAPH_SUCCESS;
}

ge::graphStatus InferShapeForArgOps(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "InferShapeForArgOps infershape is begin");
    auto x_shape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto dimension_tensor = context->GetInputTensor(INPUT_IDX_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimension_tensor);
    auto dimension_shape = context->GetInputShape(INPUT_IDX_AXIS);
    OP_CHECK_NULL_WITH_CONTEXT(context, dimension_shape);
    auto out_shape = context->GetOutputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, out_shape);
    out_shape->SetDimNum(0);
    return ArgMaxInfer(context, x_shape, out_shape);
}

ge::graphStatus InferShapeForArgOpsWithValue(gert::InferShapeContext* context)
{
    OP_LOGD(context->GetNodeName(), "infershape is begin");
    auto x_shape = context->GetInputShape(INPUT_IDX_X);
    OP_CHECK_NULL_WITH_CONTEXT(context, x_shape);
    auto indice_shape = context->GetOutputShape(OUT_IDX_INDICE);
    OP_CHECK_NULL_WITH_CONTEXT(context, indice_shape);
    auto value_shape = context->GetOutputShape(OUT_IDX_VALUE);
    OP_CHECK_NULL_WITH_CONTEXT(context, value_shape);
    return ArgMaxWithValueInfer(context, x_shape, indice_shape, value_shape);
}
}  // namespace ops
