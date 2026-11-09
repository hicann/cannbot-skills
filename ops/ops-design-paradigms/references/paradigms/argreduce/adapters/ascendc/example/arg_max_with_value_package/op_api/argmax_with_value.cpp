/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "argmax_with_value.h"
#include "opdev/aicpu/aicpu_task.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;
namespace l0op {
OP_TYPE_REGISTER(ArgMaxWithValue);

// AICORE算子kernel
std::tuple<aclTensor*, aclTensor*> Max2AiCore(const aclTensor *self,
    const int64_t dim, const bool keepdim, aclTensor *indices,
    aclTensor *out, aclOpExecutor *executor)
{
    L0_DFX(Max2AiCore, self, dim, keepdim, indices, out);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(ArgMaxWithValue,
        OP_INPUT(self),
        OP_OUTPUT(indices, out),
        OP_ATTR(dim, keepdim));
    if (ret != ACL_SUCCESS) {
        OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "ArgMaxWithValueAiCore ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }

    return std::tuple<aclTensor*, aclTensor*>(indices, out);
}

std::tuple<aclTensor*, aclTensor*> ArgMaxWithValue(const aclTensor *self, const int64_t dim,
    const bool keepdim, const op::DataType indicesDtype, aclOpExecutor *executor)
{
    L0_DFX(ArgMaxWithValue, self, dim, keepdim);
    auto out = executor->AllocTensor(self->GetDataType(), op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto indices = executor->AllocTensor(indicesDtype, op::Format::FORMAT_ND, op::Format::FORMAT_ND);
    auto ret = INFER_SHAPE(ArgMaxWithValue, OP_INPUT(self), OP_OUTPUT(indices, out), OP_ATTR(dim, keepdim));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ArgMaxWithValue InferShape failed.");
        return {};
    }

    if (self->GetViewShape().GetDimNum() == 1 && indices->GetViewShape().GetDimNum() == 1) {
        out = executor->AllocTensor(op::Shape({}), self->GetDataType(), op::Format::FORMAT_ND);
        indices = executor->AllocTensor(op::Shape({}), indicesDtype, op::Format::FORMAT_ND);
    }

    int64_t dimMax = 0;
    if (keepdim) {
        op::Shape inputShape = self->GetViewShape();
        if (dim < 0) {
            dimMax = self->GetViewShape().GetDimNum() + dim;
        } else {
            dimMax = dim;
        }
        inputShape.SetDim(dimMax, 1);
        out->SetViewShape(inputShape);
        indices->SetViewShape(inputShape);
    }

    return Max2AiCore(self, dim, keepdim, indices, out, executor);
}
}
