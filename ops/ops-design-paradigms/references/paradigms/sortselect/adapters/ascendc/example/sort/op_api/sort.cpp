/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sort.h"
#include "math/zero_op/op_api/zero_op.h"
#include "conversion/tensor_move/op_api/tensor_move.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/cast.h"
#include "opdev/platform.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "op_api/aclnn_check.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Sort);

// 本例只提供 arch35 AiCore 实现；非末轴由 L2 规范化。
static constexpr int64_t AXIS_LIMIT = 8;

static aclnnStatus SortAiCoreArch35(const aclTensor* self, bool stable, int64_t dim, bool descending, aclTensor* values,
                                    aclTensor* indices, op::DataType indicesType, aclOpExecutor* executor)
{
    L0_DFX(SortAiCoreArch35, self, stable, dim, descending, values, indices, indicesType);

    auto dimSize = static_cast<int64_t>(self->GetViewShape().GetDimNum());
    if (dim < -dimSize || dim >= dimSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim must be in range [-N, N-1]. Current dim is %ld.", dim);
        return ACLNN_ERR_PARAM_INVALID;
    }
    if ((dim < 0 ? dim + dimSize : dim) != dimSize - 1) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Sort L0 requires the last axis; use the L2 layout wrapper. "
                "Current dim is %ld.",
                dim);
        return ACLNN_ERR_PARAM_INVALID;
    }

    return ADD_TO_LAUNCHER_LIST_AICORE(Sort, OP_INPUT(self), OP_OUTPUT(values, indices),
                                       OP_ATTR(dim, descending, stable, indicesType));
}

// 处理0维场景：self cast以后viewcopy, indices是为0
static aclnnStatus HandleDimZeroTensor(const aclTensor* self, aclTensor* valuesOut, aclTensor* indicesOut,
                                       aclOpExecutor* executor)
{
    auto selfContiguous = l0op::Contiguous(self, executor);
    CHECK_RET(selfContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);
    // indices是返回0
    auto zeroIndices = l0op::ZerosLike(selfContiguous, executor);
    CHECK_RET(zeroIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto valuesCast = l0op::Cast(selfContiguous, valuesOut->GetDataType(), executor);
    CHECK_RET(valuesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto indicesCast = l0op::Cast(zeroIndices, indicesOut->GetDataType(), executor);
    CHECK_RET(indicesCast != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto viewCopyValues = l0op::TensorMove(valuesCast, valuesOut, executor);
    CHECK_RET(viewCopyValues != nullptr, ACLNN_ERR_INNER_NULLPTR);
    auto viewCopyIndices = l0op::TensorMove(indicesCast, indicesOut, executor);
    CHECK_RET(viewCopyIndices != nullptr, ACLNN_ERR_INNER_NULLPTR);

    return ACLNN_SUCCESS;
}

static std::tuple<aclTensor*, aclTensor*> AllocateSortOutputs(const aclTensor* self, const op::Shape& shape,
                                                              op::Format format, op::DataType indicesType,
                                                              aclOpExecutor* executor)
{
    auto values = executor->AllocTensor(shape, self->GetDataType(), format);
    auto indices = executor->AllocTensor(shape, indicesType, format);
    return std::tie(values, indices);
}

const std::tuple<aclTensor*, aclTensor*> Sort(const aclTensor* self, int64_t dim, bool descending, bool stable,
                                              op::DataType indicesType, aclOpExecutor* executor)
{
    L0_DFX(Sort, self, dim, descending, stable, indicesType);
    auto selfShape = self->GetViewShape();
    auto selfFormat = self->GetViewFormat();

    auto dimSize = (int64_t)(selfShape.GetDimNum());
    if (dimSize < 0 || dimSize > AXIS_LIMIT) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Tensor self dimension size must be in range [0, 8]. Current size is [%ld].",
                dimSize);
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    // 标量没有排序轴，直接复制值并返回零索引。
    if (dimSize == 0) {
        if (dim == 0 || dim == -1) {
            auto outputs = AllocateSortOutputs(self, selfShape, selfFormat, indicesType, executor);
            auto valuesOut = std::get<0>(outputs);
            auto indicesOut = std::get<1>(outputs);
            CHECK_RET(valuesOut != nullptr && indicesOut != nullptr, std::make_tuple(nullptr, nullptr));
            auto res = HandleDimZeroTensor(self, valuesOut, indicesOut, executor);
            if (res != ACLNN_SUCCESS) {
                OP_LOGE(ACLNN_ERR_INNER, "HandleDimZeroTensor error.");
                return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
            } else {
                return std::tie(valuesOut, indicesOut);
            }
        } else {
            OP_LOGE(ACLNN_ERR_PARAM_INVALID, "When dimSize == 0 , dim must be 0 or -1.");
            return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
        }
    }
    if (dim < -dimSize || dim >= dimSize) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Dim must be in range [-N, N-1]. Current dim is %ld.", dim);
        return std::tuple<aclTensor*, aclTensor*>(nullptr, nullptr);
    }
    auto outputs = AllocateSortOutputs(self, selfShape, selfFormat, indicesType, executor);
    auto values = std::get<0>(outputs);
    auto indices = std::get<1>(outputs);
    CHECK_RET(values != nullptr && indices != nullptr, std::make_tuple(nullptr, nullptr));
    auto status = SortAiCoreArch35(self, stable, dim, descending, values, indices, indicesType, executor);
    CHECK_RET(status == ACLNN_SUCCESS, std::make_tuple(nullptr, nullptr));
    return std::tie(values, indices);
}
} // namespace l0op
