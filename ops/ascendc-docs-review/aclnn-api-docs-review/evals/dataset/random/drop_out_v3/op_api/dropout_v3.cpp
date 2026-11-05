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
 * \file dropout_v3.cpp
 * \brief
 */
#include "dropout_v3.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(DropOutV3);

// AICORE算子kernel
const std::tuple<aclTensor*, aclTensor*> DropoutV3AiCore(
    const aclTensor* input, const aclTensor* optionalNoiseShape, const aclTensor* probTensor,
    const aclTensor* seedTensor, const aclTensor* offsetTensor, aclTensor* out, aclTensor* maskOut,
    aclOpExecutor* executor)
{
    L0_DFX(DropoutV3AiCore, input, optionalNoiseShape, probTensor, seedTensor, offsetTensor, out, maskOut);
    ADD_TO_LAUNCHER_LIST_AICORE(
        DropOutV3, OP_INPUT(input, optionalNoiseShape, probTensor, seedTensor, offsetTensor), OP_OUTPUT(out, maskOut));
    return std::tuple<aclTensor*, aclTensor*>(out, maskOut);
}

const std::tuple<aclTensor*, aclTensor*> DropoutV3(
    const aclTensor* input, const aclTensor* optionalNoiseShape, const aclTensor* probTensor,
    const aclTensor* seedTensor, const aclTensor* offsetTensor, aclTensor* maskOut, aclOpExecutor* executor)
{
    auto out = executor->AllocTensor(input->GetViewShape(), input->GetDataType());
    auto inputDimNum = input->GetViewShape().GetDimNum();
    FVector<int64_t> inputDimVector = {static_cast<int64_t>(inputDimNum)};
    optionalNoiseShape =
        executor->ConvertToTensor(inputDimVector.data(), inputDimVector.size(), op::DataType::DT_INT64);
    return DropoutV3AiCore(input, optionalNoiseShape, probTensor, seedTensor, offsetTensor, out, maskOut, executor);
}

} // namespace l0op
