/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sample_op.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/shape_utils.h"
#include "opdev/make_op_executor.h"
#include "opdev/platform.h"
#include "opdev/data_type_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(SampleOp);

// TODO(<op>): 按真实语义推导输出形状；示例直接沿用输入 shape
static bool SampleOpInferShape(const aclTensor* x, Shape& outShape)
{
    outShape = x->GetViewShape();
    return true;
}

// TODO(<op>): 架构/support 检查（arch22 对应 NpuArch::DAV_2201）
static bool IsAiCoreSupport(const aclTensor* x)
{
    return true;
}

static const aclTensor* SampleOpAiCore(const aclTensor* x, const aclTensor* out, aclOpExecutor* executor)
{
    L0_DFX(SampleOpAiCore, x, out);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(SampleOp, OP_INPUT(x), OP_OUTPUT(out));
    OP_CHECK(ret == ACLNN_SUCCESS,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "SampleOpAiCore failed."),
             return nullptr);

    return out;
}

const aclTensor* SampleOp(const aclTensor* x, aclOpExecutor* executor)
{
    Shape outShape;
    OP_CHECK(SampleOpInferShape(x, outShape),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Infer shape failed."),
             return nullptr);

    OP_CHECK(IsAiCoreSupport(x),
             OP_LOGE(ACLNN_ERR_PARAM_INVALID, "IsAiCoreSupport check failed."),
             return nullptr);

    auto out = executor->AllocTensor(outShape, x->GetDataType());
    OP_CHECK(out != nullptr,
             OP_LOGE(ACLNN_ERR_INNER_NULLPTR, "AllocTensor failed for output tensor."),
             return nullptr);

    return SampleOpAiCore(x, out, executor);
}

} // namespace l0op
