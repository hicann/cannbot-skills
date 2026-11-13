/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "conv3ddw.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "aclnn_kernels/common/op_error_check.h"

namespace l0op {
OP_TYPE_REGISTER(Conv3ddw);

aclnnStatus Conv3ddw(const aclTensor* x, const aclIntArray* filterSize, const aclTensor* dedy,
                     const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations, int64_t groups,
                     const char* format, bool enableHf32, const aclIntArray* outputPadding, const aclTensor* out,
                     aclOpExecutor* executor)
{
    L0_DFX(Conv3ddw, x, filterSize, dedy, strides, pads, dilations, groups, format, enableHf32, outputPadding, out);
    auto* size = executor->ConvertToTensor(filterSize, op::DataType::DT_INT32);
    if (size == nullptr)
        return ACLNN_ERR_INNER_NULLPTR;
    const_cast<aclTensor*>(size)->SetStorageFormat(op::Format::FORMAT_ND);
    const_cast<aclTensor*>(size)->SetOriginalFormat(op::Format::FORMAT_ND);
    const_cast<aclTensor*>(size)->SetViewFormat(op::Format::FORMAT_ND);
    /* 输入和属性位置与 OpDef 一致；只记录启动，不在第一段访问设备 stream。 */
    return ADD_TO_LAUNCHER_LIST_AICORE(Conv3ddw, OP_INPUT(x, size, dedy), OP_OUTPUT(out),
                                       OP_ATTR(strides, pads, dilations, groups, format, enableHf32, outputPadding));
}
} // namespace l0op
