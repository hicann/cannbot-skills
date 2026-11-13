/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclnn_conv3ddw.h"
#include "conv3ddw.h"
#include <climits>
#include <cstring>
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

namespace {
bool CheckTensor(const aclTensor* x, const aclTensor* dy, const aclTensor* out, const aclIntArray* size)
{
    using namespace op;
    if (!x || !dy || !out || !size || size->Size() != 5)
        return false;
    if (x->GetViewShape().GetDimNum() != 5 || dy->GetViewShape().GetDimNum() != 5 ||
        out->GetViewShape().GetDimNum() != 5)
        return false;
    if ((x->GetDataType() != DataType::DT_FLOAT16 && x->GetDataType() != DataType::DT_BF16) ||
        x->GetDataType() != dy->GetDataType() || out->GetDataType() != DataType::DT_FLOAT)
        return false;
    if (!IsContiguous(x) || !IsContiguous(dy) || !IsContiguous(out))
        return false;
    const auto fmt = x->GetStorageFormat();
    const auto ofmt = out->GetStorageFormat();
    if ((fmt != Format::FORMAT_NCDHW && fmt != Format::FORMAT_NDHWC) || dy->GetStorageFormat() != fmt ||
        (ofmt != Format::FORMAT_NCDHW && ofmt != Format::FORMAT_NDHWC && ofmt != Format::FORMAT_DHWCN))
        return false;
    for (size_t i = 0; i < 5; ++i)
        if ((*size)[i] <= 0 || (*size)[i] > INT32_MAX || out->GetViewShape().GetDim(i) != (*size)[i] ||
            x->GetViewShape().GetDim(i) <= 0 || dy->GetViewShape().GetDim(i) <= 0)
            return false;
    return true;
}
} // namespace

extern "C" aclnnStatus aclnnConv3ddwGetWorkspaceSize(const aclTensor* x, const aclIntArray* filterSize,
                                                     const aclTensor* outBackprop, const aclIntArray* strides,
                                                     const aclIntArray* pads, const aclIntArray* dilationsOptional,
                                                     int64_t groups, const char* dataFormatOptional, bool enableHf32,
                                                     const aclIntArray* outputPaddingOptional, const aclTensor* out,
                                                     uint64_t* workspaceSize, aclOpExecutor** executor)
{
    if (!workspaceSize || !executor)
        return ACLNN_ERR_PARAM_NULLPTR;
    L2_DFX_PHASE_1(aclnnConv3ddw,
                   DFX_IN(x, filterSize, outBackprop, strides, pads, dilationsOptional, groups, dataFormatOptional,
                          enableHf32, outputPaddingOptional),
                   DFX_OUT(out));
    *workspaceSize = 0;
    *executor = nullptr;
    if (!CheckTensor(x, outBackprop, out, filterSize) || !strides || !pads || strides->Size() != 5 ||
        pads->Size() != 6 || groups != 1 || enableHf32)
        return ACLNN_ERR_PARAM_INVALID;
    auto uniqueExecutor = CREATE_EXECUTOR();
    if (!uniqueExecutor.get())
        return ACLNN_ERR_INNER_CREATE_EXECUTOR;
    constexpr int64_t ones[5] = {1, 1, 1, 1, 1};
    constexpr int64_t zeros[5] = {0, 0, 0, 0, 0};
    const char* format = dataFormatOptional ? dataFormatOptional : "NDHWC";
    const bool ndhwc = std::strcmp(format, "NDHWC") == 0;
    if ((!ndhwc && std::strcmp(format, "NCDHW") != 0) ||
        x->GetStorageFormat() != (ndhwc ? op::Format::FORMAT_NDHWC : op::Format::FORMAT_NCDHW))
        return ACLNN_ERR_PARAM_INVALID;
    auto* size = uniqueExecutor->AllocIntArray(filterSize->GetData(), 5);
    auto* s = uniqueExecutor->AllocIntArray(strides->GetData(), 5);
    auto* p = uniqueExecutor->AllocIntArray(pads->GetData(), 6);
    if (dilationsOptional && dilationsOptional->Size() != 5)
        return ACLNN_ERR_PARAM_INVALID;
    if (outputPaddingOptional && outputPaddingOptional->Size() != 5)
        return ACLNN_ERR_PARAM_INVALID;
    auto* d = uniqueExecutor->AllocIntArray(dilationsOptional ? dilationsOptional->GetData() : ones, 5);
    auto* q = uniqueExecutor->AllocIntArray(outputPaddingOptional ? outputPaddingOptional->GetData() : zeros, 5);
    if (!size || !s || !p || !d || !q)
        return ACLNN_ERR_INNER_NULLPTR;
    for (size_t i = 0; i < 5; ++i)
        if ((*s)[i] <= 0 || (*d)[i] <= 0 || (*q)[i] != 0)
            return ACLNN_ERR_PARAM_INVALID;
    for (size_t i = 0; i < 6; ++i)
        if ((*p)[i] < 0)
            return ACLNN_ERR_PARAM_INVALID;
    const int c = ndhwc ? 4 : 1;
    if ((*s)[0] != 1 || (*s)[c] != 1 || (*d)[0] != 1 || (*d)[c] != 1)
        return ACLNN_ERR_PARAM_INVALID;
    auto result = l0op::Conv3ddw(x, size, outBackprop, s, p, d, groups, ndhwc ? "NDHWC" : "NCDHW", false, q, out,
                                 uniqueExecutor.get());
    if (result != ACLNN_SUCCESS)
        return result;
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

extern "C" aclnnStatus aclnnConv3ddw(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                     aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnConv3ddw);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}
