/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aclnn_max_dim.h"
#include "argmax_with_value.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/shape_utils.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"
#include "op_api/aclnn_check.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

constexpr size_t MAX_DIM_LEN = 8;

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_910_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_BOOL};

static const std::initializer_list<op::DataType> DTYPE_SUPPORT_GE910B_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64, op::DataType::DT_BOOL,
    op::DataType::DT_BF16};
namespace {
static const std::initializer_list<op::DataType> DTYPE_SUPPORT_REGBASE_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_INT64,
    op::DataType::DT_BOOL,  op::DataType::DT_BF16,    op::DataType::DT_INT32};
}

// 判断芯片类型是否大于等于910B
static inline bool CheckSocVersionGe910B(void)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return curArch == NpuArch::DAV_2201 || IsRegBase(curArch);
}

// 判断芯片类型是否大于等于910
static inline bool CheckSocVersionGe910(void)
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    return curArch == NpuArch::DAV_1001 || curArch == NpuArch::DAV_2201 || IsRegBase(curArch);
}

static bool CheckDtypeValid(const aclTensor* self, const aclTensor* out)
{
    std::initializer_list<DataType> CURRENT_DTYPE_SUPPORT_LIST;
    if (IsRegBase()) {
        CURRENT_DTYPE_SUPPORT_LIST = DTYPE_SUPPORT_REGBASE_LIST;
    } else {
        bool isGe910BSocVersion = CheckSocVersionGe910B();
        CURRENT_DTYPE_SUPPORT_LIST = isGe910BSocVersion ? DTYPE_SUPPORT_GE910B_LIST : DTYPE_SUPPORT_910_LIST;
    }

    OP_CHECK_DTYPE_NOT_SUPPORT(self, CURRENT_DTYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SAME(self, out, return false);
    return true;
}

static bool CheckNotNull(const aclTensor* self, const aclTensor* out, const aclTensor* indices)
{
    OP_CHECK_NULL(self, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(indices, return false);
    return true;
}

static bool CheckShape(const aclTensor* self, const aclTensor* out, const aclTensor* indices)
{
    OP_CHECK_MAX_DIM(self, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(out, MAX_DIM_LEN, return false);
    OP_CHECK_MAX_DIM(indices, MAX_DIM_LEN, return false);
    return true;
}

static bool CheckDim(const aclTensor* self, const int64_t dim)
{
    int64_t shapeSize = self->GetViewShape().GetDimNum();
    int64_t dimMin = -1 * shapeSize;
    int64_t dimMax = shapeSize - 1;
    if (shapeSize == 0) {
        dimMin = -1;
        dimMax = 0;
    }
    if ((dim > dimMax) || (dim < dimMin)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "dim should be within the range of [[%ld], [%ld]].", dimMin, dimMax);
        return false;
    }
    return true;
}

static aclnnStatus CheckParams(const aclTensor* self, const int64_t dim, const aclTensor* out, const aclTensor* indices)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(self, out, indices), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(self, out), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查shape是否支持
    CHECK_RET(CheckShape(self, out, indices), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查dim是在支持范围内
    CHECK_RET(CheckDim(self, dim), ACLNN_ERR_PARAM_INVALID);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxDimGetWorkspaceSize(const aclTensor* self, int64_t dim, bool keepdim, aclTensor* out,
                                        aclTensor* indices, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnMaxDim, DFX_IN(self, dim, keepdim), DFX_OUT(out, indices));

    auto ret = CheckParams(self, dim, out, indices);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    if (self->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto self_contiguous = l0op::Contiguous(self, uniqueExecutor.get());
    CHECK_RET(self_contiguous != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    const aclTensor* self_cast;
    bool isGe910SocVersion = CheckSocVersionGe910();
    if (self->GetDataType() == op::DataType::DT_BOOL) {
        self_cast = l0op::Cast(self_contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(self_cast != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    } else if (self->GetDataType() == op::DataType::DT_INT64 && !isGe910SocVersion) {
        self_cast = l0op::Cast(self_contiguous, op::DataType::DT_FLOAT, uniqueExecutor.get());
        CHECK_RET(self_cast != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    } else {
        self_cast = self_contiguous;
    }

    std::tuple<aclTensor*, aclTensor*> result;
    if (IsRegBase()) {
        result = l0op::ArgMaxWithValue(self_cast, dim, keepdim, indices->GetDataType(), uniqueExecutor.get());
    } else {
        result = l0op::ArgMaxWithValue(self_cast, dim, keepdim, op::DataType::DT_INT32, uniqueExecutor.get());
    }
    CHECK_RET(std::get<0>(result) != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    CHECK_RET(std::get<1>(result) != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto argmax_indices = std::get<0>(result);
    auto argmax_out = std::get<1>(result);
    CHECK_RET(CheckShapeAndScalarSame(argmax_indices, indices), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckShapeAndScalarSame(argmax_out, out), ACLNN_ERR_PARAM_INVALID);

    auto indices_res = l0op::Cast(argmax_indices, indices->GetDataType(), uniqueExecutor.get());
    CHECK_RET(indices_res != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto out_res = l0op::Cast(argmax_out, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(out_res != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto indices_view_copy_result = l0op::ViewCopy(indices_res, indices, uniqueExecutor.get());
    CHECK_RET(indices_view_copy_result != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    auto out_view_copy_result = l0op::ViewCopy(out_res, out, uniqueExecutor.get());
    CHECK_RET(out_view_copy_result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnMaxDim(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnMaxDim);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
