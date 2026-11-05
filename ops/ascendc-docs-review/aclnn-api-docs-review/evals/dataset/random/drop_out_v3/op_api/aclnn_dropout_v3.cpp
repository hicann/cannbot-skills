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
 * \file aclnn_dropout_v3.cpp
 * \brief
 */

#include "aclnn_dropout_v3.h"
#include "aclnn_kernels/cast.h"
#include "aclnn_kernels/contiguous.h"
#include "dropout_v3.h"
#include "conversion/fill//op_api/fill.h"
#include "math/zero_op/op_api/zero_op.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "op_api/aclnn_check.h"
#include "opdev/common_types.h"
#include "opdev/data_type_utils.h"
#include "opdev/format_utils.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"
#include "opdev/tensor_view_utils.h"
#include "opdev/platform.h"

using namespace op;
#ifdef __cplusplus
extern "C" {
#endif

static constexpr size_t MAX_DIM_LEN = 8;
static const int64_t BIT_NUMBER = 128;
static const int64_t UINT8_BIT_NUMBER = 8;
static const int8_t MAX_MASK_NUM = -1;

// 根据API定义，需要列出所能支持的所有dtype
static const std::initializer_list<op::DataType> ASCEND910_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16};

static const std::initializer_list<op::DataType> ARCH3510_DTYPE_SUPPORT_LIST = {
    op::DataType::DT_FLOAT, op::DataType::DT_FLOAT16, op::DataType::DT_BF16};

static const std::initializer_list<op::DataType> MASK_DTYPE_SUPPORT_LIST = {op::DataType::DT_UINT8};

static inline bool CheckNotNull(const aclTensor* input, const aclTensor* out, const aclTensor* maskOut)
{
    OP_CHECK_NULL(input, return false);
    OP_CHECK_NULL(out, return false);
    OP_CHECK_NULL(maskOut, return false);
    return true;
}

static inline bool CheckIsNullptr(const aclTensor* optionalNoiseShape)
{
    if (optionalNoiseShape != nullptr) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "currently, the input of noise_shape must be nullptr, please check.");
        return false;
    }
    return true;
}

static inline const std::initializer_list<op::DataType>& GetDtypeSupportListBySocVersion()
{
    auto curArch = GetCurrentPlatformInfo().GetCurNpuArch();
    switch (curArch) {
        case NpuArch::DAV_3510:
        case NpuArch::DAV_2201: {
            return ARCH3510_DTYPE_SUPPORT_LIST;
        }
        case NpuArch::DAV_1001: {
            return ASCEND910_DTYPE_SUPPORT_LIST;
        }
        default: {
            return ASCEND910_DTYPE_SUPPORT_LIST;
        }
    }
}

static bool CheckDtypeValid(const aclTensor* input, const aclTensor* out, const aclTensor* maskOut)
{
    OP_CHECK_RESULT_DTYPE_CAST_FAILED(input->GetDataType(), out->GetDataType(), return false);

    const std::initializer_list<op::DataType> dtypeSupportList = GetDtypeSupportListBySocVersion();
    OP_CHECK_DTYPE_NOT_SUPPORT(input, dtypeSupportList, return false);
    // 检查input的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(input, ARCH3510_DTYPE_SUPPORT_LIST, return false);

    // 检查mask的数据类型是否在支持列表内
    OP_CHECK_DTYPE_NOT_SUPPORT(maskOut, MASK_DTYPE_SUPPORT_LIST, return false);
    return true;
}

static inline bool CheckProbability(double p)
{
    if (p > 1 || p < 0) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The value of p has to be between 0 and 1, but current is %f.", p);
        return false;
    }
    return true;
}

static bool CheckShape(const aclTensor* input, const aclTensor* out, const aclTensor* maskOut)
{
    OP_CHECK_MAX_DIM(input, MAX_DIM_LEN, return false);
    OP_CHECK_SHAPE_NOT_EQUAL(input, out, return false);

    int64_t inputSize = input->GetViewShape().GetShapeSize();
    int64_t maskSize = maskOut->GetViewShape().GetShapeSize();
    int64_t needMaskSizde = (inputSize + BIT_NUMBER - 1) / BIT_NUMBER * BIT_NUMBER / UINT8_BIT_NUMBER;
    if (maskSize != needMaskSizde) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Size of maskOut has to be %ld, but current is %ld.", needMaskSizde, maskSize);
        return false;
    }
    return true;
}

static inline aclnnStatus CheckParams(
    const aclTensor* input, const aclTensor* optionalNoiseShape, double p, const aclTensor* out,
    const aclTensor* maskOut)
{
    // 1. 检查参数是否为空指针
    CHECK_RET(CheckNotNull(input, out, maskOut), ACLNN_ERR_PARAM_NULLPTR);

    // 2. 检查输入的数据类型是否在API支持的数据类型范围之内，需要根据api定义校验
    CHECK_RET(CheckDtypeValid(input, out, maskOut), ACLNN_ERR_PARAM_INVALID);

    // 3. 检查输入的optionNoiseShape是否为要求的空
    CHECK_RET(CheckIsNullptr(optionalNoiseShape), ACLNN_ERR_PARAM_INVALID);

    // 4. 检查p是否符合规则
    CHECK_RET(CheckProbability(p), ACLNN_ERR_PARAM_INVALID);

    // 5. 检查输出输出shape
    CHECK_RET(CheckShape(input, out, maskOut), ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

// 检查tuple <out, maskOut>里的元素是否为非null。true表示为非null，false表示为null
static bool CheckTupleNullptr(std::tuple<aclTensor*, aclTensor*> tensorTuple)
{
    static const int RESULT_NUM = 2;
    if (std::tuple_size<decltype(tensorTuple)>::value != RESULT_NUM) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "The length of tuple returned by DropoutV3 is not 2.");
        return false;
    }

    return (std::get<0>(tensorTuple) != nullptr) && (std::get<1>(tensorTuple) != nullptr);
}

static inline const aclTensor* FillScalar(const aclTensor* out, int8_t val, aclOpExecutor* executor)
{
    auto maskShape = out->GetViewShape();
    FVector<int64_t> maskShapeVector;
    for (size_t i = 0; i < maskShape.GetDimNum(); i++) {
        maskShapeVector.push_back(maskShape.GetDim(i));
    }
    auto dims = executor->ConvertToTensor(maskShapeVector.data(), maskShapeVector.size(), DataType::DT_INT64);
    auto shapeArray = executor->AllocIntArray(maskShapeVector.data(), maskShapeVector.size());

    FVector<int8_t> valVector = {val};
    auto valTensor = executor->ConvertToTensor(valVector.data(), valVector.size(), op::DataType::DT_INT8);
    auto mask = l0op::Fill(dims, valTensor, shapeArray, executor);
    CHECK_RET(mask != nullptr, nullptr);
    mask->SetFromWorkspace(false);
    mask->SetStorageAddr(out->GetStorageAddr());
    return out;
}

aclnnStatus aclnnDropoutV3GetWorkspaceSize(
    const aclTensor* input, const aclTensor* optionalNoiseShape, double p, int64_t seed, int64_t offset, aclTensor* out,
    aclTensor* maskOut, uint64_t* workspaceSize, aclOpExecutor** executor)
{
    L2_DFX_PHASE_1(aclnnDropoutV3, DFX_IN(input, optionalNoiseShape, p, seed, offset), DFX_OUT(out, maskOut));
    // 固定写法，创建OpExecutor
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    // 固定写法，参数检查
    auto ret = CheckParams(input, optionalNoiseShape, p, out, maskOut);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);

    if (input->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    // 固定写法，将输入input转换成连续的tensor
    auto inputContiguous = l0op::Contiguous(input, uniqueExecutor.get());
    CHECK_RET(inputContiguous != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 进行dropoutv3计算
    const aclTensor* outResult = nullptr;
    const aclTensor* maskResult = nullptr;
    if (p == 0) {
        outResult = inputContiguous;
        maskResult = FillScalar(maskOut, MAX_MASK_NUM, uniqueExecutor.get());
    } else if (p == 1) {
        outResult = l0op::ZerosLike(inputContiguous, uniqueExecutor.get());
        maskResult = FillScalar(maskOut, 0, uniqueExecutor.get());
    } else {
        FVector<double> probVector = {p};
        auto probTensor =
            uniqueExecutor.get()->ConvertToTensor(probVector.data(), probVector.size(), op::DataType::DT_DOUBLE);
        FVector<int64_t> seedVector = {seed};
        auto seedTensor =
            uniqueExecutor.get()->ConvertToTensor(seedVector.data(), seedVector.size(), op::DataType::DT_INT64);
        FVector<int64_t> offsetVector = {0, offset};
        auto offsetTensor =
            uniqueExecutor.get()->ConvertToTensor(offsetVector.data(), offsetVector.size(), op::DataType::DT_INT64);
        auto dropOutResult = l0op::DropoutV3(
            inputContiguous, optionalNoiseShape, probTensor, seedTensor, offsetTensor, maskOut, uniqueExecutor.get());
        CHECK_RET(CheckTupleNullptr(dropOutResult), ACLNN_ERR_INNER_NULLPTR);
        outResult = std::get<0>(dropOutResult);
        maskResult = std::get<1>(dropOutResult);
    }
    CHECK_RET(outResult != nullptr, ACLNN_ERR_INNER_NULLPTR);
    CHECK_RET(maskResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果转换成输出out的数据类型
    auto castOut = l0op::Cast(outResult, out->GetDataType(), uniqueExecutor.get());
    CHECK_RET(castOut != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，将计算结果拷贝到输出out上，out可能是非连续的tensor
    auto outViewCopyResult = l0op::ViewCopy(castOut, out, uniqueExecutor.get());
    CHECK_RET(outViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    auto maskViewCopyResult = l0op::ViewCopy(maskResult, maskOut, uniqueExecutor.get());
    CHECK_RET(maskViewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    // 固定写法，获取计算过程中需要使用的workspace大小
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    // 需要把 uniqueExecutor持有executor转移给executor
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnDropoutV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnDropoutV3);
    // 固定写法，调用框架能力，完成计算
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
