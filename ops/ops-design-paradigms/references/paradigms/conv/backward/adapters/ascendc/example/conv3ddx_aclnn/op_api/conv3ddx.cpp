/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * @file conv3ddx.cpp
 * @brief ACLNN L0 API 实现（l0op 风格）
 *
 * 注册算子类型、装载 OpDef 输入 0（input_size）、分配输出 y，并把 Conv3ddx
 * AICore kernel 记入 executor 的启动列表。kernel 本体（op_kernel/）与 tiling
 * （op_host/）由框架在两段式第二段统一调度 —— 本文件不触碰 stream / memcpy。
 *
 * 参考实现锚点：ops-nn conv/convolution_backward/op_api/convolutionbackward.cpp
 * 的 Conv3DBackpropInputV2 L0（GetOutputSize → ConvertToTensor 装载 input_size、
 * AllocTensor 输出、ADD_TO_LAUNCHER_LIST_AICORE）。
 */

#include "conv3ddx.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Conv3ddx);

namespace {

/**
 * @brief 标准路由：aclIntArray 值装载为 OpDef 输入 0 的常量张量
 *
 * executor 持有的宿主数组 → ConvertToTensor(int32)。OpDef/tiling 契约：
 * input_size 为 1D×5、int32/int64、ND（tiling 校验 sizeFmt == ND、长度 5）。
 * 官方卷积反向族同构（convolutionbackward.cpp GetOutputSize）。
 */
const aclTensor* Conv3ddxInputSizeTensor(const aclIntArray* inputSize, aclOpExecutor* executor)
{
    auto sizeArr = executor->AllocIntArray(inputSize->GetData(), inputSize->Size());
    OP_CHECK_NULL(sizeArr, return nullptr);
    auto inputSizeTensor = executor->ConvertToTensor(sizeArr, op::DataType::DT_INT32);
    OP_CHECK_NULL(inputSizeTensor, return nullptr);
    // ConvertToTensor 缺省 format 与 OpDef 注册（ND）对齐，显式三处设置避免缺省漂移
    const_cast<aclTensor*>(inputSizeTensor)->SetStorageFormat(op::Format::FORMAT_ND);
    const_cast<aclTensor*>(inputSizeTensor)->SetViewFormat(op::Format::FORMAT_ND);
    const_cast<aclTensor*>(inputSizeTensor)->SetOriginalFormat(op::Format::FORMAT_ND);
    return inputSizeTensor;
}

/**
 * @brief 共用 AiCore 装载：两路由唯一的 kernel 记录点
 *
 * 属性值严格按 OpDef 声明序（conv3ddx_def.cpp）：strides / pads / dilations /
 * groups / data_format / enable_hf32 —— 位置对应，不可交换。strides/pads/
 * dilations 传 executor 持有的 aclIntArray 副本（L2 已归一），dataFormat 为
 * 静态字面量（L2 已归一），groups/enableHf32 按值传递。
 */
aclnnStatus Conv3ddxAiCore(const aclTensor* inputSize, const aclTensor* filter, const aclTensor* outBackprop,
                           const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations,
                           int64_t groups, const char* dataFormat, bool enableHf32, const aclTensor* output,
                           aclOpExecutor* executor)
{
    L0_DFX(Conv3ddxAiCore, inputSize, filter, outBackprop, strides, pads, dilations, groups, dataFormat, enableHf32,
           output);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(Conv3ddx, OP_INPUT(inputSize, filter, outBackprop), OP_OUTPUT(output),
                                           OP_ATTR(strides, pads, dilations, groups, dataFormat, enableHf32));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return ret,
                                         "Conv3ddx ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return ACLNN_SUCCESS;
}

} // namespace

const aclTensor* Conv3ddx(const aclIntArray* inputSize, const aclTensor* filter, const aclTensor* outBackprop,
                          const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations,
                          int64_t groups, const char* dataFormat, bool enableHf32, aclOpExecutor* executor)
{
    OP_CHECK_NULL(inputSize, return nullptr);
    OP_CHECK_NULL(filter, return nullptr);
    OP_CHECK_NULL(outBackprop, return nullptr);
    OP_CHECK_NULL(strides, return nullptr);
    OP_CHECK_NULL(pads, return nullptr);
    OP_CHECK_NULL(dilations, return nullptr);
    OP_CHECK_NULL(dataFormat, return nullptr);
    OP_CHECK_NULL(executor, return nullptr);

    // 输出形状：input_size 元素值（唯一真值来源，spec outputs.y）
    op::Shape outShape;
    if (!Conv3ddxOutShape(inputSize, outShape)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "Conv3ddx output shape derivation failed: inputSize must hold 5 values.");
        return nullptr;
    }

    // 输出分配：dtype/format 随 out_backprop（y 与 out_backprop 严格配对，tiling
    // 校验 yFmt == dedyFmt）；kernel 写满全输出（col2im 写出前清零），无部分写语义
    auto output = executor->AllocTensor(outShape, outBackprop->GetDataType(), outBackprop->GetStorageFormat());
    OP_CHECK_NULL(output, return nullptr);

    // OpDef 输入 0：aclIntArray → 1D×5 int32 ND 常量张量（值依赖读取载体）
    auto inputSizeTensor = Conv3ddxInputSizeTensor(inputSize, executor);
    OP_CHECK_NULL(inputSizeTensor, return nullptr);

    auto ret = Conv3ddxAiCore(inputSizeTensor, filter, outBackprop, strides, pads, dilations, groups, dataFormat,
                              enableHf32, output, executor);
    if (ret != ACLNN_SUCCESS) {
        return nullptr;
    }
    return output;
}

const aclTensor* Conv3ddxTensor(const aclTensor* inputSize, const aclTensor* filter, const aclTensor* outBackprop,
                                const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations,
                                int64_t groups, const char* dataFormat, bool enableHf32, const aclTensor* out,
                                aclOpExecutor* executor)
{
    OP_CHECK_NULL(inputSize, return nullptr);
    OP_CHECK_NULL(filter, return nullptr);
    OP_CHECK_NULL(outBackprop, return nullptr);
    OP_CHECK_NULL(strides, return nullptr);
    OP_CHECK_NULL(pads, return nullptr);
    OP_CHECK_NULL(dilations, return nullptr);
    OP_CHECK_NULL(dataFormat, return nullptr);
    OP_CHECK_NULL(out, return nullptr);
    OP_CHECK_NULL(executor, return nullptr);

    // Tensor 路由：input_size 已是设备张量（1D×5 int32/int64 ND），直通 OpDef
    // 输入 0；输出形状 host 不可得（值依赖），调用方 out 直通 OP_OUTPUT —— 与
    // 生成实现的 Tensor 变体同语义；y.shape = input_size 元素值契约由 tiling 层
    // ValidateAndLoad 在值依赖读取后逐维比对执行（错配即拒绝，防 GM 越界写），
    // 几何/属性校验亦由 tiling 值依赖读取完成。
    auto ret = Conv3ddxAiCore(inputSize, filter, outBackprop, strides, pads, dilations, groups, dataFormat, enableHf32,
                              out, executor);
    if (ret != ACLNN_SUCCESS) {
        return nullptr;
    }
    return out;
}

} // namespace l0op
