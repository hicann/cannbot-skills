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
 * @file conv3dtranspose.cpp
 * @brief ACLNN L0 API 实现（l0op 风格）
 *
 * 注册算子类型（OP_TYPE_REGISTER(Conv3DTranspose)——名称与 *_def.cpp 的
 * OpDef 类名逐字一致），并把 Conv3DTranspose AICore kernel 记入 executor
 * 的启动列表。kernel 本体（op_kernel/）与 tiling（op_host/）由框架在
 * 两段式第二段统一调度 —— 本文件不触碰 stream / memcpy。
 *
 * 参考实现锚点：同族 conv3ddx 手写 ACLNN 的 Tensor 路由（值依赖 input_size
 * 设备张量 + 调用方 out 直通 OP_OUTPUT）；官方卷积反向族
 * ops-nn convolution_backward/op_api 的 L0 形态。
 */

#include "conv3dtranspose.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"
#include "opdev/shape_utils.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(Conv3DTranspose);

namespace {

/**
 * @brief 唯一的 kernel 记录点
 *
 * 属性值严格按 OpDef 声明序（conv3dtranspose_def.cpp）：strides / pads /
 * dilations / groups / data_format / output_padding / offset_x /
 * enable_hf32 —— 位置对应，不可交换。strides/pads/dilations/outputPadding
 * 传 executor 持有的 aclIntArray 副本（L2 已归一），dataFormat 为静态
 * 字面量（L2 已归一），groups/offsetX/enableHf32 按值传递。
 *
 * bias 为可选输入：nullptr 直通 OP_INPUT（框架按 OPTIONAL 输入缺省处理，
 * AppendOpArg 具空指针通路），tiling 侧 GetInputTensor(3) 得 nullptr 走
 * 无 bias 路由。
 */
aclnnStatus Conv3DTransposeAiCore(const aclTensor* inputSize, const aclTensor* x, const aclTensor* filter,
                                  const aclTensor* bias, const aclIntArray* strides, const aclIntArray* pads,
                                  const aclIntArray* dilations, int64_t groups, const char* dataFormat,
                                  const aclIntArray* outputPadding, int64_t offsetX, bool enableHf32,
                                  const aclTensor* output, aclOpExecutor* executor)
{
    L0_DFX(Conv3DTransposeAiCore, inputSize, x, filter, bias, strides, pads, dilations, groups, dataFormat,
           outputPadding, offsetX, enableHf32, output);

    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        Conv3DTranspose, OP_INPUT(inputSize, x, filter, bias), OP_OUTPUT(output),
        OP_ATTR(strides, pads, dilations, groups, dataFormat, outputPadding, offsetX, enableHf32));
    OP_CHECK_ADD_TO_LAUNCHER_LIST_AICORE(ret != ACLNN_SUCCESS, return ret,
                                         "Conv3DTranspose ADD_TO_LAUNCHER_LIST_AICORE failed.");
    return ACLNN_SUCCESS;
}

} // namespace

const aclTensor* Conv3DTranspose(const aclTensor* inputSize, const aclTensor* x, const aclTensor* filter,
                                 const aclTensor* bias, const aclIntArray* strides, const aclIntArray* pads,
                                 const aclIntArray* dilations, int64_t groups, const char* dataFormat,
                                 const aclIntArray* outputPadding, int64_t offsetX, bool enableHf32, const aclTensor* y,
                                 aclOpExecutor* executor)
{
    OP_CHECK_NULL(inputSize, return nullptr);
    OP_CHECK_NULL(x, return nullptr);
    OP_CHECK_NULL(filter, return nullptr);
    OP_CHECK_NULL(strides, return nullptr);
    OP_CHECK_NULL(pads, return nullptr);
    OP_CHECK_NULL(dilations, return nullptr);
    OP_CHECK_NULL(dataFormat, return nullptr);
    OP_CHECK_NULL(outputPadding, return nullptr);
    OP_CHECK_NULL(y, return nullptr);
    OP_CHECK_NULL(executor, return nullptr);

    // y↔x 静态契约防线（共享 helper，与 L2 校验同源）：rank 5 / dtype 随 x /
    // 存储格式与 x 严格配对。y.shape = input_size 元素值的值依赖契约由 tiling
    // 层 ValidateAndLoad 在读取设备侧 input_size 值后逐维比对执行。
    if (!Conv3DTransposeCheckOut(x, y)) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID,
                "Conv3DTranspose output contract check failed: y must be 5D, same dtype/format as x.");
        return nullptr;
    }

    // 标准路由（值依赖）：inputSize 为 L2 桥接装载的宿主常量张量（tiling 的
    // 值依赖读取需宿主地址；kernel 不消费该地址）；调用方 y 直通；bias 可为
    // nullptr（可选输入缺省）。
    auto ret = Conv3DTransposeAiCore(inputSize, x, filter, bias, strides, pads, dilations, groups, dataFormat,
                                     outputPadding, offsetX, enableHf32, y, executor);
    if (ret != ACLNN_SUCCESS) {
        return nullptr;
    }
    return y;
}

} // namespace l0op
