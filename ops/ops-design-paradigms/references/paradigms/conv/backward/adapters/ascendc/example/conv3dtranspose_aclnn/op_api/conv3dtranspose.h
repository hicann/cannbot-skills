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
 * @file conv3dtranspose.h
 * @brief ACLNN L0 API（l0op 风格）—— conv3dtranspose 三维转置卷积算子
 *
 * 职责：将 OpDef 四输入（input_size/x/filter/bias，bias 可空）+ 调用方 y 以
 * ADD_TO_LAUNCHER_LIST_AICORE 记入 executor 启动列表。本文件不触碰
 * stream / memcpy；两段式的第二段由框架统一执行。
 *
 * 契约来源（fact sheet）：
 *   - OpDef：op_host/conv3dtranspose_def.cpp —— 输入 [input_size(1D×5
 *     int32/int64 ND，值依赖), x(5D fp16/bf16 NCDHW/NDHWC), filter(5D
 *     fp16/bf16 NCDHW/NDHWC/DHWCN), bias(可选 1D fp16/fp32 ND)]，输出
 *     [y(同 x)]，属性序 [strides, pads, dilations, groups, data_format,
 *     output_padding, offset_x, enable_hf32]（offset_w 保留槽不注册为
 *     Input——bringup 证据见 def.cpp 注释）。
 *   - L2 签名契约：aclnn_conv3dtranspose.h（= REQUIREMENTS §8.2.1 /
 *     design target_aclnn_header.h 基准）——标准路由 inputSize 为设备
 *     张量（aclTensor*）。
 *   - 输出范式：y.shape = input_size 的元素值（值依赖，spec outputs.y，
 *     VariableOutput/直通形态）——host 无法从设备张量读值，调用方 y 直通
 *     OP_OUTPUT（无 AllocTensor、无 ViewCopy）；y.shape = input_size 元素值
 *     契约由 tiling 层 ValidateAndLoad 在值依赖读取后逐维比对执行（错配
 *     即拒绝，防 GM 越界写），几何/属性深校验亦由 tiling 完成。
 */

#ifndef OP_API_INC_LEVEL0_CONV3DTRANSPOSE_H_
#define OP_API_INC_LEVEL0_CONV3DTRANSPOSE_H_

#include "aclnn/aclnn_base.h"
#include "opdev/shape_utils.h"

namespace l0op {

/**
 * @brief 唯一的共享输出契约校验（L0 直通防线与 L2 校验同源）
 *
 * y 的输出形状是值依赖的（= input_size 设备张量的元素值），host 侧不可
 * 推导（paradigm-variants VariableOutput 形态：check 形 helper）——本
 * helper 只校验静态可判的 y↔x 契约：y 非空指针、rank 5、dtype 与 x 一致
 * （spec dtype_rule：等 dtype 配对下 y.dtype == x.dtype）、存储格式与 x
 * 严格配对（tiling 同口径校验 yFmt == xFmt）。y.shape == input_size 元素
 * 值的值依赖契约由 tiling 层执行。
 *
 * @param x [in] 输入特征图（dedy，dtype/format 基准）
 * @param y [in] 调用方输出张量
 * @return false 当 y 为空指针 / rank≠5 / dtype≠x / 存储格式≠x
 */
inline bool Conv3DTransposeCheckOut(const aclTensor* x, const aclTensor* y)
{
    if (x == nullptr || y == nullptr) {
        return false;
    }
    if (y->GetViewShape().GetDimNum() != 5UL) {
        return false;
    }
    if (y->GetDataType() != x->GetDataType()) {
        return false;
    }
    if (y->GetStorageFormat() != x->GetStorageFormat()) {
        return false;
    }
    return true;
}

/**
 * @brief L0 入口（标准路由）：inputSize 为 OpDef 输入 0 的装载张量
 *
 * inputSize 由 L2 的值依赖桥接装载（宿主常量张量——调用方设备张量的
 * 5 元素值经 D2H 读取后 ConvertToTensor，或调用方宿主张量直通；见
 * aclnn_conv3dtranspose.cpp 的 Conv3DTransposeInputSizeTensor）：tiling 的
 * 值依赖读取需宿主地址，kernel 侧不消费该地址。x/filter/bias 按 OpDef
 * 输入序直通（bias 可为 nullptr——可选输入缺省）；调用方 y 直通
 * OP_OUTPUT（值依赖输出，host 不可推导——与生成实现的 Tensor 路由同语义）。
 * 属性按 OpDef 声明序传入。
 *
 * @return 直通的调用方 y；失败返回 nullptr
 */
const aclTensor* Conv3DTranspose(const aclTensor* inputSize, const aclTensor* x, const aclTensor* filter,
                                 const aclTensor* bias, const aclIntArray* strides, const aclIntArray* pads,
                                 const aclIntArray* dilations, int64_t groups, const char* dataFormat,
                                 const aclIntArray* outputPadding, int64_t offsetX, bool enableHf32, const aclTensor* y,
                                 aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_CONV3DTRANSPOSE_H_
