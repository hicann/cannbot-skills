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
 * @file conv3ddx.h
 * @brief ACLNN L0 API（l0op 风格）—— conv3ddx 三维卷积输入梯度算子
 *
 * 职责：装载 OpDef 输入 0（input_size）、按 input_size 元素值分配输出 y、
 * 以 ADD_TO_LAUNCHER_LIST_AICORE 将 Conv3ddx AICore kernel 记入 executor
 * 启动列表。本文件不触碰 stream / memcpy；两段式的第二段由框架统一执行。
 *
 * 契约来源（fact sheet）：
 *   - OpDef：op_host/conv3ddx_def.cpp —— 输入 [input_size(1D×5 int32/int64 ND,
 *     值依赖), filter(5D fp16/bf16 NCDHW/NDHWC/DHWCN), out_backprop(5D
 *     fp16/bf16 NCDHW/NDHWC)]，输出 [y(同 out_backprop)]，属性序
 *     [strides, pads, dilations, groups, data_format, enable_hf32]。
 *   - L2 签名契约：build/autogen/aclnn_conv3ddx.h（生成头，逐字复制进
 *     op_api/aclnn_conv3ddx.h）—— 标准路由 inputSize 为 aclIntArray，
 *     Tensor 路由 inputSize 为设备张量。
 *   - 输出范式：y.shape = input_size 的元素值（值依赖，spec outputs.y）；
 *     标准路由 host 侧可推出完整形状 → L0 AllocTensor + L2 ViewCopy 回填
 *     调用方 out（官方 ConvolutionBackward 族同构路径）；Tensor 路由无法在
 *     host 读值 → 调用方 out 直通 OP_OUTPUT（与生成实现同语义）。
 */

#ifndef OP_API_INC_LEVEL0_CONV3DDX_H_
#define OP_API_INC_LEVEL0_CONV3DDX_H_

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"
#include "opdev/shape_utils.h"

namespace l0op {

/**
 * @brief 唯一的共享输出形状推导（L0 分配与 L2 校验同源）
 *
 * y.shape = input_size 的元素值（5 个；轴序随 data_format：NCDHW 为
 * [N, Cin, Din, Hin, Win]，NDHWC 为 [N, Din, Hin, Win, Cin] —— 元素值本身
 * 即目标轴序下的维度，shape 层面无需重排，语义轴解释由 tiling 按
 * data_format 完成）。与 spec outputs.y 的 data_dependent 规则一致。
 *
 * @param inputSize [in]  宿主侧 5 元素整型形状值（标准路由的 aclIntArray）
 * @param outShape  [out] 推导出的 y 形状
 * @return false 当 inputSize 为空或长度不为 5
 */
inline bool Conv3ddxOutShape(const aclIntArray* inputSize, op::Shape& outShape)
{
    if (inputSize == nullptr || inputSize->Size() != 5UL) {
        return false;
    }
    outShape = op::Shape({(*inputSize)[0], (*inputSize)[1], (*inputSize)[2], (*inputSize)[3], (*inputSize)[4]});
    return true;
}

/**
 * @brief L0 入口（标准路由）：inputSize 以 aclIntArray 到达
 *
 * L0 将 inputSize 装载为 1D×5 int32 ND 常量张量作为 OpDef 输入 0（tiling 经
 * GetInputTensor(0) 值依赖读取），按 Conv3ddxOutShape 分配输出（dtype/format
 * 随 outBackprop），并将 kernel 记入启动列表。
 *
 * @return L0 分配的结果张量；失败返回 nullptr
 */
const aclTensor* Conv3ddx(const aclIntArray* inputSize, const aclTensor* filter, const aclTensor* outBackprop,
                          const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations,
                          int64_t groups, const char* dataFormat, bool enableHf32, aclOpExecutor* executor);

/**
 * @brief L0 入口（Tensor 路由）：inputSize 已是 1D×5 int32/int64 ND 设备张量
 *
 * 输出形状无法在 host 侧从设备张量读出（值依赖），调用方 out 直通
 * OP_OUTPUT（与生成的 Tensor 变体同语义），形状与几何校验由 tiling 的
 * input_size 值依赖读取完成。
 *
 * @return 直通的调用方 out；失败返回 nullptr
 */
const aclTensor* Conv3ddxTensor(const aclTensor* inputSize, const aclTensor* filter, const aclTensor* outBackprop,
                                const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations,
                                int64_t groups, const char* dataFormat, bool enableHf32, const aclTensor* out,
                                aclOpExecutor* executor);

} // namespace l0op

#endif // OP_API_INC_LEVEL0_CONV3DDX_H_
