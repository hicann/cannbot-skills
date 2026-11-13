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
 * @file aclnn_conv3dtranspose.h
 * @brief conv3dtranspose 三维转置卷积算子 ACLNN L2 公共头（手写，l0op 风格）
 *
 * 签名契约（参数名/顺序/const/标量映射逐字保留，不得改动）：
 * op/tests/blackbox/design/target_aclnn_header.h —— blackbox-gen 交付的目标
 * ABI 基准（内容逐字取自 REQUIREMENTS §8.2.1 接口设计签名），冻结黑盒 CSV
 * （api_name=aclnnConv3DTranspose，张量槽 6 + 属性键 8 = 14 形参）与
 * golden.py 的 ACLNN 位置形参序均按本签名构建并经 TTK OpApiInfoKeeper
 * 正则验证（见 prechange_snapshot/MANIFEST.md 的契约消解记录）。
 *
 * 两段式设计：
 * - aclnnConv3DTransposeGetWorkspaceSize: 参数校验、构建执行计划、查询 workspace
 * - aclnnConv3DTranspose: 提交设备执行
 */

#ifndef ACLNN_CONV3DTRANSPOSE_H_
#define ACLNN_CONV3DTRANSPOSE_H_

#include "aclnn/aclnn_base.h"

#ifndef ACLNN_API
#define ACLNN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 计算执行 aclnnConv3DTranspose 所需的 workspace 大小（第一段）
 * @param inputSize [in] 必选，int32/int64，1D×5：目标输出形状（值依赖）
 * @param x [in] 必选，fp16/bf16，5D，NCDHW/NDHWC（dedy）
 * @param filter [in] 必选，fp16/bf16，5D，(Cout, Cin/G, kD, kH, kW)，NCDHW/NDHWC/DHWCN
 * @param bias [in] 可选，fp16/fp32，1D×Cin：按 y 通道广播（nullptr/空张量 = 无 bias）
 * @param offsetW [in] 可选，int8，保留（当前不启用；非空指针即拒绝）
 * @param strides [in] 必选，5 元（N/C 位必 1）
 * @param pads [in] 必选，6 元 front/back/top/bottom/left/right
 * @param dilations [in] 可选，5 元，缺省 {1,1,1,1,1}
 * @param groups [in] 可选，缺省 1
 * @param dataFormat [in] 可选，"NCDHW"/"NDHWC"，缺省 NDHWC
 * @param outputPadding [in] 可选，5 元（N/C 位必 0），缺省全 0
 * @param offsetX [in] 可选，保留，缺省 0
 * @param enableHf32 [in] 可选，缺省 false（FP32 路径开关，fp16/bf16 恒 false）
 * @param y [out] 必选，同 x dtype/format，5D；shape = inputSize 元素值（值依赖契约）
 * @param workspaceSize [out] 返回所需 workspace 大小
 * @param executor [out] 返回执行器
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnConv3DTransposeGetWorkspaceSize(
    const aclTensor* inputSize, const aclTensor* x, const aclTensor* filter, const aclTensor* bias,
    const aclTensor* offsetW, const aclIntArray* strides, const aclIntArray* pads, const aclIntArray* dilations,
    int64_t groups, char* dataFormat, const aclIntArray* outputPadding, int64_t offsetX, bool enableHf32, aclTensor* y,
    uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief 执行 conv3dtranspose 算子计算（第二段）
 * @param workspace [in] workspace 内存地址
 * @param workspaceSize [in] workspace 大小
 * @param executor [in] 执行器
 * @param stream [in] ACL 流
 * @return aclnnStatus 状态码
 */
ACLNN_API aclnnStatus aclnnConv3DTranspose(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                           aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // ACLNN_CONV3DTRANSPOSE_H_
