/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_DROPOUT_V3_H_
#define OP_API_INC_DROPOUT_V3_H_

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief aclnnDropoutV3的第一段接口，根据具体的计算流程，计算workspace大小。
 * 算子功能：训练过程中，按照概率P生成mask掩码，将输入中的元素置零，并将输出按照1/(1-p)的比例进行缩放。
 *
 * 计算图：
 * ```mermaid
 * graph LR
 *     A1[(input)]  --> B1([l0op::Contiguous])]--> C([l0op::DropoutV3])]
 *     A2[(noiseShape)]  --> B1([l0op::Contiguous])]--> C([l0op::DropoutV3])]
 *     A3((p)) --> C([l0op::DropoutV3])]
 *     A4((seed)) --> C([l0op::DropoutV3])]
 *     A5((offset)) --> C([l0op::DropoutV3])]
 *     C([l0op::DropoutV3])] --> D([l0op::ViewCopy]) -->E1[(out)]
 *     D --> E2[(maskOut)]
 * ```
 *
 * @param [in] input: npu device侧的aclTensor。
 * 数据类型支持FLOAT16、FLOAT32、BFLOAT16，且数据类型必须和out一样，数据格式支持ND，shape必须和out一样，支持非连续的Tensor。
 * @param [in] optionalNoiseShape: npu device侧的aclTensor。
 * 数据类型支持INT64，数据格式支持ND，shape为1D，用来控制生成随机数的个数，元素个数会进行128个数对齐。
 * @param [in] p: 丢弃的概率，数据类型支持DOUBLE。
 * @param [in] seed: 生成随机数的种子，数据类型支持INT64。
 * @param [in] offset: 生成随机数的偏移，数据类型支持INT64。
 * @param [in] out: npu
 * device侧的aclTensor，数据类型支持FLOAT16、FLOAT32、BFLOAT16，且数据类型必须和self一样，数据格式支持ND，shape必须和self一样。
 * @param [in] maskOut: 需要丢弃的输入数据的mask, npu
 * device侧的aclTensor，数据类型支持UINT8，数据格式支持ND，shape为1D。
 * @param [out] workspace_size: 返回用户需要在npu device侧申请的workspace大小。
 * @param [out] executor: 返回op执行器，包含算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus aclnnDropoutV3GetWorkspaceSize(
    const aclTensor* input, const aclTensor* optionalNoiseShape, double p, int64_t seedOptinal, int64_t offset, aclTensor* out,
    aclTensor* maskOut, uint64_t* workspaceSize, aclOpExecutor** executor);

/**
 * @brief aclnnDropoutV3的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnDropoutV3GetWorkspaceSize获取。
 * @param [in] stream: acl stream流。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回状态码。
 */
ACLNN_API aclnnStatus
aclnnDropoutV3(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif // OP_API_INC_DROPOUT_V3_H_
