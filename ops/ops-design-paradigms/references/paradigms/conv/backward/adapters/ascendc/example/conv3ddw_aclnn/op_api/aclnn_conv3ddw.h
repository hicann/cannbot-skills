/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLNN_CONV3DDW_H
#define ACLNN_CONV3DDW_H
#include "aclnn/acl_meta.h"
#ifdef __cplusplus
extern "C" {
#endif

/* filterSize 按 out 的物理格式解释；out 为 FP32。可选膨胀率缺省为 1，
 * outputPadding 缺省为 0；本例支持连续张量、groups=1、enableHf32=false。
 */
__attribute__((visibility("default"))) aclnnStatus aclnnConv3ddwGetWorkspaceSize(
    const aclTensor* x, const aclIntArray* filterSize, const aclTensor* outBackprop, const aclIntArray* strides,
    const aclIntArray* pads, const aclIntArray* dilationsOptional, int64_t groups, const char* dataFormatOptional,
    bool enableHf32, const aclIntArray* outputPaddingOptional, const aclTensor* out, uint64_t* workspaceSize,
    aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnConv3ddw(void* workspace, uint64_t workspaceSize,
                                                                 aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif
#endif
