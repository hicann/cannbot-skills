/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * calution: this file was generated automatically do not change it.
 */

#ifndef ACLNN_CONV3DDX_H_
#define ACLNN_CONV3DDX_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* function: aclnnConv3ddxGetWorkspaceSize
 * inputSize : required
 * filter : required
 * outBackprop : required
 * strides : required
 * pads : required
 * dilationsOptional : optional
 * groups : optional
 * dataFormatOptional : optional
 * enableHf32 : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default"))) aclnnStatus aclnnConv3ddxGetWorkspaceSize(
    const aclIntArray* inputSize, const aclTensor* filter, const aclTensor* outBackprop, const aclIntArray* strides,
    const aclIntArray* pads, const aclIntArray* dilationsOptional, int64_t groups, char* dataFormatOptional,
    bool enableHf32, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

__attribute__((visibility("default"))) aclnnStatus aclnnConv3ddxTensorGetWorkspaceSize(
    const aclTensor* inputSize, const aclTensor* filter, const aclTensor* outBackprop, const aclIntArray* strides,
    const aclIntArray* pads, const aclIntArray* dilationsOptional, int64_t groups, char* dataFormatOptional,
    bool enableHf32, const aclTensor* out, uint64_t* workspaceSize, aclOpExecutor** executor);

/* function: aclnnConv3ddx
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default"))) aclnnStatus aclnnConv3ddx(void* workspace, uint64_t workspaceSize,
                                                                 aclOpExecutor* executor, aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
