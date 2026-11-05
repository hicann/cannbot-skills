/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MERGE_SORT_CONSTANTS_H
#define MERGE_SORT_CONSTANTS_H

#include <cstdint>
#include "op_kernel/platform_util.h"

namespace MergeSortConstants {

constexpr uint32_t MERGE_LIST_MAX_NUM = 4; // MrgSort 每组最多合并四个有序段。

// 对定宽整数位模式加最小有符号值，仅翻转最高符号位。
constexpr int32_t SIGN_FLIP_ADDEND_FP32 = (-2147483647 - 1);
constexpr int16_t SIGN_FLIP_ADDEND_FP16 = -32768;

constexpr uint32_t DEALING_SORT_NUM_ONCE = 32; // Sort32 生成的初始有序段长度。

} // namespace MergeSortConstants

#endif
