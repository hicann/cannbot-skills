/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* DRAFT — 范式参考模板，未编译、未上板 */

/*!
 * \file max_pool_grad_with_argmax_apt.cpp
 * \brief 反向 kernel 入口：单路线（SIMT），32/64 位索引双路径 + TPL dtype 档
 */

#include "arch35/max_pool_grad_with_argmax_simt.h"
#include "arch35/max_pool_grad_with_argmax_tiling_key.h"

using namespace NsMaxPoolGradWithArgmax;

template <uint32_t isInt32Idx>
__global__ __aicore__ void max_pool_grad_with_argmax(GM_ADDR x, GM_ADDR grad, GM_ADDR argmax, GM_ADDR y,
                                                     GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(MaxPoolGradWithArgmaxTilingData, tilingData, tiling);
    if (tilingData.count == 0) {
        return; // 空输入短路（核数下限在 tiling 侧保证）
    }
    // 32/64 位索引双路径（kernel-entrance.md §3）：count 不超 INT32_MAX 走 32 位
    if (tilingData.count <= static_cast<int64_t>(INT32_MAX)) {
        if constexpr (isInt32Idx == TPL_INT32) {
            Process<DTYPE_GRAD, int32_t, uint32_t>(grad, argmax, y, tiling);
        } else {
            Process<DTYPE_GRAD, int64_t, uint32_t>(grad, argmax, y, tiling);
        }
    } else {
        if constexpr (isInt32Idx == TPL_INT32) {
            Process<DTYPE_GRAD, int32_t, uint64_t>(grad, argmax, y, tiling);
        } else {
            Process<DTYPE_GRAD, int64_t, uint64_t>(grad, argmax, y, tiling);
        }
    }
}
