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
 * \file max_pool_with_argmax_apt.cpp
 * \brief 正向 kernel 入口：单路线（SIMT），TilingKey 经 TPL 模板参数绑定
 */

#include "arch35/max_pool_with_argmax_simt.h"
#include "arch35/max_pool_with_argmax_tiling_key.h"

using namespace NsMaxPoolWithArgmax;

template <uint32_t format, uint32_t isInt32Idx>
__global__ __aicore__ void max_pool_with_argmax(GM_ADDR x, GM_ADDR y, GM_ADDR argmax, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    GET_TILING_DATA_WITH_STRUCT(MaxPoolWithArgmaxTilingData, tilingData, tiling);
    if (tilingData.count == 0) {
        return; // 空输出短路（核数下限在 tiling 侧保证）
    }
    if constexpr (isInt32Idx == TPL_INT32) {
        Process<DTYPE_X, int32_t, format>(x, y, argmax, tiling);
    } else {
        Process<DTYPE_X, int64_t, format>(x, y, argmax, tiling);
    }
}
