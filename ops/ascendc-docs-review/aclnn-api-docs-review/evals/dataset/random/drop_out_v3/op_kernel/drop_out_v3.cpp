/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file drop_out_v3.cpp
 * \brief
 */

#include "arch35/drop_out_v3_impl.h"

using namespace DropOutV3;

#define DROP_OUT_V3_DEFAULT_TILING_KEY 100

__global__ __aicore__ void drop_out_v3(
    GM_ADDR x, GM_ADDR noiseShape, GM_ADDR p, GM_ADDR seed, GM_ADDR offset,
    GM_ADDR y, GM_ADDR mask, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUnifiedSimtTilingDataStruct);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    GET_TILING_DATA(tilingData, tiling);
    TPipe pipe;
    if (TILING_KEY_IS(DROP_OUT_V3_DEFAULT_TILING_KEY)) {
        if constexpr (AscendC::IsSameType<DTYPE_X, float>::value) {
            DropOutV3::DropOutV3Impl<float, DTYPE_P> op;
            op.Init(p, mask, workspace, &tilingData, &pipe);
            op.Process(x, y, mask, &tilingData);
        } else if constexpr (AscendC::IsSameType<DTYPE_X, half>::value) {
            DropOutV3::DropOutV3Impl<half, DTYPE_P> op;
            op.Init(p, mask, workspace, &tilingData, &pipe);
            op.Process(x, y, mask, &tilingData);
        } else if constexpr (AscendC::IsSameType<DTYPE_X, bfloat16_t>::value) {
            DropOutV3::DropOutV3Impl<bfloat16_t, DTYPE_P> op;
            op.Init(p, mask, workspace, &tilingData, &pipe);
            op.Process(x, y, mask, &tilingData);
        }
    }
}
