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
 * \file random_uniform_v2.cpp
 * \brief
 */

#include "arch35/random_uniform_v2_op.h"
using namespace AscendC;
using namespace RandomUniformV2;

#define RANDOM_STANDARD_NORMAL_V2_DEFAULT_TILING_KEY 100

extern "C" __global__ __aicore__ void random_uniform_v2(GM_ADDR shape, GM_ADDR inOffset, GM_ADDR y, GM_ADDR outOffset,
                                                        GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(RandomUnifiedTilingDataStruct);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    AscendC::TPipe pipe;
    if (TILING_KEY_IS(RANDOM_STANDARD_NORMAL_V2_DEFAULT_TILING_KEY)) {
        RandomUniformV2::RandomUniformV2Op<DTYPE_Y> op(&pipe, &tilingData);
        op.Init(y, outOffset);
        op.Process();
    }
}
