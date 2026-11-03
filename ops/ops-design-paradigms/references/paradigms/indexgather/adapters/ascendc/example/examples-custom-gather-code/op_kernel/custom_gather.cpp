/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file custom_gather.cpp
 * \brief kernel entry for custom_gather
 *
 * MDE 5.1/7.2 设计：TilingKey 仅编码场景位 idxMode（32/64 位偏移索引）；
 * 12 组 dtype 组合由 DTYPE_INPUT / DTYPE_INDEX 宏自动实例化。
 */

#include "arch35/custom_gather_simt.h"

// MDE 5.1：idxMode 分派枚举（tiling key 宏值见 tiling_key.h）
enum class CustomGatherIdxMode : uint32_t {
    IDX32 = 0,
    IDX64 = 1
};

template <uint32_t idxMode>
__global__ __aicore__ void custom_gather(GM_ADDR input, GM_ADDR index, GM_ADDR output, GM_ADDR workspace,
                                         GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(CustomGatherTilingData);
    GET_TILING_DATA_WITH_STRUCT(CustomGatherTilingData, tilingData, tiling);
    if constexpr (idxMode == static_cast<uint32_t>(CustomGatherIdxMode::IDX32)) {
        NsCustomGather::Process<DTYPE_INPUT, DTYPE_INDEX, uint32_t>(input, index, output, &tilingData);
    } else {
        NsCustomGather::Process<DTYPE_INPUT, DTYPE_INDEX, uint64_t>(input, index, output, &tilingData);
    }
}
