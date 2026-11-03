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
 * \file custom_gather_tiling_key.h
 * \brief tiling key declare for custom_gather
 *
 * MDE §3.2 设计：TilingKey 仅编码场景位 idxMode（32/64 位偏移索引），禁止
 * 枚举 dtype；12 组 dtype 组合由 DTYPE_INPUT / DTYPE_INDEX 宏编译期实例化。
 */

#ifndef CUSTOM_GATHER_TILING_KEY_H
#define CUSTOM_GATHER_TILING_KEY_H

#include <cstdint>

#include "ascendc/host_api/tiling/template_argument.h"

// 场景模式：偏移索引位宽（MDE §3.2：outSize/元素数均 ≤ INT32_MAX 用 32 位，否则 64 位）
#define CUSTOM_GATHER_IDX32 0
#define CUSTOM_GATHER_IDX64 1

// 与 ASCENDC_TPL 模板参数一一对应的枚举（入口文件的 if constexpr 分派使用）
enum class CustomGatherTilingKey : uint32_t {
    IDX32 = CUSTOM_GATHER_IDX32,
    IDX64 = CUSTOM_GATHER_IDX64,
};

ASCENDC_TPL_ARGS_DECL(CustomGather,
                      ASCENDC_TPL_UINT_DECL(idxMode, 1, ASCENDC_TPL_UI_LIST, CUSTOM_GATHER_IDX32, CUSTOM_GATHER_IDX64));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(idxMode, ASCENDC_TPL_UI_LIST, CUSTOM_GATHER_IDX32),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(CustomGatherTilingData)),
                ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
                                     ASCENDC_TPL_UINT_SEL(idxMode, ASCENDC_TPL_UI_LIST, CUSTOM_GATHER_IDX64),
                                     ASCENDC_TPL_TILING_STRUCT_SEL(CustomGatherTilingData)));

#endif // CUSTOM_GATHER_TILING_KEY_H
