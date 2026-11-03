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
 * \file tensor_scatter_nd_update_tiling_key.h
 * \brief tiling key declare for tensor_scatter_nd_update
 *
 * 编码三个场景维度（禁止枚举 dtype，dtype 由 DTYPE_ 宏自动实例化）：
 *   - countMode：SortLib 计数类型（int32 安全与否，N <= 2^30 用 uint32 否则 int64）
 *   - keyDtype：linear_idx 的 key 类型（int16 / int32 / int64，按 totalVarElems）
 *   - deterministic：确定性模式位（v2.0 新增，MDE §11.3）
 *       TSNU_MODE_NONDET=0 非确定直接写覆盖快速路径（绑定
 *         TensorScatterNdUpdateNondetTilingData，单 key：countMode/keyDtype 为
 *         SEL 占位值 U32/I32 —— nondet 无排序无 key 落盘，不消费两维度）
 *       TSNU_MODE_DET=1    确定三段式（绑定 TensorScatterNdUpdateTilingData，v1.x 路径）
 */

#ifndef TENSOR_SCATTER_ND_UPDATE_TILING_KEY_H
#define TENSOR_SCATTER_ND_UPDATE_TILING_KEY_H

#include "ascendc/host_api/tiling/template_argument.h"

#define TSNU_COUNT_U32 0 // 计数类型 uint32_t（N <= 2^30）
#define TSNU_COUNT_I64 1 // 计数类型 int64_t（N > 2^30）

#define TSNU_KEY_I16 0 // key 类型 int16_t（totalVarElems <= 32767）
#define TSNU_KEY_I32 1 // key 类型 int32_t（<= 2^31-1）
#define TSNU_KEY_I64 2 // key 类型 int64_t（兜底）

// ── v2.0 新增：确定性模式位（MDE §11.3.1）──
#define TSNU_MODE_NONDET 0 // 非确定（直接写覆盖快速路径）
#define TSNU_MODE_DET 1    // 确定（三段式 last-write-wins，a4177ec 路径）

ASCENDC_TPL_ARGS_DECL(TensorScatterNdUpdate,
                      ASCENDC_TPL_UINT_DECL(countMode, 1, ASCENDC_TPL_UI_LIST, TSNU_COUNT_U32, TSNU_COUNT_I64),
                      ASCENDC_TPL_UINT_DECL(keyDtype, 2, ASCENDC_TPL_UI_LIST, TSNU_KEY_I16, TSNU_KEY_I32, TSNU_KEY_I64),
                      ASCENDC_TPL_UINT_DECL(deterministic, 1, ASCENDC_TPL_UI_LIST, TSNU_MODE_NONDET, TSNU_MODE_DET));

// 确定模式 SEL（6 组合，绑定 TensorScatterNdUpdateTilingData；SEL 内参数顺序与 ARGS_DECL 一致）
#define TSNU_TPL_ARGS_DET_SEL(countVal, keyVal) \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), \
                         ASCENDC_TPL_UINT_SEL(countMode, ASCENDC_TPL_UI_LIST, countVal), \
                         ASCENDC_TPL_UINT_SEL(keyDtype, ASCENDC_TPL_UI_LIST, keyVal), \
                         ASCENDC_TPL_UINT_SEL(deterministic, ASCENDC_TPL_UI_LIST, TSNU_MODE_DET), \
                         ASCENDC_TPL_TILING_STRUCT_SEL(TensorScatterNdUpdateTilingData))

// 非确定模式 SEL（1 组合：绑定 TensorScatterNdUpdateNondetTilingData；
// countMode/keyDtype 在非确定 kernel 中不消费（无 SortLib、无排序 key 落盘——线性化
// 结果在 VF 寄存器内以 int64 累积直接寻址），固定为 SEL 占位值 U32/I32，MDE §11.3.2）
#define TSNU_TPL_ARGS_NONDET_SEL() \
    ASCENDC_TPL_ARGS_SEL(ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY), \
                         ASCENDC_TPL_UINT_SEL(countMode, ASCENDC_TPL_UI_LIST, TSNU_COUNT_U32), \
                         ASCENDC_TPL_UINT_SEL(keyDtype, ASCENDC_TPL_UI_LIST, TSNU_KEY_I32), \
                         ASCENDC_TPL_UINT_SEL(deterministic, ASCENDC_TPL_UI_LIST, TSNU_MODE_NONDET), \
                         ASCENDC_TPL_TILING_STRUCT_SEL(TensorScatterNdUpdateNondetTilingData))

ASCENDC_TPL_SEL(TSNU_TPL_ARGS_DET_SEL(TSNU_COUNT_U32, TSNU_KEY_I16),
                TSNU_TPL_ARGS_DET_SEL(TSNU_COUNT_U32, TSNU_KEY_I32),
                TSNU_TPL_ARGS_DET_SEL(TSNU_COUNT_U32, TSNU_KEY_I64),
                TSNU_TPL_ARGS_DET_SEL(TSNU_COUNT_I64, TSNU_KEY_I16),
                TSNU_TPL_ARGS_DET_SEL(TSNU_COUNT_I64, TSNU_KEY_I32),
                TSNU_TPL_ARGS_DET_SEL(TSNU_COUNT_I64, TSNU_KEY_I64), TSNU_TPL_ARGS_NONDET_SEL());

#endif
