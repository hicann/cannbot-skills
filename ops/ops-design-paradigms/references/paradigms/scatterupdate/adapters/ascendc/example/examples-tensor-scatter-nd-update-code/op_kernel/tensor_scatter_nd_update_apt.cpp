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
 * \file tensor_scatter_nd_update_apt.cpp
 * \brief kernel entry for tensor_scatter_nd_update
 *
 * v2.0（MDE §11.7）：模板参数新增第三参数 deterministic（由 TPL key 的确定性位实例化），
 *   顶层 if constexpr 二分——非确定分支不实例化任何 SortLib/Phase1/Phase3 代码，
 *   确定分支体与 a4177ec 完全一致（逐字节保持）。
 *   非确定分支无 dtype 编译期门控（与 scatter_add apt 的 IsGmAtomicAddCapable 门控
 *   形成对照）——本算子全 dtype 均可走 nondet（MDE §11.6.4 对齐写原子性论证），
 *   不存在"白名单外 dtype 空体占位"分支。
 */

#include "arch35/tensor_scatter_nd_update_common.h"      // 共享工具（THREAD_NUM/打包/对齐）
#include "arch35/tensor_scatter_nd_update_det_simt.h"    // 确定三段式（v2.0 由 simt.h 改名）
#include "arch35/tensor_scatter_nd_update_nondet_simt.h" // 非确定（直接写覆盖）路径

template <uint32_t countMode, uint32_t keyDtype, uint32_t deterministic>
__global__ __aicore__ void tensor_scatter_nd_update(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR var_out,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TensorScatterNdUpdateTilingData); // 顶部一次（占位大小声明；
                                                              //  确定结构 ~232B ≥ Nondet ~168B，覆盖安全）
    if constexpr (deterministic == TSNU_MODE_NONDET) {
        // ── v2.0 非确定快速路径：单 VF 直接写覆盖 ──
        //   countMode/keyDtype 为 SEL 占位值（第 7 key 固定 U32/I32），本分支不消费。
        //   全 dtype 可达（无 dtype 白名单，MDE §11.6.4），无需编译期 dtype 门控。
        GET_TILING_DATA_WITH_STRUCT(TensorScatterNdUpdateNondetTilingData, tilingData, tiling);
        NsTensorScatterNdUpdate::NondetProcess<DTYPE_VAR, DTYPE_INDICES>(var, indices, updates, var_out, workspace,
                                                                         &tilingData);
    } else {
        // ── 确定三段式（a4177ec 现有 6 组合分派，逻辑零变更，逐字节保持）──
        GET_TILING_DATA_WITH_STRUCT(TensorScatterNdUpdateTilingData, tilingData, tiling);

        if constexpr (keyDtype == TSNU_KEY_I16) {
            if constexpr (countMode == TSNU_COUNT_U32) {
                NsTensorScatterNdUpdate::Process<DTYPE_VAR, DTYPE_INDICES, uint32_t, int16_t>(
                    var, indices, updates, var_out, workspace, &tilingData);
            } else {
                NsTensorScatterNdUpdate::Process<DTYPE_VAR, DTYPE_INDICES, int64_t, int16_t>(
                    var, indices, updates, var_out, workspace, &tilingData);
            }
        } else if constexpr (keyDtype == TSNU_KEY_I32) {
            if constexpr (countMode == TSNU_COUNT_U32) {
                NsTensorScatterNdUpdate::Process<DTYPE_VAR, DTYPE_INDICES, uint32_t, int32_t>(
                    var, indices, updates, var_out, workspace, &tilingData);
            } else {
                NsTensorScatterNdUpdate::Process<DTYPE_VAR, DTYPE_INDICES, int64_t, int32_t>(
                    var, indices, updates, var_out, workspace, &tilingData);
            }
        } else {
            if constexpr (countMode == TSNU_COUNT_U32) {
                NsTensorScatterNdUpdate::Process<DTYPE_VAR, DTYPE_INDICES, uint32_t, int64_t>(
                    var, indices, updates, var_out, workspace, &tilingData);
            } else {
                NsTensorScatterNdUpdate::Process<DTYPE_VAR, DTYPE_INDICES, int64_t, int64_t>(
                    var, indices, updates, var_out, workspace, &tilingData);
            }
        }
    }
}
