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
 * \file tensor_scatter_nd_update_common.h
 * \brief 确定性与非确定性两模板的共享工具（v2.0 自 tensor_scatter_nd_update_simt.h 拆分，MDE §11.8）
 *
 * 内容：THREAD_NUM（每核线程数）/ MAX_PACK_BYTES / AlignAllocN（UB 32B 对齐分配）/
 *       PackType<T,W> / ComputePackParam（det/nondet 共用打包逻辑，单一事实源，
 *       防两路径 W 计算漂移）。
 * 消费方：tensor_scatter_nd_update_det_simt.h（确定三段式）、
 *         tensor_scatter_nd_update_nondet_simt.h（非确定直接写覆盖路径）、
 *         tensor_scatter_nd_update_apt.cpp。
 */

#ifndef TENSOR_SCATTER_ND_UPDATE_COMMON_H
#define TENSOR_SCATTER_ND_UPDATE_COMMON_H

#include <type_traits>

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h" // GetUintDivMagicAndShift（两模板 Process 共用）
#include "simt_api/asc_simt.h"

namespace NsTensorScatterNdUpdate {
using namespace AscendC;

constexpr uint32_t THREAD_NUM = 1024;

// 字节宽打包最大位数（64 位 = 8 字节）
constexpr int64_t MAX_PACK_BYTES = 8;

// UB 32 字节对齐分配 helper（TPipe::InitBuffer 按字节分配 / LocalMemAllocator::Alloc
// 要求 N*sizeof(T) 32B 整数倍，此处向上取整元素数）
constexpr int64_t UB_ALIGN_BYTES = 32;
constexpr int64_t UB_ALIGN_MASK = UB_ALIGN_BYTES - 1;

__aicore__ inline int64_t AlignAllocN(int64_t n, int64_t elemSize)
{
    int64_t totalBytes = n * elemSize;
    return ((totalBytes + UB_ALIGN_MASK) / UB_ALIGN_BYTES) * UB_ALIGN_BYTES / elemSize;
}

// ============================================================
// 字节宽打包类型映射（纯数据搬运，无符号整数视角重解释，一条指令读写 W 个元素）
//   W==1 → T（标量）；W>1 → 按 W×sizeof(T) 字节数映射 uint16_t/uint32_t/uint64_t（最大 64 位）。
//   对齐性：linear_idx 为 sliceSize 整数倍，sliceSize%W==0 ⇒ key%W==0 ⇒ B×W 字节对齐成立。
// ============================================================
template <typename T, int W>
struct PackType {
    using type = std::conditional_t<
        (W == 1), T,
        std::conditional_t<(W * sizeof(T) == 2), uint16_t,
                           std::conditional_t<(W * sizeof(T) == 4), uint32_t,
                                              std::conditional_t<(W * sizeof(T) == static_cast<size_t>(MAX_PACK_BYTES)),
                                                                 uint64_t, T>>>>;
};

// ============================================================
// 计算打包宽度 W 与打包后的 slice 元素数 sliceVec（唯一感知 ÷2/÷4/÷8 的地方）
//   W ∈ {1,2,4,8}，取最大可整除 sliceSize 的 2 的幂，且 W×elemSize ≤ 8（最大 64 位）。
//   sliceVec = sliceSize / W（W==1 时 sliceVec == sliceSize）。
// ============================================================
__aicore__ inline int64_t ComputePackParam(int64_t sliceSize, int64_t elemSize, int64_t& sliceVec)
{
    int64_t W = 1;
    if (sliceSize > 1) {
        for (int64_t w = MAX_PACK_BYTES / elemSize; w > 1; w >>= 1) { // 1B→8, 2B→4, 4B→2, 8B→1
            if (sliceSize % w == 0) {
                W = w;
                break;
            }
        }
    }
    sliceVec = sliceSize / W;
    return W;
}

} // namespace NsTensorScatterNdUpdate

#endif
