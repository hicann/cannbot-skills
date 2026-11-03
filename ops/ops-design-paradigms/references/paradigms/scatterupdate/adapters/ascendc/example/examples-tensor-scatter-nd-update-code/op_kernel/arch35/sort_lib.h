/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * \file sort_lib.h
 * \brief SortLib public API — the only header an external operator needs to include.
 *        Include this file and call SortInvoke() to use the SIMD radix sort.
 *
 * \public
 */

#ifndef SORT_LIB_H
#define SORT_LIB_H

#include "kernel_operator.h"
#include "sort_lib/sort_radix_more_core.h"
#include "sort_lib/sort_radix_one_core.h"

namespace SortLib {

/**
 * \brief SIMD 基数排序入口（SortLib 唯一 kernel 侧对外接口）。
 *        对 x 做稳定 LSD radix sort，输出排序结果 sortedValues 与排序索引
 *        sortedIndices（permutation：sortedValues[i] == x[sortedIndices[i]]）。
 *
 * \tparam ValT        排序键类型（int8/16/32/64、uint8/16/32/64、half、bfloat16_t、float）
 * \tparam IdxT        索引输出类型（int32 或 int64），须与 tiling 侧 indexSize 对应
 * \tparam CountT      计数/偏移类型（uint32 或 int64），须与 tiling 侧 isInt32Safe 对应
 * \tparam isDescending 是否降序
 *
 * \param pipe         TPipe 实例（调用方自行构造，如 TPipe pipe;）
 * \param x            输入待排序数组（GM，长度 p.totalElements）
 * \param sortedValues 输出：排序后的值（GM，长度 p.totalElements）
 * \param sortedIndices 输出：排序索引/permutation（GM，长度 p.totalElements）
 * \param workspace    GM workspace，大小须 >= workspaceBytes，且起始地址须 32B 对齐
 * \param p            SortParams，须由 host 侧 SortTilingCompute 计算并填充
 *
 * \note p.totalElements == 0 时直接返回，不写任何输出/workspace（空输入安全）。
 * \note 排序稳定：相等元素保持原始相对顺序。
 * \note 本接口内部会分配 UB（总量对应 host 侧 tiling 传入的 ubTotalBytes 预算）。
 *       建议调用前先 pipe.Reset() 释放此前占用的 UB（保证本接口可用满 UB），
 *       调用后再 pipe.Reset() 释放本接口占用的 UB（保证后续接口可用）。
 *
 * 典型用法（host 侧算 tiling，kernel 侧调排序）：
 *   [host]   uint32_t dtypeSize   = sizeof(ValT);   // 键字节数
 *            uint32_t indexSize   = sizeof(IdxT);   // 索引字节数（4 或 8）
 *            bool     isInt32Safe = SortLib::IsInt32Safe(totalElements);
 *            uint32_t usableUb    = ubSize - SortLib::DCACHE_SIZE;  // 扣 DCACHE 后的可用 UB
 *            context->SetLocalMemorySize(usableUb);  // 设置 kernel 可用的 UB 大小
 *            SortLib::SortTilingResult r = SortLib::SortTilingCompute(
 *                totalElements, coreCount, ubSize, dtypeSize, indexSize,
 *                isInt32Safe, valueType);   // ubTotalBytes 传平台单核 UB 总大小 ubSize，内部再扣 DCACHE
 *            按 r.workspaceBytes 分配 GM workspace；
 *            把 r 的 SortParams 字段写入 tiling data；context->SetBlockDim(r.coreNumNeed);
 *            context->SetScheduleMode(1);  // 接口内部使用核间同步（SyncAll），需开启 batchmode
 * 模式使算子独占全部所需核资源，否则多流/多算子并发时可能死锁 [kernel] SortLib::SortParams p;
 *            p.numTileData=td.numTileData; p.tileCount=td.tileCount; p.activeCores=td.activeCores;
 *            p.tmpUbSize=td.tmpUbSize; p.totalElements=td.totalElements; p.isSingleCore=td.isSingleCore;
 *            pipe.Reset();   // 调用前释放此前占用的 UB，保证本接口可用满 UB
 *            SortLib::SortInvoke<ValT, IdxT, CountT, false>(&pipe, x, outVal, outIdx, ws, p);
 *            pipe.Reset();   // 调用后释放本接口占用的 UB，保证后续接口可用
 */
template <typename ValT, typename IdxT, typename CountT, bool isDescending>
__aicore__ inline void SortInvoke(AscendC::TPipe* pipe, __gm__ ValT* x, __gm__ ValT* sortedValues,
                                  __gm__ IdxT* sortedIndices, __gm__ char* workspace, const SortParams& p)
{
    if (p.totalElements == 0) {
        return; // 空输入：无元素可排序，跳过排序与 workspace 清零（避免写 0 字节 workspace 越界）
    }
    if (p.isSingleCore == 1) {
        detail::SortRadixOneCore<ValT, IdxT, isDescending> op;
        op.Init(reinterpret_cast<GM_ADDR>(x), reinterpret_cast<GM_ADDR>(sortedValues),
                reinterpret_cast<GM_ADDR>(sortedIndices), reinterpret_cast<GM_ADDR>(workspace), p, pipe);
        op.Process();
        return;
    }

    detail::SortRadixMoreCore<ValT, IdxT, CountT, isDescending> op;
    op.Init(reinterpret_cast<GM_ADDR>(x), reinterpret_cast<GM_ADDR>(sortedValues),
            reinterpret_cast<GM_ADDR>(sortedIndices), reinterpret_cast<GM_ADDR>(workspace), p, pipe);
    op.Process();
}

} // namespace SortLib

#endif
