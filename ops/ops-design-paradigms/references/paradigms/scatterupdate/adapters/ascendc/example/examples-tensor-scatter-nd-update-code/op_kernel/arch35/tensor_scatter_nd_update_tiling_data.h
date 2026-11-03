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
 * \file tensor_scatter_nd_update_tiling_data.h
 * \brief tiling data struct for tensor_scatter_nd_update
 */

#ifndef TENSOR_SCATTER_ND_UPDATE_TILING_DATA_H
#define TENSOR_SCATTER_ND_UPDATE_TILING_DATA_H

#include <cstdint>

struct TensorScatterNdUpdateTilingData {
    // 三阶段通用
    int32_t needCoreNum;   // 实际使用核数（Phase2 SortLib 决定）
    int64_t totalIndexNum; // 散点数 N = num_updates = prod(indices.shape[:-1])
    int64_t totalVarElems; // var 总元素数
    int32_t varRank;       // var 的秩 (1~8)
    int32_t indexDimNum;   // K = indices.shape[-1]（索引维数，1 <= K <= varRank）

    // Phase1 线性化
    int64_t varDims[8];    // var 各维尺寸（越界检查用，仅前 K 维有效）
    int64_t varStrides[8]; // var 各维行主序步长（元素数，仅前 K 维用于线性化）
    int64_t sliceSize;     // 每个 update 覆盖的 slice 元素数 = prod(var.shape[K:])
    int32_t useInt32Index; // 1=Phase3 索引计算用 int32（N×sliceSize <= INT32_MAX），0=int64

    // Phase2 (SortLib)
    uint32_t numTileData;  // 单 tile 数据量
    uint32_t tileCount;    // tile 总数
    uint32_t activeCores;  // SortLib 实际使用核数
    uint32_t tmpUbSize;    // SortLib 临时 UB 大小
    uint32_t isSingleCore; // 单核快路径标记（1=单核 AscendC::Sort）

    // workspace 布局（MDE §4.3，相对 usrWs 基址的偏移，单位字节）
    int64_t sortWsBytes;    // 段1 sortWorkspace 大小
    int64_t wsLinearIdxOff; // 段2 linearIdxBuffer 偏移
    int64_t wsSortedOff;    // 段3 sortedLinearIdxBuffer 偏移
    int64_t wsPermOff;      // 段4 permutation 偏移
};

// ── v2.0 新增：非确定（直接写覆盖）路径 TilingData（MDE §11.4.2）──
//   无 SortLib 参数、无 workspace 段偏移（用户 ws = 0，仅系统 ws）；
//   字段语义与确定结构同名字段一致（解析逻辑由 BaseTiling 复用，见 MDE §11.5.1）。
struct TensorScatterNdUpdateNondetTilingData {
    // ── 通用 ──
    int32_t needCoreNum;   // 实际使用核数 = blockDim = min(coreNum, max(1, ceil(N*sliceSize/1024)))
    int64_t totalIndexNum; // 散点数 N = prod(indices.shape[:-1])（与 det 同源）
    int64_t totalVarElems; // var 总元素数（空 tensor 判定用）
    int32_t varRank;       // var 的秩（1~8，DFX dump 用；kernel 线性化循环上界为 K）
    int32_t indexDimNum;   // K = indices.shape[-1]（坐标线性化循环上界）

    // ── VF 内联坐标线性化（与 det Phase1 同源参数，无 Phase1 输出 buffer）──
    int64_t varDims[8];    // var 各维尺寸（越界/负索引检查用，仅前 K 维有效）
    int64_t varStrides[8]; // var 各维行主序步长（线性化：lin = Σ_{d<K} idx[d]·varStrides[d]）
    int64_t sliceSize;     // 每 update 覆盖的 slice 元素数（==1 快路径判据 + W 计算输入）

    // ── uint32 索引域降级门控（与 det 同款判定，MDE §11.6.2 安全论证）──
    int32_t useInt32Index; // 1=flat 任务空间 int32 域（N*sliceSize <= INT32_MAX），0=int64 域
};

#endif
