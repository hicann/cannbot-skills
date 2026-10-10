/*
 * ----------------------------------------------------------------------------------------------------------
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * ----------------------------------------------------------------------------------------------------------
 */

// AdamApplyOneAssign TilingData — 按 rank 模板化，体积分两档
// 位置：math/adam_apply_one_assign/op_kernel/arch35/adam_apply_one_assign_tiling_struct.h
#ifndef ADAM_APPLY_ONE_ASSIGN_TILING_STRUCT_H_
#define ADAM_APPLY_ONE_ASSIGN_TILING_STRUCT_H_
#include <cstdint>

// === 算子特定常量 — 新算子必须修改 ===
// 这些值随算子而变，抄代码时必须替换为你的算子的实际值
constexpr int64_t MAX_INPUT_SLOTS = 10; // 本算子最大输入数
constexpr int64_t MAX_OUTPUT_SLOTS = 3; // 本算子最大输出数
constexpr int64_t PHYS_NODES = 5;       // 物理存活节点 P（= TBuf 槽位数）

struct SplitResult {
    int64_t ubSplitIdx; // UB 切分轴
    int64_t ubFactor;   // UB 主块元素数
    int64_t ubOuter;    // UB 块数（外层次数）
    int64_t ubTail;     // UB 尾块元素数
};

struct MultiCoreResult {
    int64_t usedCoreNum; // 实际使用的核数
    int64_t totalTiles;  // 总 tile 数
    int64_t mainTiles;   // 每核主块数 = CeilDiv(totalTiles, usedCoreNum)
    int64_t mainCoreNum; // 处理 mainTiles 块的核数，其余核处理 mainTiles - 1 块
};

template <int64_t RANK>
struct AdamTilingData {
    SplitResult split;
    MultiCoreResult multicore;
    int64_t rank;        // 实际 rank (1~8)，Kernel 运行期读取
    int64_t perBufBytes; // UB/P 向下对齐 32B，Kernel 用此初始化 TBuf
    int64_t maxBroShape[RANK];
    int64_t numInputs;
    int64_t numOutputs;
    int64_t inputShapes[MAX_INPUT_SLOTS][RANK];
    int64_t inputStrides[MAX_INPUT_SLOTS][RANK];
    int64_t outputShapes[MAX_OUTPUT_SLOTS][RANK];
    int64_t outputStrides[MAX_OUTPUT_SLOTS][RANK];
};
#endif // ADAM_APPLY_ONE_ASSIGN_TILING_STRUCT_H_
