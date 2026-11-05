/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SORT_REGBASE_TILING_DATA_H_
#define _SORT_REGBASE_TILING_DATA_H_

constexpr uint32_t SORT_BATCH_MERGE_MIN_ROWS = 4;

struct SortRegBaseTilingData {
    uint32_t numTileDataSize;     // 单行或单个 UB tile 的元素数，按路线解释。
    uint32_t unsortedDimParallel; // 每轮并行行数；多核 Radix 对应工作区驻留行数 P_act。
    uint32_t lastDimTileNum;      // 每行 tile 数。
    uint32_t sortLoopTimes;       // 行批次数。
    uint32_t lastDimNeedCore;     // 每行协作核数。
    // 下列字段由 schedule 复用；未使用字段不得按其他路线的含义读取。
    // sch2: globalHist 清零核数；sch1: 单个 value 队列字节数；sch0/5/6: 每批行数。
    uint32_t keyParams0;
    // sch2: exclusiveBins 清零核数；sch1: index 队列字节数；sch0: value 队列字节数；sch5/6: 批次数。
    uint32_t keyParams1;
    // sch2: 每次清零的 计数元素数（uint32/int64）；sch1: uint32 元素偏移 halfIndex；
    // sch0: index 队列字节数；sch6: rank-inverse 标志。
    uint32_t keyParams2;
    // sch2: globalHist 清零循环次数；sch1: 队列 buffer 数；sch0: 按 32 元素对齐的轴长。
    uint32_t keyParams3;
    // sch2: exclusiveBins 每核清零的 计数元素数（uint32/int64）；sch0: 队列 buffer 数。
    uint32_t keyParams4;
    // sch2: globalHist 每核清零的 计数元素数（uint32/int64）；sch0: 合批归并标志。
    uint32_t keyParams5;
    uint32_t tmpUbSize;     // Sort 高阶 API 临时 UB 字节数。
    int64_t lastAxisNum;    // 排序轴长 N。
    int64_t unsortedDimNum; // 待排序行数 M。
};
#endif
