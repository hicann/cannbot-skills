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
 * \file custom_gather_tiling_data.h
 * \brief tiling data struct for custom_gather
 *
 * 字段语义见结构体内注释（与 MDE §3.1 逐字段一致）。
 * dim/inputRank/outStrides/inStrides 均为 tiling 侧 MergeAxis() 合并后视图值。
 * 不含 threadNum（kernel 侧编译期常量）；不含 magic/shift（device 端 API 计算，
 * 禁止 host 侧计算，经 kernel Process 预计算后传 VF）。
 */

#ifndef CUSTOM_GATHER_TILING_DATA_H
#define CUSTOM_GATHER_TILING_DATA_H

#include <cstdint>

constexpr int32_t CUSTOM_GATHER_MAX_RANK = 8; // MDE 3.1：outStrides/inStrides 数组长度

#pragma pack(push, 1)
struct CustomGatherTilingData {
    int32_t dim = 0;                                  // 归一化后收集轴 ∈ [0, R)
    int32_t inputRank = 0;                            // R = input.rank = index.rank
    int64_t outSize = 0;                              // 输出总元素数 = prod(index.shape)
    int64_t gatherDimSize = 0;                        // input.shape[dim]（越界判定上界）
    int64_t inStrideDim = 0;                          // input 的 dim 维 stride（j 贡献累加）
    int64_t outStrides[CUSTOM_GATHER_MAX_RANK] = {0}; // index 累积步长（坐标解码除数）
    int64_t inStrides[CUSTOM_GATHER_MAX_RANK] = {0};  // input 累积步长（偏移累加）
};
#pragma pack(pop)

#endif // CUSTOM_GATHER_TILING_DATA_H
