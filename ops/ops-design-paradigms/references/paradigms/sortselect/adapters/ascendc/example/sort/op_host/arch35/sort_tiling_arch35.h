/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SORT_TILING_ARCH35_H
#define SORT_TILING_ARCH35_H

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "graph/graph.h"
#include "graph/utils/type_utils.h"
#include "tiling/sort/sort_tiling_intf.h"
#include "util/math_util.h"

namespace gert {
class Shape;
} // namespace gert

namespace optiling {

// =============================================================================
// 公共常量。
// =============================================================================
constexpr size_t WORK_SPACE_SIZE = 16 * 1024 * 1024;
constexpr uint32_t BIN_NUM = 256;              // 逐字节排序的桶数。
constexpr uint32_t SMALL_TILE_DATA_NUM = 1024; // 经验性的单核轴粒度；UB 容量和尾块可能使实际分块更小。
constexpr uint32_t SIMT_UB = 32768;            // 预留了32k给simt使用
constexpr int64_t RADIX_UINT32_VALUE_MAX = 0x3fffffff;        // 32 位计数的上界，高两位留作状态。
constexpr int64_t RADIX_INT64_VALUE_MAX = 0x3fffffffffffffff; // 64 位状态字也需预留两个标志位。
constexpr uint32_t SMALL_AXIS_THRESHOLD = 512;
constexpr int64_t ONE_CORE_DATA_SIZE = 2048;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t MERGE_INITIAL_RUN_SIZE = 32;
constexpr uint32_t SMALL_AXIS_MAX_DATACOPY_BLOCK_COUNT = 4095; // DataCopy 的 blockCount 硬件上限。

constexpr bool IsRadixUint32CounterRange(int64_t axisLen)
{
    return axisLen <= RADIX_UINT32_VALUE_MAX;
}
constexpr bool NeedsSignedZeroSourceOrder(ge::DataType dataType)
{
    return dataType == ge::DT_FLOAT16 || dataType == ge::DT_BF16 || dataType == ge::DT_FLOAT;
}

// =============================================================================
// 带溢出保护的算术辅助函数。
// =============================================================================
bool CeilAlignUint32(uint64_t rawSize, uint32_t alignSize, uint32_t& alignedSize);
bool CeilDivUint32(uint64_t value, uint64_t divisor, uint32_t& result);

// =============================================================================
// 通用辅助函数。
// =============================================================================

// =============================================================================
// 短轴选路。
// =============================================================================
constexpr uint32_t TWO_STAGE_RANK_INVERSE_MAX_N = 64;
// 第二阶段键 = rowId × (B × N) + rank，最大值为 B² × N - 1。
constexpr uint32_t TWO_STAGE_U16_KEY_SPACE = 65536;

enum class SmallAxisRouteKind : uint8_t {
    NONE = 0,
    INSERTION,
    TWO_STAGE
};

struct SmallAxisRoutePlan {
    SmallAxisRouteKind kind = SmallAxisRouteKind::NONE;
    uint32_t batchSize = 0;
    uint32_t batchNum = 0;
    uint32_t blockDim = 0;
    uint32_t tmpUbSize = 0;
    bool useRankInverse = false;
};

struct SmallAxisRule {
    ge::DataType dtype;
    uint32_t insertionMaxN; // 插入排序的最大轴长；零表示不支持。
    uint32_t twoStageMaxN;  // 两阶段排序的最大轴长；零表示不支持。
    // 插入路线档位为 {maxN, minSegs}，以 {0, 0} 结束。
    // 找到首个 N ≤ maxN 的档位后，再要求 fullCoreSegs ≥ minSegs。
    uint32_t insertionTiers[4][2];
    // 两阶段路线档位按 maxN 递增，以 {0, 0} 结束；查找与插入路线相同。
    uint32_t twoStageTiers[4][2];
};

constexpr SmallAxisRule kSmallAxisRules[] = {
    // 短轴门槛是经验性能策略：插入排序适合极短轴，轴长增大会增加串行比较成本。
    // 两阶段排序覆盖更宽轴长，但需足够行数摊薄 Sort 和散射开销。
    // 每个 {maxN, minSegs} 要求轴长不超过 maxN，且每个活跃核至少分到 minSegs 行。
    // 字段依次为 dtype、插入轴长上限、两阶段轴长上限、插入档位、两阶段档位。
    {ge::DT_INT64, 16, 512, {{8, 1}, {16, 4}, {0, 0}}, {{15, 8}, {128, 4}, {512, 8}, {0, 0}}},
    {ge::DT_UINT64, 16, 512, {{8, 1}, {16, 4}, {0, 0}}, {{15, 8}, {128, 4}, {512, 8}, {0, 0}}},
    {ge::DT_INT32, 11, 384, {{8, 2}, {11, 4}, {0, 0}}, {{11, 8}, {64, 4}, {384, 8}, {0, 0}}},
    {ge::DT_UINT32, 11, 384, {{8, 2}, {11, 4}, {0, 0}}, {{11, 8}, {64, 4}, {384, 8}, {0, 0}}},
    {ge::DT_INT16, 8, 192, {{4, 2}, {8, 4}, {0, 0}}, {{7, 8}, {64, 4}, {192, 12}, {0, 0}}},
    {ge::DT_UINT16, 8, 192, {{4, 2}, {8, 4}, {0, 0}}, {{7, 8}, {64, 4}, {192, 12}, {0, 0}}},
    {ge::DT_INT8, 8, 128, {{4, 2}, {8, 7}, {0, 0}}, {{3, 8}, {64, 7}, {128, 16}, {0, 0}}},
    {ge::DT_UINT8, 8, 128, {{4, 2}, {8, 7}, {0, 0}}, {{3, 8}, {64, 7}, {128, 16}, {0, 0}}},
    {ge::DT_FLOAT, 8, 24, {{4, 16}, {8, 48}, {0, 0}}, {{24, 64}, {0, 0}}},
    {ge::DT_BF16, 8, 54, {{4, 16}, {8, 48}, {0, 0}}, {{54, 64}, {0, 0}}},
    {ge::DT_FLOAT16, 8, 54, {{4, 16}, {8, 48}, {0, 0}}, {{54, 64}, {0, 0}}},
};
constexpr size_t kNumSmallAxisRules = sizeof(kSmallAxisRules) / sizeof(kSmallAxisRules[0]);

uint32_t LookupMinSegs(const uint32_t (*tiers)[2], uint32_t axisNum);
const SmallAxisRule* FindSmallAxisRule(ge::DataType dataType);
bool UseTwoStageRankInverse(uint32_t axisLen);
uint32_t ComputeInsertionBytesPerSeg(ge::DataType dataType, uint32_t axisLen, uint32_t dtypeSize,
                                     uint32_t indexDtypeSize, uint32_t blockUbSize);

// SortKthTileInfo 集中保存选路和资源预算所需参数。
struct SortKthTileInfo {
    uint32_t coreNumNeed = 0;
    uint32_t lastDimTileNum = 0;
    uint32_t unsortedDimParallel = 1;
    uint32_t ubSize = 0;
    uint32_t blockUbSize = 0;
    uint32_t dtypeSize = 0;
    uint32_t y2DtypeSize = 0;
    // Kernel 最终索引的可选位宽；零表示沿用输出 dtype 位宽。
    uint32_t finalIdxUbElemBytes = 0;
    uint32_t maxCoreNum = 0;
    uint32_t numTileDataSize = 0;
    uint32_t sortLoopTimes = 0;
    uint32_t lastDimNeedCore = 0;
    uint32_t keyParams0 = 0;
    uint32_t keyParams1 = 0;
    uint32_t keyParams2 = 0;
    uint32_t keyParams3 = 0;
    uint32_t keyParams4 = 0;
    uint32_t keyParams5 = 0;
    uint32_t tmpUbSize = 0;
    bool isDescend = false;
    ge::DataType dataType = ge::DT_UINT8;
    uint32_t isInt32 = 0;
    int64_t rank = 0;
    int64_t sortAxis = 0;
    int64_t lastAxis = 1;
    int64_t unsortedDim = 1;
    uint32_t xUbSize = 0;
    uint32_t idxUbSize = 0;
    uint32_t outputRowsPerLoop = 0;
    uint64_t oneBufferQueSize = 0;
    uint32_t oneCoreTmpUbSize = 0;
    size_t workspaceSize = 0;
};

constexpr uint32_t MERGE_SORT_MAX_AXIS_FP16 = 1024;
constexpr uint32_t MERGE_SORT_MAX_AXIS_FP32 = 4096;

bool SelectSmallAxisRoute(const SortKthTileInfo& info, SmallAxisRoutePlan& plan,
                          const SmallAxisRule* ruleOverride = nullptr);

// =============================================================================
// 基数排序。
// =============================================================================
struct RadixClearParams {
    uint32_t keyParams0 = 0;
    uint32_t keyParams1 = 0;
    uint32_t keyParams2 = 0;
    uint32_t keyParams3 = 0;
    uint32_t keyParams4 = 0;
    uint32_t keyParams5 = 0;
};

bool QuerySortTmpSizeRadix(ge::DataType dataType, uint32_t sortAxisNum, uint32_t& tmpUbSize);

uint32_t ComputeRadixRemainUb(uint32_t usableUb, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor);

void AdjustRadixTmpUb(uint32_t usableUb, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor, uint32_t blockUbSize,
                      uint32_t& tmpUbSize);
bool SelectRadixCorePlan(int64_t axisLen, int64_t unsortedDim, uint32_t maxCoreNum, uint32_t tileCapacity,
                         uint32_t& tileData, uint32_t& coresPerRow);
bool ComputeRadixTileData(int64_t axisLen, int64_t unsortedDim, uint32_t dtypeSize, uint32_t indexSize,
                          uint32_t maxCoreNum, uint32_t usableUb, uint32_t blockUbSize, uint32_t& tileData,
                          uint32_t& tmpUbSize, uint32_t& coresPerRow);
bool FillRadixKernelParams(uint32_t dtypeSize, uint32_t indexSize, uint32_t coreNumNeed, uint32_t lastDimTileNum,
                           uint32_t unsortedDimParallel, uint32_t blockUbSize, uint32_t tmpUbSize,
                           RadixClearParams& out);
bool ComputeRadixSortWorkspace(int64_t axisLen, uint32_t dtypeSize, uint32_t indexSize, uint32_t lastDimTileNum,
                               uint32_t numTileDataSize, uint32_t unsortedDimParallel, uint32_t keyParams0,
                               uint32_t keyParams1, uint32_t keyParams2, uint32_t keyParams3, uint32_t keyParams4,
                               uint32_t blockUbSize, uint64_t& workspaceSize);
bool ComputeRadixOneCoreUbSizes(int64_t lastAxis, uint32_t dtypeSize, uint32_t indexElemSize, uint32_t blockUbSize,
                                uint32_t& xUbSize, uint32_t& idxUbSize);
bool FillRadixMoreCoreInfo(SortKthTileInfo& info);

// =============================================================================
// 归并排序公共参数。
// =============================================================================

bool IsMergeSortSupported(ge::DataType dataType, int64_t axisLen);

struct MergeSortPlan {
    uint32_t alignNum = 0;
    uint32_t oneCoreRowNum = 0;
    uint32_t sortLoopTimes = 0;
    uint32_t coreNumNeed = 0;
};

bool ComputeMergeSortPlan(int64_t axisLen, int64_t unsortedDim, uint32_t blockUbSize, uint32_t tileDataNum,
                          uint32_t maxCoreNum, MergeSortPlan& plan);
bool FillMergeSortInfo(SortKthTileInfo& info, uint32_t indexDtypeSize);

// =============================================================================
// 两阶段排序。
// =============================================================================
uint32_t MaxTwoStageU16SafeBatch(uint32_t axisLen);
bool ComputeTwoStageSortTmpUb(ge::DataType dataType, uint32_t axisLen, uint32_t totalElems, uint32_t blockUbSize,
                              uint32_t& tmpUbSize);
uint64_t EstimateTwoStageUbBytes(const SortKthTileInfo& info, uint32_t totalElems, uint32_t sortTmpUb);
bool PrepareTwoStageBatchCandidate(const SortKthTileInfo& info, uint32_t candidate, uint32_t& totalElems,
                                   uint32_t& tmpUbSize, bool& useRankInverse, uint64_t& totalBytes);

struct TwoStageBatchPlan {
    uint32_t batchSize = 0;
    uint32_t batchNum = 0;
    uint32_t blockDim = 0;
    uint32_t tmpUbSize = 0;
};

bool SearchTwoStageBatchPlan(uint32_t maxBatch, std::function<bool(uint32_t, TwoStageBatchPlan&)> tryCandidate,
                             TwoStageBatchPlan& result);

// =============================================================================
// Tiling 数据转换与顶层计算。
// =============================================================================
template <typename TilingData>
void PlanToTilingData(const SortKthTileInfo& info, TilingData* tiling)
{
    tiling->numTileDataSize = info.numTileDataSize;
    tiling->unsortedDimParallel = info.unsortedDimParallel;
    tiling->lastDimTileNum = info.lastDimTileNum;
    tiling->sortLoopTimes = info.sortLoopTimes;
    tiling->lastDimNeedCore = info.lastDimNeedCore;
    tiling->keyParams0 = info.keyParams0;
    tiling->keyParams1 = info.keyParams1;
    tiling->keyParams2 = info.keyParams2;
    tiling->keyParams3 = info.keyParams3;
    tiling->keyParams4 = info.keyParams4;
    tiling->keyParams5 = info.keyParams5;
    tiling->tmpUbSize = info.tmpUbSize;
    tiling->lastAxisNum = info.lastAxis;
    tiling->unsortedDimNum = info.unsortedDim;
}

ge::graphStatus SortTilingSimt(gert::TilingContext* context, int32_t maxCoreNum);

} // namespace optiling

#endif // SORT_TILING_ARCH35_H
