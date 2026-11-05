/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sort_tiling_arch35.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "atvoss/broadcast/broadcast_tiling.h"
#include "../../op_kernel/arch35/sort_tiling_data.h"
#include "../../op_kernel/arch35/sort_tiling_key.h"
#include <cmath>
#include <functional>
#include <vector>
#include "utils/extern_math_util.h"

namespace optiling {

// =============================================================================
// 带溢出保护的算术辅助函数。
// =============================================================================
// alignSize 须非零且向上对齐可在 uint64 表示；结果超出 uint32 返回 false。
bool CeilAlignUint32(uint64_t rawSize, uint32_t alignSize, uint32_t& alignedSize)
{
    uint64_t result = Ops::Base::CeilAlign(rawSize, static_cast<uint64_t>(alignSize));
    if (result > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    alignedSize = static_cast<uint32_t>(result);
    return true;
}

// divisor 须非零；商超出 uint32 返回 false，失败时不写 result。
bool CeilDivUint32(uint64_t value, uint64_t divisor, uint32_t& result)
{
    uint64_t quotient = Ops::Base::CeilDiv(value, divisor);
    if (quotient > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    result = static_cast<uint32_t>(quotient);
    return true;
}

// =============================================================================
// 短轴选路。
// =============================================================================
uint32_t LookupMinSegs(const uint32_t (*tiers)[2], uint32_t axisNum)
{
    constexpr uint32_t kMaxTiers = 4;
    for (uint32_t i = 0; i < kMaxTiers && tiers[i][0] != 0; ++i) {
        if (axisNum <= tiers[i][0])
            return tiers[i][1];
    }
    return UINT32_MAX;
}

const SmallAxisRule* FindSmallAxisRule(ge::DataType dataType)
{
    for (size_t i = 0; i < kNumSmallAxisRules; ++i) {
        if (kSmallAxisRules[i].dtype == dataType) {
            return &kSmallAxisRules[i];
        }
    }
    return nullptr;
}

bool UseTwoStageRankInverse(uint32_t axisLen)
{
    return axisLen <= TWO_STAGE_RANK_INVERSE_MAX_N;
}

static bool ComputeBf16InsertionBytesPerSeg(uint32_t axisLen, uint64_t idxBytes, uint32_t blockUbSize,
                                            uint64_t& bytesPerSeg)
{
    uint64_t castRawBytes = 0U;
    if (ge::MulOverflow(axisLen, sizeof(int16_t), castRawBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ComputeInsertionBytesPerSeg", "axisLen", std::to_string(axisLen).c_str(),
                                              "The value of axisLen must not cause cast raw byte size overflow.");
        return false;
    }
    uint64_t castBytes = Ops::Base::CeilAlign<uint64_t>(castRawBytes, blockUbSize);
    if (castBytes == 0) {
        return false;
    }
    uint64_t castRowElems = castBytes / sizeof(int16_t);
    uint64_t valueBytes = 0U;
    if (ge::MulOverflow(castRowElems, sizeof(float), valueBytes) ||
        ge::AddOverflow(valueBytes, idxBytes, bytesPerSeg) || ge::AddOverflow(bytesPerSeg, castBytes, bytesPerSeg)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "ComputeInsertionBytesPerSeg", "bytesPerSeg",
            (std::to_string(castRowElems) + ", " + std::to_string(idxBytes) + ", " + std::to_string(castBytes)).c_str(),
            "The value of bf16 bytesPerSeg must not overflow.");
        return false;
    }
    return true;
}

// 返回每行所需 UB 字节数；0 表示容量计算失败或结果不可表示。
uint32_t ComputeInsertionBytesPerSeg(ge::DataType dataType, uint32_t axisLen, uint32_t dtypeSize,
                                     uint32_t indexDtypeSize, uint32_t blockUbSize)
{
    uint64_t valueRawBytes = 0U;
    uint64_t idxRawBytes = 0U;
    if (ge::MulOverflow(axisLen, dtypeSize, valueRawBytes) || ge::MulOverflow(axisLen, indexDtypeSize, idxRawBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ComputeInsertionBytesPerSeg", "axisLen", std::to_string(axisLen).c_str(),
                                              "The value of axisLen must not cause raw byte size overflow.");
        return 0;
    }
    uint64_t valueBytes = Ops::Base::CeilAlign<uint64_t>(valueRawBytes, blockUbSize);
    uint64_t idxBytes = Ops::Base::CeilAlign<uint64_t>(idxRawBytes, blockUbSize);
    if (valueBytes == 0 || idxBytes == 0) {
        return 0;
    }
    uint64_t bytesPerSeg = 0U;
    if (ge::AddOverflow(valueBytes, idxBytes, bytesPerSeg)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ComputeInsertionBytesPerSeg", "bytesPerSeg",
                                              (std::to_string(valueBytes) + ", " + std::to_string(idxBytes)).c_str(),
                                              "The value of valueBytes plus idxBytes must not overflow.");
        return 0;
    }
    if (dataType == ge::DT_BF16 && !ComputeBf16InsertionBytesPerSeg(axisLen, idxBytes, blockUbSize, bytesPerSeg)) {
        return 0;
    }
    if (bytesPerSeg > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("ComputeInsertionBytesPerSeg", "bytesPerSeg",
                                              std::to_string(bytesPerSeg).c_str(),
                                              "The value of bytesPerSeg must be less than or equal to uint32 max.");
        return 0;
    }
    return static_cast<uint32_t>(bytesPerSeg);
}

// =============================================================================
// 归并能力检查。
// =============================================================================
bool IsMergeSortSupported(ge::DataType dataType, int64_t axisLen)
{
    return (((dataType == ge::DT_FLOAT16) || (dataType == ge::DT_BF16)) &&
            (axisLen <= static_cast<int64_t>(MERGE_SORT_MAX_AXIS_FP16))) ||
           ((dataType == ge::DT_FLOAT) && (axisLen <= static_cast<int64_t>(MERGE_SORT_MAX_AXIS_FP32)));
}

// =============================================================================
// 基数排序。
// =============================================================================
uint32_t ComputeRadixRemainUb(uint32_t usableUb, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor)
{
    uint64_t usedUb = static_cast<uint64_t>(ubExtra) + static_cast<uint64_t>(tileFactor) * tileData;
    if (usedUb >= static_cast<uint64_t>(usableUb)) {
        return 0;
    }
    return usableUb - static_cast<uint32_t>(usedUb);
}

void AdjustRadixTmpUb(uint32_t usableUb, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor, uint32_t blockUbSize,
                      uint32_t& tmpUbSize)
{
    if (blockUbSize == 0U) {
        return;
    }
    uint32_t remainUb = ComputeRadixRemainUb(usableUb, tileData, ubExtra, tileFactor);
    remainUb = remainUb > tmpUbSize ? remainUb - tmpUbSize : 0U;
    remainUb = remainUb > blockUbSize ? remainUb - blockUbSize : 0U;
    tmpUbSize += (remainUb / blockUbSize) * blockUbSize;
}

bool QuerySortTmpSizeRadix(ge::DataType dataType, uint32_t sortAxisNum, uint32_t& tmpUbSize)
{
    std::vector<int64_t> shapeVec = {static_cast<int64_t>(sortAxisNum)};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = false;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    tmpUbSize = maxValue;
    return maxValue > 0;
}

bool SelectRadixCorePlan(int64_t axisLen, int64_t unsortedDim, uint32_t maxCoreNum, uint32_t tileCapacity,
                         uint32_t& tileData, uint32_t& coresPerRow)
{
    if (axisLen <= 0 || unsortedDim <= 0 || maxCoreNum == 0U || tileCapacity < BIN_NUM ||
        tileCapacity % BIN_NUM != 0U) {
        return false;
    }
    // 1024 用于限制过度切分；同时考虑行数和每核串行块数，不预设每行一核或每行占满全部核。
    const uint64_t n = static_cast<uint64_t>(axisLen);
    const uint64_t m = static_cast<uint64_t>(unsortedDim);
    const uint64_t maxAxisCores =
        std::min(static_cast<uint64_t>(maxCoreNum), std::max(uint64_t{1}, n / SMALL_TILE_DATA_NUM));
    uint64_t bestWaves = UINT64_MAX;
    uint64_t bestRounds = UINT64_MAX;
    uint32_t bestTile = UINT32_MAX;
    bool found = false;
    for (uint64_t candidate = 1; candidate <= maxAxisCores; ++candidate) {
        const uint64_t target = Ops::Base::CeilAlign(Ops::Base::CeilDiv(n, candidate), uint64_t{BIN_NUM});
        const uint32_t tile = static_cast<uint32_t>(std::min(target, uint64_t{tileCapacity}));
        const uint64_t tiles = Ops::Base::CeilDiv(n, uint64_t{tile});
        const uint64_t g = std::min(candidate, tiles);
        const uint64_t parallelRows = std::min(m, uint64_t{maxCoreNum} / g);
        const uint64_t rounds = Ops::Base::CeilDiv(m, parallelRows);
        const uint64_t localTiles = Ops::Base::CeilDiv(tiles, g);
        uint64_t waves = 0;
        if (ge::MulOverflow(rounds, localTiles, waves)) {
            continue;
        }
        // 优先减少串行块轮数，再比较补齐后元素数及全局批次数；这是选路启发式，并非周期模型。
        if (!found || waves < bestWaves ||
            (waves == bestWaves && (tile < bestTile || (tile == bestTile && rounds < bestRounds)))) {
            tileData = tile;
            coresPerRow = static_cast<uint32_t>(g);
            bestWaves = waves;
            bestTile = tile;
            bestRounds = rounds;
            found = true;
        }
    }
    return found;
}

bool ComputeRadixTileData(int64_t axisLen, int64_t unsortedDim, uint32_t dtypeSize, uint32_t indexSize,
                          uint32_t maxCoreNum, uint32_t usableUb, uint32_t blockUbSize, uint32_t& tileData,
                          uint32_t& tmpUbSize, uint32_t& coresPerRow)
{
    if (axisLen <= 0 || unsortedDim <= 0 || maxCoreNum == 0U || dtypeSize == 0U || indexSize == 0U ||
        blockUbSize == 0U) {
        return false;
    }
    // 固定桶区：inQueueGlobalHist_、blockExclusiveInQue_、blockHistInQue_、
    // blockUbFlagQue_、blockHistFlagUbQue_，对应下面五项；indexSize 固定为 int64 字节数。
    const uint32_t ubExtra = BIN_NUM * (indexSize + static_cast<uint32_t>(sizeof(uint16_t)) +
                                        static_cast<uint32_t>(sizeof(uint16_t)) + indexSize + indexSize);
    // 每元素区：inQueueX_、inQueueIndex_、outIdxQueue_、outValueQueue_、inputB8Que_。
    // tmpUb_（Sort scratch）另行查询并从剩余 UB 扣除。
    const uint32_t tileFactor = dtypeSize + indexSize + static_cast<uint32_t>(sizeof(uint32_t)) +
                                static_cast<uint32_t>(sizeof(uint8_t)) + static_cast<uint32_t>(sizeof(uint8_t));
    if (usableUb <= ubExtra || tileFactor == 0U) {
        return false;
    }
    const uint64_t alignedAxis = Ops::Base::CeilAlign(static_cast<uint64_t>(axisLen), uint64_t{BIN_NUM});
    uint32_t capacityTile = ((usableUb - ubExtra) / tileFactor) / BIN_NUM * BIN_NUM;
    capacityTile = static_cast<uint32_t>(std::min(alignedAxis, uint64_t{capacityTile}));
    for (; capacityTile >= BIN_NUM; capacityTile -= BIN_NUM) {
        uint32_t capacityScratch = 0;
        if (!QuerySortTmpSizeRadix(ge::DT_UINT8, capacityTile, capacityScratch)) {
            return false;
        }
        if (capacityScratch > ComputeRadixRemainUb(usableUb, capacityTile, ubExtra, tileFactor)) {
            continue;
        }
        if (!SelectRadixCorePlan(axisLen, unsortedDim, maxCoreNum, capacityTile, tileData, coresPerRow)) {
            return false;
        }
        tmpUbSize = capacityScratch;
        if (tileData != capacityTile && !QuerySortTmpSizeRadix(ge::DT_UINT8, tileData, tmpUbSize)) {
            return false;
        }
        if (tmpUbSize > ComputeRadixRemainUb(usableUb, tileData, ubExtra, tileFactor)) {
            // 实际 scratch 查询超过 UB 预算时重新规划，不能保留不可执行的候选。
            capacityTile = tileData;
            continue;
        }
        AdjustRadixTmpUb(usableUb, tileData, ubExtra, tileFactor, blockUbSize, tmpUbSize);
        return true;
    }
    return false;
}

bool FillRadixKernelParams(uint32_t dtypeSize, uint32_t indexSize, uint32_t coreNumNeed, uint32_t lastDimTileNum,
                           uint32_t unsortedDimParallel, uint32_t blockUbSize, uint32_t tmpUbSize,
                           RadixClearParams& out)
{
    if (indexSize == 0U) {
        return false;
    }
    uint32_t ubSizeNum = tmpUbSize / indexSize;
    if (ubSizeNum == 0U) {
        return false;
    }
    uint64_t allNumGlobalHist64 = static_cast<uint64_t>(BIN_NUM) * lastDimTileNum * dtypeSize * unsortedDimParallel;
    uint64_t allNumExclusiveBin64 = static_cast<uint64_t>(BIN_NUM) * dtypeSize * unsortedDimParallel;
    if (allNumGlobalHist64 > std::numeric_limits<uint32_t>::max() ||
        allNumExclusiveBin64 > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    uint32_t allNumGlobalHist = static_cast<uint32_t>(allNumGlobalHist64);
    uint32_t allNumExclusiveBin = static_cast<uint32_t>(allNumExclusiveBin64);
    // keyParams0/2/3/5 描述 globalHist 清零：核数、每次 UB 元素数、循环次数、每核元素数。
    // Kernel 的 clearCore1_/clearCoreSize1_ 中后缀 1 表示 globalHist，不是 keyParams 下标。
    uint32_t oneCoreSize = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(allNumGlobalHist), static_cast<uint64_t>(coreNumNeed)));
    out.keyParams5 = std::max(oneCoreSize, blockUbSize);
    out.keyParams0 = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(allNumGlobalHist), static_cast<uint64_t>(out.keyParams5)));
    out.keyParams3 = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(out.keyParams5), static_cast<uint64_t>(ubSizeNum)));
    out.keyParams2 = out.keyParams5 > ubSizeNum ? ubSizeNum : out.keyParams5;
    // keyParams1/4 描述 exclusiveBins 清零：核数、每核元素数；Kernel 使用后缀 0。
    // 两组大小的单位均为计数元素数（每项 indexSize 字节）；不是字节数。
    uint32_t oneCoreSize1 = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(allNumExclusiveBin), static_cast<uint64_t>(coreNumNeed)));
    out.keyParams4 = std::max(oneCoreSize1, blockUbSize);
    out.keyParams1 = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<uint64_t>(allNumExclusiveBin), static_cast<uint64_t>(out.keyParams4)));
    return true;
}

bool ComputeRadixSortWorkspace(int64_t axisLen, uint32_t dtypeSize, uint32_t indexSize, uint32_t lastDimTileNum,
                               uint32_t numTileDataSize, uint32_t unsortedDimParallel, uint32_t keyParams0,
                               uint32_t keyParams1, uint32_t keyParams2, uint32_t keyParams3, uint32_t keyParams4,
                               uint32_t blockUbSize, uint64_t& workspaceSize)
{
    uint64_t indexSize64 = static_cast<uint64_t>(indexSize);
    uint64_t blockUbSize64 = static_cast<uint64_t>(blockUbSize);
    uint64_t unsortedDimParallel64 = static_cast<uint64_t>(unsortedDimParallel);
    uint64_t axisLen64 = static_cast<uint64_t>(axisLen);

    uint64_t exclusiveBins =
        Ops::Base::CeilAlign(static_cast<uint64_t>(keyParams1) * keyParams4 * indexSize64, blockUbSize64);
    uint64_t globalHist =
        Ops::Base::CeilAlign(static_cast<uint64_t>(keyParams3) * keyParams2 * keyParams0 * indexSize64, blockUbSize64);
    uint64_t sortedIdx = Ops::Base::CeilAlign(axisLen64 * unsortedDimParallel64 * indexSize64, blockUbSize64);
    uint64_t histTile = static_cast<uint64_t>(lastDimTileNum) * BIN_NUM * unsortedDimParallel64 * sizeof(uint16_t) * 2U;
    uint64_t xB8 = Ops::Base::CeilAlign(static_cast<uint64_t>(lastDimTileNum) * numTileDataSize * unsortedDimParallel64,
                                        blockUbSize64);
    uint64_t sortedValue =
        Ops::Base::CeilAlign(axisLen64 * unsortedDimParallel64 * static_cast<uint64_t>(dtypeSize), blockUbSize64);

    workspaceSize = exclusiveBins + globalHist + sortedIdx + histTile + xB8 + sortedValue;
    return true;
}

static bool FillRadixKernelResources(SortKthTileInfo& info, uint32_t indexSize, uint32_t lastDimTileNum,
                                     uint32_t tmpUbSize)
{
    RadixClearParams clearParams;
    if (!FillRadixKernelParams(info.dtypeSize, indexSize, info.coreNumNeed, lastDimTileNum, info.unsortedDimParallel,
                               info.blockUbSize, tmpUbSize, clearParams)) {
        return false;
    }
    info.keyParams0 = clearParams.keyParams0;
    info.keyParams1 = clearParams.keyParams1;
    info.keyParams2 = clearParams.keyParams2;
    info.keyParams3 = clearParams.keyParams3;
    info.keyParams4 = clearParams.keyParams4;
    info.keyParams5 = clearParams.keyParams5;
    uint64_t sortWorkspaceSize = 0;
    if (!ComputeRadixSortWorkspace(info.lastAxis, info.dtypeSize, indexSize, lastDimTileNum, info.numTileDataSize,
                                   info.unsortedDimParallel, info.keyParams0, info.keyParams1, info.keyParams2,
                                   info.keyParams3, info.keyParams4, info.blockUbSize, sortWorkspaceSize)) {
        return false;
    }
    info.workspaceSize = static_cast<size_t>(sortWorkspaceSize + WORK_SPACE_SIZE);
    return true;
}

bool FillRadixMoreCoreInfo(SortKthTileInfo& info)
{
    if (info.lastAxis <= 0 || info.lastAxis > RADIX_INT64_VALUE_MAX) {
        return false;
    }
    // 高两位留给 lookback 状态；N <= 2^30-1 时使用 32 位计数及中间索引。
    info.isInt32 = static_cast<uint32_t>(IsRadixUint32CounterRange(info.lastAxis));
    uint32_t usableUb = info.ubSize > SIMT_UB ? info.ubSize - SIMT_UB : 0;
    const uint32_t indexSize = info.isInt32 ? sizeof(uint32_t) : sizeof(int64_t);
    uint32_t tileData = 0;
    uint32_t tmpUbSize = 0;
    uint32_t coresPerRow = 0;
    if (!ComputeRadixTileData(info.lastAxis, info.unsortedDim, info.dtypeSize, indexSize, info.maxCoreNum, usableUb,
                              info.blockUbSize, tileData, tmpUbSize, coresPerRow)) {
        return false;
    }
    // Sort 释放临时空间后，散射才可复用其中三个 uint64 槽位。
    if (tmpUbSize < 3U * sizeof(uint64_t)) {
        return false;
    }
    info.tmpUbSize = tmpUbSize;
    uint32_t lastDimTileNum = 0;
    if (!CeilDivUint32(static_cast<uint64_t>(info.lastAxis), static_cast<uint64_t>(tileData), lastDimTileNum)) {
        return false;
    }
    info.lastDimNeedCore = std::min(coresPerRow, lastDimTileNum);
    if (info.lastDimNeedCore == 0U) {
        return false;
    }
    info.unsortedDimParallel = static_cast<uint32_t>(std::min(
        static_cast<uint64_t>(info.unsortedDim), static_cast<uint64_t>(info.maxCoreNum / info.lastDimNeedCore)));
    info.numTileDataSize = tileData;
    uint64_t sortLoopTimes64 =
        (static_cast<uint64_t>(info.unsortedDim) + info.unsortedDimParallel - 1U) / info.unsortedDimParallel;
    if (sortLoopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    info.sortLoopTimes = static_cast<uint32_t>(sortLoopTimes64);
    info.coreNumNeed = info.unsortedDimParallel * info.lastDimNeedCore;
    info.lastDimTileNum = lastDimTileNum;
    return FillRadixKernelResources(info, indexSize, lastDimTileNum, tmpUbSize);
}

bool ComputeRadixOneCoreUbSizes(int64_t lastAxis, uint32_t dtypeSize, uint32_t indexElemSize, uint32_t blockUbSize,
                                uint32_t& xUbSize, uint32_t& idxUbSize)
{
    uint64_t xBytes = static_cast<uint64_t>(lastAxis) * dtypeSize;
    uint64_t idxBytes = static_cast<uint64_t>(lastAxis) * indexElemSize;
    return CeilAlignUint32(xBytes, blockUbSize, xUbSize) && CeilAlignUint32(idxBytes, blockUbSize, idxUbSize);
}

// =============================================================================
// 归并排序公共参数。
// =============================================================================
bool ComputeMergeSortPlan(int64_t axisLen, int64_t unsortedDim, uint32_t blockUbSize, uint32_t tileDataNum,
                          uint32_t maxCoreNum, MergeSortPlan& plan)
{
    uint64_t alignNum64 = Ops::Base::CeilAlign(static_cast<uint64_t>(axisLen), static_cast<uint64_t>(blockUbSize));
    if (alignNum64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) || alignNum64 == 0U) {
        return false;
    }
    plan.alignNum = static_cast<uint32_t>(alignNum64);
    uint32_t oneCoreRowNumMax = (tileDataNum / 2U) / plan.alignNum;
    oneCoreRowNumMax = oneCoreRowNumMax == 0U ? 1U : oneCoreRowNumMax;
    if (unsortedDim > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) || unsortedDim <= 0) {
        return false;
    }
    uint32_t bUnsorted = static_cast<uint32_t>(unsortedDim);
    plan.coreNumNeed = std::min(bUnsorted, maxCoreNum);
    uint32_t batchPerCore = static_cast<uint32_t>(
        Ops::Base::CeilDiv(static_cast<int64_t>(bUnsorted), static_cast<int64_t>(plan.coreNumNeed)));
    uint32_t oneCoreRowNum = batchPerCore <= oneCoreRowNumMax ? batchPerCore : oneCoreRowNumMax;
    uint64_t totalRowsPerRound = static_cast<uint64_t>(plan.coreNumNeed) * oneCoreRowNum;
    if (totalRowsPerRound == 0U) {
        return false;
    }
    uint64_t sortLoopTimes64 = (static_cast<uint64_t>(bUnsorted) + totalRowsPerRound - 1U) / totalRowsPerRound;
    if (sortLoopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    plan.oneCoreRowNum = oneCoreRowNum;
    plan.sortLoopTimes = static_cast<uint32_t>(sortLoopTimes64);
    return true;
}

bool FillMergeSortInfo(SortKthTileInfo& info, uint32_t indexDtypeSize)
{
    constexpr uint32_t SORT_TILE_DATA_NUM = 4096;
    uint32_t tileDataNum = SORT_TILE_DATA_NUM;
    MergeSortPlan plan;
    if (!ComputeMergeSortPlan(info.lastAxis, info.unsortedDim, info.blockUbSize, tileDataNum, info.maxCoreNum, plan)) {
        return false;
    }
    info.sortLoopTimes = plan.sortLoopTimes;
    info.lastDimTileNum = 1;
    info.unsortedDimParallel = plan.coreNumNeed;
    info.lastDimNeedCore = 1;
    info.numTileDataSize = static_cast<uint32_t>(info.lastAxis);
    info.coreNumNeed = plan.coreNumNeed;
    info.keyParams0 = plan.oneCoreRowNum;
    info.keyParams1 = plan.alignNum * plan.oneCoreRowNum * info.dtypeSize;
    info.keyParams2 = plan.alignNum * plan.oneCoreRowNum * indexDtypeSize;
    info.keyParams3 = plan.alignNum;
    info.keyParams4 = info.lastAxis > ONE_CORE_DATA_SIZE ? 1 : DOUBLE_BUFFER_NUM;
    info.tmpUbSize = 0;
    info.workspaceSize = WORK_SPACE_SIZE;
    return true;
}

// =============================================================================
// 两阶段排序。
// =============================================================================
bool SearchTwoStageBatchPlan(uint32_t maxBatch, std::function<bool(uint32_t, TwoStageBatchPlan&)> tryCandidate,
                             TwoStageBatchPlan& result)
{
    if (maxBatch == 0U) {
        return false;
    }
    TwoStageBatchPlan maxPlan;
    TwoStageBatchPlan bestPlan;
    bool hasMaxPlan = false;
    uint32_t maxPlanBatchesPerCore = 0;
    uint64_t bestIdleSlots = std::numeric_limits<uint64_t>::max();
    // 降序枚举：首个可行批次定义轮数基准；之后 bestPlan 始终保持该轮数，
    // 只减少尾轮空闲槽，同分保留更大批次。tryCandidate 成功时须给出非零 blockDim。
    for (uint32_t candidate = maxBatch; candidate >= 1U; --candidate) {
        TwoStageBatchPlan candidatePlan;
        if (!tryCandidate(candidate, candidatePlan)) {
            continue;
        }
        if (!hasMaxPlan) {
            maxPlan = candidatePlan;
            hasMaxPlan = true;
            maxPlanBatchesPerCore = Ops::Base::CeilDiv(maxPlan.batchNum, maxPlan.blockDim);
            bestPlan = maxPlan;
            bestIdleSlots = static_cast<uint64_t>(maxPlanBatchesPerCore) * maxPlan.blockDim - maxPlan.batchNum;
            continue;
        }
        uint32_t batchesPerCore = Ops::Base::CeilDiv(candidatePlan.batchNum, candidatePlan.blockDim);
        // 缩小批次使任务数不减；本实现先保持执行轮数，因此不跨轮数继续搜索。
        if (batchesPerCore != maxPlanBatchesPerCore) {
            break;
        }
        uint64_t idleSlots = static_cast<uint64_t>(batchesPerCore) * candidatePlan.blockDim - candidatePlan.batchNum;
        if (idleSlots < bestIdleSlots) {
            bestPlan = candidatePlan;
            bestIdleSlots = idleSlots;
        }
    }
    if (!hasMaxPlan) {
        return false;
    }
    result = bestPlan;
    return true;
}

uint32_t MaxTwoStageU16SafeBatch(uint32_t axisLen)
{
    if (axisLen == 0U || axisLen > TWO_STAGE_U16_KEY_SPACE) {
        return 0U;
    }
    uint32_t maxBatch = static_cast<uint32_t>(std::sqrt(static_cast<double>(TWO_STAGE_U16_KEY_SPACE) / axisLen));
    while (static_cast<uint64_t>(maxBatch) * maxBatch * axisLen > static_cast<uint64_t>(TWO_STAGE_U16_KEY_SPACE)) {
        --maxBatch;
    }
    while (static_cast<uint64_t>(maxBatch + 1U) * (maxBatch + 1U) * axisLen <=
           static_cast<uint64_t>(TWO_STAGE_U16_KEY_SPACE)) {
        ++maxBatch;
    }
    return maxBatch;
}

bool ComputeTwoStageSortTmpUb(ge::DataType dataType, uint32_t axisLen, uint32_t totalElems, uint32_t blockUbSize,
                              uint32_t& tmpUbSize)
{
    tmpUbSize = 0;
    QuerySortTmpSizeRadix(dataType, totalElems, tmpUbSize);
    uint32_t aligned = 0;
    if (!CeilAlignUint32(tmpUbSize, blockUbSize, aligned)) {
        return false;
    }
    tmpUbSize = aligned;
    bool useRankInverse = UseTwoStageRankInverse(axisLen);
    if (!useRankInverse) {
        uint32_t stage2TmpUbSize = 0;
        QuerySortTmpSizeRadix(ge::DT_UINT16, totalElems, stage2TmpUbSize);
        uint32_t stage2Aligned = 0;
        if (!CeilAlignUint32(stage2TmpUbSize, blockUbSize, stage2Aligned)) {
            return false;
        }
        tmpUbSize = std::max(tmpUbSize, stage2Aligned);
    }
    return true;
}

// 返回完整批次的 UB 字节数；UINT64_MAX 表示预算无效或溢出。
uint64_t EstimateTwoStageUbBytes(const SortKthTileInfo& info, uint32_t totalElems, uint32_t sortTmpUb)
{
    uint64_t valueRawBytes = 0U;
    uint64_t idxRawBytes = 0U;
    uint64_t finalIdxRawBytes = 0U;
    uint32_t finalIdxElemBytes = info.finalIdxUbElemBytes == 0U ? info.y2DtypeSize : info.finalIdxUbElemBytes;
    if (ge::MulOverflow(totalElems, info.dtypeSize, valueRawBytes) ||
        ge::MulOverflow(totalElems, sizeof(uint32_t), idxRawBytes) ||
        ge::MulOverflow(totalElems, finalIdxElemBytes, finalIdxRawBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON("EstimateTwoStageUbBytes", "totalElems",
                                              std::to_string(totalElems).c_str(),
                                              "The value of totalElems must not cause raw byte size overflow.");
        return std::numeric_limits<uint64_t>::max();
    }
    uint64_t valueBytes = Ops::Base::CeilAlign<uint64_t>(valueRawBytes, info.blockUbSize);
    uint64_t idxBytes = Ops::Base::CeilAlign<uint64_t>(idxRawBytes, info.blockUbSize);
    uint64_t finalIdxBytes = Ops::Base::CeilAlign<uint64_t>(finalIdxRawBytes, info.blockUbSize);
    if (valueBytes == 0U || idxBytes == 0U || finalIdxBytes == 0U) {
        return std::numeric_limits<uint64_t>::max();
    }
    uint32_t idxBufferCount = UseTwoStageRankInverse(static_cast<uint32_t>(info.lastAxis)) ? 2U : 3U;
    uint64_t totalBytes = 0U;
    uint64_t idxTotalBytes = 0U;
    if (ge::MulOverflow(valueBytes, 2U, totalBytes) || ge::MulOverflow(idxBytes, idxBufferCount, idxTotalBytes) ||
        ge::AddOverflow(totalBytes, idxTotalBytes, totalBytes) ||
        ge::AddOverflow(totalBytes, finalIdxBytes, totalBytes) || ge::AddOverflow(totalBytes, sortTmpUb, totalBytes)) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            "EstimateTwoStageUbBytes", "totalBytes",
            (std::to_string(valueBytes) + ", " + std::to_string(idxBytes) + ", " + std::to_string(idxBufferCount) +
             ", " + std::to_string(finalIdxBytes) + ", " + std::to_string(sortTmpUb))
                .c_str(),
            "The value of totalBytes must not overflow.");
        return std::numeric_limits<uint64_t>::max();
    }
    return totalBytes;
}

bool PrepareTwoStageBatchCandidate(const SortKthTileInfo& info, uint32_t candidate, uint32_t& totalElems,
                                   uint32_t& tmpUbSize, bool& useRankInverse, uint64_t& totalBytes)
{
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    uint64_t totalElems64 = static_cast<uint64_t>(candidate) * axisLen;
    if (totalElems64 > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    totalElems = static_cast<uint32_t>(totalElems64);
    tmpUbSize = 0;
    if (!ComputeTwoStageSortTmpUb(info.dataType, axisLen, totalElems, info.blockUbSize, tmpUbSize)) {
        return false;
    }
    useRankInverse = UseTwoStageRankInverse(axisLen);
    totalBytes = EstimateTwoStageUbBytes(info, totalElems, tmpUbSize);
    return true;
}

static bool ComputeSmallAxisInsertionBatchParams(const SortKthTileInfo& info, uint32_t axisLen, uint32_t& bytesPerSeg,
                                                 uint32_t& usableUb, uint32_t& maxBatchByUb)
{
    if (info.ubSize <= SIMT_UB) {
        return false;
    }
    bytesPerSeg =
        ComputeInsertionBytesPerSeg(info.dataType, axisLen, info.dtypeSize, info.y2DtypeSize, info.blockUbSize);
    if (bytesPerSeg == 0U) {
        return false;
    }
    usableUb = info.ubSize - SIMT_UB;
    maxBatchByUb = usableUb / bytesPerSeg;
    return true;
}

template <typename ComputeBatchNumFn>
static bool EstimateSmallAxisInsertionBatching(const SortKthTileInfo& info, uint32_t batchSizeCap,
                                               ComputeBatchNumFn computeBatchNum, SmallAxisRoutePlan& plan)
{
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    uint32_t bytesPerSeg = 0;
    uint32_t usableUb = 0;
    uint32_t maxBatchByUb = 0;
    if (!ComputeSmallAxisInsertionBatchParams(info, axisLen, bytesPerSeg, usableUb, maxBatchByUb) ||
        maxBatchByUb == 0U) {
        return false;
    }
    uint32_t batchSize = std::min({batchSizeCap, maxBatchByUb, SMALL_AXIS_MAX_DATACOPY_BLOCK_COUNT});
    if (batchSize == 0U) {
        return false;
    }
    plan.batchSize = batchSize;
    if (!computeBatchNum(batchSize, plan.batchNum)) {
        return false;
    }
    plan.blockDim = std::min(info.maxCoreNum, plan.batchNum);
    return plan.batchNum > 0U && plan.blockDim > 0U;
}

template <typename ComputeBatchNumFn>
static bool TrySmallAxisTwoStageBatchCandidate(const SortKthTileInfo& info, uint32_t candidate,
                                               ComputeBatchNumFn computeBatchNum, SmallAxisRoutePlan& plan)
{
    // 查询 Sort 临时 UB 前，先排除批次映射不支持的候选。
    plan.batchSize = candidate;
    if (!computeBatchNum(candidate, plan.batchNum)) {
        return false;
    }
    uint32_t totalElems = 0;
    uint32_t tmpUbSize = 0;
    bool useRankInverse = false;
    uint64_t totalBytes = 0;
    if (!PrepareTwoStageBatchCandidate(info, candidate, totalElems, tmpUbSize, useRankInverse, totalBytes)) {
        return false;
    }
    if (totalBytes + SIMT_UB > info.ubSize) {
        return false;
    }
    plan.blockDim = std::min(info.maxCoreNum, plan.batchNum);
    plan.tmpUbSize = tmpUbSize;
    plan.useRankInverse = useRankInverse;
    return plan.batchNum > 0U && plan.blockDim > 0U;
}

// 估算短轴两阶段排序的批次大小。
//
// 每批将 batchSize × axisLen 个元素整体排序，随后按逆排名或紧凑键恢复各行顺序。
//
// 每元素的粗略 UB 下界：两份值缓冲、idxBufferCount 份 uint32 索引及一份别名索引。
// 逆排名分支使用两份索引，紧凑键分支使用三份；最终以精确预算检查为准。
//
// 第二阶段使用 uint16 键，必须满足 batch² × axisLen - 1 ≤ UINT16_MAX。
// MaxTwoStageU16SafeBatch() 返回满足此约束的最大批次。
//
// 确定 UB 容量上界后，按既有行批次策略搜索。
template <typename ComputeBatchNumFn>
static bool EstimateSmallAxisTwoStageBatching(const SortKthTileInfo& info, uint32_t batchSizeCap,
                                              ComputeBatchNumFn computeBatchNum, SmallAxisRoutePlan& plan)
{
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    if (info.ubSize <= SIMT_UB || axisLen == 0U || batchSizeCap == 0U) {
        return false;
    }
    bool useRankInverse = UseTwoStageRankInverse(axisLen);
    // 逆排名分支需两份索引缓冲，紧凑键分支需三份。
    uint32_t idxBufferCount = useRankInverse ? 2U : 3U;
    // 粗略下界用于搜索连续行批次；最终按实际输出索引宽度调用 EstimateTwoStageUbBytes 验证。
    uint32_t finalIdxElemBytes =
        info.finalIdxUbElemBytes == 0U ? static_cast<uint32_t>(sizeof(uint32_t)) : info.finalIdxUbElemBytes;
    uint64_t minBytesPerElem = static_cast<uint64_t>(info.dtypeSize) * 2U +
                               static_cast<uint64_t>(sizeof(uint32_t)) * idxBufferCount + finalIdxElemBytes;
    uint64_t maxElemsByUb = (info.ubSize - SIMT_UB) / minBytesPerElem;
    uint64_t maxBatchByUb = maxElemsByUb / axisLen;
    uint32_t maxBatch = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(batchSizeCap), maxBatchByUb));
    if (!useRankInverse) {
        // 限制批次大小，避免第二阶段 uint16 键溢出。
        maxBatch = std::min(maxBatch, MaxTwoStageU16SafeBatch(axisLen));
    }
    TwoStageBatchPlan result;
    auto tryCandidate = [&info, &computeBatchNum](uint32_t candidate, TwoStageBatchPlan& p) -> bool {
        SmallAxisRoutePlan candidatePlan;
        if (!TrySmallAxisTwoStageBatchCandidate(info, candidate, computeBatchNum, candidatePlan)) {
            return false;
        }
        p.batchSize = candidatePlan.batchSize;
        p.batchNum = candidatePlan.batchNum;
        p.blockDim = candidatePlan.blockDim;
        p.tmpUbSize = candidatePlan.tmpUbSize;
        return true;
    };
    bool foundPlan = SearchTwoStageBatchPlan(maxBatch, tryCandidate, result);
    if (!foundPlan) {
        return false;
    }
    plan.batchSize = result.batchSize;
    plan.batchNum = result.batchNum;
    plan.blockDim = result.blockDim;
    plan.tmpUbSize = result.tmpUbSize;
    plan.useRankInverse = useRankInverse;
    return true;
}

// =============================================================================
// 短轴路线选择。
// =============================================================================
static bool SelectSmallAxisRouteImpl(const SortKthTileInfo& info, uint32_t batchSizeCap,
                                     std::function<bool(uint32_t, uint32_t&)> computeBatchNum, SmallAxisRoutePlan& plan,
                                     const SmallAxisRule* ruleOverride = nullptr)
{
    uint32_t axisLen = static_cast<uint32_t>(info.lastAxis);
    if (axisLen == 0U) {
        return false;
    }
    const SmallAxisRule* rule = ruleOverride == nullptr ? FindSmallAxisRule(info.dataType) : ruleOverride;
    if (rule == nullptr) {
        return false;
    }
    uint32_t fullCoreSegs = 0;
    if (!CeilDivUint32(static_cast<uint64_t>(info.unsortedDim), static_cast<uint64_t>(info.maxCoreNum), fullCoreSegs)) {
        return false;
    }
    // 档位门槛已筛选出适合合批的区域，先尝试两阶段以摊薄逐行开销；
    // 资源不足再尝试适合极短轴的插入。此优先级是示例策略，不是所有设备上的性能保证。
    SmallAxisRoutePlan twoStagePlan;
    if (axisLen > 1U && rule->twoStageMaxN > 0U && axisLen <= rule->twoStageMaxN && axisLen <= SMALL_AXIS_THRESHOLD &&
        fullCoreSegs >= LookupMinSegs(rule->twoStageTiers, axisLen) &&
        EstimateSmallAxisTwoStageBatching(info, batchSizeCap, computeBatchNum, twoStagePlan)) {
        plan = twoStagePlan;
        plan.kind = SmallAxisRouteKind::TWO_STAGE;
        return true;
    }
    if (axisLen > rule->insertionMaxN ||
        (axisLen > 1U && fullCoreSegs < LookupMinSegs(rule->insertionTiers, axisLen))) {
        return false;
    }
    SmallAxisRoutePlan insertionPlan;
    if (!EstimateSmallAxisInsertionBatching(info, batchSizeCap, computeBatchNum, insertionPlan)) {
        return false;
    }
    plan = insertionPlan;
    plan.kind = SmallAxisRouteKind::INSERTION;
    return true;
}

bool SelectSmallAxisRoute(const SortKthTileInfo& info, SmallAxisRoutePlan& plan, const SmallAxisRule* ruleOverride)
{
    uint32_t fullCoreSegs = 0;
    if (!CeilDivUint32(static_cast<uint64_t>(info.unsortedDim), static_cast<uint64_t>(info.maxCoreNum), fullCoreSegs)) {
        return false;
    }
    auto computeBatchNum = [&info](uint32_t batchSize, uint32_t& batchNum) -> bool {
        return CeilDivUint32(static_cast<uint64_t>(info.unsortedDim), batchSize, batchNum);
    };
    return SelectSmallAxisRouteImpl(info, fullCoreSegs, computeBatchNum, plan, ruleOverride);
}

// =============================================================================
// 参数校验。
// =============================================================================
ge::graphStatus CheckInputAndOutput(gert::TilingContext* context, SortKthTileInfo& sortTileInfo)
{
    auto platformInfo = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfo);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    OP_CHECK_IF(ubSize <= static_cast<uint64_t>(SIMT_UB),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "ubSize", std::to_string(ubSize).c_str(),
                                                      "The value of ubSize must be greater than SIMT_UB."),
                return ge::GRAPH_FAILED);
    sortTileInfo.blockUbSize = Ops::Base::GetUbBlockSize(context);
    OP_CHECK_IF(sortTileInfo.blockUbSize == 0 || ubSize > std::numeric_limits<uint32_t>::max(),
                OP_LOGE(context->GetNodeName(), "Invalid UB size or alignment."), return ge::GRAPH_FAILED);
    OP_LOGI(context->GetNodeName(), "ubSize is %ld, blockUbSize %u", ubSize, sortTileInfo.blockUbSize);
    sortTileInfo.ubSize = static_cast<uint32_t>(ubSize);
    auto inputShapePtr = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShapePtr);
    const gert::Shape& inputShape = Ops::Base::EnsureNotScalar(inputShapePtr->GetStorageShape());
    auto yStorage = context->GetOutputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage);
    const gert::Shape& outShape = Ops::Base::EnsureNotScalar(yStorage->GetStorageShape());
    auto yStorage1 = context->GetOutputShape(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, yStorage1);
    const gert::Shape& outShape1 = Ops::Base::EnsureNotScalar(yStorage1->GetStorageShape());
    OP_CHECK_IF(
        inputShape.GetShapeSize() == 0 || outShape.GetShapeSize() == 0,
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(
            context->GetNodeName(), "x, y1",
            (std::to_string(inputShape.GetShapeSize()) + ", " + std::to_string(outShape.GetShapeSize())).c_str(),
            "The values of shape sizes of x and y1 must be positive."),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(
        outShape != outShape1 || outShape != inputShape,
        OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(context->GetNodeName(), "x, y1, y2",
                                               (Ops::Base::ToString(inputShape) + ", " + Ops::Base::ToString(outShape) +
                                                ", " + Ops::Base::ToString(outShape1))
                                                   .c_str(),
                                               "The shapes of x, y1, and y2 must be the same."),
        return ge::GRAPH_FAILED);
    int32_t xDimNum = inputShape.GetDimNum();
    sortTileInfo.rank = xDimNum;
    sortTileInfo.sortAxis = xDimNum - 1;
    sortTileInfo.lastAxis = inputShape.GetDim(xDimNum - 1);
    sortTileInfo.unsortedDim = inputShape.GetShapeSize() / sortTileInfo.lastAxis;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CheckSortOutputDtypes(gert::TilingContext* context, SortKthTileInfo& sortTileInfo,
                                      ge::DataType dataType)
{
    auto outDescPtr = context->GetOutputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDescPtr);
    auto y2DType = outDescPtr->GetDataType();
    auto outDescPtr0 = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, outDescPtr0);
    auto y1DType = outDescPtr0->GetDataType();
    OP_CHECK_IF(
        (y2DType != ge::DT_INT64) && (y2DType != ge::DT_INT32),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "y2", Ops::Base::ToString(y2DType).c_str(), "INT32 or INT64"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(y1DType != dataType,
                OP_LOGE_FOR_INVALID_DTYPES_WITH_REASON(
                    context->GetNodeName(), "x, y1",
                    (Ops::Base::ToString(dataType) + ", " + Ops::Base::ToString(y1DType)).c_str(),
                    "The dtypes of x and y1 must be the same."),
                return ge::GRAPH_FAILED);
    ge::TypeUtils::GetDataTypeLength(y2DType, sortTileInfo.y2DtypeSize);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortCheckParams(gert::TilingContext* context, SortKthTileInfo& sortTileInfo)
{
    OP_CHECK_IF(CheckInputAndOutput(context, sortTileInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "CheckInputAndOutput", "GRAPH_FAILED",
                                                      "The value of CheckInputAndOutput must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    auto inputDescPtr = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputDescPtr);
    ge::DataType dataType = inputDescPtr->GetDataType();
    sortTileInfo.dataType = dataType;
    OP_CHECK_IF(
        !ge::TypeUtils::GetDataTypeLength(dataType, sortTileInfo.dtypeSize),
        OP_LOGE_FOR_INVALID_DTYPE(context->GetNodeName(), "x", Ops::Base::ToString(dataType).c_str(),
                                  "INT8, INT16, INT32, INT64, UINT8, UINT16, UINT32, UINT64, FLOAT, FLOAT16, BF16"),
        return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckSortOutputDtypes(context, sortTileInfo, dataType) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "CheckSortOutputDtypes", "GRAPH_FAILED",
                                                      "The value of CheckSortOutputDtypes must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    auto const attrs = context->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context, attrs);
    const bool* isDescending = attrs->GetAttrPointer<bool>(1);
    const int64_t* sortAxisPtr = attrs->GetAttrPointer<int64_t>(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, isDescending);
    OP_CHECK_NULL_WITH_CONTEXT(context, sortAxisPtr);
    const int64_t requestedAxis = *sortAxisPtr;
    OP_CHECK_IF(requestedAxis < -static_cast<int64_t>(sortTileInfo.rank) ||
                    requestedAxis >= static_cast<int64_t>(sortTileInfo.rank),
                OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "axis", std::to_string(requestedAxis).c_str(),
                                          "range [-dimNum, dimNum)"),
                return ge::GRAPH_FAILED);
    int32_t sortAxis = static_cast<int32_t>(requestedAxis);
    sortAxis = sortAxis < 0 ? (sortAxis + static_cast<int32_t>(sortTileInfo.rank)) : sortAxis;
    OP_CHECK_IF(sortAxis < 0 || sortAxis >= static_cast<int32_t>(sortTileInfo.rank),
                OP_LOGE_WITH_INVALID_ATTR(context->GetNodeName(), "axis", std::to_string(sortAxis).c_str(),
                                          "range [-dimNum, dimNum - 1)"),
                return ge::GRAPH_FAILED);
    sortTileInfo.sortAxis = sortAxis;
    OP_CHECK_IF(sortAxis != static_cast<int32_t>(sortTileInfo.rank) - 1,
                OP_LOGE(context->GetNodeName(), "Sort L0 requires a last-axis layout; use the L2 layout wrapper."),
                return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// UB 预算与布局辅助函数。
// =============================================================================
void SetSortTmpSize(ge::DataType dataType, uint32_t tileData, bool isDescend, SortKthTileInfo& sortTileInfo)
{
    int64_t realLen = std::min(sortTileInfo.lastAxis, static_cast<int64_t>(tileData));
    std::vector<int64_t> shapeVec = {realLen};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = isDescend;
    // 单核使用 Sort 的隐式索引重载；sourceOrder 仅在排序后用于按输出索引恢复零的符号。
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, dataType, ge::DT_UINT32, false, config, maxValue, minValue);
    OP_LOGI("RadixSortTiling", "api of sort shape is %ld, maxUb is %u", realLen, maxValue);
    sortTileInfo.tmpUbSize = maxValue;
    return;
}

bool IsRadixSortOneCore(SortKthTileInfo& sortTileInfo)
{
    if (sortTileInfo.isInt32 == static_cast<uint32_t>(0)) {
        return false;
    }
    uint32_t xUbSize = 0;
    uint32_t y2UbSize = 0;
    if (!ComputeRadixOneCoreUbSizes(sortTileInfo.lastAxis, sortTileInfo.dtypeSize,
                                    static_cast<uint32_t>(sizeof(int32_t)), sortTileInfo.blockUbSize, xUbSize,
                                    y2UbSize)) {
        return false;
    }

    // Sort 先写 uint32 索引；int64 输出还需预留另一半空间存放 Cast 结果。
    uint32_t halfNum = y2UbSize / static_cast<uint32_t>(sizeof(int32_t));
    if (sortTileInfo.y2DtypeSize == static_cast<uint32_t>(sizeof(int64_t))) {
        y2UbSize = y2UbSize * static_cast<uint32_t>(sizeof(int64_t) / sizeof(int32_t));
    }
    sortTileInfo.keyParams0 = xUbSize;
    sortTileInfo.keyParams1 = y2UbSize;
    sortTileInfo.keyParams2 = halfNum;
    sortTileInfo.keyParams3 = 1;
    int64_t sourceOrderBytes =
        NeedsSignedZeroSourceOrder(sortTileInfo.dataType) ? static_cast<int64_t>(halfNum) * sizeof(uint32_t) : 0;
    // keyParams3 表示预留 Sort 临时 UB 后，队列是否还能使用双缓冲。
    int64_t oneBufferQueSize = static_cast<int64_t>(xUbSize) * 2 + static_cast<int64_t>(y2UbSize);
    int64_t remainUb = static_cast<int64_t>(sortTileInfo.ubSize) - oneBufferQueSize - sourceOrderBytes;
    if (remainUb <= static_cast<int64_t>(0)) {
        return false;
    }
    remainUb =
        (remainUb / static_cast<int64_t>(sortTileInfo.blockUbSize)) * static_cast<int64_t>(sortTileInfo.blockUbSize);
    SetSortTmpSize(sortTileInfo.dataType, static_cast<uint32_t>(sortTileInfo.lastAxis), sortTileInfo.isDescend,
                   sortTileInfo);
    int64_t tmpUb = static_cast<int64_t>(sortTileInfo.tmpUbSize);
    OP_LOGI("RadixSortTiling", "remainUb is %ld, tmpUb is %ld", remainUb, tmpUb);
    if (tmpUb > remainUb) {
        return false;
    }

    int64_t doubleBufferRemainUb =
        static_cast<int64_t>(sortTileInfo.ubSize) - oneBufferQueSize * DOUBLE_BUFFER_NUM - sourceOrderBytes;
    doubleBufferRemainUb = (doubleBufferRemainUb / static_cast<int64_t>(sortTileInfo.blockUbSize)) *
                           static_cast<int64_t>(sortTileInfo.blockUbSize);
    if (tmpUb <= doubleBufferRemainUb) {
        sortTileInfo.keyParams3 = DOUBLE_BUFFER_NUM;
    }
    OP_LOGI("RadixSortTiling", "radix one-core bufferNum is %u", sortTileInfo.keyParams3);
    return true;
}

// =============================================================================
// 各排序策略的参数配置。
// =============================================================================
void FillSmallAxisBatched(SortKthTileInfo& sortTileInfo, const SmallAxisRoutePlan& plan)
{
    sortTileInfo.ubSize = sortTileInfo.ubSize - SIMT_UB; // 为 SIMT 核保留 32 KiB 临时空间。
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(sortTileInfo.lastAxis);
    sortTileInfo.keyParams0 = plan.batchSize;                // 每批行数。
    sortTileInfo.keyParams1 = plan.batchNum;                 // 总批次数。
    sortTileInfo.keyParams2 = plan.useRankInverse ? 1U : 0U; // 使用逆排名恢复行内顺序。
    sortTileInfo.keyParams3 = 0U;
    sortTileInfo.keyParams4 = 0U;
    sortTileInfo.coreNumNeed = plan.blockDim;
    sortTileInfo.tmpUbSize = plan.tmpUbSize;
    sortTileInfo.workspaceSize = WORK_SPACE_SIZE;
}

ge::graphStatus SetRadixOneCoreTiling(gert::TilingContext* context, SortKthTileInfo& sortTileInfo)
{
    sortTileInfo.lastDimNeedCore = static_cast<uint32_t>(1);
    sortTileInfo.numTileDataSize = static_cast<uint32_t>(sortTileInfo.lastAxis);
    sortTileInfo.lastDimTileNum = static_cast<uint32_t>(1);
    uint64_t sortLoopTimes64 =
        Ops::Base::CeilDiv(sortTileInfo.unsortedDim, static_cast<int64_t>(sortTileInfo.maxCoreNum));
    if (sortLoopTimes64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "sortLoopTimes",
                                              std::to_string(sortLoopTimes64).c_str(),
                                              "The value of sortLoopTimes must be less than or equal to uint32 max.");
        return ge::GRAPH_FAILED;
    }
    sortTileInfo.sortLoopTimes = static_cast<uint32_t>(sortLoopTimes64);
    if (sortTileInfo.sortLoopTimes > static_cast<uint32_t>(1)) {
        sortTileInfo.coreNumNeed = sortTileInfo.maxCoreNum;
    } else {
        uint32_t core = static_cast<uint32_t>(sortTileInfo.unsortedDim) % sortTileInfo.maxCoreNum;
        sortTileInfo.coreNumNeed = core == uint32_t(0) ? sortTileInfo.maxCoreNum : core;
    }
    sortTileInfo.unsortedDimParallel = sortTileInfo.coreNumNeed;
    sortTileInfo.workspaceSize = WORK_SPACE_SIZE;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SetRadixMoreCoreTiling(gert::TilingContext* context, SortKthTileInfo& info)
{
    OP_CHECK_IF(!FillRadixMoreCoreInfo(info),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "FillRadixMoreCoreInfo", "false",
                                                      "The value of FillRadixMoreCoreInfo must be true."),
                return ge::GRAPH_FAILED);
    info.ubSize = info.ubSize - SIMT_UB;
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// 参数填充与日志输出。
// =============================================================================
void FillTilingDataSort(SortKthTileInfo& info, SortRegBaseTilingData* sortTilingData)
{
    PlanToTilingData(info, sortTilingData);
    return;
}

void PrintTilingDataSort(gert::TilingContext* context, SortKthTileInfo& sortTileInfo)
{
    OP_LOGI(context->GetNodeName(),
            "realCoreNum %u, numTileDataSize %u, unsortedDimParallel %u, "
            "lastDimTileNum %u, sortLoopTimes %u, lastDimNeedCore %u, keyParams0 %u, keyParams1 %u "
            "keyParams2 %u, keyParams3 %u, keyParams4 %u, keyParams5 %u, tmpUbSize %u, "
            "lastAxisNum %ld, unsortedDimNum %ld",
            sortTileInfo.coreNumNeed, sortTileInfo.numTileDataSize, sortTileInfo.unsortedDimParallel,
            sortTileInfo.lastDimTileNum, sortTileInfo.sortLoopTimes, sortTileInfo.lastDimNeedCore,
            sortTileInfo.keyParams0, sortTileInfo.keyParams1, sortTileInfo.keyParams2, sortTileInfo.keyParams3,
            sortTileInfo.keyParams4, sortTileInfo.keyParams5, sortTileInfo.tmpUbSize, sortTileInfo.lastAxis,
            sortTileInfo.unsortedDim);
    return;
}

bool SetMergeSortTiling(SortKthTileInfo& info)
{
    // 候选不满足约束时继续尝试其他路线，不视为 Tiling 错误。
    if (!FillMergeSortInfo(info, info.y2DtypeSize)) {
        return false;
    }
    info.keyParams5 = 0U;
    // 这是已有测量采用的合批策略，不代表 dtype 或 UB 的能力边界。
    // 须与仅支持 FP16/int64 的 SortBatchedRows 实例化一致；其他类型使用逐行提案。
    if (info.dataType == ge::DT_FLOAT16 && info.y2DtypeSize == sizeof(int64_t) &&
        info.keyParams0 >= SORT_BATCH_MERGE_MIN_ROWS) {
        uint64_t batchElements = static_cast<uint64_t>(info.keyParams0) * info.keyParams3;
        constexpr uint32_t proposalBytes = sizeof(float) + sizeof(uint32_t);
        uint64_t queueBytesPerElement =
            static_cast<uint64_t>(info.keyParams4) * (2U * info.dtypeSize + info.y2DtypeSize);
        // 除 I/O 双缓冲外，合批归并还需为每行保留 B32 索引序列及两份每元素 8 字节的提案。
        constexpr uint32_t maxSortElements = std::numeric_limits<uint8_t>::max() * MERGE_INITIAL_RUN_SIZE;
        if (batchElements <= maxSortElements) {
            uint64_t requiredUb = batchElements * (queueBytesPerElement + sizeof(uint32_t) + 2U * proposalBytes);
            if (requiredUb <= info.ubSize) {
                info.keyParams5 = 1U;
            }
        }
    }
    return true;
}

// =============================================================================
// 候选路线可行性检查。
// =============================================================================
bool TrySmallAxis(SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    if (sortTileInfo.lastAxis > static_cast<int64_t>(SMALL_AXIS_THRESHOLD)) {
        return false;
    }
    SmallAxisRoutePlan plan;
    if (!SelectSmallAxisRoute(sortTileInfo, plan)) {
        return false;
    }
    if (plan.kind == SmallAxisRouteKind::TWO_STAGE) {
        schId = SORT_SCHID_6;
    } else if (plan.kind == SmallAxisRouteKind::INSERTION) {
        schId = SORT_SCHID_5;
    } else {
        return false;
    }
    FillSmallAxisBatched(sortTileInfo, plan);
    return true;
}

bool TryMerge(SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    // 性能选路阈值，与 Kernel 缓冲容量约束分开维护。
    constexpr int64_t maxFp16MergeAxis = 2048;
    constexpr int64_t maxBf16MergeAxis = 1536;
    bool useB16Merge = (sortTileInfo.dataType == ge::DT_FLOAT16 && sortTileInfo.lastAxis <= maxFp16MergeAxis) ||
                       (sortTileInfo.dataType == ge::DT_BF16 && sortTileInfo.lastAxis <= maxBf16MergeAxis);
    if (useB16Merge || IsMergeSortSupported(sortTileInfo.dataType, sortTileInfo.lastAxis)) {
        SortKthTileInfo candidate = sortTileInfo;
        if (SetMergeSortTiling(candidate)) {
            sortTileInfo = candidate;
            schId = SORT_SCHID_0;
            return true;
        }
    }

    return false;
}

bool TryRadixOneCore(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    SortKthTileInfo candidate = sortTileInfo;
    if (!IsRadixSortOneCore(candidate) || SetRadixOneCoreTiling(context, candidate) != ge::GRAPH_SUCCESS) {
        return false;
    }
    sortTileInfo = candidate;
    schId = static_cast<uint64_t>(1);
    return true;
}

// =============================================================================
// 算法选路。
// =============================================================================
ge::graphStatus SelectSortSchedule(gert::TilingContext* context, SortKthTileInfo& sortTileInfo, uint64_t& schId)
{
    // 框架错误终止选路；仅提交参数配置完整的候选。
    size_t* userWorkSpaceSize = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, userWorkSpaceSize);
    SortKthTileInfo candidate = sortTileInfo;
    uint64_t candidateSchId = schId;
    // 小轴路线先利用行批处理；未命中时优先尝试长度门槛内的浮点归并，
    // 再尝试能驻留 UB 的单核 Radix，最后使用需要跨核同步/工作区的多核 Radix。
    // 各路线仍须独立通过容量校验；优先级沿用当前调参策略，不替代目标场景实测。
    if (!TrySmallAxis(candidate, candidateSchId) && !TryMerge(candidate, candidateSchId) &&
        !TryRadixOneCore(context, candidate, candidateSchId)) {
        candidateSchId = SORT_SCHID_2;
        OP_CHECK_IF(
            SetRadixMoreCoreTiling(context, candidate) != ge::GRAPH_SUCCESS,
            OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SetRadixMoreCoreTiling", "GRAPH_FAILED",
                                                  "The value of SetRadixMoreCoreTiling must be GRAPH_SUCCESS."),
            return ge::GRAPH_FAILED);
        OP_CHECK_IF(context->SetScheduleMode(1) != ge::GRAPH_SUCCESS,
                    OP_LOGE(context->GetNodeName(), "SetScheduleMode failed."), return ge::GRAPH_FAILED);
    }
    userWorkSpaceSize[0] = candidate.workspaceSize;
    sortTileInfo = candidate;
    schId = candidateSchId;
    return ge::GRAPH_SUCCESS;
}

// =============================================================================
// Tiling 入口。
// =============================================================================
ge::graphStatus RadixSortTiling(gert::TilingContext* context, int32_t maxCoreNum)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    SortRegBaseTilingData* sortTilingData{nullptr};
    sortTilingData = context->GetTilingData<SortRegBaseTilingData>();
    OP_CHECK_IF(sortTilingData == nullptr, OP_LOGE_WITH_INVALID_INPUT(context->GetNodeName(), "tilingData"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF((memset_s(sortTilingData, sizeof(SortRegBaseTilingData), 0, sizeof(SortRegBaseTilingData)) != EOK),
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "memset_s", "not EOK",
                                                      "The value of memset_s must be EOK."),
                return ge::GRAPH_FAILED);
    SortKthTileInfo sortTileInfo;
    OP_CHECK_IF(SortCheckParams(context, sortTileInfo) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SortCheckParams", "GRAPH_FAILED",
                                                      "The value of SortCheckParams must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(maxCoreNum <= 0, OP_LOGE(context->GetNodeName(), "Vector core count must be positive."),
                return ge::GRAPH_FAILED);
    sortTileInfo.maxCoreNum = static_cast<uint32_t>(maxCoreNum);
    uint64_t isInt32 = static_cast<uint64_t>(IsRadixUint32CounterRange(sortTileInfo.lastAxis));
    const bool* isDescending = context->GetAttrs()->GetAttrPointer<bool>(1);
    uint64_t isDescend = *isDescending;
    sortTileInfo.isDescend = static_cast<bool>(isDescend);
    sortTileInfo.isInt32 = static_cast<uint32_t>(isInt32);
    OP_LOGI(context->GetNodeName(), "isInt32 is %lu, isDescend is %lu", isInt32, isDescend);
    uint64_t schId = static_cast<uint64_t>(0);
    OP_CHECK_IF(SelectSortSchedule(context, sortTileInfo, schId) != ge::GRAPH_SUCCESS,
                OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(context->GetNodeName(), "SelectSortSchedule", "GRAPH_FAILED",
                                                      "The value of SelectSortSchedule must be GRAPH_SUCCESS."),
                return ge::GRAPH_FAILED);
    // 短轴候选可能最终改选统一的 64 位多核路线；key 应编码最终位宽。
    const uint64_t tilingKey = GET_TPL_TILING_KEY(schId, sortTileInfo.isInt32, isDescend);
    OP_LOGI(context->GetNodeName(), "tilingKey is %lu, maxCoreNum %d, schId %lu", tilingKey, maxCoreNum, schId);
    OP_CHECK_IF(context->SetTilingKey(tilingKey) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetTilingKey failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetBlockDim(sortTileInfo.coreNumNeed) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetBlockDim failed."), return ge::GRAPH_FAILED);
    OP_CHECK_IF(context->SetLocalMemorySize(sortTileInfo.ubSize) != ge::GRAPH_SUCCESS,
                OP_LOGE(context->GetNodeName(), "SetLocalMemorySize failed."), return ge::GRAPH_FAILED);
    FillTilingDataSort(sortTileInfo, sortTilingData);
    PrintTilingDataSort(context, sortTileInfo);
    OP_LOGI(context->GetNodeName(), "end RadixSortTIling ");
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus SortTilingSimt(gert::TilingContext* context, int32_t maxCoreNum)
{
    return RadixSortTiling(context, maxCoreNum);
}
} // namespace optiling
