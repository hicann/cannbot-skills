/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 *
 * \file sort_lib_tiling.h
 * \brief SortLib public Tiling API — the only host-side header an external operator needs.
 *        Include this header and call SortTilingCompute() for workspace sizing and core scheduling.
 *
 * \public
 */

#ifndef SORT_LIB_TILING_H
#define SORT_LIB_TILING_H

#include "tiling/tiling_api.h"
#include <cstdint>
#include <limits>
#include <vector>

namespace SortLib {

constexpr uint32_t DCACHE_SIZE = 32 * 1024;     // UB 中预留的 DCACHE 对齐空间（字节）
constexpr uint32_t BIN_NUM = 256;               // 基数排序每轮 256 个 bin
constexpr uint32_t UINT32_BYTES = 4;            // uint32 索引/计数器字节数
constexpr uint32_t INT64_BYTES = 8;             // int64 索引/计数器字节数
constexpr int64_t INT32_SAFE_LIMIT = 1LL << 30; // 32 位计数安全上限（元素数）
constexpr uint32_t UB_ALIGN_BYTES = 32U;        // UB/workspace 32 字节对齐单位
constexpr int64_t WORKSPACE_ALIGN_BYTES = 128;  // workspace 总大小对齐单位
constexpr uint32_t EMPTY_TMP_UB_SIZE = 4096;    // 空输入占位 tmpUbSize

// SortTiling 错误码：0 表示正常，其余表示异常场景
constexpr int32_t SORT_TILING_OK = 0;                      // 正常
constexpr int32_t SORT_TILING_ERR_UB_LESS_THAN_DCACHE = 1; // ubTotalBytes <= dcacheSize
constexpr int32_t SORT_TILING_ERR_UB_INSUFFICIENT = 2;     // usableUb 不足以容纳最小 tile

/// \brief 元素数是否可用 32 位计数（<= 2^30），否则须用 64 位计数。
inline bool IsInt32Safe(int64_t totalElements)
{
    return totalElements <= INT32_SAFE_LIMIT;
}

struct SortParams {
    uint32_t numTileData = 0;  // 每 tile 最大元素数（受 UB 容量限制，由 ComputeTileData 求得）
    uint32_t tileCount = 0;    // 总 tile 数 = ceil(totalElements / numTileData)
    uint32_t activeCores = 0;  // 实际使用的 AI Core 数 = min(coreCount, tileCount)
    uint32_t tmpUbSize = 0;    // AscendC::Sort 临时 UB 空间大小（字节）
    int64_t totalElements = 0; // 待排序元素总数
    uint32_t isSingleCore = 0; // 单核/多核标志：1=单核快路径，0=多核 LSD radix sort
};

struct SortTilingResult : SortParams {
    uint32_t coreNumNeed = 0;   // 实际需要 AI Core 数（用于设置 block dim）
    int64_t workspaceBytes = 0; // 需分配的 GM workspace 字节数（空输入为 0）
    int32_t errCode = 0;        // 错误码：0=正常，非 0=异常（见 SORT_TILING_ERR_* 常量）
};

/**
 * \brief 计算 workspace 大小与核调度参数（SortLib 唯一 host 侧对外接口）。
 *        调用方在 host 侧 tiling 阶段调用本函数，把返回的 SortTilingResult 写入 tiling data；
 *        kernel 侧据此构造 SortParams 并调 SortInvoke。
 *
 * \param totalElements 待排序元素总数（须 >= 0，0 表示空输入）
 * \param coreCount     可用 AI Core 总数（须 > 0）
 * \param ubTotalBytes  该算子排序可用的 UB 总字节数（单核，尚未扣除 dcacheSize）。
 *                      须 >= dcacheSize + 内核最小 UB，其中内核最小 UB ≈
 *                      BIN_NUM×(dtypeSize + 4×indexSize + 10) + Sort 临时 UB（每个 InitBuffer
 *                      至少 32B 且 32B 对齐已计入，约 40KB+，随 dtype/indexSize 变化），
 *                      否则返回错误码 SORT_TILING_ERR_UB_*。
 * \param dtypeSize     排序键字节数（sizeof(ValT)，须为 2/4/8）
 * \param indexSize     索引输出字节数（须为 4=int32 或 8=int64）
 * \param isInt32Safe   是否可用 32 位计数（见 IsInt32Safe）
 * \param valueType     排序键的 ge::DataType（用于计算单核快路径的 UB 临时空间）
 * \param dcacheSize    内部 SIMT 实现需预留的 DCACHE 空间，必须 >= 32KB（默认 DCACHE_SIZE）
 *
 * \return SortTilingResult：errCode=0 正常；非 0 表示异常（SORT_TILING_ERR_UB_LESS_THAN_DCACHE
 *         或 SORT_TILING_ERR_UB_INSUFFICIENT），此时其余字段无效、调用方应报错。
 *         errCode=0 时：workspaceBytes 为需分配的 GM workspace 字节数（空输入为 0）、
 *         coreNumNeed 为实际需要的 AI Core 数，其余字段为传给 kernel 的 SortParams。
 */
inline SortTilingResult SortTilingCompute(int64_t totalElements, int64_t coreCount, uint64_t ubTotalBytes,
                                          uint32_t dtypeSize, uint32_t indexSize, bool isInt32Safe,
                                          ge::DataType valueType, uint32_t dcacheSize = DCACHE_SIZE);

// ==================== 内部实现（Internal，勿直接使用）====================
namespace detail {

inline int64_t CeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}
inline int64_t AlignUp(int64_t value, int64_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

inline void SetSortTmpSize(uint32_t tileData, uint32_t& tmpUbSize)
{
    int64_t realLen = static_cast<int64_t>(tileData);
    std::vector<int64_t> shapeVec = {realLen};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = false;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, ge::DT_UINT8, ge::DT_UINT32, false, config, maxValue, minValue);
    tmpUbSize = maxValue;
}

// 单核 AscendC::Sort 的临时 UB 空间（按真实 value dtype 计算，而非 uint8 字节视图）。
inline void SetSortSingleCoreTmpSize(uint32_t n, ge::DataType valueType, uint32_t& tmpUbSize)
{
    int64_t realLen = static_cast<int64_t>(n);
    std::vector<int64_t> shapeVec = {realLen};
    ge::Shape srcShape(shapeVec);
    AscendC::SortConfig config;
    config.type = AscendC::SortType::RADIX_SORT;
    config.isDescend = false;
    config.hasSrcIndex = false;
    config.hasDstIndex = true;
    uint32_t maxValue = 0;
    uint32_t minValue = 0;
    AscendC::GetSortMaxMinTmpSize(srcShape, valueType, ge::DT_UINT32, false, config, maxValue, minValue);
    tmpUbSize = maxValue;
}

inline uint32_t ComputeRemainUb(uint32_t usableUb, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor)
{
    return usableUb - (ubExtra + tileFactor * tileData);
}

inline void AdjTmpUb(uint32_t usableUb, uint32_t tileData, uint32_t ubExtra, uint32_t tileFactor, uint32_t blockUbSize,
                     uint32_t& tmpUbSize)
{
    uint32_t remainUb = ComputeRemainUb(usableUb, tileData, ubExtra, tileFactor) - tmpUbSize;
    remainUb = remainUb > blockUbSize ? (remainUb - blockUbSize) : 0U;
    uint32_t alignUb = (remainUb / blockUbSize) * blockUbSize;
    tmpUbSize = tmpUbSize + alignUb;
}

inline uint32_t ComputeTileData(uint32_t usableUb, const uint32_t dtypeSize, const uint32_t indexSize,
                                const uint32_t blockUbSize, uint32_t& tmpUbSize)
{
    uint32_t ubExtra = BIN_NUM * (indexSize + static_cast<uint32_t>(sizeof(uint16_t)) +
                                  static_cast<uint32_t>(sizeof(uint16_t)) + indexSize + indexSize);
    uint32_t tileFactor = dtypeSize + indexSize + static_cast<uint32_t>(sizeof(uint32_t)) +
                          static_cast<uint32_t>(sizeof(uint8_t)) + static_cast<uint32_t>(sizeof(uint8_t));

    if (usableUb < ubExtra) {
        return 0; // UB 不足以容纳直方图等固定开销，无法排序（避免 uint32 减法下溢）
    }
    uint32_t tileData = (usableUb - ubExtra) / tileFactor;
    tileData = (tileData / BIN_NUM) * BIN_NUM;

    uint32_t remainUb = ComputeRemainUb(usableUb, tileData, ubExtra, tileFactor);
    SetSortTmpSize(tileData, tmpUbSize);

    while (tmpUbSize > remainUb) {
        tileData = tileData - BIN_NUM;
        if (tileData == 0) {
            return 0;
        }
        remainUb = ComputeRemainUb(usableUb, tileData, ubExtra, tileFactor);
        SetSortTmpSize(tileData, tmpUbSize);
    }

    AdjTmpUb(usableUb, tileData, ubExtra, tileFactor, blockUbSize, tmpUbSize);
    return tileData;
}

// 单核快路径：小 N 时直接用 AscendC::Sort(RADIX_SORT)，0 次 SyncAll、0 GM workspace，
// 避免多核 LSD radix sort 的固定开销。命中时填充 r 并返回 true；N 过大或 UB 不足返回 false。
inline bool TrySingleCore(SortTilingResult& r, int64_t totalElements, uint32_t usableUb, uint32_t dtypeSize,
                          uint32_t indexSize, ge::DataType valueType, uint32_t blockUbSize)
{
    if (totalElements <= 0 ||
        static_cast<uint64_t>(totalElements) > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
        return false;
    }
    uint32_t n = static_cast<uint32_t>(totalElements);
    uint32_t xUbSize =
        static_cast<uint32_t>(AlignUp(static_cast<int64_t>(n) * dtypeSize, static_cast<int64_t>(blockUbSize)));
    // Sort 不带 srcIndex 时 dstIndex 固定 uint32；int64 index 时先写 uint32 到后半段再 Cast 为 int64，
    // 故索引缓冲为 uint32 结果缓冲的 1 倍(int32 index)或 2 倍(int64 index)。
    uint32_t idxUbSize32 =
        static_cast<uint32_t>(AlignUp(static_cast<int64_t>(n) * UINT32_BYTES, static_cast<int64_t>(blockUbSize)));
    uint32_t idxTotalUb = (indexSize == INT64_BYTES) ? (2U * idxUbSize32) : idxUbSize32;
    uint32_t scTmpUb = 0;
    SetSortSingleCoreTmpSize(n, valueType, scTmpUb);
    if (2ULL * xUbSize + idxTotalUb + scTmpUb > static_cast<uint64_t>(usableUb)) {
        return false;
    }
    r.isSingleCore = 1;
    r.numTileData = n;
    r.tileCount = 1;
    r.activeCores = 1;
    r.coreNumNeed = 1;
    r.tmpUbSize = scTmpUb;
    r.workspaceBytes = 0;
    r.totalElements = totalElements;
    return true;
}

// 六段 workspace 布局的总大小计算（与 kernel 侧 Init 的 offset 计算对应，详见下方契约注释）。
// ┌─ (wkBase) ───────────────────────────────────────────────┐
// │ [0] exclusiveBinsGmWk_                                     │
// │     每个 core 的 256 bin 前缀和累加器                       │
// │     Size: 256 × radixRounds × counterSize 字节               │
// │     读写: CountT (SimtGlobalOffset / CopyOutGm)            │
// ├───────────────────────────────────────────────────────────┤
// │ [1] globalHistGmWk_                                        │
// │     radixRounds 轮 × 256 bin × tileCount 的全局直方图       │
// │     Size: radixRounds × 256 × tileCount × counterSize 字节  │
// │     读写: CountT (CopyUbToGm / CopyGmToUb)                 │
// ├───────────────────────────────────────────────────────────┤
// │ [2] outIdxDbWK_                                             │
// │     索引 DoubleBuffer 工作区，存中间轮索引                    │
// │     Size: totalElements × counterSize 字节                    │
// │     写: Scatter 决定实际宽度    读: uint32_t (CopyGmToUb)    │
// ├───────────────────────────────────────────────────────────┤
// │ [3] histTileGmWk_ + histCumsumTileGmWk_                     │
// │     各 tileCount × 256 bin，uint16_t 本地直方图/前缀和        │
// │     Size: tileCount × 256 × sizeof(uint16_t) × 2            │
// │     读写: uint16_t                                          │
// ├───────────────────────────────────────────────────────────┤
// │ [4] xB8GmWk_                                                │
// │     字节视图排序键缓存 (DeInterleave 输出)                    │
// │     Size: tileCount × numTileData × sizeof(uint8_t)         │
// │     读写: uint8_t                                           │
// ├───────────────────────────────────────────────────────────┤
// │ [5] outValueDbWK_                                           │
// │     值 DoubleBuffer 工作区                                    │
// │     Size: totalElements × sizeof(ValT) (对齐后)              │
// │     读写: ValT                                               │
// └─ (wkEnd) ───────────────────────────────────────────────┘
// ⚠️ Workspace 布局契约：本六段 [0]~[5] 的总大小计算，
//    必须与 op_kernel/arch35/sort_lib/sort_radix_more_core.h 中 SortRadixMoreCore::Init
//    的六段 offset 计算逐段一致，改动任何一段公式必须同步修改另一侧，否则运行期越界。
//    变量对应：BIN_NUM==RADIX_SORT_NUM；counterSize==sizeof(CountT)；
//    blockUbSize==UB_ALIGN_BYTES；totalElements==totalDataNum_；
//    [3] 段的 ×2 对应 Init 中 histTile/histCumsum 两次 wkOffset += tileHistBytes。
inline int64_t ComputeWorkspaceBytes(int64_t totalElements, uint32_t tileCount, uint32_t numTileData,
                                     uint32_t dtypeSize, uint32_t counterSize, uint32_t radixRounds,
                                     uint32_t blockUbSize)
{
    int64_t ws = 0;
    ws += AlignUp(static_cast<int64_t>(BIN_NUM) * radixRounds * counterSize, blockUbSize); // [0]
    ws += AlignUp(static_cast<int64_t>(BIN_NUM) * tileCount * radixRounds * counterSize,   // [1]
                  blockUbSize);
    ws += AlignUp(totalElements * static_cast<int64_t>(counterSize), blockUbSize);                // [2]
    ws += AlignUp(static_cast<int64_t>(tileCount) * BIN_NUM * sizeof(uint16_t) * 2, blockUbSize); // [3]
    ws += AlignUp(static_cast<int64_t>(tileCount) * numTileData, blockUbSize);                    // [4]
    ws += AlignUp(totalElements * static_cast<int64_t>(dtypeSize), blockUbSize);                  // [5]
    return AlignUp(ws, WORKSPACE_ALIGN_BYTES);
}

} // namespace detail

// ==================== 对外接口实现 ====================
inline SortTilingResult SortTilingCompute(int64_t totalElements, int64_t coreCount, uint64_t ubTotalBytes,
                                          uint32_t dtypeSize, uint32_t indexSize, bool isInt32Safe,
                                          ge::DataType valueType, uint32_t dcacheSize)
{
    SortTilingResult r;
    r.errCode = SORT_TILING_OK; // 显式标记正常，后续异常分支覆盖为 1/2

    if (totalElements == 0) {
        r.numTileData = 1;
        r.tileCount = 1;
        r.activeCores = 1;
        r.coreNumNeed = 1;
        r.tmpUbSize = EMPTY_TMP_UB_SIZE;
        r.totalElements = 0;
        r.isSingleCore = 1; // 空输入归入单核语义（kernel 侧 SortInvoke 对 totalElements==0 直接 return）
        return r;
    }

    uint32_t blockUbSize = UB_ALIGN_BYTES;
    if (ubTotalBytes <= dcacheSize) {
        r.errCode = SORT_TILING_ERR_UB_LESS_THAN_DCACHE;
        return r;
    }
    uint32_t usableUb = static_cast<uint32_t>(ubTotalBytes - dcacheSize);

    if (detail::TrySingleCore(r, totalElements, usableUb, dtypeSize, indexSize, valueType, blockUbSize)) {
        return r;
    }

    bool use64BitCounters = !isInt32Safe;
    uint32_t counterSize = use64BitCounters ? INT64_BYTES : UINT32_BYTES;
    uint32_t radixRounds = dtypeSize; // radix sort passes: sizeof(ValT) per round

    uint32_t tmpUbSize = 0;
    uint32_t numTileData = detail::ComputeTileData(usableUb, dtypeSize, indexSize, blockUbSize, tmpUbSize);
    if (numTileData == 0) {
        r.errCode = SORT_TILING_ERR_UB_INSUFFICIENT;
        return r;
    }

    uint32_t tileCount = static_cast<uint32_t>(detail::CeilDiv(totalElements, static_cast<int64_t>(numTileData)));
    uint32_t activeCores = static_cast<uint32_t>((coreCount < tileCount) ? coreCount : tileCount);

    r.workspaceBytes = detail::ComputeWorkspaceBytes(totalElements, tileCount, numTileData, dtypeSize, counterSize,
                                                     radixRounds, blockUbSize);

    r.numTileData = numTileData;
    r.tileCount = tileCount;
    r.activeCores = activeCores;
    r.coreNumNeed = activeCores;
    r.tmpUbSize = tmpUbSize;
    r.totalElements = totalElements;

    return r;
}

} // namespace SortLib

#endif
