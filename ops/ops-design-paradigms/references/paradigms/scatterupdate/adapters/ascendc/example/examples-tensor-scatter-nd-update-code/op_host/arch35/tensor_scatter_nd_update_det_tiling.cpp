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
 * \file tensor_scatter_nd_update_det_tiling.cpp
 * \brief 确定性三段式模板实现（v2.0，MDE §11.5.2）
 *
 * v1.x TensorScatterNdUpdateTilingFunc 主体迁移（逐字节保持语义，仅做类方法化包装）：
 *   DoOpTiling       ← v1.x 步骤 1~6（通用字段（Base 成员填充）/ keyDtype 判定 /
 *                       SortLib tiling / workspace 4 段布局）
 *   GetWorkspaceSize ← v1.x 步骤 7（userWs + sysWs）
 *   PostTiling       ← v1.x 步骤 8（blockDim + SetScheduleMode(1) + SetLocalMemorySize）
 *   GetTilingKey     ← v1.x 2 参数 → 3 参数（+TSNU_MODE_DET，key 数值因新增
 *                       deterministic 位整体变化，属预期，MDE §11.3.3）
 */

#include <limits>

#include "tensor_scatter_nd_update_det_tiling.h"
#include "../../op_kernel/arch35/tensor_scatter_nd_update_tiling_data.h"
#include "../../op_kernel/arch35/tensor_scatter_nd_update_tiling_key.h"

namespace optiling {

constexpr uint32_t STATIC_UB_ESTIMATE = 512; // SIMT 参数缓冲（2 × 64B，含对齐余量）
constexpr int64_t PHASE_THREAD_NUM = 1024;   // Phase1/Phase3 每核线程数（stride-loop 粒度）
constexpr int64_t INT16_MAX_VAL = std::numeric_limits<int16_t>::max(); // 2^15-1 = 32767
constexpr int64_t INT32_MAX_VAL = std::numeric_limits<int32_t>::max(); // 2^31-1 = 2147483647

// 系统 workspace 兜底值（离线 CompileInfo 路径下 GetPlatformInfo() 为空时使用）
constexpr uint64_t WS_SYS_FALLBACK_SIZE = 16U * 1024U * 1024U;

static int64_t AlignUp128(int64_t value)
{
    return ((value + 127) / 128) * 128;
}

// ── DoOpTiling：v1.x 步骤 1~6 迁移（TilingData 填充）──
ge::graphStatus TensorScatterNdUpdateDetTiling::DoOpTiling()
{
    TensorScatterNdUpdateTilingData* tiling = context_->GetTilingData<TensorScatterNdUpdateTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tiling);
    if (memset_s(tiling, sizeof(TensorScatterNdUpdateTilingData), 0, sizeof(TensorScatterNdUpdateTilingData)) != EOK) {
        OP_LOGE(context_, "set tiling data error");
        return ge::GRAPH_FAILED;
    }

    // ── 1. 通用字段（Base GetShapeAttrsInfo 解析结果，v1.x 步骤 1~3.1 同源）──
    tiling->varRank = static_cast<int32_t>(varRank_);
    tiling->indexDimNum = indexDimNum_;
    tiling->totalIndexNum = totalIndexNum_;
    tiling->totalVarElems = totalVarElems_;
    tiling->sliceSize = sliceSize_;
    tiling->useInt32Index = useInt32Index_;
    for (int32_t d = 0; d < static_cast<int32_t>(varRank_); d++) {
        tiling->varDims[d] = varDims_[d];
        tiling->varStrides[d] = varStrides_[d];
    }

    // ── 4. key dtype 判断（linear_idx ∈ [0, totalVarElems-1]，按 totalVarElems 选最小可表示整数）──
    uint32_t keyDtype = (tiling->totalVarElems <= INT16_MAX_VAL) ? TSNU_KEY_I16 :
                        (tiling->totalVarElems <= INT32_MAX_VAL) ? TSNU_KEY_I32 :
                                                                   TSNU_KEY_I64;
    uint32_t keySize = (keyDtype == TSNU_KEY_I16) ? 2 : (keyDtype == TSNU_KEY_I32) ? 4 : 8;
    ge::DataType keyValueType = (keyDtype == TSNU_KEY_I16) ? ge::DT_INT16 :
                                (keyDtype == TSNU_KEY_I32) ? ge::DT_INT32 :
                                                             ge::DT_INT64;

    // ── 5. SortLib tiling（Phase2）：排序对象 = linear_idx（keySize 字节），置换 = int32 ──
    bool isInt32Safe = SortLib::IsInt32Safe(tiling->totalIndexNum);
    uint32_t countMode = isInt32Safe ? TSNU_COUNT_U32 : TSNU_COUNT_I64;
    auto sortR =
        SortLib::SortTilingCompute(tiling->totalIndexNum, coreNum_, static_cast<uint64_t>(ubSize_ - STATIC_UB_ESTIMATE),
                                   keySize, 4, isInt32Safe, keyValueType);
    OP_CHECK_IF(sortR.errCode != SortLib::SORT_TILING_OK,
                OP_LOGE(context_, "SortTilingCompute failed, errCode=%d", sortR.errCode), return ge::GRAPH_FAILED);
    tiling->needCoreNum = (sortR.coreNumNeed > 0) ? sortR.coreNumNeed : 1;
    tiling->numTileData = sortR.numTileData;
    tiling->tileCount = sortR.tileCount;
    tiling->activeCores = sortR.activeCores;
    tiling->tmpUbSize = sortR.tmpUbSize;
    tiling->isSingleCore = sortR.isSingleCore;

    // ── 6. workspace 布局（MDE §4.3，各段 128B 对齐）──
    int64_t n = tiling->totalIndexNum;
    tiling->sortWsBytes = sortR.workspaceBytes;
    tiling->wsLinearIdxOff = AlignUp128(tiling->sortWsBytes);
    tiling->wsSortedOff = AlignUp128(tiling->wsLinearIdxOff + n * keySize);
    tiling->wsPermOff = AlignUp128(tiling->wsSortedOff + n * keySize);
    int64_t userWsSize = tiling->wsPermOff + n * 4;
    userWsSize_ = userWsSize;

    countMode_ = countMode;
    keyDtype_ = keyDtype;
    return ge::GRAPH_SUCCESS;
}

// ── GetTilingKey：v1.x 2 参数 → 3 参数（+TSNU_MODE_DET，MDE §11.3.3）──
uint64_t TensorScatterNdUpdateDetTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(countMode_), static_cast<uint64_t>(keyDtype_),
                              static_cast<uint64_t>(TSNU_MODE_DET));
}

// ── GetWorkspaceSize：v1.x 步骤 7 迁移（userWs + sysWs）──
ge::graphStatus TensorScatterNdUpdateDetTiling::GetWorkspaceSize()
{
    uint64_t sysWorkspaceSize = WS_SYS_FALLBACK_SIZE;
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr != nullptr) {
        sysWorkspaceSize = platform_ascendc::PlatformAscendC(platformPtr).GetLibApiWorkSpaceSize();
    }
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(userWsSize_ + static_cast<int64_t>(sysWorkspaceSize));
    return ge::GRAPH_SUCCESS;
}

// ── PostTiling：v1.x 步骤 8 迁移（blockDim + SetScheduleMode(1) + SetLocalMemorySize）──
ge::graphStatus TensorScatterNdUpdateDetTiling::PostTiling()
{
    // Phase1 按散点数 N 用核（每核至少处理 PHASE_THREAD_NUM 个散点）；
    // Phase3 优化后为 flat 写（写 N×sliceSize 个元素），按写量用核，避免小 N 大 slice 被 1 核卡死；
    // SortLib 用 needCoreNum。SetBlockDim 取各阶段最大核数，多余核自然空转/跳过。
    int64_t phase1Cores = (totalIndexNum_ + PHASE_THREAD_NUM - 1) / PHASE_THREAD_NUM;
    int64_t totalWriteElems = totalIndexNum_ * sliceSize_;
    int64_t phase3Cores = (totalWriteElems + PHASE_THREAD_NUM - 1) / PHASE_THREAD_NUM;
    int64_t phaseCores = (phase1Cores > phase3Cores) ? phase1Cores : phase3Cores;
    phaseCores = (phaseCores < coreNum_) ? phaseCores : coreNum_;
    if (phaseCores < 1) {
        phaseCores = 1;
    }
    const TensorScatterNdUpdateTilingData* tiling = context_->GetTilingData<TensorScatterNdUpdateTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tiling);
    int64_t sortCores = static_cast<int64_t>(tiling->needCoreNum);
    int64_t blockDim = phaseCores;
    if (sortCores > blockDim) {
        blockDim = sortCores;
    }
    context_->SetBlockDim(static_cast<int32_t>(blockDim));
    context_->SetScheduleMode(1); // 开启 SyncAll（SortLib 核间同步依赖）

    OP_CHECK_IF(ubSize_ <= DCACHE_SIZE, OP_LOGE(context_, "ubSize %lu too small", ubSize_), return ge::GRAPH_FAILED);
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

// ── DumpTilingData：v1.x 原函数迁移（DFX，打印 TilingData 全部标量字段）──
static void DumpTilingData(gert::TilingContext* context, const TensorScatterNdUpdateTilingData* tiling)
{
    OP_LOGD(context,
            "TensorScatterNdUpdateTilingData: needCoreNum=%d, totalIndexNum=%ld, "
            "totalVarElems=%ld, varRank=%d, indexDimNum=%d, "
            "varDims=[8 elements], varStrides=[8 elements], sliceSize=%ld, "
            "numTileData=%u, tileCount=%u, activeCores=%u, tmpUbSize=%u, isSingleCore=%u, "
            "sortWsBytes=%ld, wsLinearIdxOff=%ld, wsSortedOff=%ld, wsPermOff=%ld",
            tiling->needCoreNum, tiling->totalIndexNum, tiling->totalVarElems, tiling->varRank, tiling->indexDimNum,
            tiling->sliceSize, tiling->numTileData, tiling->tileCount, tiling->activeCores, tiling->tmpUbSize,
            tiling->isSingleCore, tiling->sortWsBytes, tiling->wsLinearIdxOff, tiling->wsSortedOff, tiling->wsPermOff);
}

// ── DumpTilingInfo：基类契约 override（DoTiling 末尾调用，v1.x TilingResult 日志一并迁入）──
void TensorScatterNdUpdateDetTiling::DumpTilingInfo()
{
    const TensorScatterNdUpdateTilingData* tiling = context_->GetTilingData<TensorScatterNdUpdateTilingData>();
    DumpTilingData(context_, tiling);
    OP_LOGI(context_->GetNodeName(),
            "TilingResult: needCoreNum=%d, totalIndexNum=%ld, totalVarElems=%ld, "
            "varRank=%d, indexDimNum=%d, sliceSize=%ld, countMode=%u, keyDtype=%u, userWs=%ld",
            tiling->needCoreNum, tiling->totalIndexNum, tiling->totalVarElems, tiling->varRank, tiling->indexDimNum,
            tiling->sliceSize, countMode_, keyDtype_, userWsSize_);
}

REGISTER_OPS_TILING_TEMPLATE(TensorScatterNdUpdate, TensorScatterNdUpdateDetTiling, 10);

} // namespace optiling
