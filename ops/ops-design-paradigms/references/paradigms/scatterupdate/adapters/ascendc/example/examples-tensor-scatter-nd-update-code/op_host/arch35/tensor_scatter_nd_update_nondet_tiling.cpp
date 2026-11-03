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
 * \file tensor_scatter_nd_update_nondet_tiling.cpp
 * \brief 非确定（直接写覆盖）模板实现（v2.0，MDE §11.5.3）
 *
 * IsCapable 唯一门控：确定性开关（GetDeterministic 严格 ==0）；
 *   不满足时返回 false → 基类 DoTiling 返回 GRAPH_PARAM_INVALID → registry 降级 DetTiling
 *   （priority=10，IsCapable 恒 true，永兜底，MDE §11.2.1）。
 *   与 scatter_add 的关键差异：无 dtype 白名单（覆盖语义只需对齐单条 store 原子，
 *   10 dtype 全覆盖，MDE §11.6.4），核数按 flat 写量 N×sliceSize。
 * Workspace：用户 ws = 0（无排序/段表），仅无条件分配系统 ws（kernel 签名含
 *   GM_ADDR workspace，必须 GetWorkspaceSizes(1)，否则运行时 VEC_ERROR）。
 */

#include "tensor_scatter_nd_update_nondet_tiling.h"
#include "../../op_kernel/arch35/tensor_scatter_nd_update_tiling_data.h"
#include "../../op_kernel/arch35/tensor_scatter_nd_update_tiling_key.h"

namespace optiling {

// 非确定 VF 每核线程数（与 kernel 侧 THREAD_NUM 一致，MDE §11.5.3）
constexpr int64_t NONDET_THREAD_NUM = 1024;

// 系统 workspace 兜底值（离线 CompileInfo 路径下 GetPlatformInfo() 为空时使用）
constexpr uint64_t WS_SYS_FALLBACK_SIZE = 16U * 1024U * 1024U;

// ── IsCapable：唯一门控（覆盖语义核心差异点，MDE §11.5.3）──
bool TensorScatterNdUpdateNondetTiling::IsCapable()
{
    // 1) 确定性开关：严格 ==0 才走非确定。
    //    GetDeterministic()（CANN tiling_context.h:461，返回 int32）异常场景
    //    （无 ComputeNodeInfo / 无下发值）返回 INT32_MAX —— 非 0，自然落入确定兜底。
    if (context_->GetDeterministic() != 0) {
        return false;
    }
    // 2) 无 dtype 门控：覆盖语义只需对齐单条 store 原子（MDE §11.6.4 论证），
    //    全部 10 种 data dtype（1B/2B/4B/8B）均安全 —— 与 scatter_add 的
    //    GM 原子加白名单（7 dtype + int8/uint8/int16/bool 回退）形成对照。
    return true;
}

// ── DoOpTiling：填充 NondetTilingData（MDE §11.5.3）──
ge::graphStatus TensorScatterNdUpdateNondetTiling::DoOpTiling()
{
    // blockDim = min(coreNum, max(1, ceil(N*sliceSize/1024)))
    //   任务空间 = N*sliceSize（flat 元素级口径，与 det PostTiling 的 phase3Cores 同款；
    //   pack 后实际任务数 N*sliceVec <= N*sliceSize，核数只多不少，VF grid-stride 自然消化）
    int64_t taskCount = totalIndexNum_ * sliceSize_;
    int64_t blockNum = (taskCount + NONDET_THREAD_NUM - 1) / NONDET_THREAD_NUM;
    usedCoreNum_ = (blockNum > coreNum_) ? coreNum_ : blockNum;
    if (usedCoreNum_ < 1) {
        usedCoreNum_ = 1;
    } // 空 tensor 场景仍启动 1 核（kernel 入口早退）

    TensorScatterNdUpdateNondetTilingData* tiling = context_->GetTilingData<TensorScatterNdUpdateNondetTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context_, tiling);
    if (memset_s(tiling, sizeof(TensorScatterNdUpdateNondetTilingData), 0,
                 sizeof(TensorScatterNdUpdateNondetTilingData)) != EOK) {
        OP_LOGE(context_, "set tiling data error");
        return ge::GRAPH_FAILED;
    }
    tiling->needCoreNum = static_cast<int32_t>(usedCoreNum_);
    tiling->totalIndexNum = totalIndexNum_;
    tiling->totalVarElems = totalVarElems_;
    tiling->varRank = static_cast<int32_t>(varRank_);
    tiling->indexDimNum = indexDimNum_;
    tiling->sliceSize = sliceSize_;
    tiling->useInt32Index = useInt32Index_; // Base 统一判定（N*sliceSize <= INT32_MAX）
    for (int32_t d = 0; d < static_cast<int32_t>(varRank_); d++) {
        tiling->varDims[d] = varDims_[d]; // 仅前 K 维有效（与 det 同款约定）
        tiling->varStrides[d] = varStrides_[d];
    }
    return ge::GRAPH_SUCCESS;
}

// ── GetTilingKey：占位值与 SEL 第 7 组合严格一致（MDE §11.3.3）──
uint64_t TensorScatterNdUpdateNondetTiling::GetTilingKey() const
{
    return GET_TPL_TILING_KEY(static_cast<uint64_t>(TSNU_COUNT_U32), static_cast<uint64_t>(TSNU_KEY_I32),
                              static_cast<uint64_t>(TSNU_MODE_NONDET));
}

// ── GetWorkspaceSize：仅系统 workspace（用户 ws = 0，MDE §11.5.3）──
ge::graphStatus TensorScatterNdUpdateNondetTiling::GetWorkspaceSize()
{
    // 用户 ws = 0（无排序/段表/linearIdx/sorted/perm buffer）；kernel 签名含 GM_ADDR workspace，
    // 必须无条件 GetWorkspaceSizes(1) 分配系统 ws（框架内部依赖，否则运行时 VEC_ERROR）
    uint64_t sysWorkspaceSize = WS_SYS_FALLBACK_SIZE;
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr != nullptr) {
        sysWorkspaceSize = platform_ascendc::PlatformAscendC(platformPtr).GetLibApiWorkSpaceSize();
    }
    size_t* currentWorkspace = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

// ── PostTiling：SetBlockDim + SetLocalMemorySize（无 SyncAll 依赖，不设 ScheduleMode）──
ge::graphStatus TensorScatterNdUpdateNondetTiling::PostTiling()
{
    context_->SetBlockDim(static_cast<int32_t>(usedCoreNum_));
    // 无 SyncAll 依赖 → 不调用 SetScheduleMode(1)（与确定模板差异点：两模板各自设置，
    // 每次执行仅走其一，无冲突——两基例 v2.0 同款已验证）
    // 动态 UB（3 数组 ≈ 192B 坐标/快除参数）：与确定路径统一预留 DCache 32KB
    OP_CHECK_IF(ubSize_ <= DCACHE_SIZE, OP_LOGE(context_, "ubSize %lu too small", ubSize_), return ge::GRAPH_FAILED);
    context_->SetLocalMemorySize(static_cast<uint32_t>(ubSize_ - DCACHE_SIZE));
    return ge::GRAPH_SUCCESS;
}

// ── DumpTilingData：NondetTilingData 字段打印（DFX，对齐 det 命名）──
static void DumpTilingData(gert::TilingContext* context, const TensorScatterNdUpdateNondetTilingData* tiling)
{
    OP_LOGD(context,
            "TensorScatterNdUpdateNondetTilingData: needCoreNum=%d, totalIndexNum=%ld, "
            "totalVarElems=%ld, varRank=%d, indexDimNum=%d, "
            "varDims=[8 elements], varStrides=[8 elements], sliceSize=%ld, useInt32Index=%d",
            tiling->needCoreNum, tiling->totalIndexNum, tiling->totalVarElems, tiling->varRank, tiling->indexDimNum,
            tiling->sliceSize, tiling->useInt32Index);
}

// ── DumpTilingInfo：基类契约 override（DoTiling 末尾调用）──
void TensorScatterNdUpdateNondetTiling::DumpTilingInfo()
{
    const TensorScatterNdUpdateNondetTilingData* tiling =
        context_->GetTilingData<TensorScatterNdUpdateNondetTilingData>();
    DumpTilingData(context_, tiling);
    OP_LOGI(context_->GetNodeName(),
            "NondetTilingResult: needCoreNum=%d, totalIndexNum=%ld, totalVarElems=%ld, "
            "varRank=%d, indexDimNum=%d, sliceSize=%ld, useInt32Index=%d, userWs=0",
            tiling->needCoreNum, tiling->totalIndexNum, tiling->totalVarElems, tiling->varRank, tiling->indexDimNum,
            tiling->sliceSize, tiling->useInt32Index);
}

REGISTER_OPS_TILING_TEMPLATE(TensorScatterNdUpdate, TensorScatterNdUpdateNondetTiling, 0);

} // namespace optiling
