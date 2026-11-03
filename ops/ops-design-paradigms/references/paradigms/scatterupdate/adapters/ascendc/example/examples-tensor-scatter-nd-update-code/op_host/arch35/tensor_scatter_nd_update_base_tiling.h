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
 * \file tensor_scatter_nd_update_base_tiling.h
 * \brief 多模板注册公共基类声明（v2.0，MDE §11.5.1）
 *
 * TensorScatterNdUpdateBaseTiling：公共校验（ValidateDtype/Shape/Attr，自 v1.x tiling.cpp 迁入，
 *   逐字节保持语义）
 *   + shape 解析（varRank/indexDimNum/totalIndexNum/totalVarElems/sliceSize
 *     + varDims/varStrides 两套数组 + useInt32Index 门控判定，存成员，
 *     纯只读无副作用，降级安全）
 *   + GetPlatformInfo（coreNum/ubSize，platform info 优先、CompileInfo 兜底）
 *   + 全部纯虚接口的占位实现（子类按需 override）
 */

#ifndef TENSOR_SCATTER_ND_UPDATE_TILING_BASE_H_
#define TENSOR_SCATTER_ND_UPDATE_TILING_BASE_H_

#include "log/log.h"
#include "op_host/tiling_util.h"
#include "op_host/tiling_base.h"
#include "op_host/tiling_templates_registry.h"
#include "sort_lib_tiling.h"

namespace optiling {

// 动态 UB 的 DCache 预留（v1.x tiling.cpp:26 同值迁移，MDE §11.5.1：子类 PostTiling 统一引用）
constexpr uint32_t DCACHE_SIZE = SortLib::DCACHE_SIZE; // 32KB

// v1.x 空结构 → v2.0 填充平台信息（TilingPrepare4TensorScatterNdUpdate 写入，
// 供 GetPlatformInfo 在 platformInfoPtr 为空时兜底，MDE §11.2.3）
struct TensorScatterNdUpdateCompileInfo {
    uint64_t coreNum;
    uint64_t ubSize;
};

class TensorScatterNdUpdateBaseTiling : public Ops::Math::OpTiling::TilingBaseClass {
public:
    explicit TensorScatterNdUpdateBaseTiling(gert::TilingContext* context)
        : Ops::Math::OpTiling::TilingBaseClass(context)
    {}
    ~TensorScatterNdUpdateBaseTiling() override {}

protected:
    // ── 基类契约实现（子类按需 override；占位实现仅供编译通过）──
    bool IsCapable() override
    {
        return true;
    }
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus GetShapeAttrsInfo() override; // 公共校验 + shape 解析（v1.x 迁入）
    ge::graphStatus DoOpTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }
    uint64_t GetTilingKey() const override
    {
        return 0;
    }
    ge::graphStatus GetWorkspaceSize() override; // 占位：仅系统 ws
    ge::graphStatus PostTiling() override
    {
        return ge::GRAPH_SUCCESS;
    }

    // ── v1.x 迁入的公共校验（函数签名不变，static → 成员）──
    ge::graphStatus ValidateDtype();  // 10 种 data dtype + int32/int64 indices
    ge::graphStatus ValidateShape();  // rank/K/updates.shape（MDE §3.2.1 规则 1~4）
    ge::graphStatus ValidateAttr();   // use_locking 读取不校验（与 v1.x 一致）
    ge::graphStatus ValidateInputs(); // 调度器：Dtype → Shape → Attr

    // ── 公共解析结果（GetShapeAttrsInfo 填充，子类消费；纯只读无副作用，降级安全）──
    ge::DataType varDtype_ = ge::DT_FLOAT; // var/updates 同 dtype（校验保证）
    ge::DataType idxDtype_ = ge::DT_INT32; // indices dtype
    int64_t varRank_ = 0;
    int32_t indexDimNum_ = 0;   // K = indices.shape[-1]
    int64_t totalIndexNum_ = 0; // N = prod(indices.shape[:-1])
    int64_t totalVarElems_ = 0;
    int64_t sliceSize_ = 0;       // prod(var.shape[K:])
    int64_t varDims_[8] = {0};    // 越界检查（两子类共用）
    int64_t varStrides_[8] = {0}; // 线性化（两子类共用）
    int32_t useInt32Index_ = 0;   // (N*sliceSize <= INT32_MAX) ? 1 : 0（两子类共用）
    int64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
};

} // namespace optiling
#endif
