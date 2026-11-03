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
 * \file tensor_scatter_nd_update_tiling_base.cpp
 * \brief 多模板注册公共基类实现（v2.0，MDE §11.5.1）
 *
 * Base 职责：公共校验（ValidateDtype/Shape/Attr 自 v1.x tiling.cpp 迁入，逐字节保持语义）
 *   + shape 解析（varDims/varStrides/K/N/sliceSize/useInt32Index，纯只读成员填充，
 *     无副作用，IsCapable 降级重入安全，MDE §11.2.2）
 *   + GetPlatformInfo 双源（platform info 优先、CompileInfo 兜底，对齐 bucketize_v2）
 */

#include <limits>

#include "tensor_scatter_nd_update_base_tiling.h"

namespace optiling {

// ══ 命名常量（SE §2.2，v1.x tiling.cpp 迁入）════
constexpr int32_t VAR_INPUT_IDX = 0;
constexpr int32_t INDICES_INPUT_IDX = 1;
constexpr int32_t UPDATES_INPUT_IDX = 2;
constexpr int32_t USE_LOCKING_ATTR_IDX = 0;

constexpr int32_t MAX_VAR_RANK = 8;                                    // v1.x 同值迁移
constexpr int64_t INT32_MAX_VAL = std::numeric_limits<int32_t>::max(); // 2^31-1 = 2147483647

// 系统 workspace 兜底值（离线 CompileInfo 路径下 GetPlatformInfo() 为空时使用，bucketize_v2 同款）
constexpr uint64_t WS_SYS_FALLBACK_SIZE = 16U * 1024U * 1024U;

// ══ ValidateDtype（SE §2.2 Dtype 列 + §5.3 组合，v1.x tiling.cpp 迁入，static → 成员）══
ge::graphStatus TensorScatterNdUpdateBaseTiling::ValidateDtype()
{
    // SE §2.2: var dtype ∈ {bfloat16, bool, float16, float32, int64, int8, uint8,
    //                       float8_e5m2, float8_e4m3fn, hifloat8}
    auto varDesc = context_->GetInputDesc(VAR_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varDesc);
    ge::DataType varDtype = varDesc->GetDataType();
    OP_CHECK_IF(varDtype != ge::DT_BF16 && varDtype != ge::DT_BOOL && varDtype != ge::DT_FLOAT16 &&
                    varDtype != ge::DT_FLOAT && varDtype != ge::DT_INT64 && varDtype != ge::DT_INT8 &&
                    varDtype != ge::DT_UINT8 && varDtype != ge::DT_FLOAT8_E5M2 && varDtype != ge::DT_FLOAT8_E4M3FN &&
                    varDtype != ge::DT_HIFLOAT8,
                OP_LOGE(context_, "var dtype must be in 10 supported dtypes"), return ge::GRAPH_FAILED);

    // SE §2.2: indices dtype ∈ {int32, int64}
    auto idxDesc = context_->GetInputDesc(INDICES_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, idxDesc);
    ge::DataType idxDtype = idxDesc->GetDataType();
    OP_CHECK_IF(idxDtype != ge::DT_INT32 && idxDtype != ge::DT_INT64,
                OP_LOGE(context_, "indices dtype must be int32/int64"), return ge::GRAPH_FAILED);

    // SE §5.3: updates.dtype == var.dtype
    auto updDesc = context_->GetInputDesc(UPDATES_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updDesc);
    OP_CHECK_IF(updDesc->GetDataType() != varDtype, OP_LOGE(context_, "updates dtype must equal var dtype"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// ══ ValidateShape（SE §2.2 / §5.3 / MDE §3.2.1，v1.x tiling.cpp 迁入，static → 成员）══
ge::graphStatus TensorScatterNdUpdateBaseTiling::ValidateShape()
{
    auto varShape = context_->GetInputShape(VAR_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, varShape);
    const gert::Shape& varStorage = varShape->GetStorageShape();
    int32_t varRank = static_cast<int32_t>(varStorage.GetDimNum());

    // SE §2.2 / MDE §3.2.1: 1 <= rank(var) <= 8
    OP_CHECK_IF(varRank < 1 || varRank > MAX_VAR_RANK, OP_LOGE(context_, "var rank must be in [1, 8], got %d", varRank),
                return ge::GRAPH_FAILED);

    // MDE §3.2.1: rank(indices) >= 1
    auto idxShape = context_->GetInputShape(INDICES_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, idxShape);
    const gert::Shape& idxStorage = idxShape->GetStorageShape();
    int32_t idxRank = static_cast<int32_t>(idxStorage.GetDimNum());
    OP_CHECK_IF(idxRank < 1, OP_LOGE(context_, "indices rank must be >= 1, got %d", idxRank), return ge::GRAPH_FAILED);

    // MDE §3.2.1: 1 <= K = indices.shape[-1] <= rank(var)
    int64_t k = static_cast<int64_t>(idxStorage.GetDim(static_cast<size_t>(idxRank - 1)));
    OP_CHECK_IF(k < 1 || k > varRank, OP_LOGE(context_, "indices.shape[-1]=%ld must be in [1, %d]", k, varRank),
                return ge::GRAPH_FAILED);

    // MDE §3.2.1: updates.shape == indices.shape[:-1] + var.shape[K:]
    auto updShape = context_->GetInputShape(UPDATES_INPUT_IDX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, updShape);
    const gert::Shape& updStorage = updShape->GetStorageShape();
    int32_t expectUpdRank = (idxRank - 1) + (varRank - static_cast<int32_t>(k));
    OP_CHECK_IF(static_cast<int32_t>(updStorage.GetDimNum()) != expectUpdRank,
                OP_LOGE(context_, "updates rank must be %d, got %d", expectUpdRank,
                        static_cast<int32_t>(updStorage.GetDimNum())),
                return ge::GRAPH_FAILED);

    // 逐维校验：前 idxRank-1 维与 indices 外层一致，后 varRank-K 维与 var 尾部一致
    for (int32_t d = 0; d < idxRank - 1; d++) {
        int64_t idxDim = static_cast<int64_t>(idxStorage.GetDim(static_cast<size_t>(d)));
        int64_t updDim = static_cast<int64_t>(updStorage.GetDim(static_cast<size_t>(d)));
        if (idxDim < 0 || updDim < 0) {
            continue; // 动态 shape 未知 dim，跳过
        }
        OP_CHECK_IF(idxDim != updDim,
                    OP_LOGE(context_, "updates.size(%d)=%ld must equal indices.size(%d)=%ld", d, updDim, d, idxDim),
                    return ge::GRAPH_FAILED);
    }
    for (int32_t j = static_cast<int32_t>(k); j < varRank; j++) {
        int64_t varDim = static_cast<int64_t>(varStorage.GetDim(static_cast<size_t>(j)));
        int64_t updDim =
            static_cast<int64_t>(updStorage.GetDim(static_cast<size_t>((idxRank - 1) + (j - static_cast<int32_t>(k)))));
        if (varDim < 0 || updDim < 0) {
            continue; // 动态 shape 未知 dim，跳过
        }
        OP_CHECK_IF(varDim != updDim,
                    OP_LOGE(context_, "updates.tail.size(%d)=%ld must equal var.size(%d)=%ld",
                            j - static_cast<int32_t>(k), updDim, j, varDim),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

// ══ ValidateAttr（SE §2.4: use_locking ∈ {true, false}，函数式语义恒 false，
//   v1.x tiling.cpp 迁入：读取不校验，static → 成员）══
ge::graphStatus TensorScatterNdUpdateBaseTiling::ValidateAttr()
{
    auto attrs = context_->GetAttrs();
    if (attrs != nullptr) {
        // SE §2.4: use_locking 为 bool 属性，函数式语义下恒为 false，无额外范围约束
        const auto* useLockingPtr = attrs->GetAttrPointer<bool>(USE_LOCKING_ATTR_IDX);
        if (useLockingPtr != nullptr) {
            (void)*useLockingPtr;
        }
    }
    return ge::GRAPH_SUCCESS;
}

// ══ ValidateInputs 调度器（v1.x tiling.cpp 迁入，static → 成员；compileInfo 形参移除）══
ge::graphStatus TensorScatterNdUpdateBaseTiling::ValidateInputs()
{
    OP_CHECK_IF(ValidateDtype() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ValidateDtype failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateShape() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ValidateShape failed"),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(ValidateAttr() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ValidateAttr failed"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ══ GetPlatformInfo（bucketize_v2 / 两基例 v2.0 同款双源，MDE §11.5.1）══
ge::graphStatus TensorScatterNdUpdateBaseTiling::GetPlatformInfo()
{
    auto platformPtr = context_->GetPlatformInfo();
    if (platformPtr == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const TensorScatterNdUpdateCompileInfo*>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        coreNum_ = static_cast<int64_t>(compileInfoPtr->coreNum);
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformPtr);
        coreNum_ = ascendcPlatform.GetCoreNumAiv();
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize_);
    }
    OP_CHECK_IF(coreNum_ == 0, OP_LOGE(context_, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

// ══ GetShapeAttrsInfo（v1.x TilingFunc 步骤 1~3.1 迁入，纯只读成员填充，MDE §11.5.1）══
ge::graphStatus TensorScatterNdUpdateBaseTiling::GetShapeAttrsInfo()
{
    // 公共校验（非法输入直接 GRAPH_FAILED，不参与模板降级，与 v1.x 行为一致，MDE §11.2.2）
    OP_CHECK_IF(ValidateInputs() != ge::GRAPH_SUCCESS, OP_LOGE(context_, "ValidateInputs error"),
                return ge::GRAPH_FAILED);

    varDtype_ = context_->GetInputDesc(VAR_INPUT_IDX)->GetDataType();
    idxDtype_ = context_->GetInputDesc(INDICES_INPUT_IDX)->GetDataType();

    // ── 1. var shape：各维尺寸 + 行主序步长（v1.x 步骤 1）──
    auto varShape = context_->GetInputShape(VAR_INPUT_IDX);
    const gert::Shape& varStorage = varShape->GetStorageShape();
    varRank_ = static_cast<int64_t>(varStorage.GetDimNum());
    for (int32_t d = 0; d < static_cast<int32_t>(varRank_); d++) {
        varDims_[d] = static_cast<int64_t>(varStorage.GetDim(static_cast<size_t>(d)));
    }
    int64_t varAcc = 1;
    for (int32_t d = static_cast<int32_t>(varRank_) - 1; d >= 0; d--) {
        varStrides_[d] = varAcc;
        varAcc *= varDims_[d];
    }
    totalVarElems_ = varAcc;

    // ── 2. K 与 N（v1.x 步骤 2）──
    auto idxShape = context_->GetInputShape(INDICES_INPUT_IDX);
    const gert::Shape& idxStorage = idxShape->GetStorageShape();
    int32_t idxRank = static_cast<int32_t>(idxStorage.GetDimNum());
    indexDimNum_ = static_cast<int32_t>(idxStorage.GetDim(static_cast<size_t>(idxRank - 1)));
    int64_t n = 1;
    for (int32_t d = 0; d < idxRank - 1; d++) {
        n *= static_cast<int64_t>(idxStorage.GetDim(static_cast<size_t>(d)));
    }
    totalIndexNum_ = n;

    // ── 3. sliceSize = prod(var.shape[K:])（v1.x 步骤 3）──
    int64_t sliceSize = 1;
    for (int32_t d = indexDimNum_; d < static_cast<int32_t>(varRank_); d++) {
        sliceSize *= varDims_[d];
    }
    sliceSize_ = sliceSize;

    // ── 3.1 uint32 索引域门控（v1.x 步骤 3.1，两子类共用）──
    useInt32Index_ = (totalIndexNum_ * sliceSize_ <= INT32_MAX_VAL) ? 1 : 0;

    return ge::GRAPH_SUCCESS;
}

// ══ GetWorkspaceSize（Base 占位：仅分配系统 ws，子类 Det/Nondet 各自 override，MDE §11.5.1）══
ge::graphStatus TensorScatterNdUpdateBaseTiling::GetWorkspaceSize()
{
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

} // namespace optiling
