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
 * \file tensor_scatter_nd_update_infershape.cpp
 * \brief infer shape for tensor_scatter_nd_update
 */
#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

constexpr int32_t INDEX_INPUT_VAR = 0;
constexpr int32_t INDEX_INPUT_INDICES = 1;
constexpr int32_t INDEX_INPUT_UPDATES = 2;
constexpr int32_t INDEX_OUTPUT_VAR = 0;
constexpr int32_t MAX_VAR_RANK = 8;

static ge::graphStatus InferShapeTensorScatterNdUpdate(gert::InferShapeContext* context)
{
    const gert::Shape* varShape = context->GetInputShape(INDEX_INPUT_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varShape);
    const gert::Shape* idxShape = context->GetInputShape(INDEX_INPUT_INDICES);
    OP_CHECK_NULL_WITH_CONTEXT(context, idxShape);
    const gert::Shape* updShape = context->GetInputShape(INDEX_INPUT_UPDATES);
    OP_CHECK_NULL_WITH_CONTEXT(context, updShape);
    gert::Shape* varOutShape = context->GetOutputShape(INDEX_OUTPUT_VAR);
    OP_CHECK_NULL_WITH_CONTEXT(context, varOutShape);

    // SE §6.4: var_out.shape == var.shape（inplace 输出 shape 等于 var shape）
    *varOutShape = *varShape;

    // ── 维度校验（MDE §3.2.1，与 tiling 侧同步；未知 rank/dim 时跳过）──
    if (Ops::Base::IsUnknownRank(*varShape)) {
        return ge::GRAPH_SUCCESS;
    }
    size_t varRank = varShape->GetDimNum();

    // 1. 1 <= rank(var) <= 8
    OP_CHECK_IF(varRank < 1 || varRank > static_cast<size_t>(MAX_VAR_RANK),
                OP_LOGE(context, "var rank must be in [1, 8], got %zu", varRank), return ge::GRAPH_FAILED);

    // 2. rank(indices) >= 1
    size_t idxRank = idxShape->GetDimNum();
    OP_CHECK_IF(idxRank < 1, OP_LOGE(context, "indices rank must be >= 1, got %zu", idxRank), return ge::GRAPH_FAILED);

    // 3. 1 <= K = indices.shape[-1] <= rank(var)
    int64_t k = idxShape->GetDim(idxRank - 1);
    OP_CHECK_IF(k < 1 || k > static_cast<int64_t>(varRank),
                OP_LOGE(context, "indices.shape[-1]=%ld must be in [1, %zu]", k, varRank), return ge::GRAPH_FAILED);

    // 4. updates.shape == indices.shape[:-1] + var.shape[K:]
    size_t expectUpdRank = (idxRank - 1) + (varRank - static_cast<size_t>(k));
    OP_CHECK_IF(updShape->GetDimNum() != expectUpdRank,
                OP_LOGE(context, "updates rank must be %zu, got %zu", expectUpdRank, updShape->GetDimNum()),
                return ge::GRAPH_FAILED);

    // 逐维校验：前 idxRank-1 维与 indices 外层一致，后 varRank-K 维与 var 尾部一致
    for (size_t d = 0; d < idxRank - 1; d++) {
        int64_t idxDim = idxShape->GetDim(d);
        int64_t updDim = updShape->GetDim(d);
        if (idxDim < 0 || updDim < 0) {
            continue; // 动态 shape 未知 dim，跳过
        }
        OP_CHECK_IF(idxDim != updDim,
                    OP_LOGE(context, "updates.size(%zu)=%ld must equal indices.size(%zu)=%ld", d, updDim, d, idxDim),
                    return ge::GRAPH_FAILED);
    }
    for (size_t j = static_cast<size_t>(k); j < varRank; j++) {
        int64_t varDim = varShape->GetDim(j);
        int64_t updDim = updShape->GetDim((idxRank - 1) + (j - static_cast<size_t>(k)));
        if (varDim < 0 || updDim < 0) {
            continue; // 动态 shape 未知 dim，跳过
        }
        OP_CHECK_IF(varDim != updDim,
                    OP_LOGE(context, "updates.tail.size(%zu)=%ld must equal var.size(%zu)=%ld",
                            j - static_cast<size_t>(k), updDim, j, varDim),
                    return ge::GRAPH_FAILED);
    }

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(TensorScatterNdUpdate).InferShape(InferShapeTensorScatterNdUpdate);
} // namespace ops
