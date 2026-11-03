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
 * \file custom_gather_infershape.cpp
 * \brief infer shape for custom_gather
 *
 * Shape 推导（SE 5.4）：output.shape = index.shape。
 * 双端校验清单（MDE 3.4，与 tiling 侧成对，逐条见函数内）：rank ∈ [1,8]、
 * dim 归一化 ∈ [0, R)、index.rank == input.rank、广播约束 d != dim 时
 * index.size(d) <= input.size(d)（动态维跳过，tiling 运行期兜底）。
 * Dtype（SE 5.5）：output.dtype = input.dtype。值依赖（SE 5.8）：否。
 */

#include "register/op_impl_registry.h"
#include "log/log.h"
#include "util/shape_util.h"

using namespace ge;

namespace ops {

constexpr int32_t INPUT_INPUT = 0;
constexpr int32_t INPUT_INDEX = 1;
constexpr int32_t OUTPUT_OUTPUT = 0;
constexpr int32_t ATTR_DIM = 0;
constexpr int64_t UNKNOWN_DIM = -1;
constexpr int64_t MAX_RANK = 8;

static ge::graphStatus InferShapeCustomGather(gert::InferShapeContext* context)
{
    const gert::Shape* inputShape = context->GetInputShape(INPUT_INPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    const gert::Shape* indexShape = context->GetInputShape(INPUT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context, indexShape);
    gert::Shape* outputShape = context->GetOutputShape(OUTPUT_OUTPUT);
    OP_CHECK_NULL_WITH_CONTEXT(context, outputShape);

    // --- Unknown rank：输出 rank 无法推导（SE 5.4）---
    if (Ops::Base::IsUnknownRank(*inputShape) || Ops::Base::IsUnknownRank(*indexShape)) {
        Ops::Base::SetUnknownRank(*outputShape);
        return GRAPH_SUCCESS;
    }

    // --- SE 1.4 / MDE 9.1：rank ∈ [1, 8] ---
    size_t r = inputShape->GetDimNum();
    OP_CHECK_IF(r < 1, OP_LOGE(context, "input rank must be >= 1, got %zu", r), return ge::GRAPH_FAILED);
    OP_CHECK_IF(r > MAX_RANK, OP_LOGE(context, "input rank must be <= 8, got %zu", r), return ge::GRAPH_FAILED);

    // --- SE 5.4：dim 归一化（负值 +R）后 ∈ [0, R) ---
    int64_t dimRaw = 0;
    auto attrs = context->GetAttrs();
    if (attrs != nullptr) {
        const auto* dimPtr = attrs->GetAttrPointer<int64_t>(ATTR_DIM);
        if (dimPtr != nullptr) {
            dimRaw = *dimPtr;
        }
    }
    int64_t dim = (dimRaw < 0) ? dimRaw + static_cast<int64_t>(r) : dimRaw;
    OP_CHECK_IF(dim < 0 || dim >= static_cast<int64_t>(r), OP_LOGE(context, "dim %ld out of range [0, %zu)", dim, r),
                return ge::GRAPH_FAILED);

    // --- SE 5.4：index.rank == input.rank ---
    OP_CHECK_IF(indexShape->GetDimNum() != r,
                OP_LOGE(context, "index rank %zu != input rank %zu", indexShape->GetDimNum(), r),
                return ge::GRAPH_FAILED);

    // --- SE 5.4：广播约束（仅静态维校验；含 UNKNOWN_DIM 跳过，由 tiling 运行期兜底）---
    for (size_t d = 0; d < r; ++d) {
        if (static_cast<int64_t>(d) == dim) {
            continue;
        }
        int64_t iDim = indexShape->GetDim(d);
        int64_t pDim = inputShape->GetDim(d);
        if (iDim != UNKNOWN_DIM && pDim != UNKNOWN_DIM) {
            OP_CHECK_IF(iDim > pDim, OP_LOGE(context, "index.size(%zu)=%ld > input.size(%zu)=%ld", d, iDim, d, pDim),
                        return ge::GRAPH_FAILED);
        }
    }

    // --- SE 5.4：output.shape = index.shape（逐维拷贝，含 -1 动态维）---
    outputShape->SetDimNum(r);
    for (size_t d = 0; d < r; ++d) {
        outputShape->SetDim(d, indexShape->GetDim(d));
    }
    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(CustomGather).InferShape(InferShapeCustomGather);
} // namespace ops
