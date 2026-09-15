/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Modifications Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file flash_attention_score_infershape.cpp  (KB-asset GE op_host infershape TEMPLATE)
 * \brief GE shape/dtype inference for the FA-class arch35 op, authored as a KB template (re-expressed
 *        from the FA shape contract), NOT a raw copy of CANN
 *        ops-transformer/.../op_host/flash_attention_score_infershape.cpp.
 *
 *   ============================================================================================
 *   GENERIC SKELETON vs OP-SPECIFIC (whitebox refactor, flash_attention_score-gehost-3)
 *   ============================================================================================
 *   GENERIC GE-framework skeleton (arch-agnostic, identical shape for EVERY GE op):
 *       ge::graphStatus InferShape<Op>(gert::InferShapeContext* ctx) {
 *           read ctx->GetInputShape(i) / GetInputDesc(i) / GetAttrs();  null-guard each;
 *           [SHAPE-RELATION HOOK: compute output dims from input dims+attrs];
 *           write ctx->GetOutputShape(j)->SetDimNum/SetDim;  return GRAPH_SUCCESS;
 *       }
 *       ge::graphStatus InferDataType<Op>(gert::InferDataTypeContext* ctx) {
 *           [DTYPE-RELATION HOOK]; ctx->SetOutputDataType(j, dt);
 *       }
 *       IMPL_OP_INFERSHAPE(<Op>).InferShape(...).InferDataType(...);
 *   The context read/write plumbing + null-guards + the registration macro are GENERIC. OP-SPECIFIC =
 *   the SHAPE-RELATION HOOK (how output dims derive from input dims) + the DTYPE-RELATION HOOK. Both
 *   are fenced below with `// <<< SHAPE-RELATION HOOK >>>` / `// <<< DTYPE-RELATION HOOK >>>`.
 *
 *   A3->A5 TRANSFORM (see GE_HOST_TRANSFORM_RECIPE.md): infershape.cpp is CARRY (A3 == A5). Shape
 *   inference is ARCH-AGNOSTIC — it reads logical tensor shapes, not arch22/arch35 tiling. Grep the
 *   customer's A3 infershape for `arch22`/`arch35`/regbase: 0 refs => carry it verbatim. The ONLY A5
 *   delta is the fp8 dtype branch in the DTYPE hook (fp8 in -> bf16 out), which is a dtype-table edit,
 *   not a shape change.
 *
 *   FA shape contract (the SHAPE-RELATION HOOK content for THIS op):
 *     - softmax_max / softmax_sum : fp32, (B, N, S1, 8)  [fp8/hifloat8: last dim = 1; TND: (T,N,8)]
 *     - softmax_out               : empty (B,N,S,S) collapsed to 0s in this deliverable scope
 *     - attention_out             : same layout as query, with the D dim taken from value
 *
 *   RED LINE (port_a3_to_a5): host C++ only — NO `#include "arch35/"`, NO aclnn/aclop. The
 *   register/op_impl_registry.h + gert::InferShapeContext are the GE op-build PUBLIC framework surface.
 */
#include <string>
#include <cctype>
#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>

using namespace ge;

namespace ops {

// FA IR layout: query/key/value at input 0/1/2; head_num attr at 4, input_layout attr at 5.
static constexpr size_t FA_QUERY_IN = 0;
static constexpr size_t FA_KEY_IN = 1;
static constexpr size_t FA_VALUE_IN = 2;
static constexpr size_t FA_HEAD_NUM_ATTR = 4;
static constexpr size_t FA_LAYOUT_ATTR = 5;
static constexpr size_t FA_OUTDTYPE_ATTR = 11;
// Output IR order:
static constexpr size_t FA_SOFTMAX_MAX_OUT = 0;
static constexpr size_t FA_SOFTMAX_SUM_OUT = 1;
static constexpr size_t FA_SOFTMAX_OUT = 2;
static constexpr size_t FA_ATTENTION_OUT = 3;
// softmax-stat trailing dim: 8 for fp16/bf16/fp32, 1 for fp8/hifloat8.
static constexpr int64_t FA_SOFTMAX_STAT_DIM = 8;
static constexpr int64_t FA_SOFTMAX_STAT_DIM_FP8 = 1;

static std::string FaUpperLayout(const char *layout)
{
    std::string s = (layout != nullptr) ? std::string(layout) : std::string();
    for (auto &c : s) {
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    }
    return s;
}

struct FaShapeInfo {
    int64_t batchSize = 1;
    int64_t sequenceSize = 1;
    int64_t tokenSize = 0;
};

static bool IsSupportedFaLayout(const std::string &layout)
{
    return layout == "BSH" || layout == "BSND" || layout == "SBH" || layout == "BNSD" || layout == "TND";
}

static FaShapeInfo DeriveFaShapeInfo(const gert::Shape *queryShape, const std::string &layout)
{
    FaShapeInfo shapeInfo;
    if (layout == "SBH") {
        shapeInfo.batchSize = queryShape->GetDim(1);
        shapeInfo.sequenceSize = queryShape->GetDim(0);
    } else if (layout == "TND") {
        shapeInfo.tokenSize = queryShape->GetDim(0);
    } else if (layout == "BSND" || layout == "BSH") {
        shapeInfo.batchSize = queryShape->GetDim(0);
        shapeInfo.sequenceSize = queryShape->GetDim(1);
    } else {  // BNSD
        shapeInfo.batchSize = queryShape->GetDim(0);
        shapeInfo.sequenceSize = queryShape->GetDim(2);
    }
    return shapeInfo;
}

static ge::graphStatus SetSoftmaxStatisticsShapes(gert::InferShapeContext *context, const std::string &layout,
                                                   const FaShapeInfo &shapeInfo, int64_t headNum, int64_t statDim)
{
    gert::Shape *softmaxMaxShape = context->GetOutputShape(FA_SOFTMAX_MAX_OUT);
    gert::Shape *softmaxSumShape = context->GetOutputShape(FA_SOFTMAX_SUM_OUT);
    if (softmaxMaxShape == nullptr || softmaxSumShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    if (layout == "TND" && statDim == FA_SOFTMAX_STAT_DIM) {
        softmaxMaxShape->SetDimNum(3);
        softmaxMaxShape->SetDim(0, shapeInfo.tokenSize);
        softmaxMaxShape->SetDim(1, headNum);
        softmaxMaxShape->SetDim(2, statDim);
    } else {
        softmaxMaxShape->SetDimNum(4);
        softmaxMaxShape->SetDim(0, shapeInfo.batchSize);
        softmaxMaxShape->SetDim(1, headNum);
        softmaxMaxShape->SetDim(2, shapeInfo.sequenceSize);
        softmaxMaxShape->SetDim(3, statDim);
    }
    *softmaxSumShape = *softmaxMaxShape;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetSoftmaxOutputShape(gert::InferShapeContext *context)
{
    gert::Shape *softmaxOutShape = context->GetOutputShape(FA_SOFTMAX_OUT);
    if (softmaxOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    softmaxOutShape->SetDimNum(4);
    softmaxOutShape->SetDim(0, 0);
    softmaxOutShape->SetDim(1, 0);
    softmaxOutShape->SetDim(2, 0);
    softmaxOutShape->SetDim(3, 0);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SetAttentionOutputShape(gert::InferShapeContext *context, const gert::Shape *queryShape,
                                                const gert::Shape *keyShape, const gert::Shape *valueShape,
                                                const std::string &layout, int64_t headNum)
{
    gert::Shape *attentionOutShape = context->GetOutputShape(FA_ATTENTION_OUT);
    if (attentionOutShape == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *attentionOutShape = *queryShape;
    if (layout == "BSND" || layout == "BNSD") {
        attentionOutShape->SetDim(3, valueShape->GetDim(3));
        return ge::GRAPH_SUCCESS;
    }
    if (layout == "BSH" || layout == "SBH") {
        if (headNum == 0) {
            attentionOutShape->SetDim(2, 0);
            return ge::GRAPH_SUCCESS;
        }
        const int64_t qH = queryShape->GetDim(2);
        const int64_t dQ = qH / headNum;
        if (dQ == 0) {
            attentionOutShape->SetDim(2, 0);
            return ge::GRAPH_SUCCESS;
        }
        const int64_t kH = keyShape->GetDim(2);
        const int64_t n2 = kH / dQ;
        if (n2 == 0) {
            attentionOutShape->SetDim(2, headNum * dQ);
            return ge::GRAPH_SUCCESS;
        }
        const int64_t vH = valueShape->GetDim(2);
        const int64_t dV = vH / n2;
        attentionOutShape->SetDim(2, headNum * dV);
        return ge::GRAPH_SUCCESS;
    }
    attentionOutShape->SetDim(2, valueShape->GetDim(2));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferShapeFlashAttentionScore(gert::InferShapeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const gert::Shape *queryShape = context->GetInputShape(FA_QUERY_IN);
    const gert::Shape *keyShape   = context->GetInputShape(FA_KEY_IN);
    const gert::Shape *valueShape = context->GetInputShape(FA_VALUE_IN);
    const auto *queryDesc = context->GetInputDesc(FA_QUERY_IN);
    if (queryShape == nullptr || keyShape == nullptr || valueShape == nullptr || queryDesc == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *attrs = context->GetAttrs();
    if (attrs == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t *headNumPtr = attrs->GetInt(FA_HEAD_NUM_ATTR);
    const char *layoutPtr = attrs->GetAttrPointer<char>(FA_LAYOUT_ATTR);
    if (headNumPtr == nullptr || layoutPtr == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const int64_t headNum = *headNumPtr;
    const std::string layout = FaUpperLayout(layoutPtr);
    if (!IsSupportedFaLayout(layout)) {
        return ge::GRAPH_FAILED;
    }

    // <<< SHAPE-RELATION HOOK: FA instance = derive output shapes from q/k/v shapes + head_num/layout >>>
    // (everything from here to END SHAPE-RELATION HOOK is OP-SPECIFIC; the read/write plumbing above
    //  and below the fence is the GENERIC GE skeleton.)
    // ---- derive (B, S, T) per layout (the FA shape contract) ----
    const FaShapeInfo shapeInfo = DeriveFaShapeInfo(queryShape, layout);

    // ---- softmax_max / softmax_sum: fp32 stats. fp8 -> trailing dim 1; TND -> (T,N,dim); else (B,N,S,dim) ----
    const auto qDtype = queryDesc->GetDataType();
    const int64_t statDim = (qDtype == ge::DT_HIFLOAT8 || qDtype == ge::DT_FLOAT8_E4M3FN ||
                             qDtype == ge::DT_FLOAT8_E5M2)
                                ? FA_SOFTMAX_STAT_DIM_FP8 : FA_SOFTMAX_STAT_DIM;
    if (SetSoftmaxStatisticsShapes(context, layout, shapeInfo, headNum, statDim) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // ---- softmax_out: empty in this deliverable scope (all-zero dims) ----
    if (SetSoftmaxOutputShape(context) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    // ---- attention_out: copy query shape, then take the D dim from value (BSND/BNSD) ----
    const ge::graphStatus attentionStatus = SetAttentionOutputShape(
        context, queryShape, keyShape, valueShape, layout, headNum);
    if (attentionStatus != ge::GRAPH_SUCCESS) {
        return attentionStatus;
    }
    // <<< END SHAPE-RELATION HOOK >>>
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus InferDataTypeFlashAttentionScore(gert::InferDataTypeContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto qDtype = context->GetInputDataType(FA_QUERY_IN);
    // <<< DTYPE-RELATION HOOK: FA instance = stats fp32; fp8-in -> bf16-out; else mirror in-dtype >>>
    // (A3->A5 delta lives ENTIRELY in this hook: the fp8 branch is A5-only. A3 carries the else-branch.)
    // softmax_max / softmax_sum are always fp32.
    context->SetOutputDataType(FA_SOFTMAX_MAX_OUT, ge::DT_FLOAT);
    context->SetOutputDataType(FA_SOFTMAX_SUM_OUT, ge::DT_FLOAT);

    // fp8 / hifloat8 inputs dequantize to a bf16 attention_out + softmax_out.  [A5-only branch]
    if (qDtype == ge::DT_FLOAT8_E5M2 || qDtype == ge::DT_FLOAT8_E4M3FN || qDtype == ge::DT_HIFLOAT8) {
        context->SetOutputDataType(FA_SOFTMAX_OUT, ge::DT_BF16);
        context->SetOutputDataType(FA_ATTENTION_OUT, ge::DT_BF16);
        return ge::GRAPH_SUCCESS;
    }
    // Otherwise the float outputs mirror the input dtype.  [A3==A5 branch]
    context->SetOutputDataType(FA_SOFTMAX_OUT, qDtype);
    context->SetOutputDataType(FA_ATTENTION_OUT, qDtype);
    // <<< END DTYPE-RELATION HOOK >>>
    return ge::GRAPH_SUCCESS;
}

// GENERIC SKELETON: register the two inference fns. Only the op name <FlashAttentionScore> is op-specific.
IMPL_OP_INFERSHAPE(FlashAttentionScore)   // <<< OP-SIGNATURE: op name >>>
    .InferShape(InferShapeFlashAttentionScore)
    .InferDataType(InferDataTypeFlashAttentionScore);

}  // namespace ops
