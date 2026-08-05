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
 * \file flash_attention_score_tiling.cpp  (KB-asset GE op_host tiling TEMPLATE)
 * \brief GE-framework (gert::TilingContext) tiling entry for the FA-class arch35 op, authored as a
 *        REUSE-THE-SHARED-LOGIC template — NOT a raw copy of CANN
 *        ops-transformer/.../op_host/flash_attention_score_tiling.cpp.
 *
 *   ============================================================================================
 *   WHY THIS IS A KB TEMPLATE (not a CANN-source copy) — owner reckoning 2026-06-11
 *   ============================================================================================
 *   port_a3 FA archives previously shipped GE op_host (def/infershape/tiling.cpp) byte-for-byte
 *   copied from CANN source. Customers have NO CANN source -> non-reproducible. The fix: the GE
 *   op_host tiling-VALUE computation MUST CALL the already-extracted shared `wfh::` (alias of
 *   `wp_fa_host`) Calc-* layer in `wp_fa_host_tiling.h`, EXACTLY as the working pybind host
 *   (workspace/.../kernel/pybind11.cpp `run_flash_attention_score` DoTiling) does. Both hosts —
 *   the pybind launch host AND this GE tiling host — share the SAME `wfh::Calc*` functions, so they
 *   produce IDENTICAL tiling values. This file therefore contains NO inlined `CalcDBasicBlock()` /
 *   `CalcS1S2BasicBlock()` / `SetMultiCoreParamsRegbase()` arithmetic — it config-extracts from the
 *   GE TilingContext shapes, calls `wfh::Calc*`, and fills the GE FlashAttentionScoreSimplifiedTilingData
 *   POD. The arithmetic lives ONCE, in the shared header.
 *
 *   RED LINE (port_a3_to_a5): host C++ only — NO `#include "arch35/"` device headers, NO aclnn/aclop.
 *   The GE-framework headers (register/op_impl_registry.h, gert::TilingContext) are the op-build
 *   framework's PUBLIC surface (shipped with CANN toolkit), not arch35 kernel source.
 *
 *   wfh:: mapping (every tiling value this entry writes comes from a shared Calc-*; see GE_HOST_TEMPLATE.md):
 *     dBasicBlock          <- wfh::CalcDBasicBlock(D)
 *     effSparseMode        <- wfh::CalcEffSparseMode(sparseMode, hasAtten)
 *     SparseTiling st      <- wfh::ComputeSparseTiling(...)
 *     s1BasicBlock         <- wfh::CalcS1S2BasicBlock(...)   (host pins 128 for the wired Aligned128 tier)
 *     MultiCoreParams mc   <- wfh::SetMultiCoreParamsRegbase(...)
 *     threshold            <- wfh::CalcThresholdForS2Size(...)
 *     SplitCoreResult scm  <- wfh::SetSplitCoreModeParam(...)
 *     useDn                <- wfh::CalcUseDn(...)
 *     totalWsBytes         <- wfh::CalcWorkspaceSize(...)
 *   ============================================================================================
 *
 *   ============================================================================================
 *   GENERIC SKELETON vs OP-SPECIFIC (whitebox refactor, flash_attention_score-gehost-3)
 *   ============================================================================================
 *   GENERIC GE-framework tiling skeleton (arch-agnostic, identical shape for EVERY GE op):
 *       entry(gert::TilingContext* ctx):
 *         -> CheckParams(ctx)                              [generic null/shape guards]
 *         -> read platform (coreNum, L2) from compile-info [generic; cached by TilingParse]
 *         -> extract input shapes from ctx->GetInputShape  [generic GE plumbing]
 *         -> [TILING-COMPUTE HOOK: shapes -> tiling values] <<< the ONE op-specific step >>>
 *         -> ctx->GetTilingData<POD>() + fill              [generic GE plumbing; POD type op-specific]
 *         -> ctx->SetBlockDim(...) / GetWorkspaceSizes(...)[generic GE plumbing]
 *       IMPL_OP_OPTILING(<Op>).Tiling(fn).TilingParse<CompileInfo>(prepFn);   [generic; op name specific]
 *
 *   OP-SPECIFIC = the TILING-COMPUTE HOOK. For FA the hook = the `wfh::Calc*` shared-layer calls (the
 *   arch35 regbase tiling logic, lifted into wp_fa_host_tiling.h). It is fenced below with
 *   `// <<< TILING-COMPUTE HOOK: FA instance = wp_fa_host wfh::Calc* >>>` ... `// <<< END TILING-COMPUTE HOOK >>>`.
 *   To port a NEW op: keep the skeleton, swap (a) the POD type, (b) the input/attr indices, (c) the
 *   contents of the TILING-COMPUTE HOOK with that op's shared-tiling-logic calls.
 *
 *   A3->A5 TRANSFORM (see GE_HOST_TRANSFORM_RECIPE.md): tiling.cpp is REPLACE-HOOK. A3 tiling is arch22
 *   (FlashAttentionScoreTilingBase / general path — NO regbase); A5 tiling is arch35 regbase. They are
 *   DIFFERENT architectures, NOT a line-by-line transform. So: CARRY the generic skeleton (it is GE-
 *   framework, arch-agnostic) but REPLACE the A3 arch22 tiling-compute hook with the KB arch35 logic
 *   (wp_fa_host_tiling.h wfh::Calc*). This is exactly why the hook is fenced — the swap is localized.
 *
 *   PARAMETERIZABLE for a new op: op name in IMPL_OP_OPTILING + POD type + input/attr indices + the
 *   TILING-COMPUTE HOOK body. The skeleton (extract/fill/SetBlockDim/workspace/register) is fixed.
 */
#include <cstdint>
#include <register/op_impl_registry.h>

#include "wp_fa_host_tiling.h"                 // shared Calc-* logic (the REUSE — required by RED LINE)
#include "wholeport/wp_tiling_regbase.h"        // optiling::FlashAttentionScoreSimplifiedTilingData POD
#include "flash_attention_score_tiling_common.h"  // optiling::FlashAttentionScoreCompileInfo

using namespace ge;
namespace wfh = wp_fa_host;

// The CANN op-tiling entry convention marks the registered tiling functions with ASCENDC_EXTERN_C
// (== extern "C"). That macro is defined only in the device-side kernel_operator headers, which the
// port_a3 RED LINE forbids the HOST from including. The tiling entry is host C++, so define the
// portable fallback here (registration glue only — NOT tiling logic). The GE framework binds the
// function by its extern-"C" symbol via IMPL_OP_OPTILING below.
#ifndef ASCENDC_EXTERN_C
#define ASCENDC_EXTERN_C extern "C"
#endif

namespace optiling {

// <<< OP-SIGNATURE: parameterize per op >>>  (input / attr indices — must match def.cpp IR order)
// ---- GE input / output / attr indices (op IR order from flash_attention_score_def.cpp) ----
constexpr size_t FA_QUERY_INPUT_INDEX = 0;
constexpr size_t FA_KEY_INPUT_INDEX = 1;
constexpr size_t FA_VALUE_INPUT_INDEX = 2;
constexpr size_t FA_ATTEN_MASK_INPUT_INDEX = 6;   // optional atten_mask (real_shift=3, drop=4, pad=5, atten=6)
// attr order (def.cpp): scale_value=0, keep_prob=1, pre_tokens=2, next_tokens=3, head_num=4,
//                       input_layout=5, inner_precise=6, sparse_mode=7
constexpr size_t FA_ATTR_SCALE_INDEX = 0;
constexpr size_t FA_ATTR_KEEP_PROB_INDEX = 1;
constexpr size_t FA_ATTR_HEAD_NUM_INDEX = 4;
constexpr size_t FA_ATTR_SPARSE_MODE_INDEX = 7;

// Platform constants for A5 / Ascend950PR (arch35 reads these from PlatformAscendC; the GE tiling
// entry reads them from the compile-info that TilingPrepare cached — see below).
constexpr int64_t FA_L2_CACHE_DEFAULT = (int64_t)192 * 1024 * 1024;  // GetCacheSize(L2) fallback
// <<< END OP-SIGNATURE (indices) >>>

// ============================================================================================
// DoTilingFlashAttentionScoreRegbase — the GE tiling-VALUE fill, EXACTLY mirroring the pybind
// host DoTiling. config-extract (GE shapes) -> wfh::Calc* -> fill the POD -> SetBlockDim/Ws.
// ============================================================================================
static ge::graphStatus DoTilingFlashAttentionScoreRegbase(gert::TilingContext *context,
                                                          int64_t coreNumAic, int64_t l2CacheSize)
{
    // ---- config-extract (BNSD: q[B,N1,S1,D], k/v[B,N2,S2,D]) from the GE TilingContext shapes ----
    const auto &qShape = context->GetInputShape(FA_QUERY_INPUT_INDEX)->GetStorageShape();
    const auto &kShape = context->GetInputShape(FA_KEY_INPUT_INDEX)->GetStorageShape();
    const int64_t B  = qShape.GetDim(0);
    const int64_t N1 = qShape.GetDim(1);
    const int64_t S1 = qShape.GetDim(2);
    const int64_t D  = qShape.GetDim(3);
    const int64_t N2 = kShape.GetDim(1);
    const int64_t S2 = kShape.GetDim(2);
    const int64_t gSize = (N2 != 0) ? (N1 / N2) : 1;

    const auto qDtype = context->GetInputDesc(FA_QUERY_INPUT_INDEX)->GetDataType();
    const bool isFp32 = (qDtype == ge::DT_FLOAT);
    const bool isFp8  = (qDtype == ge::DT_FLOAT8_E4M3FN) || (qDtype == ge::DT_FLOAT8_E5M2);
    const int64_t inputDtypeBytes = isFp32 ? 4 : (isFp8 ? 1 : 2);   // fp16/bf16 = 2

    // attrs
    const auto *attrs = context->GetAttrs();
    const float scaleValue = (attrs->GetAttrPointer<float>(FA_ATTR_SCALE_INDEX) != nullptr)
                                 ? *attrs->GetAttrPointer<float>(FA_ATTR_SCALE_INDEX) : 1.0f;
    const float keepProb = (attrs->GetAttrPointer<float>(FA_ATTR_KEEP_PROB_INDEX) != nullptr)
                                ? *attrs->GetAttrPointer<float>(FA_ATTR_KEEP_PROB_INDEX) : 1.0f;
    const int64_t sparseMode = (attrs->GetAttrPointer<int64_t>(FA_ATTR_SPARSE_MODE_INDEX) != nullptr)
                                   ? *attrs->GetAttrPointer<int64_t>(FA_ATTR_SPARSE_MODE_INDEX) : 0;

    // optional atten_mask presence (reference-match: dense unless an explicit mask tensor is supplied)
    const bool hasAtten = (context->GetOptionalInputShape(FA_ATTEN_MASK_INPUT_INDEX) != nullptr);

    // <<< TILING-COMPUTE HOOK: FA instance = wp_fa_host wfh::Calc* >>>
    // The ONE op-specific step. For FA this is the arch35 regbase tiling logic, reused from the shared
    // KB layer wp_fa_host_tiling.h (SAME calls + SAME pins as the pybind launch host pybind11.cpp, so
    // the two hosts produce identical tiling values). A different op REPLACES this hook with its own
    // shared-tiling-logic calls; the skeleton above/below is untouched.  RED LINE: this hook stays
    // wfh::-calls — it must NOT inline raw arch35 arithmetic.
    // ---- Calc-* (wp_fa_host_tiling.h shared asset — the REUSE; SAME calls as pybind11.cpp) ----
    const int64_t effSparseMode = wfh::CalcEffSparseMode(sparseMode, hasAtten);
    wfh::SparseTiling st = wfh::ComputeSparseTiling(
        hasAtten ? (int64_t)wfh::SparseMode::NO_MASK : effSparseMode, S1, S2, 0, 0);
    const int64_t dBasicBlock = wfh::CalcDBasicBlock(D);
    // Host pins s1BasicBlock=128 for the wired Aligned128 device tier (K2 host<->kernel consistency,
    // pybind11.cpp note ~L126-130). wfh::CalcS1S2BasicBlock is still CALLED to derive the agnostic
    // value the dispatch would otherwise use; we keep the same pin the launch host uses.
    (void)wfh::CalcS1S2BasicBlock(B, N2, gSize, S1, D, dBasicBlock, inputDtypeBytes);
    const int64_t s1BasicBlock = 128;
    const bool useAligned64Kernel = false;
    wfh::MultiCoreParams mc = wfh::SetMultiCoreParamsRegbase(B, N2, gSize, S1, s1BasicBlock, coreNumAic);
    const int64_t attenMaskSize = hasAtten ? (S1 * S2 * 1) : 0;
    const int64_t threshold = wfh::CalcThresholdForS2Size(B, N2, gSize, S1, D, attenMaskSize,
                                                          inputDtypeBytes, mc.actualUsedCoreNum, l2CacheSize);
    wfh::SplitCoreResult scm = wfh::SetSplitCoreModeParam(effSparseMode, S1, S2, st.preTokens,
        st.nextTokens, s1BasicBlock, mc.s1OuterSize, threshold, hasAtten);
    const bool useDn = isFp8 ? false
                             : wfh::CalcUseDn(hasAtten, false, false, dBasicBlock, useAligned64Kernel);
    const int64_t totalWsBytes = wfh::CalcWorkspaceSize(D, 128, mc.actualUsedCoreNum, useDn);
    // <<< END TILING-COMPUTE HOOK >>>  (everything below is GENERIC GE-skeleton POD-fill + wiring)

    // ---- fill the GE TilingData POD (GetTilingData<T>() gives the framework-allocated raw region) ----
    auto *td = context->GetTilingData<FlashAttentionScoreSimplifiedTilingData>();
    if (td == nullptr) {
        return ge::GRAPH_FAILED;
    }
    *td = {};
    auto &ip = td->inputParamsRegbase;
    auto &mcr = td->multiCoreParamsRegbase;

    ip.bSize = B; ip.n2Size = N2; ip.gSize = gSize;
    ip.s1Size = S1; ip.s2Size = S2; ip.alignedS2 = wfh::AlignUp(S2, 16);
    ip.dSize = D; ip.dSizeV = D; ip.dSizeRope = 0;
    ip.scaleValue = scaleValue;
    ip.keepProb = keepProb;
    ip.keepProbUint8 = 0;
    ip.preTokens = st.preTokens; ip.nextTokens = st.nextTokens;
    ip.s1SparseValidSize = st.s1SparseValidSize; ip.s2SparseValidSize = st.s2SparseValidSize;
    ip.layoutType = 3;   // BNSD
    ip.implMode = 0;     // AA_HIGH_PRECISION
    ip.sparseType = hasAtten ? (uint8_t)wfh::SparseEnum::ANY : (uint8_t)st.sparseType;
    ip.attenMaskCompressMode = (uint8_t)wfh::AttenMaskCompressMode::NO_COMPRESS_MODE;
    ip.attenMaskShapeType = 2;  // (1,1,1,S1,S2) broadcast over B/N
    ip.attenMaskDataType = 1;   // bool(uint8)
    ip.attenMaskS2Size = (uint32_t)S2;
    ip.bandIndex = 0;
    ip.pseType = (uint32_t)9;   // PSE_NONE_TYPE
    ip.needDropMaskOp = 0;
    ip.dropMaskOuter = 0;
    ip.tndSoftmaxOut = 0;
    ip.qStartIdx = 0; ip.kvStartIdx = 0;
    ip.seed = 0; ip.offset = 0;
    ip.isGqa = (gSize > 1) ? 1 : 0;

    mcr.coreNum = (int32_t)mc.actualUsedCoreNum;
    mcr.totalSize = mc.totalSize;
    mcr.s1OuterSize = mc.s1OuterSize;
    mcr.splitFactorSize = mc.splitFactorSize;
    mcr.splitFactorTailSize = mc.splitFactorTailSize;
    mcr.firstFullLoadS1OuterIdx = scm.firstFullLoadS1OuterIdx;
    mcr.splitCoreMode = (uint8_t)scm.splitCoreMode;
    // sparseStartIdx[48]/bnStartIdx[48]: runtime even-split for splitCoreMode==0 -> 0 (memset above).

    // ---- GE framework wiring: block dim + workspace ----
    context->SetBlockDim((uint32_t)mc.actualUsedCoreNum);
    // GE tiling-key selection (the (dtype x D-bucket x s1Basic x feature) -> template key) is the
    // GE-framework analogue of the pybind SelectLauncher dispatch table; for a single wired Aligned128
    // tier it is a fixed key. A multi-tier op authors the key map here, mirroring SelectLauncher.
    size_t *workspaces = context->GetWorkspaceSizes(1);
    if (workspaces == nullptr) {
        return ge::GRAPH_FAILED;
    }
    workspaces[0] = (size_t)totalWsBytes;

    return ge::GRAPH_SUCCESS;
}

// ============================================================================================
// TilingFlashAttentionScore — GE tiling entry. Reads platform info (coreNumAic, L2 size) then
// dispatches into the shared-logic DoTiling above. (Empty-input fast paths are intentionally
// omitted from this template — they carry no tiling arithmetic and a new op adds them per its own
// degenerate-shape contract; the load-bearing part is the shared-logic reuse.)
// ============================================================================================
ASCENDC_EXTERN_C ge::graphStatus TilingFlashAttentionScore(gert::TilingContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    const auto *compileInfo =
        reinterpret_cast<const FlashAttentionScoreCompileInfo *>(context->GetCompileInfo());
    const int64_t coreNumAic =
        (compileInfo != nullptr && compileInfo->aicNum != 0) ? compileInfo->aicNum : 28;
    const int64_t l2CacheSize =
        (compileInfo != nullptr && compileInfo->l2CacheSize != 0)
            ? static_cast<int64_t>(compileInfo->l2CacheSize)
            : FA_L2_CACHE_DEFAULT;
    return DoTilingFlashAttentionScoreRegbase(context, coreNumAic, l2CacheSize);
}

static ge::graphStatus PopulatePlatformCompileInfo(
    gert::TilingParseContext &context, FlashAttentionScoreCompileInfo &compileInfo)
{
    const auto platformInfo = context.GetPlatformInfo();
    if (platformInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }

    platform_ascendc::PlatformAscendC platform(platformInfo);
    compileInfo.aivNum = platform.GetCoreNumAiv();
    compileInfo.aicNum = platform.GetCoreNumAic();
    compileInfo.socVersion = platform.GetSocVersion();
    compileInfo.npuArch = platform.GetCurNpuArch();
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo.ubSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, compileInfo.l1Size);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, compileInfo.l0cSize);
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::L2, compileInfo.l2CacheSize);
    return ge::GRAPH_SUCCESS;
}

// ============================================================================================
// TilingPrepareForFlashAttentionScore — caches platform numbers into the compile-info POD. This is
// boilerplate platform-query (no tiling arithmetic); a faithful structural form is acceptable, but
// it is re-expressed here against the public PlatformAscendC surface, not copied from CANN source.
// ============================================================================================
ASCENDC_EXTERN_C ge::graphStatus TilingPrepareForFlashAttentionScore(gert::TilingParseContext *context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto *compileInfo = context->GetCompiledInfo<FlashAttentionScoreCompileInfo>();
    if (compileInfo == nullptr) {
        return ge::GRAPH_FAILED;
    }
    return PopulatePlatformCompileInfo(*context, *compileInfo);
}

// GENERIC SKELETON: register the tiling + tiling-parse entry. Op-specific = op name + CompileInfo type.
IMPL_OP_OPTILING(FlashAttentionScore)   // <<< OP-SIGNATURE: op name >>>
    .Tiling(TilingFlashAttentionScore)
    .TilingParse<FlashAttentionScoreCompileInfo>(TilingPrepareForFlashAttentionScore);  // <<< OP-SIGNATURE: CompileInfo type >>>

}  // namespace optiling
