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
 * \file wp_fa_entry.h
 * \brief Phase-4 INTEGRATED whole-port entry. Templated flash_attention_score_regbase<...>
 *        modeled on arch35 flash_attention_score_entry_regbase.h INVOKE_FA_OP_IMPL_BASEAPI:
 *        single-TU MIX (g_coreType picks Cube/Vec block + dummy), KERNEL_TYPE_MIX_AIC_1_2,
 *        instantiates FlashAttentionScoreKernelTrain<Cube,Vec>, InitBaseAPI + Process().
 *        RED LINE: NO #include "arch35/", NO aclnn/aclop — wholeport wp_* only.
 *        The host (pybind11.cpp) fills FlashAttentionScoreSimplifiedTilingData and selects
 *        the (dtype,layout,D-bucket) template instantiation via the launcher table.
 */
#ifndef WP_FA_ENTRY_H_
#define WP_FA_ENTRY_H_

#include "wholeport/wp_kernel_train.h"

using namespace AscendC;
using namespace regbaseutil;
using namespace BaseApi;
using namespace optiling;

// Templated integrated kernel body. Compiled once per arch (cube/vec) via build_ascendc.py
// DYNAMIC_MODE; g_coreType resolves the block-type selection at compile time.
template <typename INPUT_T, typename T, typename OUTPUT_T,
          ImplModeEnum implMode, LayOutTypeEnum layout,
          S1TemplateType s1T, S2TemplateType s2T, DTemplateType dT, DTemplateType dvT,
          PseTypeEnum pseMode, bool hasAtten, bool hasDrop, bool hasRope>
__aicore__ inline void wp_fa_regbase_impl(
    GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR attentionOut, GM_ADDR softmaxMax, GM_ADDR softmaxSum, GM_ADDR softmaxOut,
    GM_ADDR attenMask, GM_ADDR workspace, GM_ADDR tiling,
    GM_ADDR pse = nullptr, GM_ADDR learnableSink = nullptr)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    // fa-a5-kw-21 ROOT-CAUSE FIX (large-D D>192 broken output / 507015). On __NPU_ARCH__==3510
    // (dav-c310 = A5), GetSysWorkSpacePtr() returns __get_kfc_workspace_addr() and
    // GetUserWorkspace(workspace) = GetSysWorkSpacePtr() + RESERVED_WORKSPACE(16MB) — it IGNORES the
    // passed `workspace` arg (toolkit common_impl: `(void)(workspace); return GetSysWorkSpacePtr()+16MB`).
    // Under our custom <<<>>> launch (no GE framework) the KFC/sys workspace base is NOT bound to our
    // allocated buffer, so `user` lands outside our allocation. The BMM2 L0C->GM relay (D>192 GM path,
    // C2Position=GM) writes the bmm2 result to `user+offset` -> wrong/unmapped region. D<=128 (UB path)
    // never stages output through `user` -> always worked (31/50). MUST bind the sys workspace base to
    // our buffer BEFORE GetUserWorkspace. SetSysWorkspaceForce() is the unconditional binder (the
    // deprecated SetSysWorkspace() only sets if currently nullptr). 2x2 measured (kw-21, device.o gated):
    //   {OLD/NEW host layout} x {no-SetSys} -> sum=0.00 (output never written);
    //   {any layout} x {SetSys*}            -> correct output (d256=3685.01, d512=7312.03).
    // i.e. the SetSys* call is the load-bearing fix; binding the base makes `user` valid.
    AscendC::SetSysWorkspaceForce(workspace);
    __gm__ uint8_t *user = GetUserWorkspace(workspace);
    TPipe tPipe;

    using CubeBlockType = typename std::conditional<g_coreType == AscendC::AIC,
        FANoQuantBlockCube<INPUT_T, T, OUTPUT_T, implMode, layout, s1T, s2T, dT, dvT,
            pseMode, hasAtten, hasDrop, hasRope, false, false, false, false, false>,
        FANoQuantBlockCubeDummy<INPUT_T, T, OUTPUT_T, implMode, layout, s1T, s2T, dT, dvT,
            pseMode, hasAtten, hasDrop, hasRope, false, false, false, false, false>>::type;
    using VecBlockType = typename std::conditional<g_coreType == AscendC::AIC,
        FANoQuantBlockVecDummy<INPUT_T, T, OUTPUT_T, implMode, layout, s1T, s2T, dT, dvT,
            pseMode, hasAtten, hasDrop, hasRope, false, false, false, false, false>,
        FANoQuantBlockVecTrain<INPUT_T, T, OUTPUT_T, implMode, layout, s1T, s2T, dT, dvT,
            pseMode, hasAtten, hasDrop, hasRope, false, false, false, false, false>>::type;

    // AIV copies the simplified-tiling POD GM->local; AIC gets nullptr (ssbuf sharedParams).
    FlashAttentionScoreSimplifiedTilingData tilingLocal;
    const FlashAttentionScoreSimplifiedTilingData *__restrict tilingData = nullptr;
    if ASCEND_IS_AIV {
        __gm__ uint32_t *src = reinterpret_cast<__gm__ uint32_t *>(tiling);
        uint32_t *dst = reinterpret_cast<uint32_t *>(&tilingLocal);
        for (uint32_t i = 0; i < sizeof(FlashAttentionScoreSimplifiedTilingData) / sizeof(uint32_t); ++i) {
            dst[i] = src[i];
        }
        tilingData = &tilingLocal;
    }

    FlashAttentionScoreKernelTrain<CubeBlockType, VecBlockType> op;
    // fa-a5-kw-28: thread pse + learnableSink (were nullptr). For PSE_NONE callers (dense/dropout)
    // these default to nullptr so behavior is byte-identical; the pse kernels pass the real GM ptrs.
    op.InitBaseAPI(query, key, value, /*pse*/ pse, /*dropMask*/ nullptr,
        /*paddingMask*/ nullptr, attenMask, /*prefix*/ nullptr, /*actualSeqLengths*/ nullptr,
        /*actualSeqLengthsKv*/ nullptr, /*blockTable*/ nullptr, /*queryPaddingSize*/ nullptr,
        /*kvPaddingSize*/ nullptr, /*deqScaleQ*/ nullptr, /*deqScaleK*/ nullptr,
        /*deqScaleV*/ nullptr, /*pScale*/ nullptr, /*postQuantScale*/ nullptr,
        /*postQuantOffset*/ nullptr, /*keySharedPrefix*/ nullptr, /*valueSharedPrefix*/ nullptr,
        /*actualSharedPrefixLen*/ nullptr, /*queryRope*/ nullptr, /*keyRope*/ nullptr,
        /*learnableSink*/ learnableSink, softmaxMax, softmaxSum, softmaxOut, /*softmaxLse*/ nullptr,
        attentionOut, user, tilingData, &tPipe);
    op.Process();
}

#endif // WP_FA_ENTRY_H_
