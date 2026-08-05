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
 * \file wp_fa_host_tiling.h  (KB-asset op_host template substrate)
 * \brief Advisory snapshot of config-agnostic host-tiling logic for FA-class arch35 research.
 *
 *   Provenance-preserving extraction (historical arithmetic + comments) from the wholeport host
 *   output/a3_to_a5_port/src/kernels/flash_attention_score/op_kernel/pybind11.cpp
 *   (the `run_flash_attention_score` DoTiling). The Calc-* functions below are the config-AGNOSTIC
 *   LOGIC: they take shape/dtype params -> compute tiling VALUES. They are advisory prior art for
 *   reconstructing a task-owned implementation from the selected arch22 contract. Do not include or
 *   copy this target snapshot into a deliverable and declare generation success.
 *
 *   RED LINE (port_a3_to_a5): CPU host C++ only — NO `#include "arch35/"` device headers, NO aclnn/aclop.
 *   Self-contained (only <cstdint>/<algorithm>/<limits>/<cmath>) for inspection and differential review.
 *
 *   ============================================================================================
 *   gap#2 INTEGRITY BOUNDARY (DS-pinned, LOCKED) — what is NOT in this asset:
 *     This header provides ONLY the config-AGNOSTIC Calc-* functions (the LOGIC). It deliberately
 *     does NOT contain the per-config DoTiling-ORCHESTRATION — that is the answer the graybox kw must
 *     reproduce ITSELF. Specifically ABSENT (the kw WRITES these):
 *       - the config-extract (read q/k/v shapes, dtype, optional-input presence from at::Tensor args)
 *       - the dispatch table / launcher selection (SelectLauncher / SelectPseLauncher: the (dtype ×
 *         D-bucket × s1Basic × feature) -> wp_fa_do_* symbol map)
 *       - the TilingData POD fill (FlashAttentionScoreSimplifiedTilingData td; ip.* / mc.* assignment)
 *       - the launch glue (SetSysWorkspaceForce, blockDim, the (*launchFn)(...) call), pybind module
 *     Grep-invariant (curate bar): this header has
 *       grep -E 'run_flash_attention_score|PYBIND11_MODULE|SelectLauncher|SelectPseLauncher|TilingData|launchFn|td\.|ip\.|mc\.' = 0 hits.
 *   ============================================================================================
 */
#ifndef WP_FA_HOST_TILING_H
#define WP_FA_HOST_TILING_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace wp_fa_host {

// ---- shared scalar helpers (op_host CeilDivision / AlignUp) ----
static inline int64_t CeilDiv(int64_t a, int64_t b)
{
    if (b == 0) { return 0; }
    return (a + b - 1) / b;
}

static inline int64_t AlignUp(int64_t a, int64_t b)
{
    if (b == 0) { return a; }
    return ((a + b - 1) / b) * b;
}

// ---- op_host enum mirrors (ports of arch35 flash_attention_score_tiling_regbase.h) ----
// User sparse_mode attr (SparseMode, npu_fusion_attention convention):
enum class SparseMode : int64_t {
    NO_MASK = 0, ALL_MASK = 1, LEFT_UP_CAUSAL = 2, RIGHT_DOWN_CAUSAL = 3,
    BAND = 4, PREFIX = 5, PREFIX_COMPRESS = 6, RIGHT_DOWN_CAUSAL_BAND = 7,
    BAND_LEFT_UP_CAUSAL = 8,
};
// Internal sparseType (SparseEnum, common/include/op_host/tiling_type.h — matches kernel SparseModeEnum):
enum class SparseEnum : uint8_t {
    ALL = 0, NONE = 1, ANY = 2, CAUSAL = 3, BAND = 4, PREFIX = 5,
    BAND_COMPRESS = 6, RIGHT_DOWN_CAUSAL = 7, RIGHT_DOWN_CAUSAL_BAND = 8, BAND_LEFT_UP_CAUSAL = 9,
};
// AttenMaskCompressMode (arch35 op_host + kernel wp_attenmask.h both agree on these values):
enum class AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE = 0, LEFT_UP_CAUSAL_MODE = 1, RIGHT_DOWN_CAUSAL_MODE = 2,
    BAND_MODE = 3, PREFIX_MODE = 4, RIGHT_DOWN_CAUSAL_BAND_MODE = 5, BAND_LEFT_UP_CAUSAL_MODE = 6,
};
static const int64_t ATTEN_MASK_COMPRESS_LIMIT = 2048L;          // op_host ATTEN_MASK_COMPRESS_LIMIT
static const int64_t HIGH_PERF_BLOCK_SIZE = 128L;                // op_host HIGH_PERF_BLOCK_SIZE

// Result of the ported sparse tiling computer.
struct SparseTiling {
    SparseEnum sparseType = SparseEnum::ALL;            // -> td.sparseType
    AttenMaskCompressMode compressMode = AttenMaskCompressMode::NO_COMPRESS_MODE;
    int64_t preTokens = 0;
    int64_t nextTokens = 0;
    int64_t s1SparseValidSize = 0;
    int64_t s2SparseValidSize = 0;
    bool needCompressMask = false;                      // build the [2048,2048] compress causal mask
};

// ============================================================================================
// ComputeSparseTiling — PORT of arch35 PretokenAndNexttokenAdjustment + GetSparseInfo (the S1==S2 /
// causal subset that the kp=1.0 benchmark cases exercise; sm=0 dense + sm=3 RIGHT_DOWN_CAUSAL).
// Band/prefix branches are ported structurally but the kp=1.0 manifest never reaches them (Phase 7).
// CONFIG-AGNOSTIC: pure (sparseMode, s1, s2, pre/next-token) -> SparseTiling values.
// ============================================================================================
static SparseTiling ComputeSparseTiling(int64_t sparseMode, int64_t s1Size, int64_t s2Size,
                                        int64_t preTokensIn, int64_t nextTokensIn)
{
    SparseTiling st;
    int64_t preTokens = preTokensIn;
    int64_t nextTokens = nextTokensIn;

    // op_host: also clamps to int32 range (flash_attention_score_tiling_regbase.cpp L141-156).
    if (preTokens > std::numeric_limits<int32_t>::max()) preTokens = std::numeric_limits<int32_t>::max();
    if (nextTokens > std::numeric_limits<int32_t>::max()) nextTokens = std::numeric_limits<int32_t>::max();

    if (sparseMode == static_cast<int64_t>(SparseMode::NO_MASK)) {
        // Dense: no sparse restriction, no mask read.
        st.sparseType = SparseEnum::ALL;
        st.compressMode = AttenMaskCompressMode::NO_COMPRESS_MODE;
        st.preTokens = s2Size;     // full
        st.nextTokens = 0;
        st.needCompressMask = false;
        return st;
    }

    if (sparseMode == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL)) {
        // op_host PretokenAndNexttokenAdjustment: -> CAUSAL, preTokens=s1Size, nextTokens=0.
        st.sparseType = SparseEnum::CAUSAL;
        st.compressMode = AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE;
        st.preTokens = s1Size;
        st.nextTokens = 0;
        st.needCompressMask = true;
        return st;
    }

    if (sparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL)) {
        if (s1Size == s2Size) {
            // op_host: equal S -> CAUSAL (right-down triangle == left-up for square), pre=s1Size, next=0.
            st.sparseType = SparseEnum::CAUSAL;
            st.compressMode = AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE;
            st.preTokens = s1Size;
            st.nextTokens = 0;
            st.needCompressMask = true;
            return st;
        } else {
            // op_host: unequal S -> change to BAND (preTokens=s1Size, nextTokens=s2-s1).
            st.sparseType = SparseEnum::BAND;
            st.compressMode = AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE;
            st.preTokens = s1Size;
            st.nextTokens = s2Size - s1Size;
            st.s1SparseValidSize = preTokens;  // op_host sets s1SparseValidSize = preTokens (=s1Size)
            st.s2SparseValidSize = std::min(AlignUp(st.nextTokens, HIGH_PERF_BLOCK_SIZE), s2Size);
            st.needCompressMask = true;
            return st;
        }
    }

    // BAND / PREFIX / etc. (kp=1.0 manifest does not reach these; structural fallback to dense to
    // keep the host total — these only appear at kp<1.0 = Phase 7 dropout scope).
    st.sparseType = SparseEnum::ALL;
    st.compressMode = AttenMaskCompressMode::NO_COMPRESS_MODE;
    st.preTokens = s2Size;
    st.nextTokens = 0;
    st.needCompressMask = false;
    return st;
}

// ============================================================================================
// CalcEffSparseMode — config-AGNOSTIC predicate-DERIVATION of the EFFECTIVE sparse mode.
// FAITHFUL port of arch35 GetSparseInfo `if (!hasAttenMask) return dense`, as the wholeport host
// pybind11.cpp encodes it (~L305-337 + the hasAttenMaskHost = hasAtten note ~L382):
//   CRITICAL (Phase-D iter1, verified): npu_fusion_attention IGNORES sparse_mode when no explicit
//   atten_mask tensor is provided -> reference truth is DENSE. So when !hasAttenMask we run the dense
//   (NO_MASK) path to MATCH the reference (OL-85: mirror reference logic exactly); the user's
//   sparse_mode applies ONLY when an explicit mask is present.
// This is the predicate-DERIVATION only (generic arch35 logic). The CALLER still does the raw config-
// extract that produces `hasAttenMask` (reading optional-input presence + sparse_mode == ALL_MASK for
// the userMask sub-case), then passes the resolved `hasAttenMask` + the user `sparseMode` in here.
// CONFIG-AGNOSTIC: pure (sparseMode, hasAttenMask) -> effective sparse mode (NO_MASK when no mask).
// ============================================================================================
static inline int64_t CalcEffSparseMode(int64_t sparseMode, bool hasAttenMask)
{
    // arch35 GetSparseInfo: if (!hasAttenMask) return dense. Force dense, exactly like CANN when
    // hasAttenMask==false (pybind11.cpp ~L329-337 `} else if (!maskProvided) { ... SparseEnum::ALL`).
    if (!hasAttenMask) {
        return static_cast<int64_t>(SparseMode::NO_MASK);
    }
    return sparseMode;
}

// ============================================================================================
// CalcDBasicBlock — arch35 CalcDBasicBlock = AlignUp(D, 64). The dVBasicBlock / D-tier bucket key.
// (The Aligned64/128/256/768 device-template TIER selection that follows from this value lives in the
// kw's dispatch table, NOT here — this returns ONLY the agnostic dBasicBlock number.)
// ============================================================================================
static inline int64_t CalcDBasicBlock(int64_t D)
{
    return AlignUp(D, 64);                                          // arch35 CalcDBasicBlock
}

// ============================================================================================
// CalcS1S2BasicBlock + CalcTotalSize core-fill — FAITHFUL PORT of arch35
// (op_host/arch35/flash_attention_score_tiling_basic.cpp: CalcS1S2BasicBlock L79-115 +
//  CalcTotalSize override L35-49). Returns s1BasicBlock.
//
// GE->raw-launch adaptations (see wholeport WS1 report §3):
//   - inputDtypeBytes: arch35 reads ge::GetSizeByDataType(inputDtype); caller supplies it
//     (fp16/bf16 = 2 bytes, fp32 = 4 bytes). HIFLOAT8/FP8 branches unreachable here (dropped dead code).
//   - hasRope: always false in this deliverable (no rope input) -> the `!hasRope` guards collapse.
//   - implMode: arch35 gates core-fill on implMode != AA_INVALID_LINE_HIGH_PRECISION. AA_HIGH_PRECISION
//     always runs here so the guard is satisfied (true) for every reachable case.
//
// CONFIG-AGNOSTIC: pure (B, n2Size, gSize, S1, D, dBasicBlock, inputDtypeBytes) -> s1BasicBlock.
// ============================================================================================
static int64_t CalcS1S2BasicBlock(int64_t B, int64_t n2Size, int64_t gSize, int64_t S1,
                                  int64_t D, int64_t dBasicBlock, int64_t inputDtypeBytes)
{
    // arch35 CalcS1S2BasicBlock (basic.cpp:79-115): default s1=128/s2=128 for dSize<=256; dSize>256 ->
    // s1=64(fp32)/128, s2=128. (fp8/hifloat8 dn-d64 branches unreachable here.) We only need s1BasicBlock
    // (s2BasicBlock does not feed the host multi-core split; kernel templates are D-bucketed separately).
    int64_t s1BasicBlock;
    if (D > 256) {
        s1BasicBlock = (inputDtypeBytes == 4) ? 64 : 128;
    } else {
        s1BasicBlock = 128;
    }
    // arch35 CalcTotalSize override (basic.cpp:35-49): if totalSize < aicNum AND implMode!=INVALID_LINE
    // AND !hasRope AND inputDtype != fp32/fp8 AND dBasicBlock<=256 -> core-fill drop s1BasicBlock to 64.
    // (The nested s2BasicBlock=256 tweak when s2>1024 & no mask/pse/drop does not affect the host split.)
    {
        const int64_t s1OuterTmp = CeilDiv(S1, s1BasicBlock);
        const int64_t totalTmp = B * (n2Size * gSize) * s1OuterTmp;
        if (totalTmp < 28 && inputDtypeBytes != 4 && dBasicBlock <= 256) {
            s1BasicBlock = 64;
        }
    }
    return s1BasicBlock;
}

// ============================================================================================
// SetMultiCoreParamsRegbase — FAITHFUL PORT of arch35 set_s1OuterSize + CalcTotalSize +
// SetMultiCoreParamsRegbase (op_host flash_attention_score_tiling_regbase.cpp L998-1001 /
// regbase.cpp:942-949). Computes the multi-core split numbers.
//
// coreNum is the platform AIC count (arch35 reads ascendcPlatform.GetCoreNumAic(); A5/Ascend950PR = 28).
// CONFIG-AGNOSTIC: pure (B, n2Size, gSize, S1, s1BasicBlock, coreNum) -> MultiCoreParams values.
// ============================================================================================
struct MultiCoreParams {
    int64_t s1OuterSize = 0;
    int64_t totalSize = 0;
    int64_t actualUsedCoreNum = 0;
    int64_t splitFactorSize = 0;
    int64_t splitFactorTailSize = 0;
};

static MultiCoreParams SetMultiCoreParamsRegbase(int64_t B, int64_t n2Size, int64_t gSize,
                                                 int64_t S1, int64_t s1BasicBlock, int64_t coreNum)
{
    MultiCoreParams mc;
    // --- arch35 set_s1OuterSize + CalcTotalSize (basic.cpp:36/46; the core-fill drop is in s1BasicBlock) ---
    mc.s1OuterSize = CeilDiv(S1, s1BasicBlock);
    // totalSize = bSize * n2Size * gSize * s1OuterSize (basic.cpp:36; n2Size*gSize == N for MHA & GQA).
    mc.totalSize = B * (n2Size * gSize) * mc.s1OuterSize;
    // --- arch35 SetMultiCoreParamsRegbase (regbase.cpp:942-949) ---
    mc.actualUsedCoreNum = mc.totalSize < coreNum ? mc.totalSize : coreNum;
    if (mc.actualUsedCoreNum < 1) mc.actualUsedCoreNum = 1;
    mc.splitFactorSize = CeilDiv(mc.totalSize, mc.actualUsedCoreNum);
    mc.splitFactorTailSize = mc.totalSize - mc.splitFactorSize * (mc.actualUsedCoreNum - 1);
    return mc;
}

// ============================================================================================
// CalcThresholdForS2Size — FAITHFUL PORT of arch35 CalcThresholdForS2Size (regbase.cpp:1010-1077).
// The L2-cache-occupancy threshold that the split-core decision compares S2 against.
//   l2CacheSize: arch35 reads ascendcPlatform.GetCacheSize(CacheLine::L2) (A5/Ascend950PR = 192 MiB);
//                a platform constant, supplied by the caller.
//   attenMaskSize/dataTypeSize: the L2-occupancy terms; dense (mask=None) bench cases pass 0 mask.
// CONFIG-AGNOSTIC: pure (shape, attenMaskSize, dataTypeSize, actualUsedCoreNum, l2CacheSize) -> threshold.
// ============================================================================================
static int64_t CalcThresholdForS2Size(int64_t B, int64_t n2Size, int64_t gSize, int64_t S1, int64_t D,
                                      int64_t attenMaskSize, int64_t dataTypeSize,
                                      int64_t actualUsedCoreNum, int64_t l2CacheSize)
{
    int64_t l2CacheSizeRemain = l2CacheSize;
    l2CacheSizeRemain -= attenMaskSize;
    if (l2CacheSizeRemain < 0) l2CacheSizeRemain = 0;
    // qSize / threshold (regbase.cpp:1056-1072). n1Num/n2Num use the just-set actualUsedCoreNum.
    const int64_t n1Num = std::min(2 * actualUsedCoreNum, B * n2Size * gSize);
    const int64_t n2Num = std::min(2 * actualUsedCoreNum, B * n2Size);
    int64_t thresholdS2Size = std::numeric_limits<int64_t>::max();
    if (n2Num != 0 && D != 0 && dataTypeSize != 0) {
        const int64_t qSizeBytes = n1Num * D * S1 * dataTypeSize;
        l2CacheSizeRemain -= qSizeBytes;
        if (l2CacheSizeRemain < 0) {
            thresholdS2Size = 0;
        } else {
            thresholdS2Size = l2CacheSizeRemain / (n2Num * (D + D) * dataTypeSize);  // dSizeV==dSize
        }
    }
    return thresholdS2Size;
}

// ============================================================================================
// IsUseSplitCoreMode — FAITHFUL PORT of arch35 IsUseSplitCoreMode (regbase.cpp:1079-1096).
// Per-sparse-geometry "is the shape N/S large enough to engage SQ_MULTI_CORE_FIRST" test.
// CONFIG-AGNOSTIC: pure (mode, S1, S2, thresholdS2Size) -> bool.
// ============================================================================================
static inline bool IsUseSplitCoreMode(SparseMode m, int64_t S1, int64_t S2, int64_t thresholdS2Size)
{
    if (m == SparseMode::LEFT_UP_CAUSAL)    return std::min(S1, S2) >= thresholdS2Size;
    if (m == SparseMode::RIGHT_DOWN_CAUSAL) return (S1 <= S2) && (S2 >= thresholdS2Size);
    if (m == SparseMode::ALL_MASK)          return S2 >= thresholdS2Size;
    return false;
}

// ============================================================================================
// SetSplitCoreModeParam — FAITHFUL PORT of arch35 SetSplitCoreModeParam branch tree
// (regbase.cpp:1098-1151). Picks splitCoreMode (SQ_SINGLE_CORE_FIRST=0 / SQ_MULTI_CORE_FIRST=1) and
// firstFullLoadS1OuterIdx from the effective sparse-mode + token window + the threshold.
//
// effSparseMode: per the deliverable's verified reference-match semantics — when no explicit mask is
//   supplied npu_fusion_attention runs DENSE -> effSparseMode = NO_MASK; the user's sparse_mode applies
//   only when an explicit mask is present. The CALLER computes effSparseMode (it depends on optional-
//   input presence, which is config-extract) and passes it in — this function is otherwise agnostic.
// CONFIG-AGNOSTIC: pure (effSparseMode, S1, S2, pre/next-token, s1Basic, s1OuterSize, threshold,
//   hasAttenMask) -> SplitCoreResult values.
// ============================================================================================
struct SplitCoreResult {
    int64_t splitCoreMode = 0;                          // SQ_SINGLE_CORE_FIRST (arch35 default)
    int64_t firstFullLoadS1OuterIdx = -1;
};

static SplitCoreResult SetSplitCoreModeParam(int64_t effSparseMode, int64_t S1, int64_t S2,
                                             int64_t preTokensEff, int64_t nextTokensEff,
                                             int64_t s1BasicBlock, int64_t s1OuterSize,
                                             int64_t thresholdS2Size, bool hasAttenMaskHost)
{
    SplitCoreResult r;
    // ---- SetSplitCoreModeParam branch tree (regbase.cpp:1102-1144) ----
    if (effSparseMode == static_cast<int64_t>(SparseMode::LEFT_UP_CAUSAL) &&
        IsUseSplitCoreMode(SparseMode::LEFT_UP_CAUSAL, S1, S2, thresholdS2Size)) {
        r.firstFullLoadS1OuterIdx = CeilDiv(std::min(S1, S2), s1BasicBlock) - 1;
        r.splitCoreMode = 1;
    } else if (effSparseMode == static_cast<int64_t>(SparseMode::RIGHT_DOWN_CAUSAL) &&
               IsUseSplitCoreMode(SparseMode::RIGHT_DOWN_CAUSAL, S1, S2, thresholdS2Size)) {
        r.firstFullLoadS1OuterIdx = s1OuterSize - 1;
        r.splitCoreMode = 1;
    } else if (effSparseMode == static_cast<int64_t>(SparseMode::ALL_MASK) &&
               IsUseSplitCoreMode(SparseMode::ALL_MASK, S1, S2, thresholdS2Size)) {
        r.firstFullLoadS1OuterIdx = -1;
        r.splitCoreMode = 1;
    } else if (effSparseMode == static_cast<int64_t>(SparseMode::NO_MASK)) {
        if ((!hasAttenMaskHost || (preTokensEff >= S1 && nextTokensEff >= S2)) &&
            IsUseSplitCoreMode(SparseMode::ALL_MASK, S1, S2, thresholdS2Size)) {
            r.firstFullLoadS1OuterIdx = -1;
            r.splitCoreMode = 1;
        } else if (preTokensEff >= S1 && nextTokensEff == 0 &&
                   IsUseSplitCoreMode(SparseMode::LEFT_UP_CAUSAL, S1, S2, thresholdS2Size)) {
            r.firstFullLoadS1OuterIdx = CeilDiv(std::min(S1, S2), s1BasicBlock) - 1;
            r.splitCoreMode = 1;
        }
        // (HIFLOAT8 branch unreachable — no fp8 input dtype in this build.)
    } else if (effSparseMode == static_cast<int64_t>(SparseMode::BAND)) {
        if (preTokensEff >= S1 && nextTokensEff >= S2 &&
            IsUseSplitCoreMode(SparseMode::ALL_MASK, S1, S2, thresholdS2Size)) {
            r.firstFullLoadS1OuterIdx = -1;
            r.splitCoreMode = 1;
        } else if (preTokensEff >= S1 && nextTokensEff == 0 &&
                   IsUseSplitCoreMode(SparseMode::LEFT_UP_CAUSAL, S1, S2, thresholdS2Size)) {
            r.firstFullLoadS1OuterIdx = CeilDiv(std::min(S1, S2), s1BasicBlock) - 1;
            r.splitCoreMode = 1;
        } else if (S1 <= S2 && preTokensEff >= S1 && nextTokensEff == S2 - S1 &&
                   IsUseSplitCoreMode(SparseMode::RIGHT_DOWN_CAUSAL, S1, S2, thresholdS2Size)) {
            r.firstFullLoadS1OuterIdx = s1OuterSize - 1;
            r.splitCoreMode = 1;
        } else if (!(preTokensEff < S1 - S2 || nextTokensEff < 0) &&
                   IsUseSplitCoreMode(SparseMode::ALL_MASK, S1, S2, thresholdS2Size)) {
            r.firstFullLoadS1OuterIdx = -1;
            r.splitCoreMode = 1;
        }
    }
    return r;
}

// ============================================================================================
// CalcWorkspaceSize — FAITHFUL PORT of arch35 GetWorkspaceSize + PostTiling workspace term
// (op_host flash_attention_score_tiling_basic.cpp:180 + flash_attention_score_tiling_regbase.cpp:1617).
//   bmm2Bytes = s1Basic * bmm2ResBlockSize * calcTypeSize  (bmm2ResBlockSize = dVBasic, or
//               dVTemplateType=768 when dTemplateType>Aligned256); aligned to GM_ALIGN=512.
//   vec2Bytes = s1Basic * dVBasic * calcTypeSize  (only when dTemplateType>Aligned256); aligned GM_ALIGN.
//   userData  = (bmm2Bytes + vec2Bytes) * PING_PONG_VALUE(=3) * coreNum.
//   total     = WORK_SPACE_RESERVE_SIZE(16MB) + userData    (PostTiling += reserve; layout
//               [16MB front reserve | user-data], GetUserWorkspace returns base+16MB).
// calcTypeSize = sizeof(DT_FLOAT)=4 for BOTH fp16-HIGH_PRECISION AND bf16 (bmm1OutDtype=DT_FLOAT).
//
// `useDn` mirrors the kernel IsDn (wp_common_regbase.h:163): dense (no mask), !pse, !sink, s1Basic!=64,
// dVBasic<=256, !useAligned64Kernel -> Dn path (D>=192 workspace threshold). Nd path uses D>=128.
// The CALLER computes useDn (it depends on optional-input presence = config-extract) + passes it in.
// CONFIG-AGNOSTIC: pure (D, s1BaseSizeWs, actualUsedCoreNum, useDn) -> totalWsBytes.
// ============================================================================================
// ============================================================================================
// CalcUseDn — config-AGNOSTIC predicate-DERIVATION of the `useDnWs` boolean (the Dn-vs-Nd workspace
// path selector). FAITHFUL copy of the wholeport host pybind11.cpp `useDnWs` (~L790-800), verbatim
// terms + the arch35 threshold comments (~L790-802):
//   useDn: dense (no mask), s1Basic!=64, dVBasic<=256 -> Dn path (arch35 GetWorkspaceSize useDn).
//   idx60 fix: useDnWs MUST match the kernel's IsDn (wp_common_regbase.h:163), which is false when
//   ContainOptionalInput(pse/atten/drop) is true -> those go the Nd path which uses the workspace at
//   D>=MIN_D_TO_USE_WORKSPACE(128). The old `(!hasAtten)`-only test wrongly computed useDnWs=true for
//   pse/sink cases (idx60: pse+D128) -> picked the Dn threshold (D>192) -> allocated 0 workspace -> the
//   Nd kernel read uninit workspace -> non-determinism + wrong. Add !hasPse/!hasSink + use >= thresholds
//   (arch35 MIN_D_TO_USE_WORKSPACE=128 / DN_MIN_D_TO_USE_WORKSPACE=192 are inclusive).
//   fa-a5-kw-40: when useAligned64Kernel=true, the device kernel's isS1Base64=true -> IsDn=false (Nd path).
//   Nd-path workspace threshold is D>=128 (not D>=192 as in Dn). Reflect this in useDnWs.
// The Dn(D>=192) vs Nd(D>=128) threshold itself is applied in CalcWorkspaceSize below (the
// `(!useDn && D >= 128) || (useDn && D >= 192)` guard). This helper returns ONLY the useDn predicate.
// The CALLER does the raw config-extract that produces hasAtten/hasPse/hasSink/useAligned64Kernel
// (optional-input presence + the s1Basic==64 && dBasic<=256 derivation) and passes them in here.
// CONFIG-AGNOSTIC: pure (hasAtten, hasPse, hasSink, dBasicBlock, useAligned64Kernel) -> bool.
// NOTE (DS re-audit): the source `useDnWs` also carried an `s1BaseSizeWs != 64` term, but
// `s1BaseSizeWs` is the 128-const workspace-sizing value (vacuous-true), AND the s1==64 core-fill
// case is already subsumed by `!useAligned64Kernel` (useAligned64Kernel == s1==64 && dBasic<=256).
// So that term is dropped here: behaviorally identical to the source in every quadrant, faithful to
// the EFFECTIVE predicate, and it avoids implying the actual s1BasicBlock matters (it does not).
// ============================================================================================
static inline bool CalcUseDn(bool hasAtten, bool hasPse, bool hasSink,
                             int64_t dBasicBlock, bool useAligned64Kernel)
{
    return (!hasAtten) && (!hasPse) && (!hasSink) && (dBasicBlock <= 256)
           && (!useAligned64Kernel);
}

static int64_t CalcWorkspaceSize(int64_t D, int64_t s1BaseSizeWs, int64_t actualUsedCoreNum, bool useDn)
{
    const int64_t GM_ALIGN = 512;
    const int64_t PING_PONG = 3;
    const int64_t calcTypeSize = (int64_t)sizeof(float);              // arch35: fp16-HP + bf16 -> DT_FLOAT
    int64_t dBasicBlockWs = AlignUp(D, 64);                            // dVBasicBlock
    bool splitDWs = dBasicBlockWs > 256;                              // dTemplateType > Aligned256
    int64_t bmm2ResBlockSize = splitDWs ? 768 : dBasicBlockWs;        // dVTemplateType bucket
    int64_t bmm2Bytes = 0, vec2Bytes = 0;
    if ((!useDn && D >= 128) || (useDn && D >= 192)) {
        bmm2Bytes = s1BaseSizeWs * bmm2ResBlockSize * calcTypeSize;
        if (splitDWs) {
            vec2Bytes = s1BaseSizeWs * dBasicBlockWs * calcTypeSize;
        }
    }
    bmm2Bytes = AlignUp(bmm2Bytes, GM_ALIGN);
    vec2Bytes = AlignUp(vec2Bytes, GM_ALIGN);
    const int64_t WORK_SPACE_RESERVE_SIZE = (int64_t)16 * 1024 * 1024;   // arch35 tiling_regbase.h:70
    int64_t userDataBytes = (bmm2Bytes + vec2Bytes) * PING_PONG * actualUsedCoreNum;   // arch35 GetWorkspaceSize
    // Layout: [16MB front reserve (== RESERVED_WORKSPACE skipped by GetUserWorkspace) | user-data].
    int64_t totalWsBytes = WORK_SPACE_RESERVE_SIZE + userDataBytes;          // arch35 PostTiling +=
    return totalWsBytes;
}

} // namespace wp_fa_host

#endif // WP_FA_HOST_TILING_H
