/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ATTENMASK_ARCH35_H
#define ATTENMASK_ARCH35_H

/*!
 * \file attenmask_arch35.h
 * \brief task#34 close-port of CANN arch35's WHOLE attention-mask mechanism,
 *        adapted into our library-VEC TU (NO #include "arch35/...").
 *
 * Close-port of the mask offset machinery from
 *   ~/workspace/cann/ops-transformer/attention/common/op_kernel/arch35/attenmask.h
 * specifically:
 *   - AttenMaskCompressMode enum            (attenmask.h:25-33, VERBATIM values)
 *   - AttenMaskComputeMode  enum            (attenmask.h:35-43, VERBATIM)
 *   - AttenMaskInfo struct (slimmed)        (attenmask.h:45-57)
 *   - GetAttenMaskComputeMode               (attenmask.h:108-166, VERBATIM algorithm)
 *   - ComputeOffsetForNoCompress            (attenmask.h:168-220, dense sparseMode 0)
 *   - ComputeOffsetForCausal                (attenmask.h:222-236, causal/band)
 *   - ComputeOffsetForPrefixRectangle       (attenmask.h:238-249, prefix)
 *   - ComputeAttenMaskInnerOffset           (attenmask.h:343-517, compressMode dispatch)
 *   - ComputeAttenMaskOffset                (attenmask.h:519-528)
 *
 * WHY this exists (task#34): the prior spawn applied the mask as an arch22-style
 * float ADDITIVE bias Add() — a Frankenstein shortcut (owner+main rejected). arch35's
 * coherent mechanism takes the RAW bool mask, computes a per-tile GM offset from the
 * compressMode (which subsumes dense sparseMode 0 AND sparseMode 1-4 causal/band/prefix
 * via preTokens/nextTokens), copies the bool tile, and masked-fills the scores with a
 * negative extreme (the arch35 register Select(vreg_min) semantic). This header ports
 * that offset math VERBATIM (formulas unchanged), adapting only the carrier structs:
 * the full arch35 RunInfo/ConstInfo are 60+-field structs bound to the arch35 tiling-key
 * + matmul scaffolding; we provide slim FaRunCtx/FaConstCtx carrying EXACTLY the fields
 * the offset formulas read. The mask application itself (Select-min) is done with the
 * public AscendC library VEC ops (we are library-VEC, not the arch35 __simd_vf__ VF
 * softmax paradigm — see analysis: arch35's Select lives fused inside vf_basic_block_*).
 *
 * RED LINE compliance: no #include "arch35/...", no aclnn/aclop. Pure scalar offset
 * arithmetic + public AscendC VEC application. This file lives at op_kernel/ top level
 * (same allowance as the other regbase_*.h close-port library headers).
 */

#include "kernel_operator.h"

namespace fa_atten_mask {

// ---- VERBATIM from arch35 attenmask.h:25-33 ------------------------------------
// sparseMode (host attr) maps 1:1 onto these compressMode values in CANN's host
// tiling (flash_attention_score tiling: sparse_mode -> compressMode).
enum class AttenMaskCompressMode : uint8_t {
    NO_COMPRESS_MODE             = 0,   // dense bool mask [S1,S2] (sparse_mode 0)
    LEFT_UP_CAUSAL_MODE          = 1,   // causal, left-up aligned   (sparse_mode 1/2)
    RIGHT_DOWN_CAUSAL_MODE       = 2,   // causal, right-down aligned (sparse_mode 3/4)
    BAND_MODE                    = 3,   // band (preTokens/nextTokens)
    PREFIX_MODE                  = 4,   // prefix
    RIGHT_DOWN_CAUSAL_BAND_MODE  = 5,
    BAND_LEFT_UP_CAUSAL_MODE     = 6
};

// ---- VERBATIM from arch35 attenmask.h:35-43 ------------------------------------
enum class AttenMaskComputeMode : uint8_t {
    NORMAL_MODE = 0,
    CAUSAL_OR_NEXT_ONLY_MODE,
    PRE_ONLY_MODE,
    PRE_AND_NEXT_MODE,
    NO_NEED_COMPUTE_MODE,
    PREFIX_COMPUTE_MODE,
    PREFIX_N_COMPUTE_MODE
};

// ---- slimmed from arch35 attenmask.h:45-57 -------------------------------------
// Carries exactly the mask-info fields the offset math reads (dropped the
// MLA/rope/infer/quant fields our train-only library-VEC path never touches).
struct AttenMaskInfo {
    int64_t preTokens     = 0;
    int64_t nextTokens    = 0;
    uint8_t compressMode  = 0;          // == AttenMaskCompressMode
    int64_t attenMaskS2Size = 0;        // stride of the (compressed or dense) mask GM
    int64_t attenMaskOffsetPre = 0;     // band/prefix 2nd-mask offset
    AttenMaskComputeMode computeMode = AttenMaskComputeMode::NORMAL_MODE;
};

// ---- slim carriers replacing the 60-field arch35 RunInfo/ConstInfo --------------
// FaConstCtx: the global/static block sizes the offset formulas read.
struct FaConstCtx {
    uint32_t s1BaseSize = 0;   // == BLOCK_M (query rows per base block)
    uint32_t s2BaseSize = 0;   // == BLOCK_N (kv  cols per base block)
};

// FaRunCtx: the per-tile run-time indices the offset formulas read.
struct FaRunCtx {
    int64_t s1oIdx        = 0;     // outer s1 block index (== curBx)
    int64_t s2StartIdx    = 0;     // kv start (0 for non-prefix dense)
    int64_t s2LoopCount   = 0;     // inner kv tile index (== kvTileIdx)
    int64_t vecCoreOffset = 0;     // sub-block row offset within the base block
    int64_t actualS1Size  = 0;     // qSeqLen
    int64_t actualS2Size  = 0;     // kvSeqLen
    int64_t b1SSAttenMaskOffset = 0; // per-(b) base offset for dense BS1S2 / BN2GS1S2
};

constexpr int64_t attenMaskBS1S2     = 0;  // dense mask is [S1,S2] broadcast over B/N
constexpr int64_t attenMaskBN2GS1S2  = 1;

__aicore__ inline int64_t MinI64(int64_t a, int64_t b) { return a < b ? a : b; }
__aicore__ inline int64_t MaxI64(int64_t a, int64_t b) { return a > b ? a : b; }

// ---- VERBATIM from arch35 attenmask.h:168-220 (dense, NO_COMPRESS) --------------
// For our dense case: attenMaskShapeType == BS1S2, b1SSAttenMaskOffset == 0 (mask is
// broadcast [S1,S2]), so this reduces to (s1oIdx*s1BaseSize + vecCoreOffset)*S2 +
// s2StartIdx + s2LoopCount*s2BaseSize  ==  gRow*S2 + kvColBase. Same indexing the
// arch22 patch used, but now produced by the arch35 offset formula coherently.
__aicore__ inline int64_t ComputeOffsetForNoCompress(const FaRunCtx &runInfo,
                                                     const FaConstCtx &constInfo,
                                                     const AttenMaskInfo &m)
{
    int64_t bOffset = runInfo.b1SSAttenMaskOffset;   // 0 for dense [S1,S2] broadcast
    int64_t s1Offset = runInfo.s1oIdx * constInfo.s1BaseSize + runInfo.vecCoreOffset;
    int64_t s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * constInfo.s2BaseSize;
    s1Offset *= m.attenMaskS2Size;
    return bOffset + s1Offset + s2Offset;
}

// ---- VERBATIM from arch35 attenmask.h:222-236 ----------------------------------
__aicore__ inline int64_t ComputeOffsetForCausal(int64_t delta, uint32_t s1BaseSize,
                                                 uint32_t s2BaseSize, int64_t attenMaskS2Size,
                                                 int64_t vecCoreOffset)
{
    if (delta <= 0) {
        return MinI64(-1 * delta, (int64_t)s1BaseSize) + vecCoreOffset * attenMaskS2Size;
    }
    return (MinI64(delta, (int64_t)s2BaseSize) + vecCoreOffset) * attenMaskS2Size;
}

// ---- VERBATIM from arch35 attenmask.h:238-249 ----------------------------------
__aicore__ inline int64_t ComputeOffsetForPrefixRectangle(int64_t delta, uint32_t s2BaseSize,
                                                          int64_t attenMaskS2Size)
{
    if (delta <= 0) {
        return attenMaskS2Size * attenMaskS2Size + attenMaskS2Size / 2;
    } else if (delta >= (int64_t)s2BaseSize) {
        return attenMaskS2Size * attenMaskS2Size;
    } else {
        return attenMaskS2Size * attenMaskS2Size + attenMaskS2Size / 2 - delta;
    }
}

// ---- VERBATIM algorithm from arch35 attenmask.h:108-166 ------------------------
__aicore__ inline void GetAttenMaskComputeMode(int64_t deltaCausalOrNext, int64_t deltaPre,
                                               const FaRunCtx &runInfo, const FaConstCtx &constInfo,
                                               AttenMaskInfo &m)
{
    int64_t causalOrNextFactor = deltaCausalOrNext - (int64_t)constInfo.s2BaseSize;
    if (m.compressMode == (uint8_t)AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE ||
        m.compressMode == (uint8_t)AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE) {
        if (causalOrNextFactor >= 0) {
            m.computeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
        } else {
            m.computeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
        }
        return;
    }
    if (m.compressMode == (uint8_t)AttenMaskCompressMode::BAND_MODE) {
        int64_t preFactor = deltaPre + 1 + (int64_t)constInfo.s1BaseSize;
        if (causalOrNextFactor >= 0 && preFactor <= 0) {
            m.computeMode = AttenMaskComputeMode::NO_NEED_COMPUTE_MODE;
        } else if (causalOrNextFactor < 0 && preFactor <= 0) {
            m.computeMode = AttenMaskComputeMode::CAUSAL_OR_NEXT_ONLY_MODE;
        } else if (causalOrNextFactor >= 0 && preFactor > 0) {
            m.computeMode = AttenMaskComputeMode::PRE_ONLY_MODE;
        } else {
            m.computeMode = AttenMaskComputeMode::PRE_AND_NEXT_MODE;
        }
    }
}

// ---- VERBATIM algorithm from arch35 attenmask.h:343-517 (compressMode dispatch) -
// Covers NO_COMPRESS (dense) + LEFT_UP_CAUSAL + RIGHT_DOWN_CAUSAL + BAND. The MLA/rope/
// prefix-N/infer branches of the arch35 original are dropped (train-only library path);
// PREFIX_MODE offset uses the rectangle helper. preTokens/nextTokens drive band.
__aicore__ inline int64_t ComputeAttenMaskInnerOffset(const FaRunCtx &runInfo,
                                                      const FaConstCtx &constInfo,
                                                      AttenMaskInfo &m)
{
    if (m.compressMode == (uint8_t)AttenMaskCompressMode::NO_COMPRESS_MODE) {
        return ComputeOffsetForNoCompress(runInfo, constInfo, m);
    }

    int64_t deltaCausalOrNext = 0;
    int64_t deltaPre = 0;
    int64_t deltaN = runInfo.actualS1Size - runInfo.actualS2Size;
    int64_t s1Offset = runInfo.s1oIdx * (int64_t)constInfo.s1BaseSize;
    int64_t s2Offset = runInfo.s2StartIdx + runInfo.s2LoopCount * (int64_t)constInfo.s2BaseSize;

    if (m.compressMode == (uint8_t)AttenMaskCompressMode::LEFT_UP_CAUSAL_MODE) {
        deltaCausalOrNext = s1Offset - s2Offset;
    } else if (m.compressMode == (uint8_t)AttenMaskCompressMode::RIGHT_DOWN_CAUSAL_MODE) {
        deltaCausalOrNext = s1Offset - s2Offset - deltaN;
    } else if (m.compressMode == (uint8_t)AttenMaskCompressMode::BAND_MODE) {
        deltaPre = s1Offset - s2Offset - m.preTokens - 1;
        deltaCausalOrNext = s1Offset - s2Offset + m.nextTokens;
        m.attenMaskOffsetPre = ComputeOffsetForCausal(deltaPre, constInfo.s1BaseSize,
            constInfo.s2BaseSize, m.attenMaskS2Size, runInfo.vecCoreOffset);
    } else if (m.compressMode == (uint8_t)AttenMaskCompressMode::PREFIX_MODE) {
        deltaCausalOrNext = s1Offset - s2Offset - deltaN;
    } else {
        return 0;
    }

    GetAttenMaskComputeMode(deltaCausalOrNext, deltaPre, runInfo, constInfo, m);
    return ComputeOffsetForCausal(deltaCausalOrNext, constInfo.s1BaseSize,
        constInfo.s2BaseSize, m.attenMaskS2Size, runInfo.vecCoreOffset);
}

// ---- VERBATIM from arch35 attenmask.h:519-528 ----------------------------------
__aicore__ inline int64_t ComputeAttenMaskOffset(const FaRunCtx &runInfo,
                                                 const FaConstCtx &constInfo,
                                                 AttenMaskInfo &m)
{
    return ComputeAttenMaskInnerOffset(runInfo, constInfo, m);
}

}  // namespace fa_atten_mask

#endif  // ATTENMASK_ARCH35_H
