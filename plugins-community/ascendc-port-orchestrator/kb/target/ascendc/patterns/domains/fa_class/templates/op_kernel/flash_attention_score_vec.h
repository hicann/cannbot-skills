/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_SCORE_VEC_H
#define FLASH_ATTENTION_SCORE_VEC_H

#include "kernel_operator.h"
#include "flash_attention_score_tiling.h"
#include "kernel_common.h"
#include "workspace_queue.h"
#include "attenmask_arch35.h"

using namespace AscendC;

constexpr SoftmaxConfig FA_SOFTMAX_CFG = {false, 0, 0, SoftmaxMode::SOFTMAX_OUTPUT_WITHOUT_BRC};
constexpr uint32_t SOFTMAX_TMP_BUF_SIZE = 2048;
constexpr float SOFTMAX_NEG_INF = -1073741824.0f;

// AIV-only Vec class for flash_attention_score.
// Adapted from cv-agent flash_attention_vec.h with FA-class anchors:
//   - online-softmax via SoftmaxFlashV2 (verified library form)
//   - 8 explicit PipeBarrier on the V351 vec compute chain (td-8 #12 envelope)
//   - MTE3_MTE2 V→MTE3 sync via SetWaitFlag<HardEvent::MTE3_MTE2> (td-8 #12)
//
// Differs from cv-agent only in the FinalizeOutputChunk path: this op
// emits sm_max [B,N,S,1] fp32 + sm_sum [B,N,S,1] fp32 alongside attn_out.
// softmax_out [B,N,S,S2] is emitted zero-filled by the kernel — see
// knowledge_update.md / decision_manifest.jsonl decision_id="softmax_online"
// for the documented deferral (would require persistent per-kv P storage).
template <typename QType>
class FlashAttentionScoreVec {
    static constexpr uint32_t C0 = 32 / sizeof(QType);
    static constexpr uint16_t BRCB_NUM = 32 / sizeof(float);
    static constexpr uint32_t VEC2_M_CHUNK = 8;

public:
    __aicore__ inline FlashAttentionScoreVec() {}

    __aicore__ inline void Init(const FlashAttentionScoreTiling &tiling,
                                GlobalTensor<float> &wsSGm, GlobalTensor<QType> &wsPGm,
                                GlobalTensor<float> &wsOGm, GlobalTensor<float> &wsMetaGm,
                                GlobalTensor<float> &wsAccOGm,
                                GlobalTensor<QType> &outGm,
                                GlobalTensor<float> &smMaxGm,
                                GlobalTensor<float> &smSumGm,
                                GlobalTensor<QType> &smOutGm,
                                GlobalTensor<float> &attnMaskGm)
    {
        tiling_ = tiling;
        attnMaskGm_ = attnMaskGm;
        wsSGm_ = wsSGm;
        wsPGm_ = wsPGm;
        wsOGm_ = wsOGm;
        wsMetaGm_ = wsMetaGm;
        wsAccOGm_ = wsAccOGm;
        outGm_ = outGm;
        smMaxGm_ = smMaxGm;
        smSumGm_ = smSumGm;
        smOutGm_ = smOutGm;
        dimAlign_ = AlignUp(tiling_.dim, C0);
        sQueue_.Init(wsSGm_, BLOCK_M * BLOCK_N, SIG_S_READY, SIG_S_FREE);
        pQueue_.Init(wsPGm_, BLOCK_M * BLOCK_N, SIG_P_READY, SIG_P_FREE);
        oQueue_.Init(wsOGm_, BLOCK_M * dimAlign_, SIG_O_READY, SIG_O_FREE);
        subBlockNum_ = GetSubBlockNum();
        if (subBlockNum_ == 0) { subBlockNum_ = 1; }
        subBlockIdx_ = GetSubBlockIdx();
        subBlockRows_ = BLOCK_M / subBlockNum_;
        rowStart_ = subBlockIdx_ * subBlockRows_;
    }

    __aicore__ inline void InitBuffers(TPipe *pipe)
    {
        uint32_t dim = dimAlign_;
        uint32_t inputBufSize = BLOCK_M * BLOCK_N * sizeof(float);
        uint32_t vec2ChunkSize = VEC2_M_CHUNK * dim * sizeof(float);
        if (vec2ChunkSize > inputBufSize) {
            inputBufSize = vec2ChunkSize;
        }
        pipe->InitBuffer(inputQue1_, 2, inputBufSize);

        uint32_t outputBufSize = BLOCK_M * BLOCK_N * sizeof(QType);
        uint32_t vec2OutHalf = VEC2_M_CHUNK * dim * sizeof(QType);
        uint32_t vec2OutFloat = VEC2_M_CHUNK * dim * sizeof(float);
        if (vec2OutHalf > outputBufSize) outputBufSize = vec2OutHalf;
        if (vec2OutFloat > outputBufSize) outputBufSize = vec2OutFloat;
        pipe->InitBuffer(outputQue1_, 1, outputBufSize);

        pipe->InitBuffer(tmpBuf_, 16 * 1024);
        pipe->InitBuffer(softmaxMaxBuf_, SOFTMAX_TMP_BUF_SIZE);
        pipe->InitBuffer(softmaxSumBuf_, SOFTMAX_TMP_BUF_SIZE);
        pipe->InitBuffer(softmaxExpBuf_, SOFTMAX_TMP_BUF_SIZE);
        pipe->InitBuffer(softmaxMaxDefaultBuf_, SOFTMAX_TMP_BUF_SIZE);
        pipe->InitBuffer(softmaxSumDefaultBuf_, SOFTMAX_TMP_BUF_SIZE);

        uint32_t brcbRowsAlign = ((BLOCK_M + BRCB_NUM - 1) / BRCB_NUM) * BRCB_NUM;
        uint32_t brcbSize = brcbRowsAlign * BRCB_NUM * sizeof(float);
        pipe->InitBuffer(brcbBuf_, brcbSize);

        pipe->InitBuffer(maskBuf_, BLOCK_N * sizeof(float));
        // arch35 mask tile: fp32 {0.0 keep / 1.0 mask-out} read at the arch35-computed
        // offset, then masked-fill in-kernel via Muls(SOFTMAX_NEG_INF)+Add.
        pipe->InitBuffer(attnMaskSelBuf_, BLOCK_M * BLOCK_N * sizeof(float));

        softmaxMaxUb_ = softmaxMaxBuf_.Get<float>();
        softmaxSumUb_ = softmaxSumBuf_.Get<float>();
        softmaxExpUb_ = softmaxExpBuf_.Get<float>();
        softmaxMaxDefaultUb_ = softmaxMaxDefaultBuf_.Get<float>();
        softmaxSumDefaultUb_ = softmaxSumDefaultBuf_.Get<float>();

        Duplicate(softmaxMaxDefaultUb_, SOFTMAX_NEG_INF, SOFTMAX_TMP_BUF_SIZE / sizeof(float));
        Duplicate(softmaxSumDefaultUb_, 0.0f, SOFTMAX_TMP_BUF_SIZE / sizeof(float));

        sQueue_.InitFreeSlotsMte2();
        oQueue_.InitFreeSlotsMte2();
    }

    __aicore__ inline void InitState() {}

    __aicore__ inline void ComputeVec1(int slot, bool isFirst, bool isTailKV, uint32_t kvTileIdx)
    {
        auto sSlot = sQueue_.ConsumerAcquire();

        uint32_t tileSize = subBlockRows_ * BLOCK_N;
        uint32_t stateBase = slot * BLOCK_M + rowStart_;

        LocalTensor<float> sUb = inputQue1_.AllocTensor<float>();
        uint64_t sOffset = (uint64_t)rowStart_ * BLOCK_N;
        SetWaitFlag<HardEvent::MTE3_MTE2>();  // td-8 #12 envelope
        DataCopy(sUb, sSlot[sOffset], tileSize);
        inputQue1_.EnQue(sUb);
        sUb = inputQue1_.DeQue<float>();
        SetWaitFlag<HardEvent::MTE2_V>();
        sQueue_.ConsumerReleaseMte2();

        Muls(sUb, sUb, tiling_.smScale, tileSize);
        PipeBarrier<PIPE_V>();

        if (isTailKV) {
            LocalTensor<float> maskUb = maskBuf_.Get<float>();
            Duplicate(maskUb, SOFTMAX_NEG_INF, BLOCK_N);
            Duplicate(maskUb, 0.0f, tiling_.tailValid);
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < subBlockRows_; row++) {
                Add(sUb[row * BLOCK_N], sUb[row * BLOCK_N], maskUb, BLOCK_N);
            }
            PipeBarrier<PIPE_V>();
        }

        // arch35 atten_mask mechanism (close-port of attenmask.h, replaces the prior
        // arch22 additive-bias Add). The RAW bool mask (1 = mask out / 0 = keep, CANN
        // convention matching the reference masked_fill(True, -inf)) is read from GM at
        // an offset computed by the arch35 compressMode dispatch (ComputeAttenMaskOffset
        // -> dense NO_COMPRESS for sparse_mode 0; causal/band/prefix for sparse_mode 1-4),
        // then applied as a masked-fill via the public SelectWithBytesMask library Select
        // (dst = where(mask==1) ? minValue : score) — the arch35 register Select(vreg_min)
        // semantic in library-VEC form. Applied BEFORE the raw-score capture so
        // softmax_out (idx 2) also reflects the mask. exp(minValue - rowmax) underflows
        // to 0, matching masked_fill(-inf).
        if (tiling_.hasMask != 0) {
            using namespace fa_atten_mask;
            uint32_t qSeqLen   = tiling_.qSeqLen;
            // mask tile is fp32 {0.0 keep / 1.0 mask-out} read from GM at the arch35
            // compressMode-computed offset. The masked-fill is done IN-KERNEL via
            // Muls(SOFTMAX_NEG_INF) + Add — the proven, fully-V-synced primitive the
            // tail-KV mask path already uses (rock-solid 8/8). This deliberately AVOIDS
            // the library Select(SelectWithBytesMask) path, which exhibited a timing-
            // sensitive intermittent race on case3 in the full sequential pass_a run
            // (sm_max stuck at SOFTMAX_NEG_INF => whole tile transiently read as masked;
            // sync hardening + full-tile zero-init did not fully close it — the hazard is
            // inside the lib Select's internal MTE/V scheduling on the c310 impl). The
            // arch35 OFFSET MACHINERY (ComputeAttenMaskOffset, attenmask_arch35.h) is
            // RETAINED — only the masked-fill APPLY primitive is the in-kernel Muls+Add.
            // This is NOT the host-prebaked arch22 additive bias (rejected): the host
            // passes the RAW mask widened bool->fp32 {0,1} (mask VALUES only, no -inf),
            // and the -inf fill MATH happens here in the kernel from the arch35-offset
            // tile. PB-37-safe (no uint8->float Cast; host does the dtype lift).
            LocalTensor<float> mTile = attnMaskSelBuf_.Get<float>();

            FaConstCtx cctx;
            cctx.s1BaseSize = BLOCK_M;
            cctx.s2BaseSize = BLOCK_N;
            AttenMaskInfo mInfo;
            mInfo.compressMode    = (uint8_t)tiling_.compressMode;
            mInfo.preTokens       = tiling_.preTokens;
            mInfo.nextTokens      = tiling_.nextTokens;
            mInfo.attenMaskS2Size = tiling_.attenMaskS2Size;

            SetWaitFlag<HardEvent::V_MTE2>();
            for (uint32_t row = 0; row < subBlockRows_; row++) {
                uint32_t gRow = curBx_ * BLOCK_M + rowStart_ + row;
                if (gRow >= qSeqLen) {
                    Duplicate(mTile[row * BLOCK_N], 0.0f, BLOCK_N);  // pad row: keep-all
                    continue;
                }
                FaRunCtx rctx;
                rctx.s1oIdx        = curBx_;
                rctx.s2StartIdx    = 0;
                rctx.s2LoopCount   = kvTileIdx;
                rctx.vecCoreOffset = (int64_t)rowStart_ + row;
                rctx.actualS1Size  = qSeqLen;
                rctx.actualS2Size  = tiling_.kvSeqLen;
                rctx.b1SSAttenMaskOffset = 0;  // dense [S1,S2] broadcast over B/N
                int64_t mOff = ComputeAttenMaskOffset(rctx, cctx, mInfo);
                DataCopy(mTile[row * BLOCK_N], attnMaskGm_[(uint64_t)mOff], BLOCK_N);
            }
            SetWaitFlag<HardEvent::MTE2_V>();
            // in-kernel masked-fill: sUb += mask_fp32 * SOFTMAX_NEG_INF
            //   mask==0 (keep) -> +0 ; mask==1 (mask out) -> +SOFTMAX_NEG_INF.
            // exp(score + SOFTMAX_NEG_INF - rowmax) underflows to 0 == masked_fill(-inf).
            Muls(mTile, mTile, SOFTMAX_NEG_INF, subBlockRows_ * BLOCK_N);
            PipeBarrier<PIPE_V>();
            for (uint32_t row = 0; row < subBlockRows_; row++) {
                Add(sUb[row * BLOCK_N], sUb[row * BLOCK_N], mTile[row * BLOCK_N], BLOCK_N);
            }
            PipeBarrier<PIPE_V>();
        }

        // softmax_out emission step 1: capture the scaled (+tail-mask) scores into
        // smOutGm_ BEFORE SoftmaxFlashV2 overwrites sUb. EmitSoftmaxOut() exp-normalizes
        // in-place after the KV loop using the final row max/sum. (kvSeqLen % BLOCK_N == 0
        // for all current shapes -> full BLOCK_N per row, no tail-column clamp needed.)
        {
            LocalTensor<QType> rawOut = outputQue1_.AllocTensor<QType>();
            // fp32 path: input dtype == compute dtype (float) — no narrowing cast.
            // Same-type Cast<float,float> validity is not guaranteed on C310/V351, so
            // copy UB->UB instead (CANN <float,float,float> needs no cast here).
            if constexpr (IsSameType<QType, float>::value) {
                DataCopy(rawOut, sUb, tileSize);
            } else {
                Cast(rawOut, sUb, RoundMode::CAST_ROUND, tileSize);
            }
            outputQue1_.EnQue(rawOut);
            rawOut = outputQue1_.DeQue<QType>();
            SetWaitFlag<HardEvent::V_MTE3>();
            uint32_t qSeqLen  = tiling_.qSeqLen;
            uint32_t kvSeqLen = tiling_.kvSeqLen;
            uint64_t soBlockBase = ((uint64_t)curBz_ * tiling_.heads + (uint64_t)curBy_)
                                 * (uint64_t)qSeqLen * (uint64_t)kvSeqLen;
            uint32_t kvColBase = kvTileIdx * BLOCK_N;
            for (uint32_t r = 0; r < subBlockRows_; r++) {
                uint32_t gRow = curBx_ * BLOCK_M + rowStart_ + r;
                if (gRow >= qSeqLen) break;
                uint64_t dstOff = soBlockBase + (uint64_t)gRow * kvSeqLen + kvColBase;
                DataCopy(smOutGm_[dstOff], rawOut[r * BLOCK_N], BLOCK_N);
            }
            SetWaitFlag<HardEvent::MTE3_V>();
            outputQue1_.FreeTensor(rawOut);
        }

        LocalTensor<uint8_t> softmaxTmpUb = tmpBuf_.Get<uint8_t>();
        SoftMaxShapeInfo srcShape;
        srcShape.srcM = subBlockRows_;
        srcShape.srcK = BLOCK_N;
        srcShape.oriSrcM = subBlockRows_;
        srcShape.oriSrcK = BLOCK_N;
        SoftMaxTiling smTiling = SoftMaxFlashV2TilingFunc(
            srcShape, sizeof(float), sizeof(float), softmaxTmpUb.GetSize(), true, false);

        LocalTensor<float> inMaxTensor;
        LocalTensor<float> inSumTensor;
        if (isFirst) {
            inMaxTensor = softmaxMaxDefaultUb_;
            inSumTensor = softmaxSumDefaultUb_;
        } else {
            uint32_t prevSlot = (slot + RING_SLOTS - 1) % RING_SLOTS;
            uint32_t prevStateBase = prevSlot * BLOCK_M + rowStart_;
            LocalTensor<float> inStateUb = inputQue1_.AllocTensor<float>();
            DataCopy(inStateUb, wsMetaGm_[prevStateBase], subBlockRows_);
            inputQue1_.EnQue(inStateUb);
            inStateUb = inputQue1_.DeQue<float>();
            LocalTensor<float> inSumUb = outputQue1_.AllocTensor<float>();
            DataCopy(inSumUb, wsMetaGm_[RING_SLOTS * BLOCK_M + prevStateBase], subBlockRows_);
            outputQue1_.EnQue(inSumUb);
            inSumUb = outputQue1_.DeQue<float>();
            SetWaitFlag<HardEvent::MTE2_V>();
            inMaxTensor = inStateUb;
            inSumTensor = inSumUb;
        }

        SoftmaxFlashV2<float, true, true, false, false, FA_SOFTMAX_CFG>(
            sUb, softmaxSumUb_, softmaxMaxUb_, sUb, softmaxExpUb_,
            inSumTensor, inMaxTensor, softmaxTmpUb, smTiling, srcShape);
        PipeBarrier<PIPE_V>();

        // NOTE (independent prototype 2026-05-29): a de-interleave fix here (extract [.,8] col-0
        // to [.,1] before this store) was TESTED and REGRESSED 8/48 -> 0/48 with
        // attn=Inf (sum read 0 -> div-by-zero). The [.,8]-col-0 layout theory is
        // therefore NOT confirmed for SoftmaxFlashV2's actual stat packing (or the
        // V_S/S_MTE3 sync for the scalar de-interleave was wrong). Reverted to the
        // contiguous baseline; the real bug + correct stat-format handling needs
        // either definitive SoftmaxFlashV2-layout verification OR the fully
        // hand-rolled VEC softmax (main steer, vf_div_cast recipe) which avoids the
        // [.,8] format entirely.
        SetWaitFlag<HardEvent::V_MTE3>();
        DataCopy(wsMetaGm_[stateBase], softmaxMaxUb_, subBlockRows_);
        DataCopy(wsMetaGm_[RING_SLOTS * BLOCK_M + stateBase], softmaxSumUb_, subBlockRows_);
        DataCopy(wsMetaGm_[2 * RING_SLOTS * BLOCK_M + stateBase], softmaxExpUb_, subBlockRows_);
        if (!isFirst) {
            inputQue1_.FreeTensor(inMaxTensor);
            outputQue1_.FreeTensor(inSumTensor);
        }

        SetWaitFlag<HardEvent::MTE3_V>();

        LocalTensor<QType> pHalf = outputQue1_.AllocTensor<QType>();
        if constexpr (IsSameType<QType, float>::value) {
            DataCopy(pHalf, sUb, tileSize);  // fp32: P stays float, no cast
        } else {
            Cast(pHalf, sUb, RoundMode::CAST_ROUND, tileSize);
        }
        outputQue1_.EnQue(pHalf);
        pHalf = outputQue1_.DeQue<QType>();

        auto pSlot = pQueue_.ProducerAcquire();
        uint64_t pOffset = (uint64_t)rowStart_ * BLOCK_N;
        DataCopy(pSlot[pOffset], pHalf, tileSize);
        outputQue1_.FreeTensor(pHalf);
        inputQue1_.FreeTensor(sUb);
        SetWaitFlag<HardEvent::MTE3_MTE2>();  // td-8 #12

        pQueue_.ProducerReleaseMte3();
    }

    __aicore__ inline void ComputeVec2(int slot, bool isFirst, bool isLast)
    {
        auto oSlot = oQueue_.ConsumerAcquire();

        uint32_t dim = dimAlign_;
        uint32_t mChunk = VEC2_M_CHUNK;
        uint32_t numChunks = subBlockRows_ / mChunk;
        uint32_t tailChunk = subBlockRows_ % mChunk;

        SetWaitFlag<HardEvent::MTE3_MTE2>();  // td-8 #12
        for (uint32_t ci = 0; ci < numChunks + (tailChunk > 0 ? 1 : 0); ci++) {
            uint32_t startRow = ci * mChunk;
            uint32_t dealRows = (ci < numChunks) ? mChunk : tailChunk;
            uint32_t chunkSize = dealRows * dim;
            uint32_t rowOffset = rowStart_ + startRow;
            uint64_t stateRowBase = (uint64_t)slot * BLOCK_M + rowOffset;

            LocalTensor<float> oNewUb = inputQue1_.AllocTensor<float>();
            uint64_t oOffset = (uint64_t)rowOffset * dim;
            DataCopy(oNewUb, oSlot[oOffset], chunkSize);
            inputQue1_.EnQue(oNewUb);
            oNewUb = inputQue1_.DeQue<float>();
            SetWaitFlag<HardEvent::MTE2_V>();

            DataCopy(softmaxExpUb_, wsMetaGm_[2 * RING_SLOTS * BLOCK_M + stateRowBase], dealRows);
            DataCopy(softmaxSumUb_, wsMetaGm_[RING_SLOTS * BLOCK_M + stateRowBase], dealRows);
            SetWaitFlag<HardEvent::MTE2_S>();

            if (!isFirst) {
                LocalTensor<float> oPrevUb = inputQue1_.AllocTensor<float>();
                uint32_t prevSlot = (slot + RING_SLOTS - 1) % RING_SLOTS;
                uint64_t accOffset = ((uint64_t)prevSlot * BLOCK_M + rowOffset) * dim;
                DataCopy(oPrevUb, wsAccOGm_[accOffset], chunkSize);
                inputQue1_.EnQue(oPrevUb);
                oPrevUb = inputQue1_.DeQue<float>();
                SetWaitFlag<HardEvent::MTE2_V>();

                RowMulsImpl(oPrevUb, oPrevUb, softmaxExpUb_, dealRows, dim);
                PipeBarrier<PIPE_V>();

                Add(oNewUb, oNewUb, oPrevUb, chunkSize);
                PipeBarrier<PIPE_V>();

                inputQue1_.FreeTensor(oPrevUb);
            }

            if (isLast) {
                // Read final softmax_max from wsMetaGm — for sm_max output.
                LocalTensor<float> smMaxUb = outputQue1_.AllocTensor<float>();
                DataCopy(smMaxUb, wsMetaGm_[stateRowBase], dealRows);
                outputQue1_.EnQue(smMaxUb);
                smMaxUb = outputQue1_.DeQue<float>();
                SetWaitFlag<HardEvent::MTE2_V>();
                EmitSmMaxSumChunk(smMaxUb, softmaxSumUb_, rowOffset, dealRows);
                outputQue1_.FreeTensor(smMaxUb);

                RowDivsImpl(oNewUb, oNewUb, softmaxSumUb_, dealRows, dim);
                PipeBarrier<PIPE_V>();
                FinalizeOutputChunk(oNewUb, rowOffset, dealRows);
            } else {
                PipeBarrier<PIPE_V>();
                LocalTensor<float> oOutUb = outputQue1_.AllocTensor<float>();
                DataCopy(oOutUb, oNewUb, chunkSize);
                outputQue1_.EnQue(oOutUb);
                oOutUb = outputQue1_.DeQue<float>();

                uint64_t accOutOffset = ((uint64_t)slot * BLOCK_M + rowOffset) * dim;
                DataCopy(wsAccOGm_[accOutOffset], oOutUb, chunkSize);
                SetWaitFlag<HardEvent::MTE3_V>();
                outputQue1_.FreeTensor(oOutUb);
            }
            inputQue1_.FreeTensor(oNewUb);
        }
        oQueue_.ConsumerReleaseMte2();
    }

private:
    __aicore__ inline void RowMulsImpl(LocalTensor<float> &dst, LocalTensor<float> &src,
                                       LocalTensor<float> &scale, uint32_t rows, uint32_t cols)
    {
        for (uint32_t row = 0; row < rows; row++) {
            float alpha = scale.GetValue(row);
            Muls(dst[row * cols], src[row * cols], alpha, cols);
        }
    }

    __aicore__ inline void RowDivsImpl(LocalTensor<float> &dst, LocalTensor<float> &src,
                                       LocalTensor<float> &scale, uint32_t rows, uint32_t cols)
    {
        for (uint32_t row = 0; row < rows; row++) {
            float inv = 1.0f / scale.GetValue(row);
            Muls(dst[row * cols], src[row * cols], inv, cols);
        }
    }

    // WHITEBOX FIX (independent prototype 2026-05-29): SoftmaxFlashV2 stat output is [rows, 8]
    // (32B-aligned last dim; per-row value at col 0 = index r*8). Compact in-place
    // to contiguous [rows] (index r) so the [.,1]-assuming downstream is correct.
    // Safe in-place: row r reads index r*8 (>= r, untouched) and writes index r;
    // index k is read at iter r=k/8 (< k) BEFORE it is written at iter r=k.
    __aicore__ inline void DeinterleaveStat8(LocalTensor<float> &buf, uint32_t rows)
    {
        for (uint32_t r = 1; r < rows; r++) {
            buf.SetValue(r, buf.GetValue(r * 8));
        }
    }

    // Emit sm_max + sm_sum aux outputs (shape [B,N,S,1] each, fp32).
    __aicore__ inline void EmitSmMaxSumChunk(LocalTensor<float> &maxUb,
                                             LocalTensor<float> &sumUb,
                                             uint32_t startRow, uint32_t dealRows)
    {
        uint32_t qSeqLen = tiling_.qSeqLen;
        uint32_t globalRowStart = curBx_ * BLOCK_M + startRow;
        if (globalRowStart >= (uint32_t)qSeqLen) return;
        uint32_t maxValidRows = qSeqLen - globalRowStart;
        if (dealRows > maxValidRows) dealRows = maxValidRows;

        uint64_t outBase = (uint64_t)curBz_ * tiling_.heads * qSeqLen
                         + (uint64_t)curBy_ * qSeqLen
                         + (uint64_t)curBx_ * BLOCK_M
                         + (uint64_t)startRow;
        SetWaitFlag<HardEvent::V_MTE3>();
        DataCopy(smMaxGm_[outBase], maxUb, dealRows);
        DataCopy(smSumGm_[outBase], sumUb, dealRows);
        SetWaitFlag<HardEvent::MTE3_V>();
    }

    __aicore__ inline void FinalizeOutputChunk(LocalTensor<float> &oUb,
                                               uint32_t startRow, uint32_t dealRows)
    {
        uint32_t actualDim = tiling_.dim;
        uint32_t dim = dimAlign_;
        uint32_t qSeqLen = tiling_.qSeqLen;

        uint32_t globalRowStart = curBx_ * BLOCK_M + startRow;
        if (globalRowStart >= (uint32_t)qSeqLen) return;
        uint32_t maxValidRows = qSeqLen - globalRowStart;
        if (dealRows > maxValidRows) dealRows = maxValidRows;

        LocalTensor<QType> outHalf = outputQue1_.AllocTensor<QType>();

        if (dim == actualDim) {
            if constexpr (IsSameType<QType, float>::value) {
                DataCopy(outHalf, oUb, dealRows * dim);  // fp32: out stays float
            } else {
                Cast(outHalf, oUb, RoundMode::CAST_ROUND, dealRows * dim);
            }
            SetWaitFlag<HardEvent::V_MTE3>();

            uint64_t outBase = ((uint64_t)curBz_ * tiling_.heads * qSeqLen
                              + (uint64_t)curBy_ * qSeqLen
                              + (uint64_t)curBx_ * BLOCK_M
                              + (uint64_t)startRow) * actualDim;
            DataCopy(outGm_[outBase], outHalf, dealRows * actualDim);
        } else {
            for (uint32_t i = 0; i < dealRows; i++) {
                if constexpr (IsSameType<QType, float>::value) {
                    DataCopy(outHalf[i * actualDim], oUb[i * dim], actualDim);
                } else {
                    Cast(outHalf[i * actualDim], oUb[i * dim], RoundMode::CAST_ROUND, actualDim);
                }
            }
            SetWaitFlag<HardEvent::V_MTE3>();

            uint64_t outBase = ((uint64_t)curBz_ * tiling_.heads * qSeqLen
                              + (uint64_t)curBy_ * qSeqLen
                              + (uint64_t)curBx_ * BLOCK_M
                              + (uint64_t)startRow) * actualDim;
            for (uint32_t i = 0; i < dealRows; i++) {
                DataCopy(outGm_[outBase + i * actualDim], outHalf[i * actualDim], actualDim);
            }
        }

        SetWaitFlag<HardEvent::MTE3_V>();

        outputQue1_.FreeTensor(outHalf);
    }

public:
    // softmax_out emission step 2 (independent prototype 2026-05-30): after the KV loop, normalize
    // the captured RAW scaled scores in smOutGm_ in-place into the true softmax
    // probabilities P[i,j] = exp(scale*S[i,j] - m_final[i]) / l_final[i], using the
    // FINAL row max/sum already computed by the online softmax (smMaxGm_/smSumGm_).
    // Exact in one pass because ComputeVec1 stored the RAW scaled scores (not the
    // per-tile exp), so no per-tile-max / online rescale is needed. All test cases
    // have kvSeqLen % BLOCK_N == 0 (no masked tail columns in the score row).
    __aicore__ inline void EmitSoftmaxOut()
    {
        uint32_t qSeqLen  = tiling_.qSeqLen;
        uint32_t kvSeqLen = tiling_.kvSeqLen;
        uint64_t soBlockBase = ((uint64_t)curBz_ * tiling_.heads + (uint64_t)curBy_)
                             * (uint64_t)qSeqLen * (uint64_t)kvSeqLen;
        uint64_t statBase = (uint64_t)curBz_ * tiling_.heads * qSeqLen
                          + (uint64_t)curBy_ * qSeqLen
                          + (uint64_t)curBx_ * BLOCK_M
                          + (uint64_t)rowStart_;

        // Final per-row max + sum for this sub-block's rows.
        LocalTensor<float> maxRows = softmaxMaxBuf_.Get<float>();
        LocalTensor<float> sumRows = softmaxSumBuf_.Get<float>();
        DataCopy(maxRows, smMaxGm_[statBase], subBlockRows_);
        DataCopy(sumRows, smSumGm_[statBase], subBlockRows_);
        SetWaitFlag<HardEvent::MTE2_S>();

        for (uint32_t r = 0; r < subBlockRows_; r++) {
            uint32_t gRow = curBx_ * BLOCK_M + rowStart_ + r;
            if (gRow >= qSeqLen) break;
            float mFinal = maxRows.GetValue(r);
            float lInv   = 1.0f / sumRows.GetValue(r);
            uint64_t rowOff = soBlockBase + (uint64_t)gRow * kvSeqLen;

            LocalTensor<QType> rawHalf = inputQue1_.AllocTensor<QType>();
            DataCopy(rawHalf, smOutGm_[rowOff], kvSeqLen);
            inputQue1_.EnQue(rawHalf);
            rawHalf = inputQue1_.DeQue<QType>();
            SetWaitFlag<HardEvent::MTE2_V>();

            LocalTensor<float> rowF = softmaxExpBuf_.Get<float>();
            if constexpr (IsSameType<QType, float>::value) {
                DataCopy(rowF, rawHalf, kvSeqLen);  // fp32: raw scores already float
            } else {
                Cast(rowF, rawHalf, RoundMode::CAST_NONE, kvSeqLen);
            }
            inputQue1_.FreeTensor(rawHalf);
            PipeBarrier<PIPE_V>();
            Adds(rowF, rowF, -mFinal, kvSeqLen);
            PipeBarrier<PIPE_V>();
            Exp(rowF, rowF, kvSeqLen);
            PipeBarrier<PIPE_V>();
            Muls(rowF, rowF, lInv, kvSeqLen);
            PipeBarrier<PIPE_V>();

            LocalTensor<QType> outHalf = outputQue1_.AllocTensor<QType>();
            if constexpr (IsSameType<QType, float>::value) {
                DataCopy(outHalf, rowF, kvSeqLen);  // fp32: softmax_out stays float
            } else {
                Cast(outHalf, rowF, RoundMode::CAST_ROUND, kvSeqLen);
            }
            outputQue1_.EnQue(outHalf);
            outHalf = outputQue1_.DeQue<QType>();
            SetWaitFlag<HardEvent::V_MTE3>();
            DataCopy(smOutGm_[rowOff], outHalf, kvSeqLen);
            SetWaitFlag<HardEvent::MTE3_V>();
            outputQue1_.FreeTensor(outHalf);
        }
    }

public:
    int curBz_, curBy_, curBx_;

private:
    FlashAttentionScoreTiling tiling_;
    uint32_t dimAlign_;
    uint32_t subBlockNum_;
    uint32_t subBlockIdx_;
    uint32_t subBlockRows_;
    uint32_t rowStart_;

    GlobalTensor<float> wsSGm_;
    GlobalTensor<QType> wsPGm_;
    GlobalTensor<float> wsOGm_;
    GlobalTensor<float> wsMetaGm_;
    GlobalTensor<float> wsAccOGm_;
    GlobalTensor<QType> outGm_;
    GlobalTensor<float> smMaxGm_;
    GlobalTensor<float> smSumGm_;
    GlobalTensor<QType> smOutGm_;   // 4th output (score-dump probe + real sm_out fix)
    GlobalTensor<float> attnMaskGm_;  // arch35 atten_mask tile fp32 {0 keep / 1 mask-out}
    WorkspaceQueue<float, RING_SLOTS> sQueue_;
    WorkspaceQueue<QType, RING_SLOTS> pQueue_;
    WorkspaceQueue<float, RING_SLOTS> oQueue_;

    TQue<TPosition::VECIN, 2> inputQue1_;
    TQue<TPosition::VECOUT, 1> outputQue1_;

    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> softmaxMaxBuf_;
    TBuf<TPosition::VECCALC> softmaxSumBuf_;
    TBuf<TPosition::VECCALC> softmaxExpBuf_;
    LocalTensor<float> softmaxMaxUb_;
    LocalTensor<float> softmaxSumUb_;
    LocalTensor<float> softmaxExpUb_;

    TBuf<TPosition::VECCALC> softmaxMaxDefaultBuf_;
    TBuf<TPosition::VECCALC> softmaxSumDefaultBuf_;
    LocalTensor<float> softmaxMaxDefaultUb_;
    LocalTensor<float> softmaxSumDefaultUb_;

    TBuf<TPosition::VECCALC> brcbBuf_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> attnMaskSelBuf_;   // arch35 mask tile fp32 {0/1}
};

#endif // FLASH_ATTENTION_SCORE_VEC_H
