/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_SCORE_KERNEL_H
#define FLASH_ATTENTION_SCORE_KERNEL_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "flash_attention_score_tiling.h"
#include "kernel_common.h"
#include "flash_attention_score_cube.h"
#include "flash_attention_score_vec.h"

using namespace AscendC;

// Top-level dispatcher — adapted from cv-agent flash_attention_kernel.h.
//   - MIX_AIC_1_2 sub-block normalization (cv_reference_concrete_params.md
//     §kernel_block_iteration): coreIdx = GetBlockIdx() / GetSubBlockNum()
//   - Single wsMetaGm_ workspace region (3 sub-regions: max/sum/exp), as the
//     FA-class brief mandates (no separate softmaxMax/softmaxSum GMs).
template <typename QType>
class FlashAttentionScoreKernel {
    static constexpr uint32_t C0 = 32 / sizeof(QType);

public:
    __aicore__ inline FlashAttentionScoreKernel() {}

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                GM_ADDR attnOut, GM_ADDR smMax, GM_ADDR smSum,
                                GM_ADDR smOut, GM_ADDR attenMask, GM_ADDR workspace,
                                GM_ADDR tilingGM, TPipe *pipe)
    {
        pipe_ = pipe;
        CopyTiling(&tiling_, tilingGM);

        uint32_t dim = tiling_.dim;
        uint32_t qSeqLen = tiling_.qSeqLen;
        uint32_t kvSeqLen = tiling_.kvSeqLen;
        uint32_t qSeqLenAlign = tiling_.qSeqLenAlign;
        uint32_t dimAlign = AlignUp(dim, C0);
        uint64_t qTotalElements = (uint64_t)tiling_.batch * tiling_.heads * qSeqLen * dim;
        uint64_t kvTotalElements = (uint64_t)tiling_.batch * tiling_.heads * kvSeqLen * dim;
        uint64_t smCount = (uint64_t)tiling_.batch * tiling_.heads * qSeqLen;
        uint64_t smOutCount = (uint64_t)tiling_.batch * tiling_.heads * qSeqLen * kvSeqLen;
        int totalBlocks = CeilDiv(qSeqLenAlign, BLOCK_M) * tiling_.heads * tiling_.batch;

        int coreIdx;
        if ASCEND_IS_AIC {
            coreIdx = GetBlockIdx();
        }
        if ASCEND_IS_AIV {
            // MIX_AIC_1_2: AIV uses coreIdx normalized by GetSubBlockNum().
            uint32_t subBlockNum = GetSubBlockNum();
            if (subBlockNum == 0) { subBlockNum = 1; }
            coreIdx = GetBlockIdx() / subBlockNum;
        }
        int numCores = tiling_.usedCoreNum;
        sched_.Init(totalBlocks, numCores, coreIdx);

        qGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QType *>(q), qTotalElements);
        kGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QType *>(k), kvTotalElements);
        vGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QType *>(v), kvTotalElements);
        outGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QType *>(attnOut), qTotalElements);
        smMaxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(smMax), smCount);
        smSumGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(smSum), smCount);
        smOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QType *>(smOut), smOutCount);
        // arch35 atten_mask: fp32 mask [S1, S2] {0.0 keep / 1.0 mask-out}, broadcast over
        // B/N. Host widens the raw bool mask -> fp32 {0,1} (dtype lift only, NO -inf bias
        // pre-bake; PB-37-safe). The vec kernel reads it at the arch35 compressMode-computed
        // offset (ComputeAttenMaskOffset, attenmask_arch35.h) and applies the masked-fill
        // IN-KERNEL via Muls(SOFTMAX_NEG_INF)+Add — the arch35 offset machinery with a
        // race-free apply primitive (the lib Select apply had an intermittent timing race;
        // see flash_attention_score_vec.h). NOT the arch22 host-prebaked additive shortcut.
        uint64_t maskCount = (uint64_t)qSeqLen * (uint64_t)kvSeqLen;
        attnMaskGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(attenMask), maskCount);
        // sm_out is documented-deferred: host pre-zeroes the buffer (see model_new_ascendc.py).
        // No kernel-side write — see knowledge_update.md §softmax_online for rationale.

        // Workspace layout per core — single wsMetaGm_ region with 3 sub-regions
        // (max / sum / exp), per FA-class brief anchor #1.
        uint64_t wsSSize = (uint64_t)RING_SLOTS * BLOCK_M * BLOCK_N;
        uint64_t wsPSize = (uint64_t)RING_SLOTS * BLOCK_M * BLOCK_N;
        uint64_t wsOSize = (uint64_t)RING_SLOTS * BLOCK_M * dimAlign;
        uint64_t wsMetaSize = (uint64_t)RING_SLOTS * BLOCK_M * 3;
        uint64_t wsAccOSize = (uint64_t)RING_SLOTS * BLOCK_M * dimAlign;

        uint64_t perCoreBytes = wsSSize * sizeof(float) +
                                wsPSize * sizeof(QType) +
                                wsOSize * sizeof(float) +
                                wsMetaSize * sizeof(float) +
                                wsAccOSize * sizeof(float);

        GM_ADDR wsBase = workspace + coreIdx * perCoreBytes;

        GM_ADDR wsSPtr = wsBase;
        GM_ADDR wsPPtr = wsSPtr + wsSSize * sizeof(float);
        GM_ADDR wsOPtr = wsPPtr + wsPSize * sizeof(QType);
        GM_ADDR wsMetaPtr = wsOPtr + wsOSize * sizeof(float);
        GM_ADDR wsAccOPtr = wsMetaPtr + wsMetaSize * sizeof(float);

        wsSGm_.SetGlobalBuffer((__gm__ float *)(wsSPtr), wsSSize);
        wsPGm_.SetGlobalBuffer(reinterpret_cast<__gm__ QType *>(wsPPtr), wsPSize);
        wsOGm_.SetGlobalBuffer((__gm__ float *)(wsOPtr), wsOSize);
        wsMetaGm_.SetGlobalBuffer((__gm__ float *)(wsMetaPtr), wsMetaSize);
        wsAccOGm_.SetGlobalBuffer((__gm__ float *)(wsAccOPtr), wsAccOSize);

        if ASCEND_IS_AIC {
            cubeKernel_.Init(tiling_, qGm_, kGm_, vGm_, wsSGm_, wsPGm_, wsOGm_);
            cubeKernel_.InitBuffers(*pipe_);
        }

        if ASCEND_IS_AIV {
            vecKernel_.Init(tiling_, wsSGm_, wsPGm_, wsOGm_, wsMetaGm_, wsAccOGm_,
                            outGm_, smMaxGm_, smSumGm_, smOutGm_, attnMaskGm_);
            vecKernel_.InitBuffers(pipe_);
        }
    }

    __aicore__ inline void Process()
    {
        uint32_t qSeqLenAlign = tiling_.qSeqLenAlign;
        uint32_t kvSeqLen = tiling_.kvSeqLen;
        uint32_t seqBlocks = CeilDiv(qSeqLenAlign, BLOCK_M);
        uint32_t kvLoops = CeilDiv(kvSeqLen, BLOCK_N);

        if (seqBlocks == 0 || tiling_.heads == 0) { return; }

        while (sched_.HasNext()) {
            int blockIdx = sched_.Next();

            int bx = blockIdx % seqBlocks;
            int tmp = blockIdx / seqBlocks;
            int by = tmp % tiling_.heads;
            int bz = tmp / tiling_.heads;

            if ASCEND_IS_AIC {
                cubeKernel_.LoadQ(bz, by, bx);

                for (uint32_t t = 0; t < kvLoops + PRELAUNCH; t++) {
                    if (t < kvLoops) {
                        cubeKernel_.ComputeMM1(bz, by, t);
                    }
                    if (t >= PRELAUNCH) {
                        int nowK = t - PRELAUNCH;
                        cubeKernel_.ComputeMM2(bz, by, nowK);
                    }
                }
            }

            if ASCEND_IS_AIV {
                vecKernel_.InitState();
                vecKernel_.curBz_ = bz;
                vecKernel_.curBy_ = by;
                vecKernel_.curBx_ = bx;

                int vec1Loop = 0;
                int vec2Loop = 0;

                for (uint32_t t = 0; t < kvLoops + PRELAUNCH; t++) {
                    if (t < kvLoops) {
                        int slot = t % RING_SLOTS;
                        bool isFirstVec1 = (vec1Loop == 0);
                        bool isTailKV = (t == (uint32_t)kvLoops - 1) && (tiling_.tailValid != 0);
                        vecKernel_.ComputeVec1(slot, isFirstVec1, isTailKV, t);
                        vec1Loop++;
                    }
                    if (t >= PRELAUNCH) {
                        int nowK = t - PRELAUNCH;
                        int slot = nowK % RING_SLOTS;
                        bool isFirstVec2 = (vec2Loop == 0);
                        bool isLastVec2 = (nowK == (int)kvLoops - 1);
                        vecKernel_.ComputeVec2(slot, isFirstVec2, isLastVec2);
                        vec2Loop++;
                    }
                }
                // softmax_out emission step 2: after the KV loop, exp-normalize the
                // captured raw scores in smOutGm_ using the final row max/sum.
                vecKernel_.EmitSoftmaxOut();
            }
        }
    }

private:
    TPipe *pipe_;
    FlashAttentionScoreTiling tiling_;
    BlockScheduler1D sched_;

    GlobalTensor<QType> qGm_, kGm_, vGm_, outGm_;
    GlobalTensor<float> wsSGm_, wsOGm_, wsMetaGm_, wsAccOGm_;
    GlobalTensor<QType> wsPGm_;
    GlobalTensor<float> smMaxGm_, smSumGm_;
    GlobalTensor<QType> smOutGm_;  // declared but host pre-zeros (deferred)
    GlobalTensor<float> attnMaskGm_;  // arch35 atten_mask [S1,S2] fp32 {0 keep / 1 mask-out}

    FlashAttentionScoreCube<QType> cubeKernel_;
    FlashAttentionScoreVec<QType> vecKernel_;
};

#endif // FLASH_ATTENTION_SCORE_KERNEL_H
