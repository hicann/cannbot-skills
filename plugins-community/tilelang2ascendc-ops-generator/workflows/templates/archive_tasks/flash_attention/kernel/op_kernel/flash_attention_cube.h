/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_CUBE_H
#define FLASH_ATTENTION_CUBE_H

#include "kernel_operator.h"
#include "matmul_tile.h"
#include "utils/flash_attention_tiling.h"
#include "kernel_common.h"
#include "workspace_queue.h"

using namespace AscendC;

template <typename QType>
class FlashAttentionCube {
    static constexpr uint32_t C0 = 32 / sizeof(QType);  // 16

public:
    __aicore__ inline FlashAttentionCube() {}

    __aicore__ inline void Init(const FlashAttentionTiling &tiling,
                                GlobalTensor<QType> &qGm, GlobalTensor<QType> &kGm, GlobalTensor<QType> &vGm,
                                GlobalTensor<float> &wsSGm, GlobalTensor<QType> &wsPGm,
                                GlobalTensor<float> &wsOGm)
    {
        tiling_ = tiling;
        qGm_ = qGm;
        kGm_ = kGm;
        vGm_ = vGm;
        wsSGm_ = wsSGm;
        wsPGm_ = wsPGm;
        wsOGm_ = wsOGm;
        dimAlign_ = AlignUp(tiling_.dim, C0);
        sQueue_.Init(wsSGm_, BLOCK_M * BLOCK_N, SIG_S_READY, SIG_S_FREE);
        pQueue_.Init(wsPGm_, BLOCK_M * BLOCK_N, SIG_P_READY, SIG_P_FREE);
        oQueue_.Init(wsOGm_, BLOCK_M * dimAlign_, SIG_O_READY, SIG_O_FREE);
    }

    __aicore__ inline void InitBuffers(TPipe &pipe)
    {
        uint32_t dim = tiling_.dim;
        pipe.InitBuffer(qBufL1_, BLOCK_M * dim * sizeof(QType));
        pipe.InitBuffer(kvBufL1_, BLOCK_N * dim * sizeof(QType));
        pipe.InitBuffer(pBufL1_, BLOCK_M * BLOCK_N * sizeof(QType));
        pipe.InitBuffer(queL0A_, 2, BLOCK_M * BASE_K * sizeof(QType));
        pipe.InitBuffer(queL0B_, 2, BASE_K * BLOCK_N * sizeof(QType));
        pipe.InitBuffer(queL0C_, 1, BLOCK_M * BASE_K * sizeof(float));
        pQueue_.InitFreeSlotsMte2();
    }

    __aicore__ inline void LoadQ(int bz, int by, int bx)
    {
        uint32_t dim = tiling_.dim;
        uint32_t qSeqLen = tiling_.qSeqLen;
        uint64_t qOffset = ((uint64_t)bz * tiling_.heads * qSeqLen
                          + (uint64_t)by * qSeqLen
                          + (uint64_t)bx * BLOCK_M) * dim;
        LocalTensor<QType> qL1 = qBufL1_.Get<QType>();
        LoadNdGmToNzL1(qL1, qGm_[qOffset], BLOCK_M, dim, dim);
        SetWaitFlag<HardEvent::MTE2_MTE1>();
    }

    // BMM1: S = Q @ K^T  (S shape: BLOCK_M x BLOCK_N)
    // For the tail KV tile, kvRows < BLOCK_N; we load only kvRows rows from GM
    // and use kvRowsAlign as the k-dimension in Mmad.
    __aicore__ inline void ComputeMM1(int bz, int by, int t)
    {
        auto sSlot = sQueue_.ProducerAcquire();
        uint32_t dim      = tiling_.dim;
        uint32_t kvSeqLen = tiling_.kvSeqLen;
        uint32_t tailValid = tiling_.tailValid;

        uint32_t rowStart  = t * BLOCK_N;
        uint32_t kvRows      = (tailValid != 0 && rowStart + BLOCK_N > kvSeqLen) ? tailValid : BLOCK_N;
        uint32_t kvRowsAlign = AlignUp(kvRows, C0);

        LocalTensor<QType> kvL1 = kvBufL1_.Get<QType>();
        uint64_t kOffset = ((uint64_t)bz * tiling_.heads * kvSeqLen
                          + (uint64_t)by * kvSeqLen
                          + (uint64_t)rowStart) * dim;
        LoadNdGmToNzL1(kvL1, kGm_[kOffset], kvRows, dim, dim, kvRowsAlign);
        SetWaitFlag<HardEvent::MTE2_MTE1>();

        LocalTensor<float> cL0  = queL0C_.AllocTensor<float>();
        LocalTensor<QType>  qL1  = qBufL1_.Get<QType>();

        uint32_t kTiles      = dim / BASE_K;
        uint32_t mActAlign   = AlignUp(BLOCK_M, C0);

        for (uint32_t ki = 0; ki < kTiles; ki++) {
            DoMM1Tile(cL0, qL1, kvL1,
                      (uint64_t)ki * BLOCK_M * BASE_K,
                      (uint64_t)ki * kvRowsAlign * BASE_K,
                      BASE_K, (BASE_K / C0) * (kvRowsAlign / C0),
                      mActAlign, kvRowsAlign, ki == 0);
        }

        uint32_t kRemain = dim % BASE_K;
        if (kRemain > 0) {
            uint32_t kAligned = AlignUp(kRemain, C0);
            DoMM1Tile(cL0, qL1, kvL1,
                      (uint64_t)kTiles * BLOCK_M * BASE_K,
                      (uint64_t)kTiles * kvRowsAlign * BASE_K,
                      kAligned, (kAligned / C0) * (kvRowsAlign / C0),
                      mActAlign, kvRowsAlign, kTiles == 0);
        }

        queL0C_.EnQue(cL0);
        cL0 = queL0C_.DeQue<float>();
        SetWaitFlag<HardEvent::M_FIX>();

        FixpipeParamsV220 fixParams;
        fixParams.mSize       = BLOCK_M;
        fixParams.nSize       = kvRowsAlign;
        fixParams.srcStride   = mActAlign;
        fixParams.dstStride   = BLOCK_N;
        fixParams.ndNum       = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        Fixpipe(sSlot, cL0, fixParams);

        queL0C_.FreeTensor(cL0);
        SetWaitFlag<HardEvent::FIX_MTE2>();
        sQueue_.ProducerReleaseFix();
    }

    // BMM2: O_tmp = P @ V  (O shape: BLOCK_M x dim)
    // Load a K tile (or K remainder tile) into L0B with 2D contiguous copy.
    __aicore__ inline void LoadKTileToL0B(LocalTensor<QType> &bL0,
                                           const LocalTensor<QType> &kvL1,
                                           uint64_t kL1Off, uint32_t repeatTimes)
    {
        LoadData2DParams p;
        p.startIndex  = 0;
        p.repeatTimes = repeatTimes;
        p.srcStride   = 1;
        p.dstGap      = 0;
        p.ifTranspose = false;
        LoadData(bL0, kvL1[kL1Off], p);
    }

    __aicore__ inline void ComputeMM2(int bz, int by, int t)
    {
        uint32_t dim      = tiling_.dim;
        uint32_t kvSeqLen = tiling_.kvSeqLen;
        uint32_t tailValid = tiling_.tailValid;

        auto pSlot = pQueue_.ConsumerAcquire();
        LocalTensor<QType> pL1 = pBufL1_.Get<QType>();
        LoadNdGmToNzL1(pL1, pSlot, BLOCK_M, BLOCK_N, BLOCK_N);
        SetWaitFlag<HardEvent::MTE2_MTE1>();
        pQueue_.ConsumerReleaseMte2();

        auto oSlot = oQueue_.ProducerAcquire();
        uint32_t rowStart   = t * BLOCK_N;
        uint32_t kvRows     = (tailValid != 0 && rowStart + BLOCK_N > kvSeqLen) ? tailValid : BLOCK_N;
        uint32_t kvRowsAlign = AlignUp(kvRows, C0);
        uint64_t vOffset = ((uint64_t)bz * tiling_.heads * kvSeqLen
                          + (uint64_t)by * kvSeqLen
                          + (uint64_t)rowStart) * dim;
        LocalTensor<QType> kvL1 = kvBufL1_.Get<QType>();
        uint32_t nTiles    = dim / BASE_K;
        uint32_t mActAlign = AlignUp(BLOCK_M, C0);

        for (uint32_t ni = 0; ni < nTiles; ni++) {
            uint64_t vSliceOffset = vOffset + (uint64_t)ni * BASE_K;
            ComputeMM2Tile(pL1, kvL1, kvRows, kvRowsAlign, BASE_K, vSliceOffset);
            LocalTensor<float> cL0 = queL0C_.DeQue<float>();
            FixpipeMM2Main(oSlot, cL0, ni, mActAlign);
            queL0C_.FreeTensor(cL0);
        }

        uint32_t nRemain = dim % BASE_K;
        if (nRemain > 0) {
            uint32_t nAligned = AlignUp(nRemain, C0);
            uint64_t vSliceOffset = vOffset + (uint64_t)nTiles * BASE_K;
            ComputeMM2Tile(pL1, kvL1, kvRows, kvRowsAlign, nAligned, vSliceOffset);
            LocalTensor<float> cL0 = queL0C_.DeQue<float>();
            FixpipeMM2Remainder(oSlot, cL0, nTiles, nAligned);
            queL0C_.FreeTensor(cL0);
        }

        SetWaitFlag<HardEvent::FIX_MTE2>();
        oQueue_.ProducerReleaseFix();
    }

private:
    // Helper: one MM1 tile — load Q/K slices, Mmad, accumulate into cL0
    __aicore__ inline void DoMM1Tile(LocalTensor<float> &cL0, LocalTensor<QType> &qL1,
                                      LocalTensor<QType> &kvL1,
                                      uint64_t qL1Off, uint64_t kL1Off,
                                      uint32_t kSize, uint32_t kRepeatTimes,
                                      uint32_t mActAlign, uint32_t kvRowsAlign,
                                      bool isFirst)
    {
        SetWaitFlag<HardEvent::M_MTE1>();
        LocalTensor<QType> aL0 = queL0A_.AllocTensor<QType>();
        {
            for (uint32_t i = 0; i < mActAlign / C0; i++) {
                LoadData2DParams p;
                p.startIndex  = i;
                p.repeatTimes = kSize / C0;
                p.srcStride   = mActAlign / C0;
                p.dstGap      = 0;
                p.ifTranspose = false;
                LoadData(aL0[C0 * i * kSize], qL1[qL1Off], p);
            }
        }
        queL0A_.EnQue(aL0);

        SetWaitFlag<HardEvent::M_MTE1>();
        LocalTensor<QType> bL0 = queL0B_.AllocTensor<QType>();
        LoadKTileToL0B(bL0, kvL1, kL1Off, kRepeatTimes);
        queL0B_.EnQue(bL0);

        aL0 = queL0A_.DeQue<QType>();
        bL0 = queL0B_.DeQue<QType>();

        MmadParams mp;
        mp.m = BLOCK_M;
        mp.n = kvRowsAlign;
        mp.k = kSize;
        mp.cmatrixInitVal = isFirst;
        mp.cmatrixSource  = false;
        SetWaitFlag<HardEvent::MTE1_M>();
        PipeBarrier<PIPE_M>();
        Mmad(cL0, aL0, bL0, cL0, mp);
        queL0A_.FreeTensor(aL0);
        queL0B_.FreeTensor(bL0);
    }

    // Helper: one MM2 tile — load P/V slices, Mmad, EnQue cL0
    __aicore__ inline void ComputeMM2Tile(LocalTensor<QType> &pL1, LocalTensor<QType> &kvL1,
                                           uint32_t kvRows, uint32_t kvRowsAlign,
                                           uint32_t nSize, uint64_t vSliceOffset)
    {
        SetWaitFlag<HardEvent::M_MTE1>();
        LocalTensor<QType> aL0 = queL0A_.AllocTensor<QType>();
        LoadNzL1ToZzL0A(aL0, pL1, BLOCK_M, kvRowsAlign, BLOCK_M);
        queL0A_.EnQue(aL0);

        LoadNdGmToNzL1(kvL1, vGm_[vSliceOffset], kvRows, nSize, tiling_.dim, kvRowsAlign);
        SetWaitFlag<HardEvent::MTE2_MTE1>();
        SetWaitFlag<HardEvent::M_MTE1>();
        LocalTensor<QType> bL0 = queL0B_.AllocTensor<QType>();
        LoadNzL1ToZnL0B(bL0, kvL1, kvRowsAlign, nSize, kvRowsAlign);
        queL0B_.EnQue(bL0);

        aL0 = queL0A_.DeQue<QType>();
        bL0 = queL0B_.DeQue<QType>();

        LocalTensor<float> cL0 = queL0C_.AllocTensor<float>();
        MmadParams mp;
        mp.m = BLOCK_M;
        mp.n = nSize;
        mp.k = kvRowsAlign;
        mp.cmatrixInitVal = true;
        mp.cmatrixSource  = false;
        SetWaitFlag<HardEvent::MTE1_M>();
        PipeBarrier<PIPE_M>();
        Mmad(cL0, aL0, bL0, mp);
        queL0C_.EnQue(cL0);
        queL0A_.FreeTensor(aL0);
        queL0B_.FreeTensor(bL0);
    }

    // Helper: Fixpipe for main MM2 tiles
    __aicore__ inline void FixpipeMM2Main(LocalTensor<float> &oSlot, LocalTensor<float> &cL0,
                                           uint32_t ni, uint32_t mActAlign)
    {
        SetWaitFlag<HardEvent::M_FIX>();
        uint64_t wsBase = (uint64_t)ni * BASE_K;
        FixpipeParamsV220 fixParams;
        fixParams.mSize       = BLOCK_M;
        fixParams.nSize       = BASE_K;
        fixParams.srcStride   = mActAlign;
        fixParams.dstStride   = dimAlign_;
        fixParams.ndNum       = 1;
        fixParams.srcNdStride = 0;
        fixParams.dstNdStride = 0;
        Fixpipe(oSlot[wsBase], cL0, fixParams);
        SetWaitFlag<HardEvent::FIX_MTE2>();
    }

    // Helper: Fixpipe for remainder MM2 tile
    __aicore__ inline void FixpipeMM2Remainder(LocalTensor<float> &oSlot, LocalTensor<float> &cL0,
                                                uint32_t nTiles, uint32_t nAligned)
    {
        SetWaitFlag<HardEvent::M_FIX>();
        uint64_t wsBase = (uint64_t)nTiles * BASE_K;
        FixpipeNzL0cToNdGm(oSlot[wsBase], cL0, nAligned, BLOCK_M);
        SetWaitFlag<HardEvent::FIX_MTE2>();
    }

    FlashAttentionTiling tiling_;
    uint32_t dimAlign_;

    GlobalTensor<QType> qGm_, kGm_, vGm_;
    GlobalTensor<float> wsSGm_;
    GlobalTensor<QType> wsPGm_;
    GlobalTensor<float> wsOGm_;
    WorkspaceQueue<float, RING_SLOTS> sQueue_;
    WorkspaceQueue<QType, RING_SLOTS> pQueue_;
    WorkspaceQueue<float, RING_SLOTS> oQueue_;

    TBuf<TPosition::A1> qBufL1_;
    TBuf<TPosition::A1> kvBufL1_;
    TBuf<TPosition::A1> pBufL1_;

    TQue<TPosition::A2, 2> queL0A_;
    TQue<TPosition::B2, 2> queL0B_;
    TQue<TPosition::CO1, 1> queL0C_;
};

#endif // FLASH_ATTENTION_CUBE_H
