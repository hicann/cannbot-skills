/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASH_ATTENTION_SCORE_CUBE_H
#define FLASH_ATTENTION_SCORE_CUBE_H

#include "kernel_operator.h"
#include "regbase_copyin.h"         // task#34: regbase GM->L1 nd2nz (matched pair w/ L0 3D-im2col)
#include "regbase_matmul.h"         // task#34: regbase matmul-API (MatmulBase/Full/N + L0 DB)
#include "regbase_fixpipe_out.h"    // FixpipeParamsC310 helpers
#include "flash_attention_score_tiling.h"
#include "kernel_common.h"
#include "workspace_queue.h"

using namespace AscendC;
using namespace fa_base_matmul;

// AIC-only Cube class for flash_attention_score — task#34 REGBASE skeleton (BSH + fp16).
//
// Replaces the hand-rolled membase Mmad micro-op loop (matmul_tile.h LoadData2D +
// per-K-tile Mmad + serial fence) with the arch35 regbase matmul-API:
//   - BufferManager<L0A/L0B/L0C> + BuffersPolicyDB → hardware-managed L0 double-buffer
//   - MatmulBase<INPUT_T,INPUT_T,float, baseM,baseN,baseK, MK, KN> → K-iteration +
//     L0 addressing offloaded to the matmul-API (MAC stays busy, no AIC scalar churn)
//   - FixpipeParamsC310<ROW_MAJOR> (NZ2ND) for L0C→GM
//
// Cross-core sync (WorkspaceQueue S/P/O ring buffers), LoadQ, and the Process-loop
// interface (Init/InitBuffers/LoadQ/ComputeMM1/ComputeMM2) are UNCHANGED so the
// orchestrator (flash_attention_score_kernel.h) needs no edit.
//
// Perf rationale (independent prototype msprof on membase 086f0460): AIC scalar-bound 57-61% /
// MAC util 4-11% (cube starvation). The regbase API removes the per-tile scalar
// address/loop/sync work from the AIC scalar pipe.
template <typename QType>
class FlashAttentionScoreCube {
    static constexpr uint32_t C0 = 32 / sizeof(QType);  // 16 for half / bfloat16
    // L0 region sizes (bytes) — match the A5 build CMake (l0a=64K, l0b=64K, l0c=256K).
    // Each region is double-buffered by BuffersPolicyDB → 2 slots of half the region.
    static constexpr uint32_t L0A_BYTES = 64 * 1024;
    static constexpr uint32_t L0B_BYTES = 64 * 1024;
    static constexpr uint32_t L0C_BYTES = 256 * 1024;
    // Regbase template base dims. BLOCK_M=64, BLOCK_N=64, dim<=128 → single-K/N.
    static constexpr uint32_t BASE_M = 128;
    static constexpr uint32_t BASE_N = 128;
    static constexpr uint32_t BASE_K = 128;

public:
    __aicore__ inline FlashAttentionScoreCube() {}

    __aicore__ inline void Init(const FlashAttentionScoreTiling &tiling,
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
        // L1 staging buffers (nd2nz). Kept identical to the membase form.
        pipe.InitBuffer(qBufL1_, BLOCK_M * dim * sizeof(QType));
        pipe.InitBuffer(kvBufL1_, BLOCK_N * dim * sizeof(QType));
        pipe.InitBuffer(pBufL1_, BLOCK_M * BLOCK_N * sizeof(QType));

        // Regbase L0 managers + double-buffer policies (AIC only).
        if ASCEND_IS_AIC {
            l0aMgr_.Init(&pipe, L0A_BYTES);
            l0bMgr_.Init(&pipe, L0B_BYTES);
            l0cMgr_.Init(&pipe, L0C_BYTES);
            l0aDb_.Init(l0aMgr_, L0A_BYTES / 2);
            l0bDb_.Init(l0bMgr_, L0B_BYTES / 2);
            l0cDb_.Init(l0cMgr_, L0C_BYTES / 2);
        }

        // Inverse-side init: cube is the consumer of P (vec writes P → cube reads).
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
        // GM->L1 regbase nd2nz (matched pair w/ LoadDataToL0A<MK>). srcOrgWidth=dim
        // (BNSD-canonical Q row is [S,D] contiguous). dstNzC0Stride=AlignUp(BLOCK_M,16).
        RegbaseCopyToL1Nd2Nz(qL1, qGm_[qOffset], BLOCK_M, dim, dim);
        SetWaitFlag<HardEvent::MTE2_MTE1>();
    }

    // BMM1: S = Q @ K^T (output: BLOCK_M x kvRowsAlign) via regbase MatmulBase.
    __aicore__ inline void ComputeMM1(int bz, int by, int t)
    {
        auto sSlot = sQueue_.ProducerAcquire();
        uint32_t dim       = tiling_.dim;
        uint32_t kvSeqLen  = tiling_.kvSeqLen;
        uint32_t tailValid = tiling_.tailValid;

        uint32_t rowStart    = t * BLOCK_N;
        uint32_t kvRows      = (tailValid != 0 && rowStart + BLOCK_N > kvSeqLen) ? tailValid : BLOCK_N;
        uint32_t kvRowsAlign = AlignUp(kvRows, C0);

        // Load K tile to L1 via regbase nd2nz (matched pair w/ LoadDataToL0B<KN>).
        // K laid [kvRows x dim] natural; KN-load's enTranspose handles K^T.
        LocalTensor<QType> kvL1 = kvBufL1_.Get<QType>();
        uint64_t kOffset = ((uint64_t)bz * tiling_.heads * kvSeqLen
                          + (uint64_t)by * kvSeqLen
                          + (uint64_t)rowStart) * dim;
        RegbaseCopyToL1Nd2Nz(kvL1, kGm_[kOffset], kvRows, dim, dim);
        SetWaitFlag<HardEvent::MTE2_MTE1>();

        LocalTensor<QType> qL1 = qBufL1_.Get<QType>();

        // Acquire an L0C slot (regbase double-buffer; producer=M, consumer=FIX).
        Buffer<BufferType::L0C> resL0C = l0cDb_.Get();
        resL0C.Wait<HardEvent::FIX_M>();

        // S = Q @ K^T : A=Q (MK, no transpose), B=K (KN, isRightTranspose → K^T).
        // singleK=dim (<=128) → MatmulBase resolves to MatmulFull (single-K, single-N).
        // singleN = kvRowsAlign (16-aligned): MmadParams.n must be 16-aligned for the
        // L0C NZ fragment layout (real-kvRows tried -> regressed; arch35's s2RealSize is
        // already s2BaseSize-aligned in its common path).
        MMParam param = MakeMMParam(BLOCK_M, kvRowsAlign, dim,
                                    /*isLeftTranspose=*/false, /*isRightTranspose=*/true);
        MatmulBase<QType, QType, float, BASE_M, BASE_N, BASE_K, ABLayout::MK, ABLayout::KN>(
            qL1, kvL1, l0aDb_, l0bDb_, resL0C.GetTensor<float>(), param);

        resL0C.Set<HardEvent::M_FIX>();
        resL0C.Wait<HardEvent::M_FIX>();

        // L0C(NZ float) → wsSGm_(ND float) via FixpipeParamsC310 ROW_MAJOR (NZ2ND).
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixp;
        fixp.nSize       = (kvRowsAlign + 7) >> 3 << 3;           // N 8-elem aligned
        fixp.mSize       = (BLOCK_M + 1) >> 1 << 1;                // even M for dualDstCtl
        fixp.srcStride   = ((fixp.mSize + 15) / 16) * 16;          // L0C NZ fragment stride
        fixp.dstStride   = BLOCK_N;                                // ND row stride in wsSGm
        // dualDstCtl=0: single GM destination (vec reads [BLOCK_M x BLOCK_N] contiguous).
        // arch35 uses dualDstCtl=1 only because it writes to UB split across two AIV
        // sub-blocks (mix-core); our S workspace is one GM region, NOT M/2-split.
        fixp.dualDstCtl  = 0;
        fixp.params.ndNum       = 1;
        fixp.params.srcNdStride = 0;
        fixp.params.dstNdStride = 0;
        Fixpipe<float, float, PFA_CFG_ROW_MAJOR_GM>(sSlot, resL0C.GetTensor<float>(), fixp);

        resL0C.Set<HardEvent::FIX_M>();
        SetWaitFlag<HardEvent::FIX_MTE2>();
        sQueue_.ProducerReleaseFix();
    }

    // BMM2: O_tmp = P @ V (output: BLOCK_M x dim) via regbase MatmulBase.
    __aicore__ inline void ComputeMM2(int bz, int by, int t)
    {
        uint32_t dim       = tiling_.dim;
        uint32_t kvSeqLen  = tiling_.kvSeqLen;
        uint32_t tailValid = tiling_.tailValid;

        auto pSlot = pQueue_.ConsumerAcquire();

        // Load P tile to L1 via regbase nd2nz (A operand, MK). P[BLOCK_M x BLOCK_N],
        // org row width = BLOCK_N.
        LocalTensor<QType> pL1 = pBufL1_.Get<QType>();
        RegbaseCopyToL1Nd2Nz(pL1, pSlot, BLOCK_M, BLOCK_N, BLOCK_N);
        SetWaitFlag<HardEvent::MTE2_MTE1>();
        pQueue_.ConsumerReleaseMte2();

        auto oSlot = oQueue_.ProducerAcquire();

        uint32_t rowStart    = t * BLOCK_N;
        uint32_t kvRows      = (tailValid != 0 && rowStart + BLOCK_N > kvSeqLen) ? tailValid : BLOCK_N;
        uint32_t kvRowsAlign = AlignUp(kvRows, C0);

        // Load V tile to L1 via regbase nd2nz (B operand, KN). V[kvRows x dim] natural.
        uint64_t vOffset = ((uint64_t)bz * tiling_.heads * kvSeqLen
                          + (uint64_t)by * kvSeqLen
                          + (uint64_t)rowStart) * dim;
        LocalTensor<QType> kvL1 = kvBufL1_.Get<QType>();
        RegbaseCopyToL1Nd2Nz(kvL1, vGm_[vOffset], kvRows, dim, dim);
        SetWaitFlag<HardEvent::MTE2_MTE1>();

        Buffer<BufferType::L0C> resL0C = l0cDb_.Get();
        resL0C.Wait<HardEvent::FIX_M>();

        // O = P @ V : A=P (MK, no transpose), B=V (KN, no transpose).
        // singleK = kvRowsAlign (16-aligned contraction dim). singleN=dim.
        MMParam param = MakeMMParam(BLOCK_M, dim, kvRowsAlign,
                                    /*isLeftTranspose=*/false, /*isRightTranspose=*/false);
        MatmulBase<QType, QType, float, BASE_M, BASE_N, BASE_K, ABLayout::MK, ABLayout::KN>(
            pL1, kvL1, l0aDb_, l0bDb_, resL0C.GetTensor<float>(), param);

        resL0C.Set<HardEvent::M_FIX>();
        resL0C.Wait<HardEvent::M_FIX>();

        // L0C(NZ float) → wsOGm_(ND float, row stride dimAlign_).
        FixpipeParamsC310<CO2Layout::ROW_MAJOR> fixp;
        fixp.nSize       = (dim + 7) >> 3 << 3;
        fixp.mSize       = (BLOCK_M + 1) >> 1 << 1;
        fixp.srcStride   = ((fixp.mSize + 15) / 16) * 16;
        fixp.dstStride   = dimAlign_;
        fixp.dualDstCtl  = 0;  // single GM destination (see ComputeMM1 note)
        fixp.params.ndNum       = 1;
        fixp.params.srcNdStride = 0;
        fixp.params.dstNdStride = 0;
        Fixpipe<float, float, PFA_CFG_ROW_MAJOR_GM>(oSlot, resL0C.GetTensor<float>(), fixp);

        resL0C.Set<HardEvent::FIX_M>();
        SetWaitFlag<HardEvent::FIX_MTE2>();
        oQueue_.ProducerReleaseFix();
    }

private:
    FlashAttentionScoreTiling tiling_;
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

    // Regbase L0 managers + double-buffer policies.
    BufferManager<BufferType::L0A> l0aMgr_;
    BufferManager<BufferType::L0B> l0bMgr_;
    BufferManager<BufferType::L0C> l0cMgr_;
    BuffersPolicyDB<BufferType::L0A> l0aDb_;
    BuffersPolicyDB<BufferType::L0B> l0bDb_;
    BuffersPolicyDB<BufferType::L0C> l0cDb_;
};

#endif // FLASH_ATTENTION_SCORE_CUBE_H
