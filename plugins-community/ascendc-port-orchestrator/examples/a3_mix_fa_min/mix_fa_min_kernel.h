/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MIX_FA_MIN_WITNESS_KERNEL_H
#define MIX_FA_MIN_WITNESS_KERNEL_H

// ============================================================================
// a3 MIX_AIC_1_2 cross-core SYNC-WITNESS — handshake-only (NOT an operator).
//
//   Target : arch22 / 220x / soc=Ascend910_9382 / dav-2201 / CANN 9.0.0.
//   Purpose: BUILD + RUN to WITNESS the a3 cube+vector MIX_AIC_1_2 AIC<->AIV
//            cross-core handshake close WITHOUT deadlock. The compute between
//            the flags is a TRIVIAL PLACEHOLDER (identity DataCopy) — there is
//            deliberately NOTHING here liftable for a real operator. Learn that
//            the handshake works, then GENERATE your own kernel+host from the KB
//            template P-P116 using CANN/catlass library primitives. See README.
//
// Structure kept (this is the load-bearing witness — genuine, not mocked):
//   S = Q @ K^T                    (cube #1, AIC)   [seq,seq]  GENUINE MatmulImpl
//   P = identity(S)                (vector, AIV)    PLACEHOLDER — trivial copy
//   O = P @ V                      (cube #2, AIC)   [seq,d]    GENUINE MatmulImpl
//
// Cross-sync chain (2 handshakes, DISTINCT non-zero flag ids per PB-35):
//   AIC: cube#1 -> CrossCoreSetFlag<MODE2,PIPE_FIX>(FLAG_S)   (S ready, broadcast)
//        CrossCoreWaitFlag(FLAG_P)                            (block for P)
//        cube#2                                               (O = P@V)
//   AIV: CrossCoreWaitFlag(FLAG_S)                            (block for S)
//        identity(S)->P ; CrossCoreSetFlag<MODE2,PIPE_MTE3>(FLAG_P)
//
// Load-bearing invariants (do NOT regress — see README + P-P116 / PB-34/35/55):
//   * Both matmuls are GENUINE cube via MatmulImpl<> + IterateAll<sync=true>
//     (NOT async KFC Iterate()/GetTensor() -> that is PB-34 deadlock on a3).
//     They are kept because the handshake is only witnessable if REAL cube work
//     grabs the FFTS sync slots on either side of the flags.
//   * NO KERNEL_TASK_TYPE_DEFAULT macro on arch22 (arch35-only -> 107000 at
//     RegisterAscendBinary, PB-28). The default MIX dispatch already reaches
//     AIC+AIV; ASCEND_IS_AIC / ASCEND_IS_AIV partition the work.
//   * FORWARD FLAG_S is BROADCAST (one AIC set releases both AIV subblocks).
//     REVERSE FLAG_P is per-subblock-COUNTED: BOTH AIV subblocks must
//     CrossCoreSetFlag(FLAG_P) or the single AIC WaitFlag hangs forever (PB-55).
//   * The AIV op between the flags is a PLACEHOLDER identity copy — the real
//     softmax step-sequence + its RowReduce/Align8/FloorPow2 helpers are gone.
//     There is no real-op compute to lift; only the handshake is on display.
// ============================================================================

#include "kernel_operator.h"
#include "lib/matmul_intf.h"   // MatmulImpl / MatmulType are not transitive

using namespace AscendC;
using namespace matmul;

// ---- cross-core handshake constants (user flag range 1..7; NEVER 0, PB-35) --
constexpr uint8_t  MIX_SYNC_MODE2 = 2;   // AIC<->AIV MIX 1:2 handshake mode
constexpr uint16_t FLAG_S         = 4;   // AIC->AIV : S ready
constexpr uint16_t FLAG_P         = 5;   // AIV->AIC : P ready

// ---- matmul static config (covers both QK^T and P@V single-tile shapes) --
template <typename T>
__aicore__ inline constexpr MatmulApiStaticTiling MakeMixCfg() {
    MatmulApiStaticTiling t{};
    t.cfg = CFG_NORM;
    t.usedCoreNum = 1;
    t.baseM = 128;
    t.baseN = 128;
    t.baseK = std::is_same_v<T, float> ? 64 : 128;
    t.depthA1 = 8;
    t.depthB1 = 8;
    t.stepM = 1;
    t.stepN = 1;
    t.stepKa = 1;
    t.stepKb = 1;
    t.dbL0A = 1;
    t.dbL0B = 1;
    t.dbL0C = 1;
    t.iterateOrder = 0;
    t.isBias = 0;
    t.transLength = 0;
    t.shareMode = 0;
    t.shareL1Size = 0;
    t.shareL0CSize = 0;
    t.shareUbSize = 0;
    t.depthAL1CacheUB = 0;
    t.depthBL1CacheUB = 0;
    return t;
}

// AIC cube #1 : S[seq,seq] = Q[seq,d] @ K[seq,d]^T  (transpose-B via runtime bool).
template <typename T>
__aicore__ inline void MixCubeQK(GM_ADDR q, GM_ADDR k, GM_ADDR s,
                                 int32_t seq, int32_t d) {
    static constexpr auto MM_CFG = MakeMixCfg<T>();
    using AT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
    using BT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
    using CT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
    using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;

    TPipe pipe;
    TCubeTiling tiling{};
    // M_=seq, N_=seq, K_=d.  A=Q[seq,d], B stored [seq,d]=[N_,K_] transposed.
    tiling.M = seq; tiling.N = seq; tiling.Ka = d; tiling.Kb = d;
    tiling.singleCoreM = seq; tiling.singleCoreN = seq; tiling.singleCoreK = d;

    MatmulImpl<AT, BT, CT, BiasT, MM_CFG> mm;
    mm.Init(&tiling, &pipe);

    GlobalTensor<T> qG, kG, sG;
    qG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(q), static_cast<int64_t>(seq) * d);
    kG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(k), static_cast<int64_t>(seq) * d);
    sG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(s), static_cast<int64_t>(seq) * seq);

    mm.SetTensorA(qG, /*isTransposeA=*/false);
    mm.SetTensorB(kG, /*isTransposeB=*/true);   // runtime bool drives the transpose
    mm.SetSingleShape(seq, seq, d);
    mm.template IterateAll<true>(sG, 0, false, false, false);  // sync=true, no atomic
    mm.End();
}

// AIC cube #2 : O[seq,d] = P[seq,seq] @ V[seq,d]  (both normal, no transpose).
template <typename T>
__aicore__ inline void MixCubePV(GM_ADDR p, GM_ADDR v, GM_ADDR o,
                                 int32_t seq, int32_t d) {
    static constexpr auto MM_CFG = MakeMixCfg<T>();
    using AT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
    using BT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
    using CT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
    using BiasT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;

    TPipe pipe;
    TCubeTiling tiling{};
    // M_=seq, N_=d, K_=seq.  A=P[seq,seq], B=V[seq,d]=[K_,N_] normal.
    tiling.M = seq; tiling.N = d; tiling.Ka = seq; tiling.Kb = seq;
    tiling.singleCoreM = seq; tiling.singleCoreN = d; tiling.singleCoreK = seq;

    MatmulImpl<AT, BT, CT, BiasT, MM_CFG> mm;
    mm.Init(&tiling, &pipe);

    GlobalTensor<T> pG, vG, oG;
    pG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p), static_cast<int64_t>(seq) * seq);
    vG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(v), static_cast<int64_t>(seq) * d);
    oG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(o), static_cast<int64_t>(seq) * d);

    mm.SetTensorA(pG, /*isTransposeA=*/false);
    mm.SetTensorB(vG, /*isTransposeB=*/false);
    mm.SetSingleShape(seq, d, seq);
    mm.template IterateAll<true>(oG, 0, false, false, false);
    mm.End();
}

// AIV PLACEHOLDER : P[rows,cols] = identity(S[rows,cols]) — a trivial per-row
// GM->UB->GM DataCopy. This is NOT a softmax and NOT any real-op compute; the
// original numerically-stable softmax step-sequence and its RowReduce/Align8/
// FloorPow2 helpers were DELETED so there is nothing liftable here. Its ONLY
// job is to run a trivial vector op between CrossCoreWaitFlag(FLAG_S) and
// CrossCoreSetFlag(FLAG_P) so the MIX_AIC_1_2 handshake is exercised and
// witnessable. BOTH AIV subblocks run this identical benign copy (PB-55: the
// reverse AIV->AIC handshake is per-subblock counted, not broadcast).
template <typename T>
__aicore__ inline void MixPlaceholder(GM_ADDR s, GM_ADDR p,
                                      int32_t rows, int32_t cols) {
    TPipe pipe;
    TBuf<TPosition::VECCALC> rowBuf;
    pipe.InitBuffer(rowBuf, cols * sizeof(T));

    GlobalTensor<T> sG, pG;
    sG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(s), static_cast<int64_t>(rows) * cols);
    pG.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(p), static_cast<int64_t>(rows) * cols);

    event_t eMte2Mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
    event_t eMte3Mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    LocalTensor<T> row = rowBuf.Get<T>();
    for (int32_t r = 0; r < rows; ++r) {
        DataCopy(row, sG[static_cast<int64_t>(r) * cols], cols);   // GM S row -> UB
        SetFlag<HardEvent::MTE2_MTE3>(eMte2Mte3);
        WaitFlag<HardEvent::MTE2_MTE3>(eMte2Mte3);
        DataCopy(pG[static_cast<int64_t>(r) * cols], row, cols);   // UB -> GM P row (identity)
        SetFlag<HardEvent::MTE3_MTE2>(eMte3Mte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eMte3Mte2);                 // before next iter reuses row
    }
}

#endif  // MIX_FA_MIN_WITNESS_KERNEL_H
