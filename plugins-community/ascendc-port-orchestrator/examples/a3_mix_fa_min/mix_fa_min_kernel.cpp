/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// a3 MIX_AIC_1_2 cross-core SYNC-WITNESS — handshake-only entry (NOT an op).
//
// The full cube+vector MIX cross-sync chain, single-tile, fixed shape, with the
// AIV compute replaced by a TRIVIAL PLACEHOLDER (identity copy). NO
// KERNEL_TASK_TYPE_DEFAULT macro (arch35-only -> 107000 on arch22). The default
// MIX launch reaches BOTH AIC and AIV; ASCEND_IS_AIC / ASCEND_IS_AIV partition.
//
//   AIC : S = Q@K^T (cube#1, GENUINE) -> notify AIV (FLAG_S, broadcast)
//         wait AIV (FLAG_P)           -> O = P@V (cube#2, GENUINE)
//   AIV : wait AIC (FLAG_S)           -> P = identity(S) [PLACEHOLDER] -> notify AIC (FLAG_P)
//
// PB-55 (the load-bearing rule): the REVERSE AIV->AIC handshake is per-subblock
// COUNTED, not broadcast. BOTH AIV subblocks run the (identical -> benign)
// placeholder copy and BOTH CrossCoreSetFlag(FLAG_P); a single-setter reverse
// hangs the AIC CrossCoreWaitFlag forever. The FORWARD AIC->AIV FLAG_S is
// broadcast (one set releases both AIVs). This asymmetry is the whole witness.
// The compute is a placeholder ON PURPOSE — there is nothing here to copy for a
// real op; only the deadlock-free handshake is on display.
// ============================================================================
#include "kernel_operator.h"
#include "mix_fa_min_kernel.h"

extern "C" __global__ __aicore__ void mix_fa_min_example_fp16(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR o,
    GM_ADDR s_gm, GM_ADDR p_gm,
    int32_t seq, int32_t d) {
    if ASCEND_IS_AIC {
        MixCubeQK<half>(q, k, s_gm, seq, d);                 // S = Q @ K^T (genuine cube)
        // fixpipe drained (IterateAll sync=true); publish S to the paired AIVs.
        CrossCoreSetFlag<MIX_SYNC_MODE2, PIPE_FIX>(FLAG_S);
        CrossCoreWaitFlag(FLAG_P);                           // block until P ready
        MixCubePV<half>(p_gm, v, o, seq, d);                 // O = P @ V (genuine cube)
    }
    if ASCEND_IS_AIV {
        CrossCoreWaitFlag(FLAG_S);                           // block until S ready
        // PB-55: BOTH subblocks run the placeholder identity copy (identical,
        // benign) and BOTH raise FLAG_P — the reverse handshake is counted.
        MixPlaceholder<half>(s_gm, p_gm, seq, seq);
        CrossCoreSetFlag<MIX_SYNC_MODE2, PIPE_MTE3>(FLAG_P);
    }
}
