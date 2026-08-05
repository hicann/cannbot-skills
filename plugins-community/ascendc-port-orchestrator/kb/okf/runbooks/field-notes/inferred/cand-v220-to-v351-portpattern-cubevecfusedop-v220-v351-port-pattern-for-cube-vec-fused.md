---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220→V351 port pattern for cube+vec fused ops — TWO V351 sync paradigms (forward FA-class vs backward gradient)"
description: "applies_to: soc_pair=V220→V351 (Ascend910_V220 source → Ascend950PR_9579 V351 target); cann=9.0.0; bisheng=15.0.5; op_class=non-FA-fused-cube-vec fused-quant-matmul fused-norm-matmul indexer-attention"
phenomenon: build_failure
signal:
  - "applies_to: soc_pair=V220→V351 (Ascend910_V220 source → Ascend950PR_9579 V351 target); cann=9.0.0; bisheng=15.0.5; op_class=non-FA-fused-cube-vec | fused-quant-"
confidence: inferred
status: stub
original_id: CAND-V220-to-V351-PortPattern-CubeVecFusedOp
timestamp_inferred: true
tags: [candidate, inferred, processbaseblock, taskidmod2, sync_aiv_inner_flag2, sync_v6_to_c3_flag, pipe_mte2, cand-v220-to-v351-portpattern-cubevecfusedop]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc_pair=V220→V351 (Ascend910_V220 source → Ascend950PR_9579 V351 target); cann=9.0.0; bisheng=15.0.5; op_class=non-FA-fused-cube-vec | fused-quant-matmul | fused-norm-matmul | indexer-attention-non-softmax-class | backward-gradient-cube-vec`
`verified_on: forward path = lightning_indexer arch22→arch35 (CANN .../lightning_indexer/op_kernel/ diff 2026-05-24); backward path = sparse_lightning_indexer_grad_kl_loss arch35 (CANN .../sparse_lightning_indexer_grad_kl_loss/op_kernel/arch35/ direct grep 2026-05-24 — independent review catch + main correction)`
`unverified_on: kw runtime PASS yet for LIG_grad backward port using these patterns — bg orch bmz9tfk7b in flight`

**Principle — V351 has TWO sync paradigms** (critical clarification post independent review catch 2026-05-24 21:39Z):

V351 cube+vec fused ops do NOT all share the same cross-core sync paradigm. The path depends on the **forward-vs-backward** axis of the op class:

**Path A — Forward FA-class fused op (mode=4 KFC-internal, drop per-block sync)**:
Forward attention-class ops with Q×K@V tile-scheduling. Evidence: `lightning_indexer/op_kernel/arch35/lightning_indexer_kernel.h:623-624,655-656` defines `QLI_SYNC_MODE4 = 4` (in `lightning_indexer_common.h:70`) and uses it for **outer-loop dual-flag setup + teardown ONLY**. `ProcessBaseBlock` per-block has **NO CrossCoreSetFlag/WaitFlag calls** — per-block sync is delegated to matmul library + RegBase MicroAPI primitives via KFC channel.

**Path B — Backward gradient or scatter/gather-heavy fused op (mode=2 manual per-block, rotating dual-flag indexed by taskIdMod2)**:
Backward gradient ops keep V220-style manual per-block handshake, but enhanced with V351 features: **dual-flag rotating by `taskIdMod2`** for pipeline-depth-2 producer-consumer, plus **per-stage typed flags**. Evidence: `sparse_lightning_indexer_grad_kl_loss/op_kernel/arch35/sparse_lightning_indexer_grad_kl_loss_cube_block.h:41` defines `static constexpr uint8_t SYNC_MODE = 2`. `kernel_base.h:633-664` shows 30+ `CrossCoreSetFlag<2, PIPE>(...)` and `CrossCoreWaitFlag<2, PIPE>(...)` sites with:
- Dual-flag arrays: `SYNC_MM2_TO_V1_FLAG[0,1]`, `SYNC_GATHER_TO_MM12_FLAG[0,1]`, `SYNC_C3_TO_V7_FLAG[0,1]`
- Per-stage typed flags: `SYNC_AIV_INNER_FLAG2`, `SYNC_V6_TO_C3_FLAG`
- Per-block usage: `SYNC_*_FLAG[pRunInfo.kTaskIdMod2]` rotates between [0] and [1] per task

**Decision criterion** (which paradigm to use for V220→V351 port):
- Forward attention-class (softmax + Q×K@V + per-row online softmax) → **Path A** (mode=4 KFC-internal)
- Backward gradient (GEMM-reduce + scatter + gather + relu_grad / etc.) → **Path B** (mode=2 manual rotating dual-flag)
- Forward non-attention fused (e.g. fused-norm + cube, fused-quant + cube, GMSQ_v2 path) → **Path A** mode=4 with outer-only dual-flag (verified by GMSQ_v2 commit f8fecd70 — different from sync_aic_aiv_modes=4 but same paradigm of "outer-only KFC-internal")
- Heavy scatter/gather in middle of pipeline → **Path B** (mode=2 manual)

**Concrete delta (forward `lightning_indexer_kernel.h::ProcessBaseBlock` evidence — same op, V220 vs V351)**:

V220 (arch22, manual per-block ping-pong):
```cpp
template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::ProcessBaseBlock(...) {
    if ASCEND_IS_AIC {
        CrossCoreWaitFlag(constInfo.syncV1C1);
        matmulService.ComputeMm1(runInfo);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_FIX>(
            constInfo.syncC1V1);
    } else {
        CrossCoreWaitFlag(constInfo.syncC1V1);
        vectorService.ProcessVec(runInfo);
        CrossCoreSetFlag<LICommon::ConstInfo::FIA_SYNC_MODE2, PIPE_MTE2>(
            constInfo.syncV1C1);
    }
}
```

V351 (arch35, KFC-internal):
```cpp
template <typename LIT>
__aicore__ inline void LightningIndexerKernel<LIT>::ProcessBaseBlock(...) {
    if ASCEND_IS_AIC {
        matmulService.ComputeMm1(runInfo);   // NO manual sync — KFC-internal
    } else {
        vectorService.ProcessVec1(runInfo);  // NO manual sync — KFC-internal
        if (runInfo.isLastS2InnerLoop) {
            vectorService.ProcessTopK(runInfo);
        }
    }
}
```

Outer-loop setup/teardown (both arches, but V351 uses mode=4 + dual-flag indexed events):

V220:
```cpp
if ASCEND_IS_AIV {
    vectorService.AllocEventID();
    CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);  // 2x prime the pipe
    CrossCoreSetFlag<FIA_SYNC_MODE2, PIPE_MTE2>(constInfo.syncV1C1);
} else {
    matmulService.AllocEventID();
}
// ... loop ...
if ASCEND_IS_AIC {
    matmulService.FreeEventID();
    CrossCoreWaitFlag(constInfo.syncV1C1);     // 2x drain
    CrossCoreWaitFlag(constInfo.syncV1C1);
}
```

V351:
```cpp
if ASCEND_IS_AIV {
    vectorService.AllocEventID();
    CrossCoreSetFlag<QLI_SYNC_MODE4, PIPE_V>(CROSS_VC_EVENT + 0);
    CrossCoreSetFlag<QLI_SYNC_MODE4, PIPE_V>(CROSS_VC_EVENT + 1);
} else {
    matmulService.AllocEventID();
}
// ... loop with NO per-block manual sync ...
if ASCEND_IS_AIC {
    matmulService.FreeEventID();
    CrossCoreWaitFlag<QLI_SYNC_MODE4, PIPE_FIX>(CROSS_VC_EVENT + 0);
    CrossCoreWaitFlag<QLI_SYNC_MODE4, PIPE_FIX>(CROSS_VC_EVENT + 1);
}
```

**Port-recipe checklist** (for V220→V351 cube+vec fused op, applicable to LIG_grad / quant-matmul / fused-norm-matmul backward classes):

1. **Sync mode constant**: replace `<MODE2>` template param in `CrossCoreSetFlag/WaitFlag` with `<MODE4>`. (V220 → V351 hardware sync infrastructure change. Sync constant value differs per op-family but the mode parameter is always V351=4.)
2. **PIPE on CrossCoreSetFlag for AIV-side**: V220 uses `PIPE_MTE2`; V351 uses `PIPE_V`. AIC-side keeps `PIPE_FIX` both arches.
3. **Flag IDs**: V220 typically uses named per-op flags (`syncV1C1`, `syncC1V1`); V351 uses event-indexed dual-flag (`CROSS_VC_EVENT + 0/+1`). Outer-loop setup primes BOTH flags; teardown drains BOTH.
4. **Per-block CrossCore calls in inner loop**: DELETE them entirely on V351. The matmul library + RegBase MicroAPI primitives (`MicroVAdd`/`MicroVMul`/etc.) handle KFC-internal sync. Only outer-loop setup + teardown remains.
5. **Service classes (`matmulService` / `vectorService`)**: V351 versions live in arch35/ subdir. Cube/vec primitives use `vf/` (vector fission) for typed primitives. The service-class API at V351 ports the V220 API surface to RegBase MicroAPI internals — caller code stays similar, callees differ.
6. **Per-pipe define for AIC vs AIV TU**: V351 build requires per-source-file compile-flag isolation (`-DASCENDC_MATMUL_AICORE` on AIC.cpp, `-DASCEND_VEC_AICORE` on AIV.cpp). See OL-176 / EC-58 for KFC sync per-pass-defines pattern.
7. **Sync paradigm note for non-FA-class fused ops**: per OL-185, op classification stays L2/L3 (NOT L4) when no softmax/attention/online-softmax-tile-scheduling. Calibration anchor remains flat_quant (shipped 8/8 via L2 path).

**Reject_cond** — do NOT apply when:
- Op is FA-forward class (softmax/attention with Q×K@V) → L4 path, different sync requirements (see CAND-V220-V351-FA-DIFF-1).
- Op is pure-VEC (no cube stage) → no cross-core sync needed at all.
- Op uses `KERNEL_TYPE_MIX_AIC_2_2` instead of `MIX_AIC_1_2` → may have different KFC channel layout, re-verify.

**Symptom anchor** (LIG_grad port 2026-05-24 in-flight):
- Worker fa_fused_mixed_fp16 V220 port to V351 hung at `CrossCoreWaitFlag` spin (independent prototype T1.12 iter 1/2/3 cumulative falsification) when applying V351 mode=4 to per-block sync calls that V351 doesn't need. Fix path = drop per-block, only outer dual-flag setup.
- LIG_grad worker attempted V220 line-port (mode=2 manual per-block) → would have hit same V220-only paradigm on V351 hardware. Plugin fix P0gg (commit f9b98ea3) now routes LIG_grad to L2 with this pattern as expected calibration.

**Other instances (predicted)**:
- LIG_grad (backward gradient, GEMM+gather+scatter+reduce class) — in-flight verification (bg orch bmz9tfk7b)
- Future V220→V351 backward gradients of non-FA fused ops (e.g. attention_grad sans softmax, MoE-finalize backward, fused-quant-matmul backward, fused-norm-matmul backward)
- Quant-matmul forward V220→V351 (e.g. `quant_batch_matmul_v3` port — has identical MIX_AIC_1_2 + matmul library shape)

**Promote when**:
1. LIG_grad ships via this pattern (commit SHA + verification.json pass_a 8/8) → verifies pattern beyond forward op evidence
2. A SECOND non-FA-class V220→V351 port (e.g. quant_batch_matmul_v3, fused_norm_matmul, attention_grad_no_softmax) applies the recipe + ships clean

**Cross-link**: forward `lightning_indexer/op_kernel/` arch22+arch35 pair (CANN reference source; NOT customer-readable — KB body above is self-contained); flat_quant calibration anchor (OL-185, commit 7b3c7bf3 on origin/main); CAND-V351-AIV-WholeReduceMax (related V351 AIV gotcha, commit 5260fd68); CAND-V351-arch35-RegBase-service-class-skeleton (complementary — V351 service-class detail patterns).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-V220-to-V351-PortPattern-CubeVecFusedOp，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
