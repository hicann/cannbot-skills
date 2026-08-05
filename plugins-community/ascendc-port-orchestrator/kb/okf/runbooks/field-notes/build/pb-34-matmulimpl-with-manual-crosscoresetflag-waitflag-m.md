---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`MatmulImpl<>` with manual `CrossCoreSetFlag`/`WaitFlag` + `MIX_AIC_1_2` deadlock on V220 [V220, mixed-mode-sync]"
description: "applies_to: soc=Ascend910_9382 (V220 A2/A3 single-die); cann=9.0.0+; op_class=mixed_aic_aiv_with_high_level_matmul_library; macro=KERNEL_TYPE_MIX_AIC_1_2"
phenomenon: build_failure
signal:
  - "Mixed cube+vec kernel dispatched via KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2). AIC half uses the high-level MatmulImpl<> / MatmulClient<> / KFC library"
confidence: single_run
original_id: PB-34
timestamp_inferred: true
tags: [507035, 507014, crosscoresetflag, waitflag, mix_aic_1_2, ascendc, pb-34]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382 (V220 A2/A3 single-die); cann=9.0.0+; op_class=mixed_aic_aiv_with_high_level_matmul_library; macro=KERNEL_TYPE_MIX_AIC_1_2`
`verified_on: a5_ops:3_FusionAttention case_b27a259d (kw-3, 2026-05-07) — independently witnessed in cann_learn offline scan 2026-05-21 (run_id 5f1f559cb8fa)`
`verified_does_not_reproduce_on: Ascend950PR_9579 (V351 / A5) — probe_a5_v300_fa_sync 2026-05-23 — Pattern A runs clean: MatmulImpl<> + manual CrossCoreSetFlag<0x2>(FLAG_AIC_DONE=0) + MIX_AIC_1_2 + 16×16×16 fp16 mm.IterateAll + AIV Muls(*,2.0) all complete in 0.036ms steady state with bit-exact y output and non-zero matmul C output. V220 FFTS sync-slot conflict does NOT reproduce on V351 hardware. Cross-ref: workspace/probe_a5_v300_fa_sync/PROBE_REPORT.md`
`verified_does_not_reproduce_on (FULL-OP scale): Ascend950PR (V351 / A5, CANN 9.1.T500) — chunk_gated_delta_rule (GDN) light-port 2026-06-15 — the FULL multi-stage upstream V220 kernel (8 matmul::MatmulImpl<> instances ×3 cube stages + KERNEL_TYPE_MIX_AIC_1_2 + manual CrossCoreSetFlag<0x2,PIPE_FIX|MTE3>/WaitFlag handshakes + SyncAll + sequential UT-inverse) COMPILED FIRST-TRY on bisheng dav-c310 + RAN without hang + 122/122 T1 PASS. This confirms the no-reproduce verdict at PRODUCTION FULL-OP scale, not just the trivial micro-probe — definitively falsifying the "MatmulImpl<>+MIX+manual-CrossCore needs structural regbase rewrite on arch35" inference for the full-op case. Consequence: for a V220 cube-MIX fused op, the DEFAULT A5 route is a LIGHT PORT (keep MatmulImpl<> + the manual flag chain; adapt only the ACLRT_LAUNCH entry + host tiling), NOT a hand-rolled tile-Mmad rewrite.`

- **Severity**: HIGH (build + register-binary both succeed, kernel launches, then hard-hangs forever; vector core fault `LaunchAscendKernel 507035` or AICore timeout `507014` depending on which side starves first; no degraded-mode workaround — the only safe response is REWRITE to one of the two valid patterns below).
- **Status**: CONFIRMED 2026-05-07 (3_FusionAttention kw-3 first-witnessed); CODIFIED 2026-05-21 after cann_learn extracted CAND-FA1 with hard-do-not-apply clause naming this exact combination. This PB documents the negative-evidence side of CAND-FA1's pattern — CAND-FA1 says "do not apply when MatmulImpl<> is in use", this PB says "and here's specifically what goes wrong if you do".
- **Symptom**: Mixed cube+vec kernel dispatched via `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`. AIC half uses the high-level `MatmulImpl<>` / `MatmulClient<>` / `KFC` library template for matmul stages. AIV half pairs with the AIC stages via `CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_X_DONE)` + `CrossCoreWaitFlag<0x2>(FLAG_X_DONE)`. Build succeeds, register-binary succeeds, kernel launches — then either AIV hangs at `CrossCoreWaitFlag(FLAG_MM_DONE)` waiting on a flag the cube side never publishes, OR AIC hangs at its `Iterate()`/`GetTensor()` body waiting on KFC-internal events that don't fire because the user-owned flag chain has captured the same hardware sync slots. Result on the host: `LaunchAscendKernel` returns `507035` (vector core abnormal exit) or `507014` (AICore timeout), depending on which engine starves first.
- **Root cause**: `MatmulImpl<>` runs its own internal cross-core synchronization through the KFC (Kernel Framework Client) protocol, consuming the same FFTS flag-ID hardware slots that user-owned `CrossCoreSetFlag<0x2>(0..7)` calls allocate from. The user-owned and library-owned slot lifetimes are incompatible: KFC expects exclusive ownership of the AIC↔AIV handshake space inside one `Iterate()` call, but the user-owned flag chain re-enters the same slots between AIV stages, corrupting KFC's internal flag-count state machine. The hang is not a software bug in either side individually — it is a hardware-level resource conflict on the FFTS sync slots.
- **Fix — pick exactly ONE of two valid V220 mixed-mode patterns; never mix them**:
  1. **Pattern A — tile-MMAD primitives + manual CrossCore** (CAND-FA1's recommendation). Replace `MatmulImpl<>` instantiations with raw tile-MMAD calls: `LoadData2D` / `LoadData3D` for L1→L0A/B fills, `Mad<>` for the multiply-accumulate, `FixpipeOut` for L0C→GM write-back. Keep the existing `CrossCoreSetFlag<0x2, PIPE_FIX>(...)` / `CrossCoreWaitFlag<0x2>(...)` chain — it works (verified by CANN's own `flash_attention_score` arch22 source).
  2. **Pattern B — `MatmulImpl<>` with KFC-implicit sync ONLY, NO manual CrossCore**. Keep the `MatmulImpl<>` instantiations. Remove ALL `CrossCoreSetFlag<0x2>` and `CrossCoreWaitFlag<0x2>` calls from BOTH the AIC and AIV sides of the kernel. Stage handoff must go through the matmul library's own queue/callback surface (`SetTensorA/SetTensorB/Iterate/GetTensor` on AIC, mirroring `GetTensorC` consumers on AIV). This pattern is only viable when the cross-stage handoff fits inside one `Iterate()` boundary — multi-stage FA pipelines often do NOT, which is why CAND-FA1 favors Pattern A.
- **Detection** (pre-build static guard):
  ```bash
  # Hard-stop: same kernel file mentions BOTH MatmulImpl AND CrossCoreSetFlag<0x2 — PB-34 collision
  for f in workspace/<op>/kernel/*.{h,cpp}; do
      grep -lq "MatmulImpl<\|MatmulClient<" "$f" \
        && grep -lq "CrossCoreSetFlag<0x2\|CrossCoreWaitFlag<0x2" "$f" \
        && echo "PB-34 violation candidate: $f"
  done
  ```
  Runtime smoking-gun: `LaunchAscendKernel` returns `507035` (vec) or `507014` (cube) on a kernel that built + registered cleanly. If the kernel ALSO previously ran fine on AIV_ONLY fallback, that confirms the issue is in the cube+vec sync surface, not the math.
- **Anti-pattern (DO NOT EMIT)**:
  ```cpp
  // BAD — Pattern A and Pattern B mixed → V220 deadlock
  MatmulImpl<AT, BT, CT, /*BIAS=*/CT, MM_CFG_STATIC> mm;  // library cube
  mm.Init(...); mm.SetTensorA(...); mm.SetTensorB(...); mm.Iterate(); mm.GetTensor(...);
  CrossCoreSetFlag<0x2, PIPE_FIX>(FLAG_MM_DONE);          // user-owned flag on top of KFC
  // ... on AIV side ...
  CrossCoreWaitFlag<0x2>(FLAG_MM_DONE);                   // hangs forever
  ```
- **Evidence**:
  - 3_FusionAttention kw-3 (2026-05-07) `case_b27a259d`: cube+vec MIX_AIC_1_2 + `MatmulImpl<>` + `CrossCoreWaitFlag(FLAG_MM1_DONE)` — AICore timeout 507014, AIVec stuck on `FLAG_MM1_DONE`. kw-3's defensive response: wrap the entire mixed entry in `#if __NPU_ARCH__ >= 3510` and route V220 traffic through an AIV-only fallback (achieved 0.04× CANN perf — wrong root-cause fix).
  - cann_learn offline scan 2026-05-21 (run_id `5f1f559cb8fa`): CAND-FA1 extracted from CANN `flash_attention_score` arch22 source, including hard-do-not-apply clause "kernel must NOT instantiate `MatmulImpl<>` / `MatmulClient` / KFC" — this PB-34 documents the SPECIFIC failure mode that clause exists to forbid.
  - 3_FusionAttention kw-1 (2026-05-20) `fusion_attention_fused_kernels.cpp`: emitted `MatmulImpl<>` (line 493/547 of `fusion_attention_kernel.h`) AND `CrossCoreSetFlag<0x2, PIPE_MTE3>(FLAG_CANON_DONE)` (line 103 of fused_kernels.cpp) — the exact PB-34 collision. Built clean but `LaunchAscendKernel 507035` at runtime on every test case (1/61 PASS).
  - **chunk_gated_delta_rule (GDN) light-port (2026-06-15, A5 / arch35 / CANN 9.1.T500) — FULL-OP no-reproduce witness**: the upstream V220 ChunkGatedDeltaRule kernel (8 `matmul::MatmulImpl<MatmulType<GM,ND,bf16,transpose>>` across 3 cube stages, `KERNEL_TYPE_MIX_AIC_1_2`, manual `CrossCoreSetFlag<0x2,PIPE_FIX|MTE3>`/`WaitFlag` handshakes, `SyncAll`, sequential UT-inverse) compiled FIRST-TRY on bisheng dav-c310 and ran without hang — `122/122 T1 PASS`, perf ~89–121µs. The exact PB-34 collision pattern (MatmulImpl<> + manual CrossCoreSetFlag<0x2> in MIX_AIC_1_2) is BENIGN on V351 at full-op scale. This is the positive negative-evidence: the deadlock is V220-specific FFTS slot behavior; on A5 the same code is the recommended light-port route.
- **Cross-reference**: `patterns/unverified/candidates.md` CAND-FA1 (Pattern A recommendation + hard-do-not-apply clause this PB enforces); PB-28 (over-generalization of which was the historical pretext for the bogus `__NPU_ARCH__ >= 3510` defensive guard that masked PB-34 on V220 by routing around it); `ascend950pr.md` § Cross-core sync (`MAX_REVERSE_DEPTH = 16` slot-count; user-owned 0..7 vs reserved 8..10 BarrierFlag IDs).

<!-- 迁移自 porter kb/target/ascendc/（PB-34，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
