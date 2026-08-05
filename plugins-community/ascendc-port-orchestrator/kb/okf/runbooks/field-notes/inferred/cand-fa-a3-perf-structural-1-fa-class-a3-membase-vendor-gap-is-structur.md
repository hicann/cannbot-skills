---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "FA-class A3 (membase) vendor-gap is structural — matmul::Matmul async cube + zero-CrossCore vs hand-Mmad MIX_AIC_1_2; in-paradigm scalar levers cap at lever-1"
description: "applies_to: soc=Ascend910_9382 (V220 A3); cann=9.0.0+; op_class=fa_class_membase_mix_aic; layout=BNSD/BSH/SBH derived-from: a5_ops whitebox CANN-A3-source comparison (3_FusionAttention_n9bis vs ops-tr"
phenomenon: build_failure
signal:
  - "FA-class A3/V220 membase kernel below vendor npu_fusion_attention, profile shows aiv_scalar_ratio high (>0.5) + aic_mac_ratio low (<0.15). Use to decide whether"
confidence: inferred
status: stub
original_id: CAND-FA-A3-PERF-STRUCTURAL-1
timestamp_inferred: true
tags: [candidate, inferred, npu_fusion_attention, aiv_scalar_ratio, aic_mac_ratio, iteratebmm1, iteratebmm2, cand-fa-a3-perf-structural-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_9382 (V220 A3); cann=9.0.0+; op_class=fa_class_membase_mix_aic; layout=BNSD/BSH/SBH`
`derived-from: a5_ops whitebox CANN-A3-source comparison (3_FusionAttention_n9bis vs ops-transformer/attention/flash_attention_score/op_kernel/arch22/flash_attention_score_s1s2_bn2gs1.h, 2026-05-30, NPU2 device-time)`
`verified_on: a5_ops (3_FusionAttention_n9bis msprof A/B: ours 0.25x vendor S=1024 = 236us vs npu_fusion_attention 58-61us; aic_mac 0.089 vs vendor 0.315; aiv_scalar 0.776 vs vendor 0.283)`
`unverified_on: matmul::Matmul library FULL-FA port perf on A3 (POC 2026-05-30: the lib COMPILES standalone on V220 but DEADLOCKS at runtime — 507014 KFC-workspace-bootstrap gap; standalone >0.7x NOT achieved, see Recommendation)`

**Trigger**: FA-class A3/V220 membase kernel below vendor `npu_fusion_attention`, profile shows `aiv_scalar_ratio` high (>0.5) + `aic_mac_ratio` low (<0.15). Use to decide whether further hand-tuning is worth it vs accepting the membase ceiling.

**Finding (CANN-source-grounded, file:line both sides)**: the ~4x vendor gap is STRUCTURAL, not a tunable scalar loop:
1. **Cube dispatch**: vendor uses `matmul::Matmul<>` library (`flash_attention_score_s1s2_bn2gs1.h:98-119` decl; `IterateBmm1` L1086-1106 / `IterateBmm2` L2116-2155 call `IterateAll<false>` = single async call doing internal L1-reuse of Q, L0A/L0B double-buffer, K-tiling pipeline, library-managed Fixpipe+fences). Cube runs ASYNC while vector computes (3-stage pipeline bmm1[t] || vec1[t-1] || bmm2[t-2]). OURS (`fusion_attention_cube.h` ComputeMM1 L85-175 / ComputeMM2 L195-270) = hand `Mmad`+fence SYNCHRONOUS loop; AIC stop-and-waits per WorkspaceQueue slot. => explains aic_mac 0.089 vs 0.315 (our cube is WAITING, not computing).
2. **CrossCore fence count**: vendor = 0 `CrossCoreSetFlag/WaitFlag` in bn2gs1 (~13 `WaitIterateAll` per Q-block, amortized over the tile, same kernel body drives cube+vec). OURS = `workspace_queue.h:46-81` fires 12 hardware CrossCore ops/KV-tile => **96 per Q-block at S=1024** (48/side). Each AIV `CrossCoreWaitFlag` stalls the scalar pipe waiting for AIC Fixpipe => dominant source of aiv_scalar 0.776.
3. **Root cause = architecture**: we use MIX_AIC_1_2 with SEPARATE AIC/AIV bodies (require hardware CrossCore flags to communicate); vendor runs cube-dispatch + vector in a SINGLE body via the matmul library's async interface (zero CrossCore flags).

**Recommendation**:
- In-paradigm (membase hand-Mmad) actionable levers are EXHAUSTED at lever-1 (RowMulsImpl Brcb+Mul vectorization, shipped). The only remaining in-paradigm lever is RowDivsImpl Brcb+Div vectorization, but (a) expected <5% (dominated by CrossCore stalls, not the 256 GetValue ops/Q-block) and (b) it FAILED 3 ways on V220: shared softmaxExpUb_ scratch -> NaN, dedicated recipBuf_ -> NaN (deeper RowMuls->RowDivs Vec2 pipeline hazard in `!isFirst&&isLast` tile), Brcb+Div -> UB-alignment fault 507015 (`Div` BinaryRepeatParams stride convention differs from `Mul` on V220 — open sub-issue if pursued).
- To close the >=80% structural gap: port FA-class A3 to `matmul::Matmul<>` library (`CFG_EXCEED` config) = single-body async-cube redesign (DEBT-20 `-DASCENDC_MATMUL_AICORE` flag isolation + full kernel restructure). NOT a one-line edit. This is the A3 high-perf path (still membase/arch22, NOT regbase/A5). **Owner gate (2026-05-30): validate the matmul library on A3 hits >0.7x vendor via a SMALL experiment BEFORE the full rewrite — guard against it being another membase ceiling.**
- **POC RESULT (2026-05-30, empirical — gate verdict NO-GO for standalone kernel rewrite)**: a minimal QK^T-only `matmul::Matmul<>` (KFC path, `matmul_intf.h`, `KERNEL_TYPE_MIX_AIC_1_2`, no manual CrossCoreSetFlag) in a standalone pybind11 kernel **COMPILES + links on V220** but **DEADLOCKS at runtime = aicore timeout 507014**. Root cause: the KFC path needs the CANN operator framework's workspace bootstrap (`SetSysWorkspaceForce(workspace)` + auto_gen `WORKSPACE_PARAM_OFFSET` + framework-allocated workspace layout matching `KfcCommServer::Init()`); without it the AIC-side KFC server never processes the AIV cube requests → AIV hangs → timeout. This is a NEW V220 hazard distinct from PB-34 (`MatmulImpl<>`+manual-CrossCore) and PB-35 (event-id collision): **KFC standalone workspace-bootstrap failure**. Implication: realizing the matmul-library high-perf path on A3 requires registering FA as a proper CANN operator (op_host tiling fn + framework workspace + working `REGIST_MATMUL_OBJ` bootstrap) — substantially MORE than a kernel-level rewrite, and off the port_a3/DEBT-110 product mainline. The standalone pybind11 architecture's 0.25x is therefore an EMPIRICALLY-CONFIRMED ceiling (not just source-inferred). The POC did NOT measure its own aic_mac (deadlocked before running) — vendor 0.355 / hand-Mmad 0.090 stand as the only measured cube ratios.

**Risks before promotion**: structural claim is source-comparison + msprof grounded. The matmul::Matmul-port is now EMPIRICALLY tested at POC level (compiles-but-deadlocks-507014 standalone — see POC RESULT); the FULL framework-integrated port perf remains unmeasured (would require the CANN-operator-registration effort). RowDivsImpl Div-alignment is an open V220 question. Cross-validate the aic_mac/fence root-cause on a second FA-class membase op (GQA) before promoting to OL.

**Other instances predicted**: any FA-class / cube+vec fused op on V220 membase MIX_AIC_1_2 (GQA, sparse-FA, NSA) — same hand-Mmad-vs-library structural gap.

**Cross-ref**: CAND-FA-TILESIZE-1 (the tile-size lever — real 1.6-1.7x internal but does NOT close vendor gap; this CAND explains WHY the residual is structural), OL-196 (membase=A3/V220/arch22 vs regbase=A5/V351/arch35), PB-34 (MatmulImpl + manual CrossCoreSet/Wait deadlock — a DIFFERENT failure mode, relevant when attempting the library port), PB-35 (FA pipe-sync event-id collision in MIX_AIC_1_2), DEBT-20 (per-source MATMUL_AICORE flag isolation needed for the library port), CAND-FA-MULTI-LAUNCH-PERF-GAP (the 5-delta perf comparison — overlapping root cause).

> **UPDATE 2026-05-30 (independent prototype, source-derived — the "507014=ceiling/needs-framework-registration" conclusion above is OVERTURNED as a mechanism)**: the standalone-KFC deadlock is NOT a ceiling. See **CAND-KFC-standalone-bootstrap-teardown** below — the 507014 is a 2-layer KFC lifecycle issue (workspace bootstrap + RAII-destructor teardown), both replicable standalone. The earlier POC concluded "ceiling" because it had the teardown wrong (`mm.End()` does NOT send SERVICE_QUIT). Standalone matmul-lib is reachable. (Owner meta-diagnosis 2026-05-30: a running same-chip CANN reference ⇒ "standalone won't start" = mechanism not fully read, not a ceiling.) **Status: mechanism source-solid; the FIX is NOT verified until a standalone kernel applies it and RUNS without 507014.**

## CAND-KFC-standalone-bootstrap-teardown (standalone matmul-lib/KFC reachable; overturns the 507014-ceiling)

**Source provenance**: CANN dav_c310 (V351/A5) + dav_c220 (V220/A3) `kfc/` headers + `kernel_operator_common_impl.h`, owner-authorized white-box read 2026-05-30. KFC lifecycle pattern is common across both archs. **Customer-runnable: this CAND states the MECHANISM + the standalone fix; no CANN path required to apply it.**

**Problem**: `matmul::Matmul<>` library (KFC path, `KERNEL_TYPE_MIX_AIC_1_2`) in a standalone pybind11 kernel (no CANN-op framework) deadlocks at runtime → aicore timeout 507014 / Exit 124. Two earlier POCs split on the cause (one "bootstrap fails", one "layer-3 teardown blocked") and one concluded a hard ceiling. Both are reconciled below; neither is a ceiling.

**Mechanism — 2 KFC lifecycle layers, both replicable standalone**:
1. **Bootstrap**: the AIC KFC server + AIV client message buffers live at offsets into a GM `workspace` (`KfcCommServer::Init(workspace, i)` → `GetMsgHead(workspace, i)`). Standalone recipe: allocate a GM workspace sized for the KFC msg ring + matmul L1/L0 scratch, call `SetSysWorkspaceForce(workspace)` so `GetSysWorkSpacePtr()` returns it, THEN `REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), bmm1, tiling, bmm2, tiling)` (+ `matmul::InitL1Buffer`). If the workspace layout doesn't match `KfcCommServer::Init`'s expected offsets, the AIC server never services AIV cube requests → AIV hangs at the FIRST iterate.
2. **Teardown**: `SERVICE_QUIT` (0xfd00) exits the AIC server `while(isRun)` loop. It is posted by the **AIV-side `~KfcCommClient()` DESTRUCTOR** (RAII scope-exit): `AllocMessage()` + `KfcMsgMakeFlag(SERVICE_QUIT, 0)` + `dcci` to GM. **It is NOT sent by `mm.End()`.** Standalone fix: let the matmul / KfcCommClient object destruct at AIV kernel scope-exit (correct RAII lifetime — do not keep it alive past the kernel body); software KFC (`enableHardWare=false`); MIX_NUM/subblock conditions (in 1:2 AIC:AIV mode only the right AIV subblock sends, plus a `CrossCoreWaitFlag(KFC_SYNC_ID)` handshake). Missing teardown → AIC `while(isRun)` hangs forever → 507014.

**Implication**: standalone matmul-lib/KFC is achievable (workspace bootstrap + RAII destructor teardown) — NOT a ceiling requiring full CANN-operator registration. Unblocks the FA-A5 regbase perf path AND the FA-A3 matmul-lib path (shared blocker, shared answer).

**Verification status (DISCIPLINE)**: MECHANISM is source-derived and solid. This is **NOT a verified fix** until a standalone kernel applies it and a measured run returns WITHOUT 507014/124. Do not mark "solved" pre-run. Flow: codify (this CAND) → apply (RAII destructor scope + workspace bootstrap) → measured run → close.

**Cross-ref**: overturns the 507014-ceiling conclusion in the CAND above; PB-34 (MatmulImpl+manual-CrossCore — different failure mode); DEBT-20 (`-DASCENDC_MATMUL_AICORE` per-source flag isolation, still needed for the library compile); OL-196 (membase/regbase); CAND-V351-arch35-RegBase-service-class-skeleton (the regbase service-class that uses this KFC).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-A3-PERF-STRUCTURAL-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
