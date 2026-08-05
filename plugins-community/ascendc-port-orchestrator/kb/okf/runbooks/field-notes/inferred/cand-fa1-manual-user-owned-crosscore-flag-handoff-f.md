---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Manual user-owned CrossCore flag handoff for mixed AIC/AIV producer-consumer stages (NOT for kernels using high-level Matmul<> library internals)"
description: "applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=mixed_aic_aiv_fused_kernel_with_user_owned_cross_engine_handoff derived-from: cann-source (FA-class fused-attention reference struct"
phenomenon: build_failure
signal:
  - "mixed-mode kernel dispatched via KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2) with AIC half compiled from __DAV_C220_CUBE__ and AIV half from __DAV_C220_VE"
confidence: inferred
status: stub
original_id: CAND-FA1
timestamp_inferred: true
tags: [candidate, inferred, nd2nzparams, mmad, loaddata2dparams, loaddatawithtranspose, loadnzl1tozzl0a, cand-fa1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=mixed_aic_aiv_fused_kernel_with_user_owned_cross_engine_handoff`
`derived-from: cann-source (FA-class fused-attention reference structure, 2026-05-10 revise-cl3)`
`verified_on: cann ops-transformer FA reference — top-level kernel file (cube/vec flag chain ~80 lines, three named user-owned flags); arch cross-core-sync header (FlagID = uint16_t; MAX_REVERSE_DEPTH = 16 array-slot count = 15 reuses + 1 initial state, consistent with ascend950pr.md "同一 flagId 最大计数 15 次"; FFTS_MAX_FLAG = 7); reserved-barrier IDs at 8/9/10 are a cann-source-derived REFINEMENT of the public 0–10 range documented in ascend950pr.md, not a replacement`
`refuted_on: a5_ops:3_FusionAttention:case_b27a259d — kernel mixes MatmulImpl + manual CrossCore which violates this pattern's hard-do-not-apply clause; the manual CrossCoreWaitFlag path hard-hung. This is NEGATIVE evidence reinforcing the exclusion, NOT positive validation. The specific failure mode (AICore timeout 507014 / LaunchAscendKernel 507035) is codified separately as PB-34 in PLATFORM_BUGS.md — read it before emitting any MIX_AIC_1_2 kernel that touches MatmulImpl/MatmulClient/KFC.`

`empirically_validated_on (2026-05-21, partial evidence chain — a5_ops:3_FusionAttention kw-4 cycle 3 run buksn5pky):`
- **`Nd2NzParams` field shape for D-aligned fp16 tiles (D%16==0)**: `Nd2NzParams{ndNum=1, nValue=S, dValue=D, srcNdMatrixStride=0, srcDValue=D, dstNzC0Stride=D/16, dstNzNStride=16, dstNzMatrixStride=0}` correctly performs ND→NZ during `DataCopy(l1, gm, params)` GM→L1. Evidence: with this shape, the first-`Mmad` fault evolved `507015` (iter 1+2, wrong shape) → `0x8000004000` L0B read/write conflict (iter 3 Phase 1, shape correct but sync missing). The transition is unambiguous evidence that the shape passes the L1-decode stage.
- **`LoadData2DParams` with `ifTranspose=true` for K^T side** (matmul1 = Q @ K^T case): mechanically equivalent to `LoadDataWithTranspose`; built clean on V220 with `LoadData2DParams{startIndex=0, repeatTimes=(baseM/16)*(baseK/16), srcStride=1, sid=0, dstGap=0, ifTranspose=true, addrMode=0}`. **SUPERSEDED 2026-05-28** by `fa_class/cv_reference_concrete_params.md#decision_id-qk_load_form` (cv-agent ComputeMM1 dual-operand `ifTranspose=false` + Mmad k-contraction along D=C0). This `ifTranspose=true` form COMPILES clean and accepts QK^T magnitude but, when paired asymmetrically with a plain 3DParamsV2 A-load (`LoadNzL1ToZzL0A`), produces a layout-permute on the `[BLOCK_M × BLOCK_N]` tile — `attn_out` `abs_max` tracks ref within ~1% but element-wise `max_diff ~1.3-1.6` on the FA-A3 6-case canonical (P-P99 corollary: A/B contraction axes must source the same axis). Use the `qk_load_form` decision; do NOT emit `ifTranspose=true` in fresh code. Kept here for historical evidence of the V220 compile-clean signal that masked the precision bug.
- **Event-ID allocation — `N >= 4` Fix PARTIALLY REFUTED, deeper deadlock unresolved**: prior hypothesis "`event_t(0)` collides with cross-core `FLAG_CANON_DONE`; use `N >= 4` to dodge" — the collision IS real (codified as PB-35), but the kw-5 cycle (3_FusionAttention iter 4, 2026-05-21T~14:00Z) empirically falsified that `N >= 4` is sufficient. Tested three distinct schemes: (a) `event_t(0)` baseline (silent hang); (b) raw `event_t(2,3,4)` for mm1 + `event_t(5,6,7)` for mm2 with distinct IDs ≥ 2 (silent hang, same signature); (c) `GetTPipePtr()->FetchEventID(HardEvent::X)` canonical runtime allocation (silent hang, same signature). ALL three produce identical "kernel enqueues + torch.npu.synchronize() hangs past 45s, no aicore exception" symptom. The visible event-ID collision is one layer; the actual deadlock root cause is deeper — open hypotheses include Cross* sync uniformity per HardEvent class, MIX_AIC_1_2 cube-internal-sync incompatibility with the CrossCoreSetFlag chain, and FFTSCNT mailbox semantics interacting with cube-internal HardEvent flags. See PB-35 Evidence row 3 + new candidate CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP for the full evidence chain.
- **Outstanding for canonical promotion**: full Pass A case_3 `PASS_T1` + measured perf vs CANN baseline. Iter 4 attempted convergence with corrected event-ID scheme — FAILED (silent hang persists across all 3 sync schemes). Promotion to canonical `P-P` / `OL` entry BLOCKED by the unresolved MIX_AIC_1_2 sync-infra gap; baseline kernel remains AIV-only VEC fallback delivering case_3 PASS_T1 (ours_mere=1.999e-6 < cann_mere=2.227e-6) with 60 deterministic `_OutOfScope` skips. Next attempted convergence requires either `aog-fused-optimizer` investigation of MIX_AIC_1_2 sync semantics OR `aog-cann-learner` Mode 5 extraction of CANN ops_transformer arch22 `flash_attention_score` kernel structure.

**Trigger**: mixed-mode kernel dispatched via `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)` with AIC half compiled from `__DAV_C220_CUBE__` and AIV half from `__DAV_C220_VEC__`, decomposing into ≥2 user-owned producer-consumer stages where cube and vector exchange GM-resident intermediates and a kernel-wide `SyncAll<true>()` is too coarse. The kernel must NOT instantiate the high-level AscendC `Matmul<>` template (`MatmulImpl` / `MatmulClient` / KFC) — see hard-exclusion below.

**Recommendation**: pair `AscendC::CrossCoreSetFlag<0x2, PIPE>(flagId)` with `AscendC::CrossCoreWaitFlag<0x2>(flagId)`. **MODE template argument must appear identically on both sides** — this satisfies the canonical "SetFlag 和 WaitFlag 必须参数完全一致" rule in `ascend950pr.md`. `CrossCoreWaitFlag`'s MODE is defaulted in the public header, so a bare `CrossCoreWaitFlag(id)` call resolves to the same EventID; for KB readability ALWAYS write the explicit `<0x2>` on the wait so the pairing is visually unambiguous. MODE `0x2` is the AIC↔AIV 1:2 paired-sync mode inside `KERNEL_TYPE_MIX_AIC_1_2`; release reaches only the paired sub-blocks of the opposite engine, NOT a whole-device broadcast.

Pipe selection (verified against the FA reference):
- `PIPE_FIX` when the producer is AIC writing its output through the FIX pipe to GM
- `PIPE_MTE3` when the producer is AIV writing data through MTE3 to GM
- Pick the pipe whose retirement must precede the consumer's read.

Flag-ID ownership (cann-source-derived refinement of the public `0–10` range documented in `ascend950pr.md`):
- IDs `0..7` (`FFTS_MAX_FLAG = 7`) are the user-owned range used by this pattern.
- IDs `8`, `9`, `10` are reserved by `BarrierFlag` specializations for inter-block / inter-subblock barriers. Earlier local examples using `0x8` predate this carve-out — they were not in conflict only because no barrier specialization was active; new code MUST stay in `0..7`.
- Per-flagId count budget: canonical KB documents **15 reuses** (`ascend950pr.md` "同一 flagId 最大计数 15 次"); the cann-source constant `MAX_REVERSE_DEPTH = 16` is the underlying slot-array size = 15 reusable counts + 1 initial state. Treat 15 as the publicly-bounded reuse limit.

**Concrete anchor** (three-flag QK → softmax → PV chain; cube ladder built on tile-MMAD primitives, NOT `Matmul<>`):
```cpp
constexpr uint32_t cube

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
