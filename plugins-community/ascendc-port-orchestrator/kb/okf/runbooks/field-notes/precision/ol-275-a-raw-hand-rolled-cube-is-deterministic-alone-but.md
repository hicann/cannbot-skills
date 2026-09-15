---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A raw hand-rolled cube is deterministic ALONE, but a multi-cube pipeline self-poisons (silent wrong-result, not a hang) unless it uses a managed cube lifecycle — and a stub-AIV MIX co-schedule perturbs even a single cube"
description: "applies_to: soc=Ascend910_V220; cann=9.1.0; bisheng=n/a; op_class=attention"
phenomenon: precision_issue
signal:
  - "applies_to: soc=Ascend910_V220; cann=9.1.0; bisheng=n/a; op_class=attention"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-275
timestamp_inferred: true
tags: [507014, 507015, 000276, ascendc, ol-275]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_V220; cann=9.1.0; bisheng=n/a; op_class=attention`
`verified_on: soc=Ascend910_V220; cann=9.1.0 (FA-a3 DEBT-36 white-box, 2026-07-12)`
`unverified_on: soc=Ascend950PR (V351 / A5) — V220 evidence; the arch35 analog is OL-223 / PB-45 (Reset-safe fences) territory, do not assume transfer`

**Where this sits vs existing entries**: PB-35 is the *hang* form of the V220 hand-rolled-cube-under-MIX wall (event-flag rendezvous never completes). THIS OL is the *silent-corruption / non-determinism* form of the same wall — the symptom PB-35 and OL-223 only note in passing ("M-tail cross-core non-determinism is the irreducible PB-35 wall for a hand-rolled cube; the deterministic path is a library"). It gives that non-determinism a clean, grep-able characterization plus the mitigation decision rule.

**Principle (self-poison)**: On V220 a **single** raw cube (hand-rolled `Mmad` / `Fixpipe`, no `REGIST_MATMUL_OBJ`) launched ALONE is bit-deterministic. But a **2+-cube** pipeline (`cube → vec → cube`) **self-poisons**: ~50% of runs produce wrong output, and the corruption moves run-to-run **between** the two cubes (anti-correlation — when one cube is right the other is wrong, and vice-versa). This is unmanaged persistent cube / FFTS on-core state left by one cube and read by the next — it is **NOT a hang** (kernel completes, output is just wrong) and it is **NOT a stream-ordering or data-cache-coherence bug**: it **survives a full `aclrtSynchronizeStream` between the two launches** (a host barrier that would have fixed either of those).

**Principle (stub-AIV perturbs a single cube)**: even a SINGLE raw cube dispatched under `KERNEL_TYPE_MIX_AIC_1_2` **with an empty / no-op AIV branch** is ~12% non-deterministic; the empty-AIV co-schedule itself (the mixed-dispatch spinning up an idle vector partition), not any AIV compute, perturbs the cube's state.

**Mitigation decision rule (applies to ANY V220 hand-rolled cube)**:
- Cube needs no vector epilogue → dispatch **`AIC_ONLY`**, never `MIX_AIC_1_2`-with-empty-AIV. This made the single cube 12/12 deterministic and removed the multi-cube anti-correlation.
- Two-or-more cubes that MUST co-reside → determinism requires a **managed cube lifecycle** (`REGIST_MATMUL_OBJ` → ClearWorkspace + managed L0/L1), which must be **bootstrapped by the CANN op-framework** (the standalone `build_ascendc.py` KFC path still 507014-deadlocks — see CAND-KFC-standalone-bootstrap-teardown). On this harness the practical deterministic route is the library/op-framework path, matching P-P102 / OL-235.
- **Refuted mitigation (grep guard)**: partitioning `SetSysWorkspaceForce` into per-launch non-overlapping regions is **inert** for a hand-rolled `FaFracMM`-style cube's determinism, because that kernel **never calls `GetSysWorkSpacePtr`** — it never reads the reserved base, so the partition cannot influence its output. Before proposing per-launch sys-workspace partitioning as a *determinism* fix, grep the kernel for `GetSysWorkSpacePtr`; if absent it is a no-op. (This is distinct from EC-68 / CAND-FA-A5-KFC-WORKSPACE, where a cube that DOES call `GetUserWorkspace` / `GetSysWorkSpacePtr` genuinely needs `SetSysWorkspaceForce` first to avoid an OOB `507015` crash — a crash bug, not a determinism bug.)

**Evidence** (FA-a3 DEBT-36 white-box, 2026-07-12, Ascend910_V220 / CANN 9.1.0):
- raw multi-launch cube pipeline = **2/12** out-vs-torch; **each cube alone = 20/20**; corruption anti-correlates between the two cubes run-to-run.
- **5 manual mitigations refuted**: producer-side / consumer-side / cube-output `DataCacheCleanAndInvalid`, full `aclrtSynchronizeStream` between launches, and per-launch distinct sys-workspace (inert per the grep guard above).
- single cube under `MIX_AIC_1_2` + empty AIV branch ≈ **12% non-deterministic**; same cube re-dispatched `AIC_ONLY` = **12/12** deterministic (and removed the multi-cube anti-correlation).
- **Managed-lifecycle resolution at the SAME shape**: the vendor `FlashAttentionScore` V220 op (`REGIST_MATMUL_OBJ` ×14 + `KERNEL_TYPE_MIX_AIC_1_2` + `KFC_L1_RESERVER_SIZE 0`, launched via the op-framework that bootstraps FFTS/workspace-sync), exercised through `torch_npu.npu_fusion_attention` at the SAME 64×64 fp16 shape where the raw multi-launch scored 2/12, is **22/22 bit-identical** (12 in-process + 10 fresh-process), `max_abs_diff` 0.000276 (fp16 floor); **independently re-verified 10/10 bit-identical**. → confirms the self-poison is purely unmanaged cube state, resolved by KFC's managed lifecycle once the op-framework bootstraps it.
- **First SHIPPED verified_on:a3 cube op via the `AIC_ONLY` route (1_BatchMatmul, DEBT-206, 2026-07-13, Ascend910_9382 / CANN 9.1.0)**: the mitigation rule above ("`AIC_ONLY` when no vec epilogue") landed a real archived op — a pure cube-only batched matmul (direct `MatmulImpl<>`, single-AIC-per-batch strided dispatch, NON-KFC synchronous `IterateAll<sync=true>` + `End()`), which guards the cube body with `ASCEND_IS_AIC` so only the AIC executes the matmul. It sidesteps the 507014 MIX/KFC wall entirely (no `REGIST_MATMUL_OBJ`, no msg-ring), builds + runs clean on the standalone `build_ascendc.py`, is det 5/5 bit-identical, and ships PARTIAL_PERSIST (46/51 strict, 50/51 inclusive; the 1 FAIL is fp32 reference-ub — see OL-276). This is the existence proof that the `AIC_ONLY` single-cube path is not merely deterministic in a probe but carries an end-to-end archive — the first a3 cube op. (Caveat: the `ASCEND_IS_AIC` body-guard + `AIC_ONLY` intent were applied together with a constexpr `MatmulApiStaticTiling` on-stack `TCubeTiling`, and the pair cleared an earlier `507015` aicore-MPU fault; which of the two cleared it was NOT independently A/B-isolated, so read both as cube-only hygiene, not a single attributed 507015 fix.)
- **Compiler-generated Pattern-A witness = a second managed path that resolves self-poison on V220 (FA-a3, 2026-07-14, Ascend910_9382 / CANN 9.1.0)**: an independent AscendC compiler generated a cube+vector FA with tile-MMAD, manual cross-core flags, and a host FFTS descriptor. At the same 64×64 fp16 shape where the raw hand-rolled multi-cube pipeline was deterministic only 2/12 times, the generated FA was byte-identical in 5/5 in-process and 3/3 fresh-process runs and matched the reference. The decisive launch mechanism is host `rtGetC2cCtrlAddr(&fftsAddr, &fftsLen)` → pass `fftsAddr` as the first kernel argument → kernel `AscendC::SetSyncBaseAddr(fftsAddr)`. The standalone harness can support this mechanism in principle but its host translation unit currently lacks the required CANN include path; treat this as DEBT-210(d′), a compile-path gap that requires escalation. Bound: correctness and determinism for two shapes only, with no performance claim; this is the Pattern-A path, not the `MatmulImpl`/KFC deadlock path (PB-34).

**Other instances (predicted)**: any V220 multi-cube hand-rolled pipeline (stacked matmuls, attention `QK^T → softmax → PV`, conv/GEMM chains); any cube-only op wrapped in `MIX` dispatch with a stub/early-return AIV.

**Cross-ref**: PB-35 (the hang form of the same V220 hand-rolled-cube-under-MIX wall + the `event_t` collision detail), OL-223 / PB-45 (the arch35/V351 Reset-safe-fence resolution of the hand-rolled-cube sync problem), CAND-KFC-standalone-bootstrap-teardown (why the managed lifecycle needs the op-framework, not the standalone build), P-P102 / OL-235 (manual-`Mmad` default + why the library-cube path is unbuildable through `build_ascendc.py`), EC-68 / CAND-FA-A5-KFC-WORKSPACE (the DISTINCT `SetSysWorkspaceForce` *crash* requirement for cubes that DO call `GetUserWorkspace`).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-275（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
