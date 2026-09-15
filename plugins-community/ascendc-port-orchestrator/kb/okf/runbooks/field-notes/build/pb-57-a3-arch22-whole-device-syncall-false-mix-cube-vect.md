---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "a3/arch22 whole-device `SyncAll<false>()` MIX (cube+vector) requires `KERNEL_TYPE_MIX_AIC_1_1` (1:1) — the no-macro 1:2 default makes idle AIV cores skip the barrier loop → 507014 deadlock (narrows PB-28/A-P34)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "a MIX (cube+vector) kernel that uses whole-device SyncAll<false>() as its cross-core barrier and emits NO task-type macro hangs at torch.npu.synchronize() with"
confidence: single_run
original_id: PB-57
timestamp_inferred: true
tags: [107000, 507014, kernel_type_mix_aic_1_1, ascendc, pb-57]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
  soc: Ascend910_9382
  cann: 9.1.0
  op_class: attention-fwd
  macro: KERNEL_TYPE_MIX_AIC_1_1
  note: multi_stage_mix_aic_aiv (cube+vector) with whole-device SyncAll<false>()
```
- **Status**: CONFIRMED (2026-07-20, device-verified on a3/Ascend910_9382 — root cause pinned + empirically flipped on a SINGLE variable = the task-type macro). GDR forward gen3, archive `output/fqa_gated_delta_rule/`.
- **Severity**: HIGH — builds + packs clean (bisheng OK, no 107000 at `RegisterAscendBinary`); hangs only at RUNTIME as aicore-timeout 507014, easily misdiagnosed as an algorithm/layout bug.
- **Symptom**: a MIX (cube+vector) kernel that uses whole-device `SyncAll<false>()` as its cross-core barrier and emits NO task-type macro hangs at `torch.npu.synchronize()` with aicore-timeout 507014; adding `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` as the first entry statement → EXIT=0, runs, no hang.
- **Mechanism**: no task-type macro → default arch22 MIX dispatch = **1:2**, so AIV `GetBlockNum()` (8) ≠ AIC (4). A per-work-item loop `for (bh = GetBlockIdx(); bh < nHead; bh += GetBlockNum())` is entered a different number of times per core: AIV cores with `GetBlockIdx() >= nHead` never enter → issue **zero** `SyncAll` calls, while AIC + low-index AIV cores issue N each. Whole-device `SyncAll<false>()` waits for ALL cores → the cores that skipped never arrive → permanent cross-core deadlock (507014). Pinning **1:1** (`MIX_AIC_1_1`) gives AIC/AIV identical `GetBlockNum()` → symmetric barrier count → no deadlock.
- **Narrows PB-28 / A-P34**: those state "`KERNEL_TASK_TYPE_DEFAULT` is arch35-only, rejected on Ascend910_9382 → 107000; emit NO task-type macro on arch22." That is over-broad. It holds for `KERNEL_TYPE_AIV_ONLY`/`AIC_ONLY` (A-P34) and `MIX_AIC_1_2` (PB-28/PB-40), but is REFUTED for `MIX_AIC_1_1`: the customer cv-reference emits exactly this macro, builds `SOC_VERSION=Ascend910_9382`, and runs clean (device-measured). PB-28 already declares `MIX_AIC_1_1` out-of-scope of the 107000 ban — this entry promotes that exemption to a positive decision rule.
- **Decision rule**: (a) MIX + whole-device `SyncAll<false>()` + per-core work-item loop → emit `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` (1:1 mandatory; the 1:2 default breaks SyncAll symmetry). (b) MIX + scoped `CrossCoreSetFlag`/`WaitFlag` MODE2 handshake → default 1:2 is DEADLOCK-free, NO macro (that path is built around the 1:2 asymmetry, e.g. P-P116) — **BUT deadlock-free does not guarantee determinism; see (b')**. (c) NEVER mix "no macro (1:2)" with "whole-device SyncAll" — that is the GDR gen3 507014 trap.
- **(b') Generated scoped-handshake caveat — 1:2 is deadlock-free but can be RUN-TO-RUN NON-DETERMINISTIC at high parallelism**: a generated fused MIX kernel with `MIX_AIC_1_2` plus many per-id `CrossCoreSetFlag<0x2>` handshakes flipped bit outputs across runs above roughly 12 concurrent MIX blocks, consistent with FFTS flag-pool aliasing. **Fix**: select the balanced `MIX_AIC_1_1` template. No whole-device `SyncAll` is needed; pairing balance alone removed the nondeterminism. Keep the per-id flags and WAR barriers unchanged. Scope is limited to the measured generated-kernel lane; behavior of the hand-written scoped-1:2 path (P-P116) remains unverified.
- **Evidence**: GDR forward gen3 (a3, Ascend910_9382, 2026-07-20). No-macro + 15 whole-device `SyncAll` → 507014 on case0 `[1,64,4,128]`. Add `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1)` as first entry statement → EXIT=0, runs, no hang. cv-reference (same macro) runs clean on the same chip. Archive `output/fqa_gated_delta_rule/` (verification.json + SYNCFIX_RUN_LOG.md).
- **Evidence (b')**: generated GDR forward kernel (a3, Ascend910_9382, 2026-07-21). `MIX_AIC_1_2` plus 11 per-id `CrossCoreSetFlag<0x2>` handshakes was deadlock-free but nondeterministic at high parallelism (fp64 gate 15/16). Selecting `MIX_AIC_1_1` made all 16 cases deterministic over N=30 runs and preserved measured device performance.
- **Cross-ref**: PB-28 (the `MIX_AIC_1_1`-out-of-scope note this entry promotes to a rule), A-P34 (the `*_ONLY` macro ban, still valid), PB-40 (`MIX_AIC_1_2` 107000 at teardown on arch22), PB-55 (single-core both-AIV-subblocks-must-Set — the scoped-handshake sibling), `patterns/domains/fa_class_a3_mix_template.md` §(c) (P-P116, the scoped-CrossCore MODE2 alternative), `fa_class/cross_core_sync.md`, P-P117 (the chunked-GDR pattern that surfaced this).

<!-- 迁移自 porter kb/target/ascendc/（PB-57，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
