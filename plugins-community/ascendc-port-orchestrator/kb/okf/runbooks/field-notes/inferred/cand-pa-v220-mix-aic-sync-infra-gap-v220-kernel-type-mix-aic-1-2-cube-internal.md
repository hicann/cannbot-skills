---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220 `KERNEL_TYPE_MIX_AIC_1_2` cube-internal pipe sync deadlocks regardless of event-ID scheme — root cause deeper than event-ID allocation"
description: "applies_to: soc=Ascend910_9382 (V220 A2/A3); cann=9.0.0+; op_class=mixed_aic_aiv_pattern_a_tile_mmad; macro=KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2) verified_on: a5_ops:3_FusionAttention kw-3"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_9382 (V220 A2/A3); cann=9.0.0+; op_class=mixed_aic_aiv_pattern_a_tile_mmad; macro=KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)"
confidence: inferred
status: stub
original_id: CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP
timestamp_inferred: true
tags: [candidate, inferred, kernel_type_mix_aic_1_2, datacopy, loaddata2d, mmad, cand-pa-v220-mix-aic-sync-infra-gap]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_9382 (V220 A2/A3); cann=9.0.0+; op_class=mixed_aic_aiv_pattern_a_tile_mmad; macro=KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`
`verified_on: a5_ops:3_FusionAttention kw-3/kw-4/kw-5 iter chain 2026-05-21 — 4 distinct sync schemes tested over 4 worker iters; ALL produce identical silent-hang at torch.npu.synchronize()`
`v351_reproduces: NO (probe_a5_v300_fa_sync 2026-05-23) — Pattern A on V351 (MatmulImpl<> + manual CrossCoreSetFlag<0x2>(FLAG_AIC_DONE=0) + MIX_AIC_1_2 + 16×16×16 fp16 mm.IterateAll + AIV Muls(*,2.0)) completes in 0.036ms steady-state with bit-exact AIV output and non-zero matmul C output across 3 trials. The "deeper sync-infra gap" hypothesis applies to V220-specific FFTSCNT mailbox semantics, NOT to V351. PB-35 cube-internal pipe sync (event_t for HardEvent::M_FIX) remains unverified on V351 — follow-up probe required.`
`derived-from: empirical kw-5 falsification of CAND-FA1's "use event_t(N >= 4)" event-ID-collision hypothesis (codified separately as PB-35)`

**Hypothesis falsified by this candidate**: "Low event IDs (0/1) collide with AIC↔AIV CrossCoreSetFlag chain at flag IDs 0..3; using event IDs ≥ 4 (or canonical `GetTPipePtr()->FetchEventID()`) for cube-internal pipe sync resolves the deadlock."

**Empirical evidence chain** (4 schemes, 4 worker iters):

| Iter / cycle | Scheme | Outcome |
|---|---|---|
| iter 1 (kw-2 cycle 1) | `DataCopy` ND GM→L1 + `LoadData2D` L1→L0 (no ND→NZ step) | `error code 507015 aicore exception` on first `Mmad` (layout fault, not sync) |
| iter 2 (kw-3 cycle 2) | `DataCopy(l1, gm, Nd2NzParams{...dstNzC0Stride=S})` + `LoadData2D` (wrong stride) | Same generic 507015 (still layout fault) |
| iter 3 Phase 1 (kw-4) | Corrected `Nd2NzParams{...dstNzC0Stride=D/16}` + `LoadData2D` (no pipe sync) | `0x8000004000 L0B read/write conflict in MTE` (sync genuinely missing — layout fixed) |
| iter 3 Phase 2 (kw-4) | Above + `SetFlag/WaitFlag<HardEvent::MTE2_MTE1\|MTE1_M\|M_FIX>(event_t(0))` | Silent hang at `torch.npu.synchronize()` past 90s (PB-35 codifies as "event_t(0) collides with FLAG_CANON_DONE") |
| iter 4 Phase 1 (kw-5) | Same + raw `event_t(2,3,4)` for mm1 + `event_t(5,6,7)` for mm2 (distinct IDs, all ≥ 2) | Silent hang, identical signature |
| iter 4 Phase 2 (kw-5) | Same + `GetTPipePtr()->FetchEventID(HardEvent::X)` canonical runtime allocation | Silent hang, identical signature |

Three layer transitions across the chain (each iter peeled one layer): Layer 1 (iter 1) — ND vs NZ layout (solved by `Nd2NzParams` shape); Layer 2 (iter 3 Phase 1) — cross-pipe sync absent (MTE1→M RAW hazard); Layer 3 (iter 3 Phase 2 onward) — even with valid sync events, the deadlock persists. Falsifies the simple "event-ID collision" hypothesis.

**Strong candidate root-cause hypotheses for fo investigation**:
1. **Cross* sync uniformity per HardEvent class**: AIC↔AIV CrossCoreSetFlag chain may impose barriers that collide with `HardEvent::M_FIX` regardless of event ID (the cross-core semantics are uniform per HardEvent class, not per event ID). All three schemes fail because they all use `HardEvent::M_FIX` which IS the FIX-pipe event being driven externally by the AIV→AIC chain.
2. **MIX_AIC_1_2 may require uniform Cross* sync across ALL pipe events on the cube side**, not local SetFlag/WaitFlag mixed with CrossCoreSetFlag for cross-core sync. Cube-internal pipe sync via local `SetFlag<HardEvent::X>` may be incompatible with the mixed-mode dispatch loop.
3. **FFTSCNT mailbox semantics**: per the `kfc_dispatch_failure_followup.md` root cause investigation, MIX_AIC_1_2 may have mailbox-counter semantics that prevent any cube tile-MMAD with internal sync regardless of event ID scheme.

**Scope**:
- Applies only to `KERNEL_TYPE_MIX_AIC_1_2` launches (single-launch mixed cube + vector core).
- Non-mixed cube-only `KERNEL_TYPE_AIC_ONLY` launches NOT tested this evidence chain — may behave differently.
- Tested only on case_3 [4,64,512] BSH fp16 head=8 (S=64, D=64). Smaller / larger shapes not probed; conclusion likely holds across shapes given the failure is at the cube-launch level not at compute.

**Mitigation today (until fo investigation lands)**:
- Stay on AIV-only VEC fallback path for fp16 cube-eligible shapes. AIV-only path delivers 3_FusionAttention case_3 PASS_T1 with ours_mere=1.999e-6 beats CANN (cann_mere=2.227e-6, 11% better). Performance via AIV path is bandwidth-limited but acceptable for the in-scope shape.
- Workspace baseline kernel uses VEC-only path; do NOT attempt MIX_AIC_1_2 cube tile-MMAD with user-owned pipe sync until KB seeding lands.

**Escalation paths (mutually exclusive — pick one)**:
1. **`aog-fused-optimizer` + KB seeding**: Investigate alternate launch modes (e.g. `KERNEL_TYPE_AIC_ONLY` separate from AIV softmax, with explicit GM hand-off via CrossCoreSetFlag<0x2>). Estimated 8-12 fo iters once API_CATALOG.md gains entries for cube tile primitives.
2. **`aog-cann-learner` Mode 5 extraction**: Run dedicated CANN learner against CANN's `ops-transformer/op_kernel/flash_attention_score/arch22/` source to extract V220 cube workflow into `patterns/unverified/candidates.md` as a verified pattern (with explicit event-ID allocation and CrossCoreSetFlag/HardEvent interaction documented).
3. **Pause for authoritative documentation**: Wait for an AscendC mixed-mode programming page on hiascend.com (currently no public doc covers MIX_AIC_1_2 cube-internal pipe sync rules).

**HYPOTHESIS STATUS UPDATED 2026-05-23 — V220 Pattern C status UNVERIFIED (NOT closed); A5/V351 cross-arch probe (commit `56444ff8`) invalidated V220 Pattern A as cross-arch finding**:

#### Prior (PR #117, 2026-05-22) claim — partially RETRACTED 2026-05-23

PR #117 codified my V220 Pattern C probe as "Class B falsification — Pattern C structurally blocked on V220" and authored the verdict "all 3 V220 mixed AIC+AIV patterns now empirically falsified". **Both claims are now flagged for re-examination per main agent's A5/V351 probe finding (commit `56444ff8` 2026-05-23)**:

1. **My V220 Pattern C probe was likely misdesigned** (same defect as main's V351 Pattern C probe per their PROBE_REPORT.md): used `SetFlag<HardEvent::MTE3_MTE2>` from AIC + `WaitFlag<HardEvent::MTE3_MTE2>` from AIV — but `HardEvent::*` is **intra-core** pipe-sync semantics. AIC and AIV are different cores → the WaitFlag on AIV waits on a pipe-event AIC never raises on the AIV's pipe register. Hang is from the misdesign, NOT a V220 architectural block of single-launch fused. To genuinely probe Pattern C on V220 the cross-core sync mechanism is `CrossCoreSetFlag<0x2>(flagId)` not `SetFlag<HardEvent>`.

2. **V220 Pattern A status unchanged** (PB-34 + PB-35 still valid V220-only). Pattern B unwound from production for unrelated reasons. But **the "all 3 patterns falsified" claim was overstating** — only A is empirically falsified on V220 by the 5-iter chain; B/C are weaker claims (probe-design defect or unverified).

3. **CAND-FA-MULTI-LAUNCH-PERF-GAP V220 "real ceiling at 0.014× CANN" claim still holds via Pattern A falsification alone** (no Pattern C contribution needed); but the phrasing "multi-launch is FORCED architecture" was too strong — should be "multi-launch is the path of least resistance given Pattern A's confirmed V220 deadlock; Pattern C on V220 remains genuinely unverified due to probe-design defect".

#### A5/V351 cross-arch finding (commit `56444ff8` 2026-05-23) — confirmed by main agent's `probe_a5_v300_fa_sync`

Pattern A on V351/Ascend950PR_9579 + CANN B103 runs clean (0.036ms, bit-exact, 3 deterministic trials). V220 Pattern A deadlock does NOT reproduce on V351. This invalidates any "V220 → V351 inheritance" of mixed-mode-deadlock assumption. See PB-34's new `verified_does_not_reproduce_on: V351` line.

#### Current status

- **V220**: Pattern A confirmed deadlocking (PB-34/PB-35 valid). Pattern B/C status — Pattern B unwound for unrelated reasons; Pattern C UNVERIFIED (probe was misdesigned). Multi-launch architecture remains the practical V220 choice given Pattern A block, but the door is NOT formally closed on Pattern C.
- **V351**: Pattern A runs clean. Single-launch fused FA is **viable on V351**. Use single-launch architecture for any V351 / A5 fused-attention port.
- **Real V220 ceiling**: 0.014× CANN @ S=1024 (PR #114 measured) stays correct, supported by Pattern A deadlock (PB-34) — does NOT rely on Pattern C "falsification".

#### Follow-up needed (genuine V220 Pattern C falsification)

A correctly-designed V220 Pattern C probe must use cross-core sync (`CrossCoreSetFlag<0x2>(flagId)` on AIC + `CrossCoreWaitFlag<0x2>(flagId)` on AIV), not `HardEvent`-based intra-core sync. The original PR #116 probe source at `docs/design/fa_delta5_pattern_c_probe_snapshot/probe_pattern_c_kernel.h` is **invalidated as a Pattern C reproducer**; future probe should follow PB-35's reserved-flag-IDs convention (cross-core flags 0..3, cube-internal events ≥4 if pipe sync also needed). Cost: ~60-90 min for proper probe.

#### Cost paid for this codification

- PR #117 (~45 min) shipped a partially-wrong claim that was caught + corrected within ~3h by main's A5 probe.
- Net KB still benefits from the negative finding (probe-design defect is a learned lesson, codified below), even though the original "falsified" claim was wrong.
- **Lesson per OL-175**: probe-result codification needs probe-design-validation step. My V220 Class B hang shape (silent timeout, no error) is identical to a misdesigned-sync probe AND a real cross-core deadlock — they're not distinguishable without checking sync intrinsic semantics first.

#### Cross-ref

- [CAND-FA-MULTI-LAUNCH-PERF-GAP §5](candidates.md#CAND-FA-MULTI-LAUNCH-PERF-GAP) (re-scoped 2026-05-23 to V220-only; V351 has different path)
- PB-34, PB-35 (still valid Pattern A V220-only falsifications)
- `workspace/probe_a5_v300_fa_sync/PROBE_REPORT.md` (main's A5 probe + identification of intra-core vs cross-core sync defect)
- `docs/design/fa_delta5_pattern_c_probe_snapshot/probe_pattern_c_kernel.h` (my malformed V220 probe — preserved as reproducer for the defect, NOT as falsification evidence)
- OL-175 (failure-framing discipline — applies here: "Class B fail" was claimed without distinguishing misdesign from real deadlock)

**Hard do-not-apply**:
- Do NOT use `KERNEL_TYPE_MIX_AIC_1_2` with user-owned cube tile-MMAD + local `SetFlag/WaitFlag<HardEvent::*>` pipe sync until this candidate resolves. Compiles clean, runtime deadlocks.
- Do NOT interpret PB-35's "Use IDs ≥ 4" Fix as a complete solution — it closes the visible event-ID collision but not the underlying sync-infra gap.

**Cross-ref**:
- PB-35 (the visible event-ID collision; this candidate explains why the PB-35 Fix is incomplete)
- PB-34 (the related cube+vec sync minefield — Matmul library vs user-owned flags)
- CAND-FA1 (Pattern A recommendation; this candidate is the open-hypothesis follow-up for CAND-FA1's empirical_validated_on event-ID section)
- OL-159 (FA-class independent prototype structural-rewrite-needed — the larger framing under which this gap lives)
- `output/.../workspace/3_FusionAttention/kfc_dispatch_failure_followup.md` (prior fo workstream tracking the FFTSCNT mailbox angle)
- API_CATALOG.md (currently zero entries for cube tile-MMAD primitives — known gap)

**Promotion path**: candidate stays here until (a) one of the 3 escalation paths resolves the deadlock with reproducible evidence on case_3, OR (b) the AIV-only fallback path is formally declared the canonical V220 FA strategy and Pattern A on V220 is documented as architecturally infeasible. Either outcome promotes to OL-class entry.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PA-V220-MIX-AIC-SYNC-INFRA-GAP，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
