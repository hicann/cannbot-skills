---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "MIX dispatch runs ONE binary on BOTH AIC and AIV — every core-type-specific op (TQue/InitBuffer/Mmad/UB) must sit behind an `ASCEND_IS_AIC`/`ASCEND_IS_AIV` guard, and cross-stage `SyncAll<false>()` barriers must be count- and trip-aligned on both paths"
description: "Provenance: contributed by the customer's Kimi-K3 + a5_ops gated_delta_rule CV-fusion rewrite (PR #200, author liuyu15819). Reconciled onto latest main (DS 2026-07-20): kept as a genuine increment com"
phenomenon: build_failure
signal:
  - "Provenance: contributed by the customer's Kimi-K3 + a5_ops gated_delta_rule CV-fusion rewrite (PR #200, author liuyu15819). Reconciled onto latest main (DS 2026"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-278
timestamp_inferred: true
tags: [507014, ascend_is_aic, ascend_is_aiv, ascendc, ol-278]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

> Provenance: contributed by the customer's Kimi-K3 + a5_ops gated_delta_rule CV-fusion rewrite (PR #200, author liuyu15819). Reconciled onto latest main (DS 2026-07-20): kept as a genuine increment complementary to PB-57 (the 1:1-ratio facet); applies_to preserved as-declared (910B2C / 220x / CANN 8.5.1).

`applies_to: soc=Ascend910 (910B2C, arch 220x); cann=8.5.1; bisheng=n/a; kernel_type=KERNEL_TYPE_MIX_AIC_1_1; op_class=mixed_cube_vec`
`verified_on: soc=Ascend910 (910B2C, arch 220x); cann=8.5.1 (gated_delta_rule CV-fusion rewrite, 2026-07-20)`
`unverified_on: soc=Ascend950PR (351x/A5); cann=9.x — the guard requirement is dispatch-semantics-level and expected to transfer, but barrier/ffts details differ (see OL-223/PB-45 for the arch35 sync family); validate on first A5 MIX port`

**Principle (the deadlock mechanism)**: under `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_*)` the SAME `__global__` image is scheduled on BOTH the cube (AIC) and vector (AIV) partitions. Any core-type-specific operation that is not compile-time-gated executes on the wrong core and wedges it **silently** — 100% AICore spin, no exception, no error code, host blocks in synchronize. The unguarded items that actually bit: `TQue<TPosition::A1/B1/A2/B2/CO1>` `AllocTensor/EnQue/DeQue/FreeTensor`, `TPipe::InitBuffer` for those queues (must be guarded too — initializing an A1 queue on AIV is itself illegal), `Mmad`/`LoadData`/`Fixpipe`, and conversely UB `TBuf`/VEC ops on the AIC path. Diagnosis shortcut: if a kernel is correct under `KERNEL_TYPE_AIC_ONLY` but hangs under `MIX_*`, the bug is launch-mode structure (missing guards / barrier misalignment), NOT the kernel math — do not bisect matmul parameters first (that was the initial misdiagnosis here).

**Guard pattern (concrete anchor)**:
```cpp
if ASCEND_IS_AIC { pipe_.InitBuffer(a1_, 1, SZ); /* A1/B1/A2/B2/CO1 only */ }
else             { pipe_.InitBuffer(ub_, UB_SZ); /* UB only */ }
__aicore__ inline void Process() {
    if ASCEND_IS_AIC { ProcessAIC(); } else { ProcessAIV(); }
    SyncAll<false>();              // AIC+AIV global barrier — `<false>` is mandatory
}
```

**Barrier alignment rules (all three required, each independently a deadlock)**:
1. `SyncAll<false>()` — the no-arg template defaults to `isAIVOnly=true` (AIV-only barrier); the AIC+AIV barrier requires the explicit `<false>`.
2. **Count parity** — the AIC path and AIV path must execute the SAME number of `SyncAll<false>()` calls in the SAME order (an extra call on one side deadlocks the fleet).
3. **Trip-count uniformity** — every core (AIC and AIV) must run the same loop trip count when barriers are inside a loop: use a padded loop (`slots = ceil(work/cores)`, out-of-range cores skip the work body via a `valid` flag but still reach every barrier).

**Pairing fact (probe-verified)**: under `MIX_AIC_1_1`, `GetBlockIdx()`/`GetBlockNum()` return identical values on the paired AIC and AIV (AIV core *i* partners AIC core *i*) — verified by having AIV write both values to GM and reading back (4 cores: idx 0..3, num 4). So one head→core assignment formula can be shared verbatim by both paths.

**Distinguish from the sibling MIX walls (different mechanisms, different fixes)**:
- **PB-57 (the ratio facet — complementary)**: on arch22 whole-device `SyncAll<false>()` REQUIRES the `MIX_AIC_1_1` 1:1 ratio; the default 1:2 dispatch gives AIC `GetBlockNum()` != AIV, so surplus AIV cores skip the barrier and the fleet hangs (507014). OL-278 covers the ORTHOGONAL facets (core-type guards + barrier count/trip alignment + AIV↔AIC pairing) that apply ONCE the ratio is 1:1.
- **`cv_lowering.md:42` (attention-perf scope — NOT a conflict)**: cv_lowering forbids `MIX_AIC_1_1` **for attention** (1 AIV = under-parallel — a PERFORMANCE rule, scoped to the seq×seq-softmax scoped-CrossCore-handshake attention path, 1:2, P-P116). That does NOT govern whole-device `SyncAll<false>()` ops (GDR / chunk-recurrent), where `MIX_AIC_1_1` is a CORRECTNESS requirement (PB-57; the 1:2 default deadlocks 507014). Different sync primitive → not contradictory. Pick by sync-primitive × op-class: attention / scoped-CrossCore → 1:2; chunk-recurrent / whole-device-SyncAll → 1:1. (cv_lowering.md:42 carries a mirror scope-note pointing back here.)
- **PB-35**: cube-internal `SetFlag<HardEvent>(event_t(0..3))` colliding with cross-core flag IDs — an event-ID allocation bug, not a guard bug.
- **OL-275**: hand-rolled cube under MIX with a STUB/no-op AIV self-poisons (silent wrong-result, CANN 9.1.0 standalone-KFC path) — a determinism bug, not a hang; and its `AIC_ONLY` mitigation applies when there is NO real vector work. The present entry covers the opposite regime: REAL AIC + REAL AIV work (true CV fusion) through the torch-op launch path (`EXEC_KERNEL_CMD`, which op-framework-bootstraps the FFTS descriptor — the same "managed bootstrap" OL-275 credits for its vendor-op determinism). 16/16 cases PASS + stable MERE here, consistent with that bootstrap theory.
- **OL-213**: forbids `SyncAll` in AIC-ONLY pipelines (zero-work cores never arrive) — scoped to no-MIX; in MIX the cross-core barrier is the intended primitive, subject to the alignment rules above.

**Evidence**: gated_delta_rule CV-fusion rewrite (2026-07-20, 910B2C / CANN 8.5.1): unguarded `MIX_AIC_1_0` → instant 100%-AICore deadlock; guarded `MIX_AIC_1_1` skeleton passed first try; full 8-matmul + vector-stage pipeline with 6 `SyncAll<false>()`/chunk → 16/16 precision PASS (MERE < 9.8e-4) and geomean 8.85× vs PyTorch reference. A barrier-count mismatch introduced mid-development (7 vs 6) reproduced the deadlock until realigned.

**Other instances (predicted)**: any hand-rolled MIX kernel (attention fwd/bwd, chunk-recurrent/linear-attention, conv+epilogue, matmul+fused-activation) on 910B-class chips; the probe for AIV↔AIC pairing generalizes to any `MIX_AIC_1_N` ratio bring-up.

**Cross-ref**: `cv_lowering.md` (the canonical AIC/AIV core-type guard skeleton this entry's diagnostics build on — see its §guard, not re-stated here; and its :42 attention-perf `MIX_AIC_1_1`-forbidden rule, scope-reconciled in the sibling-walls list above), PB-57 (MIX_AIC_1_1 1:1-ratio requirement for whole-device SyncAll — the ratio facet OL-278's guard/barrier facets complement), P-P117 (the a3 gated_delta_rule-fwd MIX kernel that runs on this guarded base), P-P120 (the Neumann cmatrixInitVal L0C-fold, in `a3_mix_small_matmul_cube.md`), OL-213 (AIC-only SyncAll prohibition), OL-275 (stub-AIV determinism wall), PB-35 (event-ID collision hang), P-P102 / cube_vector_fusion.md (canonical MIX structure incl. WorkspaceQueue alternative to global barriers).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-278（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
