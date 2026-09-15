---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "High-level reduce/broadcast primitives run on arch versions absent from their public arch-guard — they fall through to the common implementation; verify by compiling, don't avoid them on an \"unlisted\" SoC"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-230
timestamp_inferred: true
tags: [ascendc, ol-230]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.1.T500`

**Principle.** Some high-level AscendC library primitives carry a public arch-guard whose explicit list does NOT name every SoC they actually support. When the running arch is absent from the listed set, dispatch falls through to the **common implementation** — so the primitive still compiles and runs correctly. An "unlisted" arch in a guard is therefore NOT evidence the primitive is unavailable: confirm by compiling against the target's headers, not by reading the guard list. Avoiding such a primitive on a wrongly-assumed "unsupported" SoC forces a hand-rolled scalar fallback that leaves the vector pipe idle.

**Concrete anchor (verified on Ascend950PR / `__DAV_C310__`, CANN 9.1.T500):**
- `Sum<T, reduceDim>(dstLocal, srcLocal, tmpLocal, SumParams{outter, inner, n})` — high-level batched reduction. Header `lib/reduce/sum.h`; the common-impl it falls to is `sum_common_impl.h:40`. The public arch-guard list does not name c310, yet it builds + runs.
- `Broadcast<T, 2, 1>(...)` — same story (compiles + runs on c310 via common-impl).

**Why it matters.** `Sum` is the primitive for **batched per-block reductions** that unblock serial-scan / per-row-reduction kernels: replacing per-element `V→S→V ReduceSum` round-trips (scalar-pipe-bound) with one batched `Sum<float>` moves the work onto the otherwise-idle vector pipe.

### Evidence
- selective_scan backward SIMD, 2026-06-19 (Ascend950PR_957b, CANN 9.1.T500, msprof device-time): swapping per-l `ReduceSum` round-trips for one batched `Sum<float>` (+ vectorized combine) raised `aiv_vec_ratio` 0.631 → 0.982 and cut device-time **2.47×** (large fp32 2653 → 1074 µs), precision unchanged 30/30 vs fp64 autograd.
- selective_scan **forward** SIMD, 2026-06-19 (same env): the SAME lever transfers — an N-wide vector scan storing aligned `xall[l*N]` + one batched `Sum<float>[L,N]→[L]` raised `aiv_vec_ratio` **0.044 → 0.975** and cut device-time **3.80×**, making the forward SIMD **beat the production SIMT 1.27–1.62×** with identical precision. The forward gain was larger because the baseline was more scalar-bound (vec pipe 95.6% idle).

### Other instances (predicted)
- Any serial-scan / per-row-reduction SIMD kernel on A5 that currently collects `ReduceSum` per step in the scalar pipe (cumulative ops, RNN / linear-recurrence backward, attention row-reductions) — batch the reduction with `Sum`.
- Any kernel needing `Broadcast` (or another high-level reduce/select primitive) on A5 where the arch-guard list omits c310 — try compiling before falling back to a manual loop.

### Cross-references
- OL-20 (msprof vec_ratio reading), OL-225 / OL-226 (UB coherence / SyncAll co-residency), OL-103 / OL-109 (per-dtype precision tiers the optimized kernel must still hold).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-230（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
