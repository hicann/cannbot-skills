---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Tiled reduction+normalization via GM workspace (3-pass streaming) when the reduction span exceeds UB"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=normalization / sampling / long-vector-reduce unverified_on: soc=Ascend910_V220 (A3 — UB is 192KB not 256KB; tile budget must be re-check"
confidence: single_run
original_id: P-P107
timestamp_inferred: true
tags: [sort, optimization, div, max_v_ub, globalmax, globalsum, exp, p-p107, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=normalization / sampling / long-vector-reduce`
`unverified_on: soc=Ascend910_V220 (A3 — UB is 192KB not 256KB; tile budget must be re-checked, and DataCopyPad UB→GM is the V220-crashing EC-23 direction, not the free V351 path)`

**Problem**: A per-row normalization — softmax, L1/L2-normalization, online row-stats — needs a reduction (global max, then global sum) followed by an elementwise re-application (`div`) over a vector of length V. When V exceeds what fits in one UB allocation, a fixed `MAX_V_UB`-sized buffer silently overflows at V > cap: the elementwise pass writes/reads past the buffer, producing garbage output or a 507035 vector-core exception when later O(V)-shaped scalar loops walk the overflowing region.

**Pattern** — three streaming passes through a per-row GM workspace, carrying only two scalars (`globalMax`, `globalSum`) across tiles:

```
// V split into TILE_SIZE tiles; softmaxWkGm = V*sizeof(float) per row
Pass 1 (find global max):  per tile: DataCopy GM→UB → Cast fp32 → ReduceMax(calcIndex=false)
                           → keep running globalMax across tiles
Pass 2 (exp + sum, MATERIALIZE): per tile: load → Cast → Sub(globalMax) → Exp
                           → DataCopyPad exp → GM workspace (must persist for Pass 3)
                           → ReduceSum → accumulate running globalSum
Pass 3 (normalize):        per tile: DataCopyPad exp ← GM → Div(globalSum) → DataCopyPad → GM
```

Only the two scalar accumulators cross the tile boundary; the exp values are materialized to GM once (recomputing `Exp` in Pass 3 would require `globalMax` again — materializing once is cheaper and avoids a 4th pass). This is the standalone (non-matmul) form of the A3 upstream tiled-softmax idiom (`GetRowMax` → `SoftMaxFstCompute` → `SoftMaxSecCompute`), lifted to a GM-resident V.

**UB budget**: ~5 fp32 tiles (raw / fp32 / tmp / reduce-work / reduce-out) × TILE_SIZE×4B + one per-row GM workspace of V×4B.

**Concrete anchor** (top_k_top_p_sample `TiledSoftmaxToGM`, `top_k_top_p_sample_kernel.h:119-218`):
```cpp
// Pass 1 running max
ReduceMax(rmax, fp32, rwork, count, /*calcIndex=*/false);
SetFlag<HardEvent::V_S>(EVENT_ID0); WaitFlag<HardEvent::V_S>(EVENT_ID0);  // reduce→scalar read
if (rmax.GetValue(0) > globalMax) globalMax = rmax.GetValue(0);
// Pass 2: Sub(globalMax)→Exp→DataCopyPad to GM→ReduceSum → globalSum
// Pass 3: DataCopyPad from GM→Div(globalSum)→DataCopyPad to GM
```

**Pipe sync**: every tile transition fences the producer→consumer pipe pair — `MTE2_V` (load→compute), `V_MTE3` (compute→store), `MTE3_MTE2` (store→next-tile-load); reduce→scalar reads add `V_S`/`S_V` (generic VEC↔Scalar discipline: P-P64). UB↔GM `DataCopyPad` works on V351 (EC-23); on V220 the UB→GM direction crashes (EC-23) — use the pybind pre-pad shim.

**Evidence**: top_k_top_p_sample A3→A5 kw-5 V-tiling rewrite (2026-06-25, Ascend950PR_9579, CANN 9.0.0, workspace `top_k_top_p_sample_kernel.h` md5 a441279d0c56ea4f9959451ad34ff733): 32/32 T1 PASS — fp16 18/18 + bf16 14/14, V=256..32000 across all 4 algorithm branches (no-Q / A / B / C). Replaced a fixed `MAX_V_UB=2048` kernel that was 16/16 PASS at V=256 but garbage at V>4096 and 507035-crashed in Branch C at V≥8192 (O(V²) scalar loop on the overflowing UB buffer). Determinism best_effort, 3/3 cases bit-exact.

**Other instances (predicted)**: large-vocab classifier softmax (V=vocab > UB), long-context row normalization, nucleus/top-P sampling over large vocab, any L1/L2 row-normalization over V > UB. (FlashAttention uses the online-softmax variant with `delta` correction + running max-rescale — this is the simpler non-fused, non-matmul form for standalone normalization where the whole vector is reducible in one pass.)

**Stop condition**: if V fits in UB with room for all 5 buffers, do a single-pass in-UB softmax (no GM round-trip). The 3-pass GM form pays 3× GM traffic per element — only worth it when V genuinely exceeds UB.

**Related**: P-P109 (hardware Sort+cumsum nucleus finding over the GM-resident softmax this produces — the correct O(V log²V) selection for top-P / Branch-C), P-P108 (iterative GM selection — valid ONLY for bounded top-K (k≪V); ANTI-PATTERN for nucleus/top-P), P-P57 (UB-resident small-k selection — the V≤UB sibling), P-P64 (VEC↔Scalar fence), EC-23 (DataCopyPad UB↔GM on V351), P-P42 (hardware Sort when full sorted order, not just normalization, is needed).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/sort.md（P-P107，convert_patterns_to_okf.py）。confidence 未升格。 -->
