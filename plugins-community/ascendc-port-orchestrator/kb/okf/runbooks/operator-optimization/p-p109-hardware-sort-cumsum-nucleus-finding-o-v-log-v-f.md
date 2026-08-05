---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Hardware Sort + cumsum nucleus finding (O(V log²V)) for top-P / Branch-C sampling — the correct replacement for P-P108's O(V²) iterative selection"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=sampling / nucleus / topk / long-vector-reduce Problem: nucleus (top-P) sampling needs the smallest prefix of the descending-sorted proba"
confidence: single_run
original_id: P-P109
timestamp_inferred: true
tags: [sort, optimization, toppnum, top_k_top_p_sample, processbranchc_q, tiledqsampleongm, tiledargmaxongm, p-p109, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=sampling / nucleus / topk / long-vector-reduce`

**Problem**: nucleus (top-P) sampling needs the smallest prefix of the descending-sorted probability distribution whose cumulative sum first exceeds `p`. P-P108's iterative max-selection is O(V²) — it re-scans all V elements per selected token, and the prefix length `topPNum ≈ p·V` can approach V (measured 134 ms @ V=32000). The correct complexity is O(V log²V): sort once, then walk a cumsum.

**Pattern** — sort-then-cumsum, value+index carried together, merge while accumulating:

```
1. SEGMENT SORT (V → runs of sorted (value,index)):
   split the V softmax values into fixed chunks (~1024 elements each, a segment sized to one Sort
   pass in UB; tail-pad the last chunk with -inf);
   build a parallel index tensor [0,1,...,V-1] = original vocabulary index;
   per chunk: Concat(value, workbuf) then Sort<float, /*isDescend=*/true>(sorted, value, index, tmp)
              → (value,index) DESCENDING (largest probability first);
   store each chunk's sorted (value,index) run back to GM (paired/interleaved).

2. k-WAY MERGE + CUMSUM (find topPNum):
   running cumsum scalar `cs` (init 0);
   merge the sorted runs in descending order:
     ≤4 runs  → one MrgSort4Que pass with cumsum enabled;
     >4 runs  → iterative MrgSort/MrgSortMQue tree-merge (ping-pong GM buf A ↔ B) until ≤4 runs,
                then MrgSort4Que;
   on each merged tile emitted in descending order:
     GatherMask to de-interleave the (value,index) stream into separate value[] and index[] tensors;
     ReduceSum the value[] tile → increment `cs`;
     if cs > p: the nucleus boundary falls inside this tile — walk it element-by-element to find the
                exact first position where cs crosses p → that count is topPNum; STOP;
     else: emit the whole tile to sortValGm/sortIdxGm, continue merging.

3. EMIT: the prefix [0..topPNum) of the globally-descending-sorted stream is the nucleus:
   sortValGm[0..topPNum] = descending softmax values;
   sortIdxGm[0..topPNum] = matching ORIGINAL vocabulary indices;
   topPNum               = the cumsum cutoff count.
```

**Drop-in contract** (replaces ONLY the selection block): a consumer that expects `sortValGm[0..topPNum]` (descending values) + `sortIdxGm[0..topPNum]` (matching original indices) + a scalar `topPNum` can switch from P-P108's iterative loop to this Sort path with NO downstream change. For `top_k_top_p_sample` (`top_k_top_p_sample_kernel.h`), `ProcessBranchC_Q`'s post-selection code — `TiledSoftmaxOnGM(sortValGm,…)` → `TiledQSampleOnGM` → `TiledArgmaxOnGM` → `WriteIdx` — consumes exactly these three artifacts, so the Sort replacement swaps only the O(V²) selection loop (the `for(kk=0;kk<V)` block) and reuses everything after it.

**Sort API + staging** (public AscendC, V351): `Sort<T,isDescend>` over ~1024-element chunks; `MrgSort4Que` (≤4-way) and `MrgSort`/`MrgSortMQue` (>4-way tree-merge) for the merge; `GatherMask` to split the interleaved (value,index) merge output; `ReduceSum` for the per-tile cumsum increment; `DataCopyPad` for GM↔UB staging; `PipeBarrier<PIPE_V>` between Sort/MrgSort/GatherMask stages (they share the V pipe). Carry the original index as a `uint32_t` parallel tensor (Sort/MrgSort preserve the value↔index pairing); widen to `int64_t` only at write-out. GM workspace: one V×4B region for the softmax input (P-P107), plus two for the sorted (value,index) output — the same `B*3V` layout the iterative path already uses, so no pybind workspace widening is required to adopt this.

**Determinism**: descending `Sort` + deterministic k-way merge → a stable, unique nucleus prefix; per-row dispatch (one AIV per row, or strip-mined `row = bid; row < B; row += nblk`) with no atomicAdd and no cross-row reduction → deterministic by construction. **Tie-order caveat**: AscendC `Sort` ASC tie direction is reversed vs PyTorch stable (P-P60); for DESC nucleus sampling the impact is limited to equal-probability boundary tokens and is usually acceptable, but verify against your reference if exact tie indices matter.

**Evidence** (same NPU, same session, back-to-back — CLAUDE.md perf A/B rule; re-confirmed 2026-06-26, NPU 0, Ascend950PR, fp16, `perf_counter`+sync):
- Our **O(V²) iterative kernel** (P-P108, current `top_k_top_p_sample` Branch C): 8.0 / 10.0 / 17.0 / 28.0 / **134.3 ms** at V = 1024 / 4096 / 8192 / 12288 / **32000** (133.3 ms bf16 @ 32000) — clean quadratic scaling (`perf_branch_analysis.py`); cross-checked 127.9 ms @ V=32000 (`perf_vs_torchnpu.py`).
- **Sort+cumsum decomposition** (torch.npu `sort`(descending) + `cumsum` + `where` + 2nd `softmax` + `gather` + `argmax`, each op dispatching to CANN hardware Sort/reduce — the SAME algorithm realized as separate CANN calls): 0.46 / 0.26 / **5.98 ms** at V = 4096 / 8192 / **32000**.
- **Ratio @ V=32000 Branch C: ~21× (134 vs 6 ms); 38× @ V=8192; 11× @ V=4096** — gap widens with V (O(V²) vs O(V log²V)).
- Correctness: kernel Branch-C indices exactly match the sort+cumsum decomposition on a B=4 V=2048 sanity case (`[706,100,445,1290]`) → same algorithm, different realization.

**Honesty caveat**: the 5.98 ms is the **decomposed** sort approach (≈10 separate CANN op launches, each with per-launch overhead), so it is an *upper bound* on the Sort path's cost. A single **fused** AscendC Sort+cumsum kernel (the deferred re-impl, DEBT-169) would be faster still — widening the gap beyond 21×. First-party *our-AscendC-Sort* kernel timing is the DEBT-169 follow-up; this entry's 21× is the algorithmic lower bound proven by the decomposition.

**Stop condition / when to use**: use this for ANY nucleus/top-P/large-K selection where the count is unbounded (≈ p·V) or V≫UB. For bounded top-K (k≤1024, k≪V) the simpler P-P57/P-P108 iterative form is acceptable. If only a single argmax is needed (no top-K/top-P), collapse to one ReduceMax (OL-252).

**Related**: P-P108 (the deprecated O(V²) iterative anti-pattern this replaces), P-P107 (produces the GM-resident softmax this sorts), P-P42 (hardware Sort pipeline — the primitive layer), P-P60 (Sort ASC tie-break anti-pattern — verify tie order if exact indices matter), P-P82 (cross-AIV merge of sorted top-K buffers — if you parallelize the merge across AIVs), DEBT-169 (the fused AscendC Sort+cumsum re-impl of this op's Branch C).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/sort.md（P-P109，convert_patterns_to_okf.py）。confidence 未升格。 -->
