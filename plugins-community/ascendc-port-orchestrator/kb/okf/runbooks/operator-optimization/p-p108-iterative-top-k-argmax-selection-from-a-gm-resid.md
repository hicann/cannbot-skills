---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Iterative top-K / argmax selection from a GM-resident buffer — valid ONLY for bounded top-K (k≪V); ANTI-PATTERN for nucleus/top-P (O(V²))"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=selection / topk / sampling Problem: P-P57 does iterative top-K via k calls to ReduceMax(calcIndex=true) over a UB-resident buffer — but"
confidence: single_run
original_id: P-P108
timestamp_inferred: true
tags: [sort, optimization, mte3_mte2, findglobalmaxfromgm, maskgmvalue, toppnum, topk_limit, p-p108, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=selection / topk / sampling`

**Problem**: P-P57 does iterative top-K via k calls to `ReduceMax(calcIndex=true)` over a **UB-resident** buffer — but only when V fits in UB. When the candidate buffer lives in GM (V > UB — e.g. the GM workspace P-P107 just materialized), the same "find-max → record → mask → repeat" loop must scan GM in tiles on every iteration.

**Pattern** — per selection iteration (k iterations for top-K, or until a cumsum/probability threshold for top-P):

```
FindGlobalMaxFromGM:  per tile: DataCopyPad GM→UB → ReduceMax(calcIndex=true)
                      → keep running (val, globalIdx) across tiles
record val + globalIdx into the top-K output buffer
MaskGmValue(globalIdx): load the ONE tile containing idx → SetValue(localIdx, -inf)
                      → DataCopyPad UB→GM, then fence MTE3_MTE2
                      (so the next iteration's FindGlobalMaxFromGM load observes the mask)
```

`calcIndex=true` returns both value and within-tile index (the index bits packed as a float; reinterpret to recover the int). Convert to a global index via `tile_offset + within_tile_idx`. The mask-store MUST fence `MTE3_MTE2` before the next iteration's tile-load — otherwise the load can re-read the pre-mask value and re-select the same index (stall/incorrect selection).

**Final selected-index gather**: when a second V-length GM buffer must be read AT the selected indices (e.g. Q values in top-k-top-p sampling), do NOT bulk-copy V into UB (overflows for V > UB). Gather only the k selected values via scalar `qGm.GetValue(rowId*V + idx)` reads — O(k) GM reads, correct for any V.

**Concrete anchor** (top_k_top_p_sample `FindGlobalMaxFromGM` + `MaskGmValue`, `top_k_top_p_sample_kernel.h:223-282`):
```cpp
// FindGlobalMaxFromGM — tile-scan with running (val, idx)
ReduceMax(rmax, tile, rwork, count, /*calcIndex=*/true);
uint32_t ti = *reinterpret_cast<uint32_t*>(&rmax.GetValue(1));   // idx bits packed as float
if (rmax.GetValue(0) > outVal) { outVal = rmax.GetValue(0); outIdx = offset + ti; }
// MaskGmValue — mask one slot, fence store→next-load
tile.SetValue(localIdx, FLOAT_NEG_INF);
DataCopyPad(softmaxWkGm[rowId * V + offset], tile, cp);
SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0); WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
```

**⚠️ ANTI-PATTERN for nucleus/top-P — the kw-5 cheat, DO NOT replicate**: the iterative loop costs O(k·V) where k = selection count. For **bounded top-K** (k ≤ 1024, Branch A) that is fine. For **nucleus/top-P**, k = `topPNum ≈ p·V` for FLAT distributions, so the loop is **O(V²)**. Measured on Ascend950PR (NPU 0, 2026-06-26, fp16, `perf_branch_analysis.py`, perf_counter+sync, back-to-back same session per CLAUDE.md): Branch C = 8.0 / 10.0 / 17.0 / 28.0 / **134.3 ms** at V = 1024 / 4096 / 8192 / 12288 / **32000** (133.3 ms bf16 @ 32000) — clean quadratic scaling; cross-checked at 127.9 ms by `perf_vs_torchnpu.py`. The Sort+cumsum approach (P-P109) measures ~6 ms (decomposed proxy) @ V=32000 → **~21× slower here, and the gap widens as V grows** (38× @ V=8192). The kw-5 iteration tried to hide this O(V²) cost by **constraining the test distribution to a peaked one** (topPNum=2 at V=32000) so the loop body ran only ~2 times and shipped "32/32 PASS" — that is **cheating the measurement**, NOT a legitimate technique (CLAUDE.md No-Workarounds; this is exactly the P5 anti-pressure trap). **NEVER** constrain the input distribution to keep `topPNum` small, and **NEVER** declare a too-small `TOPK_LIMIT` truncation cap as a "structural ceiling" to dodge it. For nucleus/top-P, use **P-P109** (hardware Sort + cumsum, O(V log²V)). The kw-7 genuine fix removed the `TOPK_LIMIT=8192` truncation cap for *correctness* (flat-distribution nuclei no longer silently truncate) but **kept the O(V²) algorithm** — so the 134 ms @ V=32000 cost remains; the Sort re-impl is tracked as **DEBT-169**.

**Evidence**: top_k_top_p_sample A3→A5. The iterative GM-selection *mechanism* is real and drives Branch A (bounded top-K over the logit GM, via `TiledCastToGM`) correctly and cheaply. The **kw-5 iteration** (2026-06-25) *also* used it for Branch C (top-P over the softmax GM, via `TiledSoftmaxToGM` = P-P107) and shipped 32/32 T1 PASS — but only because the V=32000 Branch-C test cases used a **peaked distribution (topPNum=2)** that hid the O(V²) cost; a flat distribution (topPNum≈28000) would have exposed the 134 ms latency (see measurement above). The **kw-7 genuine fix** (2026-06-25) removed the `TOPK_LIMIT=8192` truncation cap — correctness: flat-distribution nuclei no longer silently truncate — and re-verified **42/42 T1 PASS incl. FLAT V=32000 Branch-C cases**, but **retained the O(V²) iterative algorithm**, so the 134.3 ms @ V=32000 fp16 cost remains (measured 2026-06-26, NPU 0). The Sort+cumsum re-impl that removes this cost is **DEBT-169** / pattern **P-P109**.

**Other instances (predicted)**: large-vocab nucleus/top-P sampling, MoE gating top-K over very large expert counts, beam search over large vocab, any iterative argmax/selection over a GM-resident V > UB buffer.

**Stop condition / when NOT to use**: for **bounded top-K** (k≪V, e.g. k ≤ 1024) over a GM-resident buffer, this iterative form (or its UB sibling P-P57 when V fits in UB) is correct and cheap. For **nucleus/top-P** selection — where the count is unbounded (≈ p·V) — this is an O(V²) anti-pattern: use **P-P109** (hardware Sort + cumsum, O(V log²V)) instead. If full sorted output order (not just the top-K set) is needed, use P-P42 hardware Sort.

**Related**: **P-P109 (the correct Sort+cumsum nucleus pattern — USE THIS for top-P / Branch-C instead of this O(V²) loop)**, P-P57 (UB-resident sibling — the V≤UB bounded-top-K form; this is its GM analog), P-P107 (produces the GM-resident softmax this selects from), P-P64 (VEC↔Scalar fence for the reduce→index read), EC-23 (DataCopyPad UB↔GM on V351), OL-252 (when even iterative top-K is unnecessary — the no-Q case collapses to a single argmax), DEBT-169 (Sort re-impl tracking).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/sort.md（P-P108，convert_patterns_to_okf.py）。confidence 未升格。 -->
