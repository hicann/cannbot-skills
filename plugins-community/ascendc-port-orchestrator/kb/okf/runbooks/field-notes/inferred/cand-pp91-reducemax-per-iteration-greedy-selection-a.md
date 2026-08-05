---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "ReduceMax-per-iteration greedy selection as sort-replacement under tight UB budget"
description: "applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=greedy_selection verified_on: soc=Ascend910_9382; cann=9.0.0-beta.2 unverified_on: soc=Ascend950PR_9589 (has 256KB UB — might fit full"
phenomenon: build_failure
signal:
  - "op whose reference algorithm is \"sort by score DESC → iterate greedy over sorted list → suppress IoU≥threshold neighbors\". When N_max × sizeof(dtype) > UB / 2,"
confidence: inferred
status: stub
original_id: CAND-PP91
timestamp_inferred: true
tags: [candidate, inferred, num_selected, cand-pp91]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220; cann=9.0.0; bisheng=n/a; op_class=greedy_selection`
`verified_on: soc=Ascend910_9382; cann=9.0.0-beta.2`
`unverified_on: soc=Ascend950PR_9589 (has 256KB UB — might fit full sort for larger N; greedy selection may still be preferable for tile-locality reasons independent of UB pressure)`

**Trigger**: op whose reference algorithm is "sort by score DESC → iterate greedy over sorted list → suppress IoU≥threshold neighbors". When `N_max × sizeof(dtype) > UB / 2`, pre-sorting the full element list is infeasible — the sort workspace alone exceeds the UB budget.

**Recommendation**: replace the pre-sort with a ReduceMax-per-iteration greedy loop. Each iteration: (1) tiled ReduceMax over persistent UB scores buffer to find the current maximum-score element, (2) compute pairwise IoU against that element for all candidates in tiled chunks, (3) suppress candidates whose IoU ≥ threshold by setting their scores to -inf in the persistent buffer. No sort workspace needed — total UB = persistent scores buffer (N_max × sizeof(dtype)) + IoU compute tile (~2KB per chunk). For V220 (192KB UB) with fp32 N≤32768, this uses ~160KB (83% utilization).

**Concrete anchor** (canonical V220 NMS inner loop):
```cpp
// Per-iteration: find max-score element via tiled ReduceMax
float best_score = -INFINITY;
int   best_idx   = -1;
for (int tile = 0; tile < num_tiles; ++tile) {
    auto tile_max = WholeReduceMax(scores[tile], scores[tile], tile_len, 1);
    float tile_best;  tile_max.GetValue(0, tile_best);
    if (tile_best > best_score) { best_score = tile_best; /* track idx */ }
}
if (best_score < score_threshold) break;  // early exit
// IoU suppress: for each tile, compute IoU vs best box, mask scores to -inf
```

**Single-AIV determinism**: with `nblk=1` (single-AIV execution), strict `>` across tiles and linear-forward in-tile scan gives deterministic output matching PyTorch stable-sort tie-break semantics (lowest index wins on tie). No cross-core communication, no atomicAdd.

**Promote when**: a second greedy-selection op on V220 (or another platform with tight UB) confirms the ReduceMax-per-iteration approach independently. Candidate promotion candidates include: top-K with dynamic K (where K varies per row and full-sort is wasteful), iterative beam search, WBF (weighted boxes fusion).

**Evidence**: op#30 NMS a3 ds kw-1 (2026-05-07, Ascend910_9382 V220, CANN 9.0.0-beta.2). N_max=32768 fp32 — pre-sort needs ~256KB (exceeds 192KB UB). ReduceMax-greedy uses ~160KB UB (83% utilization). Pass B vs Python CPU reference: 31/31 bit-exact (set-equivalence comparison on `selected_indices[:num_selected]` + bit-exact `num_selected`). Single-AIV, deterministic by construction.

**Source**: op#30 NMS a3 ds kw-1 (2026-05-07). 1-op evidence.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP91，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
