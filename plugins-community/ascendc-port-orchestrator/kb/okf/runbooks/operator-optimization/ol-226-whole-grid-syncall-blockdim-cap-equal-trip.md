---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Whole-grid SyncAll requires all blocks co-resident — cap blockDim at min(work, cores) and tile extra work into a per-block loop with no-op-guarded slots so every block makes the SAME SyncAll count"
description: "Whole-grid SyncAll needs all blocks co-resident: cap blockDim at min(work, cores) and give every block the same trip count with no-op-guarded slots so SyncAll counts match."
confidence: single_run
original_id: OL-226
classified_by: llm-assisted
timestamp_inferred: true
tags: [sync-correctness, optimization, ol-226, syncall, blockdim, deadlock]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Applies to**: Ascend950PR / CANN 9.1.T500 / all op classes using a whole-grid SyncAll barrier. Verified on GDN chunk_gated_delta_rule (catlass), 2026-06-16.

**Principle**: `AscendC::SyncAll()` is a whole-grid barrier — **every launched block must reach the same SyncAll for any to proceed**. Two distinct failure modes:

1. **Over-subscription deadlock**: if `blockDim > physical AIC cores`, the runtime schedules blocks in waves; wave-2 blocks have not started when wave-1 blocks hit the barrier → wave-1 waits for blocks that are not co-resident → hang. Fix: cap `blockDim = min(work_items, CORES)` on the host.
2. **Count-mismatch deadlock**: once work exceeds the capped block count, the extra work is tiled into a **per-block loop** (each block handles a strided set of items). If that loop's trip count differs across blocks (e.g. `ceil(N/grid)` items but the last block has fewer), the blocks make **different numbers of SyncAll calls** → barrier N on one block pairs with barrier N+1 on another → hang. Fix: give **every block the SAME trip count** (`ceil(N/grid)`), and **no-op-guard** the slots where `item >= N` (execute the loop iteration but skip the work) so the SyncAll count stays equal across all blocks.

**Decision rule**: using whole-grid `SyncAll` AND `work_items > CORES`: cap `gridDim = min(work_items, CORES)`; loop `headsPerBlock = ceil(work_items / gridDim)` iterations per block; mark `active = (item < work_items)` and guard every compute step with it, but **never** skip the loop iteration or the SyncAll itself. Prefer this over `blockDim = work_items` (which over-subscribes) and over a `break`-early loop (which mismatches counts).

**Concrete anchor**:
```cpp
// blockDim capped at min(Nv, CORES) on host; each block processes a STRIDED set of heads.
// ALL blocks execute the SAME headsPerBlock iterations (slots with head>=Nv are no-op-guarded)
// so SyncAll counts match.
const uint32_t headsPerBlock = (Nv + gridDim - 1) / gridDim;   // ceil — same for every block
for (uint32_t hs = 0; hs < headsPerBlock; ++hs) {
  const uint32_t n = baseHead + hs * gridDim;
  const bool headActive = (n < Nv);                            // no-op-guard, do NOT skip iter
  /* ... compute guarded by headActive ... */
  AscendC::SyncAll<false>();                                   // every block hits this exactly headsPerBlock times
}
```

**Evidence**: GDN `chunk_gated_delta_rule` on catlass (A5/V351, CANN 9.1.T500, 2026-06-16): one head per block. For `Nv >= 32` (> physical cores), launching `blockDim = Nv` deadlocked at the inter-phase `SyncAll`. Head-tiling (`gdn.cpp:46-52, 70-76`) capped `gridDim = min(Nv, CORES)`, looped `ceil(Nv/gridDim)` head-slots per block, and no-op-guarded `head >= Nv` — resolving the hang.
