---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Loop reordering for pooling — traverse (id,ih) once per (n,od,oh) group, not once per output point"
description: "Assign one block per (n,od,oh) group so all outW output points share one (id,ih) input traversal; benefit scales with outW. 3.39× over a per-output-point TQue kernel for adaptive_avg_pool3d."
confidence: single_run
original_id: OL-249
classified_by: llm-assisted
timestamp_inferred: true
tags: [pooling, optimization, ol-249, loop-reordering, window-reduction]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**When to use:** any window-based reduction with multiple output points (adaptive_avg_pool3d, adaptive_max_pool3d, similar pooling ops) where several output points share the same `(n,od,oh)` prefix, i.e. `outW ≥ 2`.

**The problem.** A naive kernel assigns one block per output point `(n,od,oh,ow)`. Each block independently traverses its `(id,ih,iw)` window, so the same input row `(n,id,ih,:)` is loaded once **per `ow`**. For `outW > 1` the input row is loaded `outW`× redundantly.

**The transform.** Change block assignment from "per output point" to "per `(n,od,oh)` group". All `outW` output points sharing that prefix are computed by ONE block that traverses `(id,ih)` once, distributing each loaded column to all overlapping `ow` accumulators. Key design elements:
1. **Flat accumulator buffer** `sumAll[outW * cAlign]` — all `outW` partial sums in one contiguous UB buffer.
2. **Per-(id,ih): load the full inW row** (multi-column tiling) then **distribute** each `iw` column to the `ow` accumulators whose window covers `iw`.
3. **After the (id,ih) loop:** `Muls(1/winVol)` normalize → `pipe_barrier(PIPE_ALL)` (OL-247) → `DataCopy` write all `outW` results.
4. **Block distribution:** `N × outD × outH` blocks (not `N × outD × outH × outW`).

**Decision rule / tradeoffs:**
- `outW ≥ 2`: loop reordering wins; benefit scales `outW`× (the `(id,ih)` traversal reduction factor).
- `outW = 1` (single-element output / `output_size` spatial dim = 1): loop reordering is slightly WORSE — the flat buffer + Duplicate-zero overhead has no traversal reduction to amortize. Keep per-output-point blocking.
- **Double-buffering** adds 11-28% on top for large work units, but for loop-reorder kernels the per-`(id,ih,tile)` work is dominated by the inner `ow` Add-distribution loop, so its benefit is marginal.
- **Multi-column tiling** (loading `dw = we−ws` contiguous columns per DataCopy) should be combined with loop reordering — it cuts DataCopy calls per `(id,ih)` from `inW` to `ceil(inW / maxTileCols)`.

### Concrete anchor
```cpp
// One block per (n, od, oh); all outW accumulators in a flat buffer:
//   sumAll[(ow * cAlign) + c] holds the partial sum for output point (n,od,oh,ow), channel c
Duplicate<float>(sumAll, 0.0f, owCount * cAlign);   // zero once per group
for (id = ws_d .. we_d) {
    for (ih = ws_h .. we_h) {
        for (tileStart = 0; tileStart < inW; tileStart += maxCols) {   // load full inW row
            cols = min(maxCols, inW - tileStart);
            // DataCopy the tile, then distribute each iw column to overlapping ow accumulators
        }
    }
}
```

### Evidence
adaptive_avg_pool3d optimization (Ascend950PR, CANN 9.1.T500, 2026-06-17): **3.39× improvement over the original per-output-point TQue kernel.**

### Related
- OL-247 (pipe_barrier(PIPE_ALL) between VEC normalize and the DataCopy write).
