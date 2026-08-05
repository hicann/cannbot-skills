---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Manual prefetch for MTE2-VEC overlap when per-tile compute is light"
description: "On light-compute tile-loop kernels, TQue auto-prefetch fails to overlap MTE2 with VEC; issue tile (t+1)'s GM load before consuming tile t (explicit async-issue prologue+per-iter). Not the same as bumping IN_QUE_DEPTH."
confidence: single_run
original_id: OL-115
classified_by: llm-assisted
timestamp_inferred: true
tags: [pipeline-overlap, optimization, ol-115, prefetch, mte2-vec]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**When to use:** a tile-loop kernel where per-tile VEC compute is light (~4-8K cycles for elementwise + a simple reduce on 4096 elements), `IN_QUE_DEPTH=2`, and msprof shows `aiv_mte2_ratio < 0.5` (MTE2 idle exposed) and `aiv_vec_ratio < 0.75` (VEC under-utilized). Loaded by aog-kernel-optimizer (honest A/B perf below target, aiv_vec_ratio < 0.7) or by aog-kernel-worker when writing a thin-compute tile loop from scratch.

**Pattern.** Automatic queue-based prefetch (TQue with `EnQue` after issue) is INSUFFICIENT to overlap MTE2 with VEC when per-tile compute is short. Fix with **explicit async-issue prefetch**: issue tile (t+1)'s GM load BEFORE consuming tile t, structured as prologue + per-iter issue-then-consume.

**Important distinction (tradeoff).** This is NOT the same as bumping `IN_QUE_DEPTH=4`. On light-compute kernels a deeper queue regresses perf — see the counter-example 29_DynamicQuant ko-1 Opt3 REVERT (+7% perf loss). A deeper queue alone doesn't help if issue+consume stays sequential within the iter.

### Concrete anchor (per-row tile-loop skeleton)
```cpp
// PROLOGUE: issue tile 0's load before entering the loop
xQue.AllocTensor<T>();
DataCopy[Pad](localBuf_0, gm[r * stride], ...);  // tile 0
xQue.EnQue(localBuf_0);

for (int32_t t = 0; t < tile_count; ++t) {
    // ASYNC ISSUE: tile (t+1)'s load BEFORE consuming tile t
    if (t + 1 < tile_count) {
        xQue.AllocTensor<T>();
        DataCopy[Pad](localBuf_next, gm[r * stride + (t+1) * TILE], ...);
        xQue.EnQue(localBuf_next);
    }
    // CONSUME: tile t (its load was issued in the previous iter / prologue)
    LocalTensor<T> in_t = xQue.DeQue<T>();
    Cast(workBuf, in_t, RoundMode::CAST_NONE, cnt);
    Abs(tmpBuf, workBuf, cnt);
    ReduceMax<float>(maxAccum[t * 8], tmpBuf, scratch, cnt_aligned, false);
    xQue.FreeTensor(in_t);
    // the issue for (t+1) overlaps this VEC compute
}
```
The same pattern applies to Pass-2 emit loops (load tile (t+1) for quantize while emitting tile t).

### Evidence
- 29_DynamicQuant ko-2 (2026-05-02): applied to both Pass 1 and Pass 2 of the two-pass tile loop. msprof case 12 [4096, 11008] fp16:
  - BEFORE (ko-1 Iter4): aiv_vec_ratio=0.700, aiv_mte2_ratio=0.480, duration 227 µs
  - AFTER (ko-2 Iter1): aiv_vec_ratio=**0.861** (beats CANN 0.82), aiv_mte2_ratio=**0.622**, duration **182 µs** (-20%)
  - Honest A/B mean 0.339x → 0.609x (+79.6%); crossed the 0.6x threshold in a single iter
  - Precision floor maintained (Pass A 42/42 + Pass B 11/11 OL-83); bit-exact count 3 → 7

### Other instances (predicted)
- Any thin-compute per-row tile-loop: softmax (Exp+Sum+Div), layer norm (Sum+Sum²+scale), L2-norm (Sum²+sqrt+scale), elementwise quant (Cast+Muls+clamp).

### Related
- OL-114 (two-pass tile loop — the structure this prefetch is commonly applied to)
- OL-63 (TQue depth default)
