---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Global pooling (output_size=1) catastrophic performance — O(winVol) per-output-point vs CANN's O(1) integral-image approach"
description: "For global pooling (output_size scalar, window spans the whole spatial input), a per-output-point scan is O(winVol) and loses 39-286x to CANN's O(1) integral-image (prefix-sum) algorithm on large windows; integral-image is the correct algorithm, and a multi-core parallel spatial reduction narrows the gap to 5-10x."
original_id: OL-241
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [pooling, performance, ol-241, integral-image, prefix-sum, global-pooling, optimization]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**When to consult:** implementing or porting a pooling op (adaptive_avg_pool3d, adaptive_max_pool3d) that supports global pooling (`output_size=1` / `[1,1,1]`). `applies_to: soc=Ascend950PR; cann=9.1.T500; op_class=pooling`. `verified_on: cann=9.1.T500 (adaptive_avg_pool3d, 2026-06-22/23)`.

### Principle

When `output_size=1` or `[1,1,1]` (global pooling), the window spans the entire spatial input — `winVol = inD*inH*inW`. A straightforward per-output-point kernel traverses ALL input elements inside the window for EACH output point, yielding **O(winVol) per-output** complexity. CANN's native `F.adaptive_avg_pool3d` uses an **integral-image (prefix-sum) algorithm** achieving **O(1) per-output-point** regardless of window size: precompute cumulative sums, then compute any window average as `(sum[end] - sum[start]) / winVol`.

The gap grows with window volume (CANN reference stays ~17-24 us, confirming O(1)):

| winVol | Our O(winVol) | CANN O(1) | Ratio |
|--------|--------------|-----------|-------|
| 343 (7^3 pool) | 69.8 us | 19.2 us | 3.6x |
| 2,048 (32^3/ch=512) | 64.5 us | 16.8 us | 3.8x |
| 32,768 (32^3, batch=2) | 890.0 us | 22.7 us | 39x |
| 262,144 (32x64x128, output=[1,1,1]) | 6,784.9 us | 23.7 us | 286x |

### Decision rule

- **Pooling ops where window size >= 100 elements AND output_size is a scalar (no spatial dims)** -> integral-image is the correct algorithm. A per-output-point scan is categorically wrong.
- **Non-global pooling** (output_size has spatial dims >= 2) -> O(winVol) per-output-point is acceptable, though loop reordering (OL-249) can amortize traversal.
- When porting from V220/A3 source: check whether the source already has an integral-image variant. If not, the CANN native kernel will dominate on global-pooling shapes — document this as a known performance gap rather than repeatedly optimizing the wrong algorithm.

### Concrete anchor

```cpp
// WRONG for global pooling (output_size=1, winVol=32K):
//   for each (id,ih,iw) in window: sum += input[id][ih][iw]  // 32K iters per output point
//   output = sum / winVol
//
// RIGHT (integral-image approach):
//   prefixSum[d][h][w] = prefixSum[d-1][h][w] + prefixSum[d][h-1][w] + ...
//   window_sum = prefixSum[end] - prefixSum[start_range]   // O(1) per window
//   output = window_sum / winVol
```

### Mitigation: MODE_GLOBAL v3 — multi-core parallel spatial reduction (2026-06-22)

While integral-image is the correct long-term solution, a **multi-core parallel spatial reduction** reduces the 39-286x gap to **5-10x** (a 6-31x improvement) without changing the algorithm fundamentally.

## 证据
- adaptive_avg_pool3d (2026-06-22 v2->v3 multi-core spatial reduction: 39-286x -> 5-10x gap; 2026-06-23 host-cache + convenience-wrapper: overall 0.585 -> 0.715x vs CANN native, 50-case wall-clock).
