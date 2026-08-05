---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Cap scan/recurrence chunk length CH = min(L, LN_CAP/N) so the per-chunk UB working set stays bounded as state width N grows"
description: "Per-chunk UB buffers sized N×CH overflow UB at large N if CH is fixed; make CH N-adaptive with LN_CAP the proven N×CH (4096): N=16→256, N=64→64. Guard non-mult-of-8 N fail-loud."
confidence: single_run
original_id: OL-253
classified_by: llm-assisted
timestamp_inferred: true
tags: [tiling, optimization, ol-253, scan, ub-management]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**When to use:** an L-chunked scan/recurrence whose per-chunk UB buffers (`[l*N+n]` layout) are sized `N * CH`, so the working set scales with `N * CH` and a chunk length tuned for one state width breaks at larger N.

**The problem.** A `CH` tuned for a small state width — e.g. `CH=256` at `N=16` → `N*CH=4096`, the proven UB-fitting point — **overflows UB (error 507035)** at larger N if left fixed: `N=64, CH=256` → 16384 × ~12 buffers × 4B ≈ 789 KB ≫ 248 KB UB.

**The fix.** Make `CH` N-adaptive: `CH = min(L, max(1, LN_CAP / N))` with `LN_CAP` = the proven UB-fitting `N*CH` (e.g. 4096). Then N=16 → 256 (unchanged, no regression for the original customer), N=32 → 128, N=64 → 64. More (smaller) chunks at larger N, but each cross-chunk carry is the EXACT recurrence state, so correctness is preserved. Pair with a `dstate % 8` (granule) host guard that **fails LOUD** rather than 507035-crashing on a non-granule N.

### Concrete anchor
```cpp
constexpr int32_t LN_CAP = 4096;
int32_t chCap = LN_CAP / N_;
if (chCap < 1)          chCap = 1;
if (chCap > CHUNK_MAX)  chCap = CHUNK_MAX;
CH_ = (L_ < chCap) ? L_ : chCap;
```

### Evidence
selective_scan fwd-SIMD + bwd-SIMD (Ascend950PR, CANN 9.1.T500, bisheng=AIV, 2026-06-24, PR #52/#53). Fixed `CH=256` overflowed at N=64; N-adaptive CH → N∈{8..64} build + run within UB, customer N=16 unchanged. Non-mult-of-8 N (17/33) guarded fail-loud (full support would need an Npad layout; realistic Mamba dstate is a multiple of 8).

### Other instances (predicted)
Any tiled op whose UB footprint scales with a runtime dim the tiling was first tuned for at one value — chunked attention (head-dim), tiled normalization (feature-dim), any `[tile * D]` UB layout.
