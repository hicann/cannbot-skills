---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "da Vinci VEC throughput tiers — Muls (Tier 0) is 8-16× faster than Div (Tier 4)"
description: "fp32 VEC throughput runs from Muls 8-16 elem/cyc (Tier 0) down to Div(vec,vec) 0.5-1 (Tier 4); count Tier-3/4 ops and PipeBarriers per row and minimize — normalize with Muls(x,1/rms), not Divs."
phenomenon: perf_regression
signal:
  - "a SIMD kernel's per-row inner loop uses Divs/Div/Sqrt or emits >5 PipeBarrier<PIPE_V> per row for N≤256"
confidence: single_run
original_id: OL-257
classified_by: llm-assisted
timestamp_inferred: true
tags: [vec-throughput, perf-model, ol-257, davinci, pipebarrier]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发
A SIMD kernel is slower than expected and its per-row inner loop contains divide/sqrt ops or many pipeline barriers. da Vinci VEC instructions differ by up to ~16× in throughput, so the count of expensive ops per row directly bounds perf.

Approximate fp32 VEC throughput tiers:

| Tier | Operations | Approx elem/cycle | Per-row guidance |
|---|---|---|---|
| Tier 0 (fastest) | `Muls(vec, scalar)` | 8-16 | use wherever possible |
| Tier 1 (standard) | `Add`, `Mul`, `Sub`, `Cast` | 4-8 | normal vector work |
| Tier 2 (moderate) | `Abs`, `Max`, `Min`, `Relu` | 2-4 | acceptable |
| Tier 3 (expensive) | `Divs(vec, scalar)`, `Sqrt` | 1-2 | limit to ≤1 per row |
| Tier 4 (avoid) | `Div(vec, vec)` | 0.5-1 | avoid in perf paths |
| Fixed cost | `PipeBarrier<PIPE_V>` | ~10-50 cyc drain | target ≤5 per row for N≤256 |

## 根因 / 教训
Decision rule:
1. Count Tier-3/4 ops per row. If >1, look for conversion opportunities (see OL-256: `Divs(x,rms)` → `Muls(x,1/rms)` is precision-safe in fp32).
2. Count `PipeBarrier<PIPE_V>` per row. If >5 for N≤256, see OL-245 (regbase eliminates per-op barriers between chained VEC ops).
3. A normalization step (`x / rms`) should be `Muls(x, 1/rms)`, not `Divs(x, rms)`.

Concrete anchor:
```cpp
// EXPENSIVE (~400-800 cycles for N=256):
Divs(work, work, rms, N);         // Tier 3
PipeBarrier<PIPE_V>();
Div(work, work, scaleFp32, N);    // Tier 4

// CHEAP (~16-32 cycles for N=256):
float invRms = 1.0f / rms;
Muls(work, work, invRms, N);      // Tier 0
```

Evidence: add_rms_norm_quant V1→V2 A/B and selective_scan fwd perf-loop iters 1-4 (Ascend950PR_957b, 2026-06-23/24). Cross-refs: OL-256 (the Tier-0 substitute for Tier-3), OL-245 (regbase / barrier elimination), OL-258 (TQue double-buffering hides MTE latency).
