---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "For deeply memory-bound (OI<1) kernels, optimize data movement not compute"
description: "When a kernel is deeply memory-bound (OI < 1), spend the budget on reducing GM reads and overlapping MTE2/VEC, not on compute optimizations — the VEC unit is not the bottleneck."
confidence: single_run
original_id: ROOFLINE_MODEL.md#ridge-point-sg-memory-bound
classified_by: llm-assisted
timestamp_inferred: true
tags: [roofline, optimization, memory-bound, mte2, tque-prefetch]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

For kernels with OI < 1 (deeply memory-bound, e.g. SG kernels), direct all optimization effort at data movement:

1. **Reduce GM reads** — data reuse, loop reorder.
2. **Maximize MTE2/VEC overlap** — TQue prefetch so compute hides behind loads.
3. **Do NOT chase compute optimizations** — the VEC unit is not the bottleneck; making the math faster buys nothing while the kernel waits on HBM.

This ordering follows directly from the roofline classification: below the ridge, wall time is set by `bytes / bandwidth`, so only cutting bytes moved (or better overlapping it) shifts the ceiling.
