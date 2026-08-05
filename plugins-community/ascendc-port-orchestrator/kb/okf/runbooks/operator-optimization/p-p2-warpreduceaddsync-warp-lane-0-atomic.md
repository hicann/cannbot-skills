---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "WarpReduceAddSync + warp-lane-0 atomic"
description: "cpp // Anti-pattern: per-thread atomicAdd → 512 calls/block // Correct: warp reduce then lane-0 atomicAdd → 16 calls/block float warp_sum = Simt::WarpReduceAddSync(partial_sum); Simt::ThreadBarrier();"
severity: high
confidence: single_run
original_id: P-P2
timestamp_inferred: true
tags: [scatter_add, optimization, p-p2, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

```cpp
// Anti-pattern: per-thread atomicAdd → 512 calls/block
// Correct: warp reduce then lane-0 atomicAdd → 16 calls/block
float warp_sum = Simt::WarpReduceAddSync(partial_sum);
Simt::ThreadBarrier();
if (threadIdx.x % 32 == 0 && warp_sum != 0.0f)
    atomicAdd(dst, warp_sum);
```

**Precondition**: dst must be pre-zeroed on the host side. Measured SG backward: 0.865ms → 0.121ms (7.1x).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P2，convert_patterns_to_okf.py）。confidence 未升格。 -->
