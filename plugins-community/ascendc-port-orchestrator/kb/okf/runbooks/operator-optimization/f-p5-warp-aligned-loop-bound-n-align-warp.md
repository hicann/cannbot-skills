---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Warp-aligned loop bound (n_align_warp)"
description: "This is a mandatory correctness requirement for cooperative-group programming, not an optional optimization. When the loop body contains __shfl / __shfl_xor / ThreadBarrier, the loop bound must be ali"
severity: high
confidence: single_run
original_id: F-P5
timestamp_inferred: true
tags: [precision, optimization, __shfl, __shfl_xor, threadbarrier, f-p5, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**This is a mandatory correctness requirement for cooperative-group programming, not an optional optimization.**

When the loop body contains `__shfl` / `__shfl_xor` / `ThreadBarrier`, the loop bound must be aligned to the group size:
```cpp
uint64_t n_align = ((n + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE;
for (uint64_t idx = ...; idx < n_align; idx += stride) {
    if (idx < n) { /* normal processing */ }
    else { result = ILLEGAL; }  // out-of-range thread marked invalid but still participates in __shfl
}
```

**Root cause**: `__shfl` requires all lanes in the group to execute the same instruction simultaneously. If some lanes exit the loop → deadlock.

## Anti-Patterns

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（F-P5，convert_patterns_to_okf.py）。confidence 未升格。 -->
