---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Cooperative-group parallel traversal + shuffle reduction"
description: "Transform single-thread sequential traversal into N-thread cooperative parallel + __shfl_xor divide-and-conquer reduction. cpp auto rank = threadIdx.x % GROUP_SIZE; for (uint32_t pos = rank; pos < arr"
severity: high
confidence: single_run
original_id: P-P13
timestamp_inferred: true
tags: [cooperative, optimization, __shfl_xor, p-p13, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

Transform single-thread sequential traversal into N-thread cooperative parallel + `__shfl_xor` divide-and-conquer reduction.

```cpp
auto rank = threadIdx.x % GROUP_SIZE;
for (uint32_t pos = rank; pos < array_size; pos += GROUP_SIZE) { ... }
for (int32_t offset = GROUP_SIZE / 2; offset > 0; offset /= 2) {
    auto other = __shfl_xor(val, offset, GROUP_SIZE);
    if (other < val) val = other;  // or +=, max, etc.
}
```

Applicable to any scenario requiring min/max/sum within a warp.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/cooperative.md（P-P13，convert_patterns_to_okf.py）。confidence 未升格。 -->
