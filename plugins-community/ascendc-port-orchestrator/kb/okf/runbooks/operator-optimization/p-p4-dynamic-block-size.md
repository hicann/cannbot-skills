---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Dynamic block size"
description: "cpp __aicore__ inline uint32_t calc_block_size(int dim, int divisor) { int raw = std::min(dim / divisor, 1024); return ((std::max(raw, 1) + 31) / 32) 32; // round to warp } Note: The function must be"
severity: medium
confidence: single_run
original_id: P-P4
timestamp_inferred: true
tags: [thread_utilization, optimization, __aicore__, p-p4, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

```cpp
__aicore__ inline uint32_t calc_block_size(int dim, int divisor) {
    int raw = std::min(dim / divisor, 1024);
    return ((std::max(raw, 1) + 31) / 32) * 32;  // round to warp
}
```

**Note**: The function must be annotated with `__aicore__`.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/thread_utilization.md（P-P4，convert_patterns_to_okf.py）。confidence 未升格。 -->
