---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Dynamic numBlocks"
description: "cpp // Pooling: saturate all AIV cores constexpr uint32_t MAX_AIV_CORES = 56; // 28 AICore x 2 AIV // SG: each block handles one token uint32_t fwd_blk = token_num grid_y; Distinction: Pooling shares"
severity: high
confidence: single_run
original_id: P-P1
timestamp_inferred: true
tags: [thread_utilization, optimization, p-p1, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

```cpp
// Pooling: saturate all AIV cores
constexpr uint32_t MAX_AIV_CORES = 56;  // 28 AICore x 2 AIV

// SG: each block handles one token
uint32_t fwd_blk = token_num * grid_y;
```

**Distinction**: Pooling shares work by stride -> saturate all cores. SG has each block own one output slice -> numBlocks = number of work items.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/thread_utilization.md（P-P1，convert_patterns_to_okf.py）。confidence 未升格。 -->
