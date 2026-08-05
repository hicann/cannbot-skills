---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Cooperative-group value transport"
description: "GROUP_SIZE threads share the work of a large vector copy: cpp for (uint32_t j = rank; j < dim; j += GROUP_SIZE) { dst[pos dim + j] = src[idx dim + j]; } At dim=128: 128 stores → 16 threads each do 8 s"
severity: medium
confidence: single_run
original_id: P-P16
timestamp_inferred: true
tags: [cooperative, optimization, p-p16, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

GROUP_SIZE threads share the work of a large vector copy:
```cpp
for (uint32_t j = rank; j < dim; j += GROUP_SIZE) {
    dst[pos * dim + j] = src[idx * dim + j];
}
```

At dim=128: 128 stores → 16 threads each do 8 stores. Applicable to any large embedding/feature vector transport.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/cooperative.md（P-P16，convert_patterns_to_okf.py）。confidence 未升格。 -->
