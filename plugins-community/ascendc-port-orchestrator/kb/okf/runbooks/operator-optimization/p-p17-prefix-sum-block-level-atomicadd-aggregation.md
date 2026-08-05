---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Prefix-sum + block-level atomicAdd aggregation"
description: "Three-level aggregation to reduce global atomics: 1. In-group prefix sum (__shfl_up): each thread obtains local_offset 2. Group leader → UB atomicAdd (__ubuf__): once per group (512/32 = 16 times) 3."
severity: high
confidence: single_run
original_id: P-P17
timestamp_inferred: true
tags: [scatter_add, optimization, __shfl_up, __ubuf__, p-p17, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

Three-level aggregation to reduce global atomics:
1. **In-group prefix sum** (`__shfl_up`): each thread obtains local_offset
2. **Group leader → UB atomicAdd** (`__ubuf__`): once per group (512/32 = 16 times)
3. **Block leader → global atomicAdd**: once per whole block

512 global atomicAdd → 1. **Directly applicable to Pooling backward atomicAdd optimization.**

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P17，convert_patterns_to_okf.py）。confidence 未升格。 -->
