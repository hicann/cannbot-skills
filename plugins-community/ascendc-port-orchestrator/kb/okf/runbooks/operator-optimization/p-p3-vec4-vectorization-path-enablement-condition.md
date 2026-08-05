---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "vec4 vectorization path enablement condition"
description: "Anti-pattern: if (hidden_dim % 4 == 0 && grid_y > 1) — grid_y > 1 is a redundant gate. Correct: if (hidden_dim % 4 == 0) — when grid_y==1, block_y=0 is entirely correct."
severity: medium
confidence: single_run
original_id: P-P3
timestamp_inferred: true
tags: [memory_access, optimization, p-p3, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Anti-pattern**: `if (hidden_dim % 4 == 0 && grid_y > 1)` — `grid_y > 1` is a redundant gate.

**Correct**: `if (hidden_dim % 4 == 0)` — when grid_y==1, block_y=0 is entirely correct.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P3，convert_patterns_to_okf.py）。confidence 未升格。 -->
