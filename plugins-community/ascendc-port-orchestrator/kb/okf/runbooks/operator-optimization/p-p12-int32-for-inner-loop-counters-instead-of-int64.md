---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "int32 for inner-loop counters instead of int64"
description: "Use int for variables whose range is within int32 (loop counter j, thread_idx_emb, etc.). edge_in[i] emb_dim must stay int64. Effect: fwd -3%, bwd -2%."
severity: medium
confidence: single_run
original_id: P-P12
timestamp_inferred: true
tags: [memory_access, optimization, int, p-p12, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

Use `int` for variables whose range is within int32 (loop counter j, thread_idx_emb, etc.). `edge_in[i] * emb_dim` must stay int64. Effect: fwd -3%, bwd -2%.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P12，convert_patterns_to_okf.py）。confidence 未升格。 -->
