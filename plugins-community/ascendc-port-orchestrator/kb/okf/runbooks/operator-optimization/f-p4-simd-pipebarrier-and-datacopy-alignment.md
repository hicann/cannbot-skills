---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMD PipeBarrier and DataCopy alignment"
description: "Anti-pattern: PipeBarrier<PIPE_MTE2>() — fine-grained barrier can cause data races. Correct pattern: PipeBarrier<PIPE_ALL>() guarantees correctness. Optimal: use TQue depth=2 double buffering. DataCop"
severity: high
confidence: single_run
original_id: F-P4
timestamp_inferred: true
tags: [platform_compat, optimization, datacopypad, f-p4, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Anti-pattern**: `PipeBarrier<PIPE_MTE2>()` — fine-grained barrier can cause data races.

**Correct pattern**: `PipeBarrier<PIPE_ALL>()` guarantees correctness. Optimal: use TQue depth=2 double buffering.

**DataCopy alignment requirement**: fp32: %8==0, fp16/bf16: %16==0. When unaligned, fall back to SIMT or use `DataCopyPad`.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（F-P4，convert_patterns_to_okf.py）。confidence 未升格。 -->
