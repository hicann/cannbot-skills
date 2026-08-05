---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "View-returning ops have ~0 reference cost — benchmark vs materialized copy"
description: "torch.split/narrow/slice return tensor views (~0.009ms), not copies. Our AscendC kernel does actual data movement. Direct latency comparison is misleading — our kernel is \"slower\" but does real work."
phenomenon: perf_regression
signal:
  - "when benchmarking split, narrow, slice, or any op that returns views"
confidence: single_run
original_id: OL-39
timestamp_inferred: true
tags: [ascendc, measurement, ol-39]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
when benchmarking split, narrow, slice, or any op that returns views

## 教训 / 根因
torch.split/narrow/slice return tensor views (~0.009ms), not copies. Our AscendC kernel does actual data movement. Direct latency comparison is misleading — our kernel is "slower" but does real work. For fair evaluation, compare against the cost that would force materialization (.contiguous() on the view).

## 证据
Split 14 benchmark (2026-04-09): torch.split ~0.009ms vs AscendC 0.018-0.96ms

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-39（category=measurement，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
