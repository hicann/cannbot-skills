---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Cost analysis of norm + activation fusion"
description: "In norm + activation fusion, the activation (e.g. Swish: `Muls(neg_x, x, -scale)` → `Exp` → `Adds(1)` → `Reciprocal` → `Mul`) adds only 5 VEC instructions to the normalize pass and **contributes almos"
phenomenon: perf_regression
signal:
  - "planning to fuse norm (GroupNorm/LayerNorm/RmsNorm) + activation (Swish/GELU/SiLU) into a single kernel"
confidence: single_run
original_id: OL-69
timestamp_inferred: true
tags: [exp, reciprocal, mul, ascendc, performance, ol-69]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
planning to fuse norm (GroupNorm/LayerNorm/RmsNorm) + activation (Swish/GELU/SiLU) into a single kernel

## 教训 / 根因
In norm + activation fusion, the activation (e.g. Swish: `Muls(neg_x, x, -scale)` → `Exp` → `Adds(1)` → `Reciprocal` → `Mul`) adds only 5 VEC instructions to the normalize pass and **contributes almost nothing to end-to-end latency**. Reason: the kernel is bandwidth-bound on DataCopy, and the VEC instructions are hidden inside the pipeline. Fusion gain comes mainly from **eliminating the intermediate GM round-trip** (norm output and activation input are the same data).

## 证据
2_GroupNormSwish vs Pure PyTorch on the same-hardware baseline: 2.05x speedup; compared against standalone GroupNorm L1 at 0.76x, fusion yields ~2.7x net improvement (vs separate norm + separate activation). E3 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-69（category=performance，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
