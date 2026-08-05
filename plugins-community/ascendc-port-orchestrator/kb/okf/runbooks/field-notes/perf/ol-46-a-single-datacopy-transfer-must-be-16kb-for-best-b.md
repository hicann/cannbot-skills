---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "A single DataCopy transfer must be ≥16KB for best bandwidth"
description: "Single-transfer sizes <16KB have significantly reduced bandwidth utilization. Grow TILE_SIZE as large as possible so each DataCopy moves ≥16KB. fp32 TILE=4096 (16KB), fp16/bf16 TILE=8192 (16KB) are re"
phenomenon: perf_regression
signal:
  - "tile-size selection for DataCopy in a SIMD kernel"
confidence: single_run
original_id: OL-46
timestamp_inferred: true
tags: [ascendc, measurement, ol-46]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
tile-size selection for DataCopy in a SIMD kernel

## 教训 / 根因
Single-transfer sizes <16KB have significantly reduced bandwidth utilization. Grow TILE_SIZE as large as possible so each DataCopy moves ≥16KB. fp32 TILE=4096 (16KB), fp16/bf16 TILE=8192 (16KB) are reasonable lower bounds.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-46（category=measurement，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
