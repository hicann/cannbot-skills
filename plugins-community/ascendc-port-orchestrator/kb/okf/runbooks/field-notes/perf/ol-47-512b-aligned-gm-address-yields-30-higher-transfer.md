---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "512B-aligned GM address yields 30% higher transfer bandwidth"
description: "For GM→UB transfers, 512B-aligned GM addresses have 30% higher bandwidth than 32B-aligned. When allocating output tensors in the pybind layer, ensure the start address is 512B-aligned (allocate with e"
phenomenon: perf_regression
signal:
  - "output tensor allocation or DataCopy GM-offset computation"
confidence: single_run
original_id: OL-47
timestamp_inferred: true
tags: [ascendc, measurement, ol-47]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
output tensor allocation or DataCopy GM-offset computation

## 教训 / 根因
For GM→UB transfers, 512B-aligned GM addresses have 30% higher bandwidth than 32B-aligned. When allocating output tensors in the pybind layer, ensure the start address is 512B-aligned (allocate with extra padding then narrow). Note: this optimization is most effective on A2-series products.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-47（category=measurement，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
