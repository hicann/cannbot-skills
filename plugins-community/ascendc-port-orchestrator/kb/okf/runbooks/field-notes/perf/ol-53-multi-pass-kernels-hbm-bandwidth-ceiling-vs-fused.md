---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Multi-pass kernels — HBM bandwidth ceiling vs fused CANN"
description: "A multi-pass kernel reads each element from HBM N times (N=number of passes). A fused CANN kernel reads once. The performance ceiling of an N-pass kernel is ~1/N of fused, regardless of VEC optimizati"
phenomenon: perf_regression
signal:
  - "AscendC kernel performance <0.5x of CANN reference AND kernel requires multiple passes over input data"
confidence: single_run
original_id: OL-53
timestamp_inferred: true
tags: [ascendc, performance_analysis, ol-53]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
AscendC kernel performance <0.5x of CANN reference AND kernel requires multiple passes over input data

## 教训 / 根因
A multi-pass kernel reads each element from HBM N times (N=number of passes). A fused CANN kernel reads once. The performance ceiling of an N-pass kernel is ~1/N of fused, regardless of VEC optimization. For per-row reductions that need a global statistic before element-wise processing (e.g., quantization needs max before scaling), 2-pass is mandatory unless rows fit in single UB tile. When <0.5x vs CANN with a multi-pass kernel, check if the algorithm requires multi-pass before investing in VEC tuning — the bottleneck is HBM bandwidth, not compute.

## 证据
DynamicQuant 2-pass (find max, then quantize) → 0.25x of CANN's fused npu_dynamic_quant

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-53（category=performance_analysis，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
