---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Reduction-op selection — WholeReduceMax vs binary fold by data size"
description: "Reduction schemes differ greatly in perf. Official guidance: small data use WholeReduceMax (single instruction, low latency), large data use binary accumulation (BinaryFold mode). Choose ReduceSum/Max/Min impl by data size."
confidence: single_run
original_id: OL-52
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-52, reduction, wholereducemax, binaryfold, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**场景 / Trigger**: implementing ReduceSum / ReduceMax / ReduceMin.

**选型**: Different reduction schemes have large perf differences. Official guidance — choose based on data size:
- **Small data** → `WholeReduceMax` (single instruction, low latency).
- **Large data** → binary accumulation (**BinaryFold** mode).

The current `BinaryFoldReduceMax` is a reasonable choice but may be simplified in small-data scenarios.

**Source**: hiascend.com best practices (2026-04).
