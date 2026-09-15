---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "CANN perf gap is a knowledge gap, not a hardware limit"
description: "paradigm: ascendc"
phenomenon: perf_regression
signal:
  - "when analyzing the cause of a performance gap"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-42
timestamp_inferred: true
tags: [ascendc, ol-42]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: trust_calibration
- **Loaded by**: Analyzer, Generator
- **Trigger**: when analyzing the cause of a performance gap
- **Lesson**: CANN uses exactly the same hardware capabilities and AscendC APIs. If our op runs <1.0x, the reason is missing optimization knowledge (DMA batching, cache-line coalescing, hybrid SIMT+SIMD strategies), not CANN having a "secret hardware interface". CANN source (~/workspace/cann/) contains these optimization techniques and can be studied in op-gen mode. Perf gap = closable knowledge gap.
- **Evidence**: Gather fp16 dim=last 0.16x is not a platform limit — CANN achieves 1.0x on the same hardware, proving the AscendC API fully supports it

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-42（category=trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
