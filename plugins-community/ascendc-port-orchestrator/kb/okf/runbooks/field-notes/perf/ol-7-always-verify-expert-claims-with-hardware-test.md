---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Always verify expert claims with hardware test"
description: "paradigm: ascendc"
phenomenon: perf_regression
signal:
  - "when an expert makes a hardware performance claim without providing measurement data"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-7
timestamp_inferred: true
tags: [ascendc, ol-7]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: trust_calibration
- **Loaded by**: Optimizer, Planner
- **Trigger**: when an expert makes a hardware performance claim without providing measurement data
- **Lesson**: An expert stated that 128-bit loads are slower than 64-bit on Ascend950PR. Hardware testing (`tests/load_width_test/`) showed the opposite: 128-bit is 1.2x-2.1x faster than 32-bit for sequential reads (1MB: 1.17x, 16MB: 1.39x, 64MB: 2.12x). The expert's claim may have been valid for a different access pattern (random/indirect indexing), but for the sequential reads in our kernels, wider loads are strictly better. Always run a targeted micro-benchmark before accepting or rejecting a hardware claim.
- **Evidence**: EXPERT_FEEDBACK.md E7-3 (128-bit实测数据 / 128-bit measured), `tests/load_width_test/`

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-7（category=trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
