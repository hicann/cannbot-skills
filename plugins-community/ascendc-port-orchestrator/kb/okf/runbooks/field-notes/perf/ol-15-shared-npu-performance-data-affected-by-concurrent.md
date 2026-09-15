---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Shared NPU — performance data affected by concurrent users"
description: "paradigm: ascendc"
phenomenon: perf_regression
signal:
  - "when running performance benchmarks on A5 server"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-15
timestamp_inferred: true
tags: [ascendc, ol-15]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: environment
- **Loaded by**: QA
- **Trigger**: when running performance benchmarks on A5 server
- **Lesson**: A5 server is shared. Our container binds one NPU but other processes may use it. Before performance tests: check `npu-smi info` for other processes. If busy: try another NPU or wait/retry. Never trust a single benchmark run on shared infra.
- **Evidence**: A5 server 198.51.100.35 shared infrastructure

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-15（category=environment，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
