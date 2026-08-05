---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Shared NPU Contention"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Benchmark results vary wildly between runs"
confidence: single_run
original_id: PB-10
timestamp_inferred: true
tags: [ascendc, pb-10]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Benchmark results vary wildly between runs
- **Affected**: A5 server (shared infrastructure)
- **Workaround**: Run `npu-smi info` before benchmarking, check for other processes
- **Evidence**: OPERATIONAL_KNOWLEDGE.md OL-15

<!-- 迁移自 porter kb/target/ascendc/（PB-10，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
