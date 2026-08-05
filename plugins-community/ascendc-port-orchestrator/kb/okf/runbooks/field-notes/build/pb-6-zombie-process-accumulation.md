---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Zombie Process Accumulation"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Training/benchmark hangs, resource exhaustion after multiple runs"
confidence: single_run
original_id: PB-6
timestamp_inferred: true
tags: [ascendc, pb-6]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Training/benchmark hangs, resource exhaustion after multiple runs
- **Affected**: Docker containers on A5 server
- **Workaround**: **Always restart container before every experiment**
- **Evidence**: 2280 zombies found after E13h

## Archived

### PB-7 (duplicate, line 68): Shared NPU Contention
- **Archived**: 2026-04-09. Reason: duplicate ID with PB-7 (line 43, merge_mix_obj). Content moved to PB-10.

<!-- 迁移自 porter kb/target/ascendc/（PB-6，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
