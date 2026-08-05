---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "TQue<VECIN,2> Data Corruption"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "99.5% elements corrupted when using TQue with depth 2"
confidence: single_run
original_id: PB-2
timestamp_inferred: true
tags: [ascendc, pb-2]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: 99.5% elements corrupted when using TQue with depth 2
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Use TQue<VECIN,4> (depth 4 works correctly)
- **Status**: OPEN
- **Evidence**: hardware/target/ascend950pr.md, E13 test data

<!-- 迁移自 porter kb/target/ascendc/（PB-2，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
