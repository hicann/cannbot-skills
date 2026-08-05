---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "-O2 Required for NPU (No -O0)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Kernel may produce wrong results or crash with -O0 on NPU"
confidence: single_run
original_id: PB-5
timestamp_inferred: true
tags: [ascendc, pb-5]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Kernel may produce wrong results or crash with -O0 on NPU
- **Affected**: All NPU builds
- **Workaround**: Always use -O2 for NPU builds (-O0 only for CPU debug mode)
- **Status**: By design (bisheng optimizations required for correct codegen)

<!-- 迁移自 porter kb/target/ascendc/（PB-5，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
