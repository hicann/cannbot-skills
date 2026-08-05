---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Typed Kernel Entry Crash (CANN 9.0.0)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Error 507035 on kernel launch with typed entry points (e.g., _fp32 suffix)"
confidence: single_run
original_id: PB-1
timestamp_inferred: true
tags: [507035, _fp32, ascendc, pb-1]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Error 507035 on kernel launch with typed entry points (e.g., `_fp32` suffix)
- **Affected**: CANN 9.0.0 with bisheng 2026-03-21
- **Workaround**: Use legacy untyped entry points (single dispatcher .cpp, cast inside kernel)
- **Status**: OPEN (not fixed in CANN 9.0.T501)
- **Evidence**: OPERATIONAL_KNOWLEDGE.md OL-16

<!-- 迁移自 porter kb/target/ascendc/（PB-1，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
