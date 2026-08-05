---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "NPU Device 0 Post-Reboot Failure"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Error 507033 on device 0 after server reboot (2026-04-01)"
confidence: single_run
original_id: PB-3
timestamp_inferred: true
tags: [507033, ascendc, pb-3]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Symptom**: Error 507033 on device 0 after server reboot (2026-04-01)
- **Affected**: A5 server 198.51.100.35, device 0 only
- **Workaround**: Use devices 1-4, 7
- **Status**: Hardware issue, may require RMA

## Bisheng Compiler Bugs

<!-- 迁移自 porter kb/target/ascendc/（PB-3，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
