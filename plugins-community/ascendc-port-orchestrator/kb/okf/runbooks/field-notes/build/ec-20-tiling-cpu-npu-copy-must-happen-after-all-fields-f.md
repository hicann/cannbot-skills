---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Tiling CPU→NPU copy must happen AFTER all fields finalized"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Wrong results — tiling field has stale value on NPU"
confidence: single_run
original_id: EC-20
timestamp_inferred: true
tags: [ascendc, ec-20]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: Wrong results — tiling field has stale value on NPU
- **Root cause**: If pybind writes `tiling.field = X` after `tiling_npu = tiling_cpu.to(device)`, the NPU copy has old value
- **Fix**: Finalize ALL tiling fields, then copy once
- **Evidence**: Pad V2 mode routing bug (2026-04-09)

<!-- 迁移自 porter kb/target/ascendc/（EC-20，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
