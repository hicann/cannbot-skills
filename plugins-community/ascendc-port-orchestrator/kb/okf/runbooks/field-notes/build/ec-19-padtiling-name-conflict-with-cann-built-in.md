---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "PadTiling name conflict with CANN built-in"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "error: reference to 'PadTiling' is ambiguous"
confidence: single_run
original_id: EC-19
timestamp_inferred: true
tags: [padtiling, using, padoptiling, mypadtiling, ascendc, ec-19]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: `error: reference to 'PadTiling' is ambiguous`
- **Root cause**: CANN `kernel_tiling.h` defines `PadTiling` in `AscendC::tiling` namespace and imports it via `using`. Custom struct with same name conflicts.
- **Fix**: Rename custom tiling struct to unique name (e.g., `PadOpTiling`, `MyPadTiling`)
- **Evidence**: Pad V2 first build (2026-04-09)

<!-- 迁移自 porter kb/target/ascendc/（EC-19，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
