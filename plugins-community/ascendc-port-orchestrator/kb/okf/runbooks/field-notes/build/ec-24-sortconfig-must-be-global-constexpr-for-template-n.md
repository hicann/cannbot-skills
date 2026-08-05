---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "SortConfig must be global constexpr for template NTTP"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "constexpr SortConfig cfg = {...} inside a function body → \"cannot use as template non-type parameter\" compile error."
confidence: single_run
original_id: EC-24
timestamp_inferred: true
tags: [sortconfig, ascendc, ec-24]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: `constexpr SortConfig cfg = {...}` inside a function body → "cannot use as template non-type parameter" compile error.
- **Root cause**: C++ template constraint — `const SortConfig&` NTTP requires the variable to have external linkage / global scope.
- **Fix**: Declare `SortConfig` at namespace/global scope, outside any function body.
- **Evidence**: Sort kernel development (2026-04-14).

<!-- 迁移自 porter kb/target/ascendc/（EC-24，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
