---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Advanced Sort API is device-only (__NPU_ARCH__ guard)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Sort<T, isReuse, config>() compilation fails with \"SortConfig not found\" or \"SortType not found\"."
confidence: single_run
original_id: EC-25
timestamp_inferred: true
tags: [__npu_arch__, ascendc, ec-25]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: `Sort<T, isReuse, config>()` compilation fails with "SortConfig not found" or "SortType not found".
- **Root cause**: The advanced Sort API (`adv_api/sort/sort.h`) is only available when `__NPU_ARCH__` is defined during device compilation. Host build stubs don't include these declarations.
- **Fix**: Guard Sort API usage with `#if defined(__NPU_ARCH__) && (__NPU_ARCH__ > 0)`. In host compilation, the function can be a stub or not compiled.
- **Evidence**: Sort kernel development (2026-04-14).

<!-- 迁移自 porter kb/target/ascendc/（EC-25，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
