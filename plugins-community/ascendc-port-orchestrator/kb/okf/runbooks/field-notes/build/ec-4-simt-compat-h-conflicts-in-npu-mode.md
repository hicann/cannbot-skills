---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`simt_compat.h` conflicts in NPU mode"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-4
timestamp_inferred: true
tags: [blockdim, threadidx, ascendc, ec-4]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: redefinition of 'blockDim' as different kind of symbol
  ```
  or:
  ```
  error: expected unqualified-id
  ```
  (when `#define blockDim` macro clashes with CANN's built-in `blockDim` in NPU mode)
- **Root cause**: `simt_compat.h` defines `blockDim` and `threadIdx` as macros that map to raw CPU-mode globals (`g_threadDimX`, `g_threadIdxX`). In NPU mode, CANN provides its own built-in `blockDim`/`threadIdx` — the macros collide with these built-ins. The header must only be included in CPU debug builds.
- **Fix**:
  ```cpp
  // BEFORE (fails on NPU):
  #include "simt_compat.h"    // unconditional include → macro conflicts

  // AFTER (conditional):
  #if defined(ASCENDC_CPU_DEBUG)
  #include "simt_compat.h"
  #endif
  ```
  The guard works because:
  - CPU debug mode: `ASCENDC_CPU_DEBUG` is defined by tikicpulib CMake target → macros active
  - NPU mode: `ASCENDC_CPU_DEBUG` is not defined → header skipped, CANN built-ins used
- **Related**: None (project-specific compatibility layer)

<!-- 迁移自 porter kb/target/ascendc/（EC-4，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
