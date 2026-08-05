---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "aclrtlaunch Undefined Reference (Linker)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "undefined reference to 'aclrtlaunch_xxx(...)'"
confidence: single_run
original_id: EC-10
timestamp_inferred: true
tags: [ascendc, ec-10]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: `undefined reference to 'aclrtlaunch_xxx(...)'`
- **Root cause**: Auto-generated `host_stub.cpp` exports kernel launch functions as **C symbols** (no mangling). Test code that declares them without `extern "C"` gets C++ mangled names → linker mismatch.
- **Fix**:
  ```cpp
  // ❌ Wrong — C++ mangling
  uint32_t aclrtlaunch_my_kernel(uint32_t, void*, void*, void*, int);

  // ✅ Correct — C linkage
  extern "C" {
  uint32_t aclrtlaunch_my_kernel(uint32_t, void*, void*, void*, int);
  }
  ```
- **Related**: PB-8 in PLATFORM_BUGS.md

<!-- 迁移自 porter kb/target/ascendc/（EC-10，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
