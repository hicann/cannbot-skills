---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Missing `#include <kernel_operator.h>`"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-8
timestamp_inferred: true
tags: [ascendc, ec-8]
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
  error: unknown type name 'GM_ADDR'
  error: unknown type name '__gm__'
  error: use of undeclared identifier 'atomicAdd'
  error: unknown type name 'bfloat16_t'
  error: no member named 'VF_CALL' in namespace 'AscendC::Simt'
  ```
  (cascade of errors — types, macros, and functions all undefined)
- **Root cause**: `kernel_operator.h` is the master header for AscendC. It pulls in all CANN types (`GM_ADDR`, `__gm__`, `bfloat16_t`, `half`), SIMT APIs (`Simt::VF_CALL`, `Simt::Dim3`), SIMD APIs (`DataCopy`, `Cast`), atomics (`atomicAdd`), and platform macros (`LAUNCH_BOUND`, `__aicore__`). Without it, nothing AscendC-specific compiles.
- **Fix**:
  ```cpp
  // BEFORE (cascade of errors):
  #include <cstdint>
  // missing kernel_operator.h

  // AFTER (compiles):
  #include <kernel_operator.h>    // ✅ MUST be first AscendC include
  #include <cstdint>
  ```
- **Rule**: Every `.h` and `.cpp` file that uses any AscendC type or API must include `<kernel_operator.h>` as its first AscendC include. Standard library headers (`<cstdint>`, `<cstring>`) can come before or after.
- **Related**: None (basic AscendC requirement)

<!-- 迁移自 porter kb/target/ascendc/（EC-8，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
