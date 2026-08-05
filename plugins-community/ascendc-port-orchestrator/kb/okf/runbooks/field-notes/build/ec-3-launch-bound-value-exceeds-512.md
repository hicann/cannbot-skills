---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`LAUNCH_BOUND` value exceeds 512"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-3
timestamp_inferred: true
tags: [launch_bound, ascendc, ec-3]
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
  error: 'LAUNCH_BOUND' attribute parameter 1024 exceeds maximum allowed value 512
  ```
  or at runtime: incorrect results / register spilling when LAUNCH_BOUND > 512 with complex kernel logic.
- **Root cause**: Ascend950PR supports LAUNCH_BOUND up to 2048 in theory, but **512 is the practical maximum** for kernels with non-trivial register usage. At 512 threads, each thread gets 64 registers (128KB register file / 512 threads / 4 bytes). Higher thread counts reduce per-thread registers, causing spills to slower memory and often incorrect codegen.
- **Fix**:
  ```cpp
  // BEFORE (risky or fails):
  LAUNCH_BOUND(1024) inline void kernel_vf(...) { ... }

  // AFTER (safe default):
  LAUNCH_BOUND(512) inline void kernel_vf(...) { ... }

  // Define as named constant:
  constexpr uint32_t OP_THREAD_NUM = 512;
  LAUNCH_BOUND(OP_THREAD_NUM) inline void kernel_vf(...) { ... }
  ```
- **Note**: source `__launch_bounds__(1024)` must be reduced to 512 when migrating. The dispatcher `Simt::Dim3{OP_THREAD_NUM}` must match the LAUNCH_BOUND value.
- **Related**: P-P5 (LAUNCH_BOUND + LAUNCH_CHECK)

<!-- 迁移自 porter kb/target/ascendc/（EC-3，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
