---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`GM_ADDR` needs typed pointer cast"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-2
timestamp_inferred: true
tags: [gm_addr, ascendc, ec-2]
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
  error: cannot initialize a variable of type '__gm__ float *' with an lvalue of type 'GM_ADDR' (aka 'uint8_t * __attribute__((address_space(1)))')
  ```
  or:
  ```
  error: subscript of pointer to type '__gm__ uint8_t' ... is not allowed
  ```
- **Root cause**: `GM_ADDR` is `__gm__ uint8_t*`. Kernel VF functions receive all GM pointers as untyped `GM_ADDR`. To access data as a specific type, you must cast with `reinterpret_cast<__gm__ T*>`. The `__gm__` qualifier must be preserved through the cast.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  __gm__ float* input = input_gm;              // type mismatch
  float val = input_gm[i];                      // subscript on uint8_t*

  // AFTER (compiles):
  __gm__ float* input = reinterpret_cast<__gm__ float*>(input_gm);
  float val = input[i];                          // correct typed access

  // For const pointers:
  __gm__ const int* edge_in = reinterpret_cast<__gm__ const int*>(edge_in_gm);
  ```
- **Related**: P-P5 (LAUNCH_BOUND + LAUNCH_CHECK — kernel launch pattern)

<!-- 迁移自 porter kb/target/ascendc/（EC-2，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
