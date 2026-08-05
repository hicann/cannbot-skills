---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Missing `__aicore__` on helper function"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-1
timestamp_inferred: true
tags: [__aicore__, __host__, ascendc, ec-1]
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
  error: calling a __host__ function("helper_func") from a __aicore__ function("kernel_vf") is not allowed
  ```
- **Root cause**: All functions called inside `__simt_vf__ __aicore__` kernel VF functions must themselves be decorated with `__aicore__`. Bisheng treats undecorated functions as `__host__`-only, and cross-domain calls are forbidden.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  inline float compute_weight(float x) { return x * 0.5f; }

  // AFTER (compiles):
  __aicore__ inline float compute_weight(float x) { return x * 0.5f; }
  ```
- **Note**: Template helper functions also need `__aicore__`:
  ```cpp
  template <typename T>
  __aicore__ inline float simt_to_float(T v) { return static_cast<float>(v); }
  ```
- **Related**: None (basic AscendC requirement)

<!-- 迁移自 porter kb/target/ascendc/（EC-1，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
