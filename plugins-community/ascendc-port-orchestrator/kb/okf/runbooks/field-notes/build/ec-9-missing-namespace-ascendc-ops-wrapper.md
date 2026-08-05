---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Missing `namespace ascendc_ops {}` wrapper"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-9
timestamp_inferred: true
tags: [ascendc, ec-9]
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
  error: redefinition of 'ITER'
  error: redefinition of 'simt_to_float'
  error: use of undeclared identifier 'POOLING_FWD_THREAD_NUM'
  ```
  (name collisions between kernel files, or missing constant/helper definitions when files are compiled together)
- **Root cause**: All kernel code in this project must be wrapped in `namespace ascendc_ops { ... }`. Without the namespace: (1) macros like `ITER(x,y)` and helper templates like `simt_to_float` collide when multiple kernel headers are included in the same translation unit; (2) dispatcher `.cpp` files use `using namespace ascendc_ops;` to access kernel VF functions and constants — if the VF functions are in the global namespace, `using namespace ascendc_ops;` finds nothing.
- **Fix**:
  ```cpp
  // BEFORE (collisions, missing symbols):
  #include <kernel_operator.h>
  using namespace AscendC;

  #define ITER(x, y) (((x) + (y) - 1) / (y))

  template <typename T>
  __simt_vf__ __aicore__
  LAUNCH_BOUND(512) inline void my_kernel_vf(GM_ADDR input_gm, ...) { ... }

  // AFTER (namespaced):
  #include <kernel_operator.h>

  namespace ascendc_ops {
  using namespace AscendC;

  #define ITER(x, y) (((x) + (y) - 1) / (y))

  template <typename T>
  __simt_vf__ __aicore__
  LAUNCH_BOUND(512) inline void my_kernel_vf(GM_ADDR input_gm, ...) { ... }

  }  // namespace ascendc_ops
  ```
  Corresponding dispatcher file:
  ```cpp
  #include "my_kernel.h"
  using namespace ascendc_ops;

  extern "C" __global__ __aicore__ void my_kernel_fp32(GM_ADDR input_gm, ...) {
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Simt::VF_CALL<my_kernel_vf<float>>(
        Simt::Dim3{512}, input_gm, ..., GetBlockIdx(), GetBlockNum());
  }
  ```
- **Note**: `extern "C" __global__` dispatcher functions are in the global namespace (required by CANN runtime). Only the VF functions, helpers, and constants go inside `namespace ascendc_ops`.
- **Related**: None (project convention for multi-file compilation)

<!-- 迁移自 porter kb/target/ascendc/（EC-9，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
