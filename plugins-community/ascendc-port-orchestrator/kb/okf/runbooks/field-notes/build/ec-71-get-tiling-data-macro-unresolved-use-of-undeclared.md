---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`GET_TILING_DATA` macro unresolved (`use of undeclared identifier 'tilingData'`) in a GENERATED arch35 kernel compiled by the workspace verify-build → define a POD TilingData struct before the algorithm `#include` + load the GM tiling blob with the FA `CopyTiling<T>` byte-copy helper"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=port_a3_to_a5"
phenomenon: build_failure
signal:
  - "a workspace-side verify-build (NPUKernelBench / build_ascendc.py — the per-op pybind/ACLRT_LAUNCH_KERNEL build, NOT the on-host CANN ops-nn-build pipeline) of a"
confidence: single_run
original_id: EC-71
timestamp_inferred: true
tags: [get_tiling_data, aclrt_launch_kernel, ascendc, ec-71]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.1.T500`

**Symptom**: a workspace-side verify-build (NPUKernelBench / `build_ascendc.py` — the per-op pybind/`ACLRT_LAUNCH_KERNEL` build, NOT the on-host CANN ops-nn-build pipeline) of a *generated* arch35 kernel fails with `error: use of undeclared identifier 'tilingData'` at the `GET_TILING_DATA(...)` call site.

**Root cause**: `GET_TILING_DATA` resolves only through the synthesized tiling-data header that the CANN ops-nn-build pipeline emits from the `BEGIN_TILING_DATA_DEF` / `REGISTER_TILING_DATA_CLASS` macros in `op_host/<op>_tiling.h`. The workspace verify-build does NOT process op_host tiling registration, so the macro (and the `tilingData` symbol it expands to) does not exist in that translation unit.

This is distinct from EC-52 (apt.cpp mistakenly `#include`s `<op>_tiling.h` → file not found) and EC-54 (PR4778 ship artifacts misplaced under `kernel/` → move them to `op_kernel/`). Here the kernel is *meant* to build in the verify-build TU and genuinely needs the tiling fields there — so neither "remove the include" nor "move to op_kernel/" applies.

**Fix** (the FA-proven portable substitute): define a POD TilingData struct mirroring the tiling fields, place it in the build TU **before** the algorithm `#include`, and byte-copy the GM tiling blob into a stack instance with a `CopyTiling<T>` helper (reinterpret the GM blob int32-wise into the stack POD):

```cpp
struct MyTilingData { int32_t f0; int32_t f1; /* ... mirror op_host layout ... */ };

template <typename T>
__aicore__ inline void CopyTiling(T* dst, GM_ADDR tilingGM) {
    auto src = reinterpret_cast<__gm__ int32_t*>(tilingGM);
    auto d   = reinterpret_cast<int32_t*>(dst);
    for (uint32_t i = 0; i < sizeof(T) / sizeof(int32_t); ++i) d[i] = src[i];
}
// in the entry, BEFORE using fields:
MyTilingData td; CopyTiling(&td, tilingGM);
#include "<op>_algorithm.h"   // POD must be defined ABOVE this include
```

**Evidence**: recurrent_gated_delta_rule kw-1 iter-1 (2026-06-18, port_a3_to_a5, A5 Ascend950PR_957b, CANN 9.1.T500): generated AIV-only recurrent-decode kernel tripped `undeclared identifier tilingData` on the first verify-build; adopting the FA `CopyTiling<T>` byte-copy helper + POD-TilingData-before-include let the build proceed and the kernel reached 30/30 T1 PASS. The pattern is the FA `kernel_common.h` `CopyTiling<T>` helper reused unchanged.

**Other instances (predicted)**: any *generated* (non-ship-artifact) arch35 / port_a3 kernel that needs tiling fields and is compiled by the workspace verify-build rather than the ops-nn-build pipeline — recurrent / SSM / linear-attention family especially, where the kernel is authored from arch22 algorithm source.

**Cross-reference**: EC-52 (include-resolution variant), EC-54 (layout-misplacement variant), EC-68 (ACLRT_LAUNCH workspace base — same verify-build/host-stub class), OL-141 (L1 mechanical port).

<!-- 迁移自 porter kb/target/ascendc/（EC-71，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
