---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "bisheng `half` macro undefined in `aic_obj` compile target — direct-include `kernel_operator.h` in .cpp"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "error: unknown type name 'half'; did you mean 'half2'? at the __gm__ half reinterpret_cast site, in the aic_obj build target — even though the op is KERNEL_TASK"
confidence: single_run
original_id: EC-43
timestamp_inferred: true
tags: [half, aic_obj, ascendc, ec-43]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Severity**: HIGH (compile-time blocker, error message mentions a similar-but-wrong type which sends debug down the wrong path)
- **Status**: CONFIRMED 2026-05-01 op#2 SwiGLU kw-2 build iter 1
- **Affected**: Ascend950PR / CANN 9.0.0 b103 / bisheng 2026-03-21. SIMD multi-core class kernel (`__global__ __aicore__` + `class.Init().Process()` pattern), kernel.h uses `__gm__ half*` reinterpret_cast.
- **Symptom**: `error: unknown type name 'half'; did you mean 'half2'?` at the `__gm__ half*` reinterpret_cast site, in the `aic_obj` build target — even though the op is `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` (no AIC kernels). Same pattern in older built kernels in `output/npukernelbench/src/kernels/*/kernel/` compiles fine, suggesting context-dependent.
- **Root cause hypothesis (not fully bisected)**:
  - `__clang_cce_types.h:19` defines `#define half __cce_half`
  - `__clang_cce_runtime_wrapper.h:139` does `#undef half` then `typedef __cce_half half;` immediately after — so `half` becomes an actual type alias.
  - For the `aic_obj` compile target, `runtime_wrapper.h`'s typedef path may not be reached (likely guarded by `#if !defined(CCE_NO_HALF)` or analogous macro polarity that differs per target). The `#undef` in `runtime_wrapper.h` ran BUT the typedef did not fire → `half` is left as undefined identifier.
- **Workaround (canonical, verified)**: add `#include "kernel_operator.h"` **directly** at the top of the `<op>_kernels.cpp` file (the dispatcher .cpp), not just transitively via `#include "<op>_kernel.h"`. The direct include reaches `runtime_wrapper.h` early enough in the `aic_obj` target's preprocessor stack that the typedef fires before the kernel.h's `__gm__ half*` site.
- **Detection rule**: if you see "unknown type name 'half'; did you mean 'half2'?" in `aic_obj` build output, FIRST add `#include "kernel_operator.h"` at top of the dispatcher .cpp. Do NOT chase the suggested 'half2' — it's a wrong rabbit hole.
- **Reference templates that have the direct include**: `output/npukernelbench/src/kernels/11_GroupNorm/kernel/groupnorm_kernels.cpp:1` (working), `output/npukernelbench/src/kernels/2_SwiGLU/kernel/swiglu_kernels.cpp:1` (post-fix). Static check `missing_kernel_operator` was added 2026-05-01 to flag .cpp files without the direct include.
- **Evidence**: op#2 SwiGLU kw-2 (2026-05-01) — initial Phase B kernel (synthesized from scratch, only `#include "<op>_kernel.h"` in .cpp) failed compile in `aic_obj` target with the unknown-type-name error. Workaround applied (direct include) → build PASS first try.

<!-- 迁移自 porter kb/target/ascendc/（EC-43，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
