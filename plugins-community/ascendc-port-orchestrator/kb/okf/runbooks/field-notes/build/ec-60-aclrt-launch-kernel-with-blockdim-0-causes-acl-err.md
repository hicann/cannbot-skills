---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "ACLRT_LAUNCH_KERNEL with blockDim=0 causes ACL_ERROR_RT_PARAM_INVALID (107000) on V220"
description: "applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=n/a; op_class=all"
phenomenon: build_failure
signal:
  - "Kernel compiles cleanly (static_check 10/10 PASS), but every launch fails with ACL_ERROR_RT_PARAM_INVALID (107000)."
confidence: single_run
original_id: EC-60
timestamp_inferred: true
tags: [107000, ascendc, ec-60]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=n/a; op_class=all`

- **Severity**: CRITICAL (kernel never launches, but compilation succeeds silently)
- **Status**: CONFIRMED 2026-05-22 26_AvgPool3d a3-ds
- **Symptom**: Kernel compiles cleanly (static_check 10/10 PASS), but every launch fails with `ACL_ERROR_RT_PARAM_INVALID (107000)`.
- **Root cause**: `ACLRT_LAUNCH_KERNEL(kernel_name)(0, stream, args...)` passes blockDim=0. ACL runtime rejects it; kernel's `GetBlockNum()` returns 0, causing division-by-zero in per-block work distribution.
- **Fix**: Replace `ACLRT_LAUNCH_KERNEL` with explicit `extern "C"` declarations + dynamic nblk computation (floor at 1, cap at 56 for V220).
- **Detection**: grep for `ACLRT_LAUNCH_KERNEL.*\(0,` in pybind11.cpp.
- **Evidence**: 26_AvgPool3d a3-ds kw-1 (2026-05-22, Ascend910_9382 V220).
- **Cross-ref**: PB-28 (KERNEL_TASK_TYPE_DEFAULT is arch35-only — also produces 107000 on V220).

<!-- 迁移自 porter kb/target/ascendc/（EC-60，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
