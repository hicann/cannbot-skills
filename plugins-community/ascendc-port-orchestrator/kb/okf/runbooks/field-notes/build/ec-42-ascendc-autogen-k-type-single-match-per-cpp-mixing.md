---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "AscendC autogen K_TYPE single-match-per-cpp — mixing AIV and AIC kernels in one .cpp silently mis-registers all but the first"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "build succeeds with no error, optionally a [WARNING]: Multiple kernel functions are detected. It is recommended to define only one kernel function per file. lin"
confidence: single_run
original_id: EC-42
timestamp_inferred: true
tags: [ascendc, ec-42]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Severity**: CRITICAL (silent corruption, no runtime error)
- **Status**: OPEN (CANN 9.0.0 build pipeline)
- **Symptom**: build succeeds with no error, optionally a `[WARNING]: Multiple kernel functions are detected. It is recommended to define only one kernel function per file.` line in build log. At runtime, the kernels meant to be AIC (e.g. cube via `MatmulImpl`) silently produce no output — destination buffer reads as uninitialized memory. Same .cpp's first-declared kernel (whose `KERNEL_TASK_TYPE_DEFAULT(...)` ran first) runs correctly; subsequent kernels of a different task type write nothing.
- **Trigger pattern**: kernel author puts kernels of DIFFERENT task types (`KERNEL_TYPE_AIV_ONLY` and `KERNEL_TYPE_AIC_ONLY`) in the same .cpp file, expecting per-`extern "C"` `KERNEL_TASK_TYPE_DEFAULT(...)` to apply locally to each kernel. The macro looks like a per-function attribute but is actually a file-scope global.
- **Root cause**: `cann-9.0.0/tools/tikcpp/ascendc_kernel_cmake/legacy_modules/util/extract_host_stub.py::find_kernel_type_by_source` uses `re.search` (single-match) on `__enable_feature_for_compile_default = X;` (the macro injected by `KERNEL_TASK_TYPE_DEFAULT`). Only the FIRST match is taken and applied to ALL kernels in that file. Mixing types silently mis-registers all but the first.
- **Fix**: split kernels into separate .cpp files by task type. Build script `kernel_dir.glob("*.cpp")` (excluding `pybind11.cpp`) picks up multiple files independently, each with its own first-and-only `KERNEL_TASK_TYPE_DEFAULT(...)` line. Naming convention: `<op>_aiv_kernels.cpp` + `<op>_aic_kernels.cpp` per the op#7 ConvStandard2d precedent.
- **Detection rule**: if your kernel set has BOTH `KERNEL_TYPE_AIV_ONLY` and `KERNEL_TYPE_AIC_ONLY`, you MUST split. The build warning is logged but easy to miss.
- **Diagnostic when output is silent zeros**: re-deploy a minimal one-cube-call kernel in its own .cpp first to confirm the cube path itself works, before debugging algorithm. If isolated cube works but combined doesn't → K_TYPE trap is the root cause.
- **Evidence**:
  - op#7 ConvStandard2d Opt1 (aog-kernel-optimizer ko-1, 2026-04-29). Single combined `conv2d_kernels.cpp` with AIV im2col + AIC cube + AIV bias → 50/50 precision PASS appeared but cube output was zeros (debug `4×8 @ 8×16 = ones×8` returned zeros). Split into `conv2d_aiv_kernels.cpp` + `conv2d_aic_kernels.cpp` → cube ran correctly, returned 8.0 as expected, end-to-end 50/50 + 16/16 PASS, perf median 0.087× → 0.155× → 0.705×.
  - op#6 QuantMatmul kw (Phase E backfill 2026-05-07). Two-launch quant-matmul: AIC int8 GEMM → workspace int32 → AIV dequant → output T. Single `quantmatmul_kernels.cpp` with both `[aicore]` and `[aivec]` `extern "C"` entry points; `device_aiv.o` was empty post-build, AIV dequant never ran, output buffer was uninitialized. Detection: `ls kernels_aiv_device_dir/` empty when AIV launches were expected. Fix: split into `quantmatmul_kernels.cpp` (AIC) + `quantmatmul_aiv_kernels.cpp` (AIV) → 50/50 PASS. Confirms generalization beyond conv-shaped two-launch ops; applies to the textbook quant-matmul shape on Ascend950PR.
- **Cross-ref**: OL-91 (cube playbook), P-P68 (single-AIC GEMM template). Any kernel layering AIV stages around an AIC cube call needs split-cpp.

<!-- 迁移自 porter kb/target/ascendc/（EC-42，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
