---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`std::string` in pybind11 crashes with `basic_string null` SIGSEGV on V220 ARM64 (bisheng/GCC ABI mismatch)"
description: "applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=n/a; op_class=all (pybind11 host code)"
phenomenon: build_failure
signal:
  - "pybind11 module loads successfully, but any call to the wrapped function crashes with basic_string null SIGSEGV (exit code 139). Same binary works on x86 host,"
confidence: single_run
original_id: EC-63
timestamp_inferred: true
tags: [ascendc, ec-63]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382 (V220); cann=9.0.0; bisheng=n/a; op_class=all (pybind11 host code)`

- **Severity**: CRITICAL (kernel compiles, pybind11 loads, crashes at first call with SIGSEGV)
- **Status**: CONFIRMED 2026-05-22 28_Interpolate a3-ds
- **Symptom**: pybind11 module loads successfully, but any call to the wrapped function crashes with `basic_string null` SIGSEGV (exit code 139). Same binary works on x86 host, crashes on V220 ARM64.
- **Root cause**: GCC (host, x86) and bisheng (device, ARM64) have different std::string ABIs. The `std::string` parameter in pybind11 function signature is constructed by pybind11 from Python str, but the underlying memory layout differs between host and device.
- **Fix**: Replace `const std::string&` parameters with `const char*` or `py::str` in pybind11 wrappers. Convert to C string before passing to kernel launch.
- **Detection**: grep for `std::string` in kernel/pybind11.cpp. Any match → replace with C-string alternative.
- **Evidence**: 28_Interpolate a3-ds (2026-05-22, V220): `interpolate_forward(..., const std::string& mode_str, ...)` crashed with SIGSEGV 139 on A3 NPU0. Kernel compiled and built successfully.
- **Cross-ref**: OL-180 (CANN env init for .so loading) — both are pybind11-level V220-specific host issues.

<!-- 迁移自 porter kb/target/ascendc/（EC-63，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
