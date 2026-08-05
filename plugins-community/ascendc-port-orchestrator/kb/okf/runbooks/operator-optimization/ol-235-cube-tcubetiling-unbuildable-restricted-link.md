---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Restricted launch-lib link line makes a matmul-library cube kernel structurally unbuildable — this is the harness-linkage reason cube-class ports default to manual Mmad"
description: "A launch shared lib linking only the device kernel + framework libs never has the CANN host tiling lib on its link line, so a matmul::Matmul cube kernel cannot compute its TCubeTiling — use manual AscendC::Mmad (needs no host tiling)."
original_id: OL-235
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [cube-mix, tcubetiling, ol-235, port-a3-to-a5, link-line, manual-mmad, optimization]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**When to consult:** porting any CUBE_MIX op (matmul / attention / conv / rnn / gmm) through the launch-shared-lib harness (pybind / `ACLRT_LAUNCH_KERNEL` with a restricted link line). `applies_to: soc=Ascend950PR/V351; cann=9.1.T500; mode=port_a3_to_a5`. `verified_on: cann=9.1.T500 — deformable_conv2d port_a3 2026-06-20`.

### Principle

A `matmul::Matmul<>` + `REGIST_MATMUL_OBJ` cube kernel needs a host-computed `TCubeTiling`, produced by the CANN host tiling lib (`matmul_tiling::MatmulApiTiling::GetTiling` from `tiling/tiling_api.h`). In a harness whose launch shared library links ONLY the device kernel object + framework libs, the host tiling lib's symbols are never on that link line — so the `TCubeTiling` host computation cannot be performed at the launch layer, and the matmul-library cube path is **structurally unbuildable** there regardless of correctness.

This is the concrete linkage reason behind the manual-`AscendC::Mmad` default for cube-class `port_a3_to_a5` ops (P-P102) — it generalizes beyond the FA numerical/deadlock evidence to ALL CUBE_MIX ports built through this harness. **The fix is not "find the tiling lib" — it is to use manual `Mmad` (which needs no host `TCubeTiling`), per P-P102.**

### Concrete anchor

`build_ascendc.py`'s autogen CMake links the pybind module against a fixed, restricted set:
```cmake
add_library(pybind11_lib SHARED ".../pybind11.cpp")
target_link_libraries(pybind11_lib PRIVATE
  kernels torch_npu m dl)          # no host tiling lib (no MatmulApiTiling / libtiling)
```
`host_config.cmake` / `host_intf.cmake` are pulled in by `ascendc.cmake` for the DEVICE `ascendc_library(kernels)` target, not for the pybind target. Remote probe (CANN 9.1.T500): `tiling_api.h` + `libplatform.so` are present, but no host `libtiling` exporting `MatmulApiTiling` is on the pybind link path.

### Scope distinction vs OL-220 (read before assuming a contradiction)

OL-220 shows host `MultiCoreMatmulTiling` / `TCubeTiling` (from `tiling/matrix/bmm_tiling.h`) building clean (chunk_gated_delta_rule, 122/122) — but through the **hand-authored `ascendc_library` flow with a separately-linked `op_host` target** that DOES get the host tiling include/link path. The two are not in conflict: host matmul tiling CAN link when you author the host target's link line yourself (OL-220 flow); it does NOT link when the launch lib's link line is fixed by `build_ascendc.py` to `kernels torch_npu m dl` (this entry). The discriminator is WHERE the `TCubeTiling` computation lives and whether that target's link line includes the tiling lib.

## 证据
- deformable_conv2d (modulated DCNv2, CUBE_MIX) port_a3 2026-06-20 (empirical pybind link-line probe + `build_ascendc.py` source read).
