---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Build recipe for a cube+vec MIX kernel via ascendc_library — non-empty CMAKE_BUILD_TYPE, post-ascendc_library include scoping, and where MultiCoreMatmulTiling lives"
description: "Building a MIX cube+vec kernel via ascendc_library needs non-empty CMAKE_BUILD_TYPE, ascendc_include_directories called AFTER ascendc_library, and MultiCoreMatmulTiling from bmm_tiling.h."
phenomenon: build_failure
signal:
  - "A MIX (cube+vec) ascendc_library build aborts silently or with a misleading message: merge_mix_obj.sh exits non-zero, a \"non-existent target ..._interface\" error, or MultiCoreMatmulTiling / TCubeTiling not found"
confidence: single_run
original_id: OL-220
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, ol-220, mix-aic-aiv, ascendc-library, cmake, matmul-tiling]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

Ascend950PR / CANN 9.1.T500, MIX_AIC_AIV kernel. Verified on chunk_gated_delta_rule (GDN) light-port, 2026-06-15 (built clean, 122/122 T1 PASS, perf ~89–121µs). Building a MIX (cube+vec) AscendC kernel through the CMake `ascendc_library` flow on this CANN version has three non-obvious requirements that each abort the build silently or with a misleading message if missed.

## 根因 / 教训

1. **`CMAKE_BUILD_TYPE` MUST be non-empty (`-DCMAKE_BUILD_TYPE=Release`)**. An empty build type makes the MIX object-merge step (`merge_mix_obj.sh --build-type`) receive an empty `--build-type` arg and abort under `set -e`. The failure is silent (the merge script just exits non-zero) and does NOT clearly point at the build type — always pass `-DCMAKE_BUILD_TYPE=Release` (or another non-empty value) when configuring.

2. **`ascendc_include_directories(<lib> <scope> <dirs>)` MUST be called AFTER `ascendc_library(<lib> ...)`**. `ascendc_library` is what creates the lib's `<lib>_interface` target; calling `ascendc_include_directories` before it (or with bare dirs and no preceding library) errors `"non-existent target ..._interface"`.

3. **`MultiCoreMatmulTiling` / `TCubeTiling` host include path + namespace**. `MultiCoreMatmulTiling` lives in `tiling/matrix/bmm_tiling.h`, namespace `matmul_tiling`, and needs `ascendc/include/highlevel_api` on the HOST include path. Qualify the enums explicitly: `matmul_tiling::{TPosition, CubeFormat, DataType}`. `TCubeTiling` uses PLAIN struct fields (e.g. `.dbL0C = 1`, no `set_*` setters) and exposes an `AscendC::tiling::TCubeTiling& GetTiling(...)` overload.

```cmake
# minimal MIX build skeleton (order matters)
add_compile_definitions(...)
ascendc_library(my_op_kernel STATIC ${KERNEL_SRCS})          # creates my_op_kernel_interface
ascendc_include_directories(my_op_kernel PRIVATE ${INC_DIRS}) # AFTER the library, not before
# host side: add ascendc/include/highlevel_api for bmm_tiling.h
target_include_directories(my_op_host PRIVATE ${ASCEND_HOME}/.../ascendc/include/highlevel_api)
```
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release ...   # non-empty build type — empty aborts merge_mix_obj.sh
```

**Other instances (predicted)**: any V220→A5 light-port of a cube-MIX fused op (FA-class, GDN/linear-attention family, fused norm+matmul) that keeps `matmul::MatmulImpl<>` and builds through `ascendc_library`.
