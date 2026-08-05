---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`matmul_intf.h` is NOT included transitively from `kernel_operator.h` — must be explicit `#include` OR per-pass interface-library propagation [V220+V351, ALL_MODES, build-include + KFC-sync-activation]"
description: "applies_to: soc=Ascend910_9382 (V220) + Ascend950PR_9579 (V351); cann=9.0.0+; bisheng=ascendc.cmake DYNAMIC_MODE; op_class=mixed_aic_aiv_with_REGIST_MATMUL_OBJ"
phenomenon: build_failure
signal:
  - "Build fails with error: ‘MatmulImpl’ has not been declared OR ‘REGIST_MATMUL_OBJ’ does not name a type even though the kernel #include <kernel_operator.h>. Addi"
confidence: single_run
original_id: EC-58
timestamp_inferred: true
tags: [ascendc, ec-58]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382 (V220) + Ascend950PR_9579 (V351); cann=9.0.0+; bisheng=ascendc.cmake DYNAMIC_MODE; op_class=mixed_aic_aiv_with_REGIST_MATMUL_OBJ`
`verified_on: a5_ops:3_FusionAttention 2026-05-21 + probe_a5_v300_fa_sync 2026-05-23 — both architectures need explicit matmul_intf.h plumbing`

- **Symptom**: Build fails with `error: ‘MatmulImpl’ has not been declared` OR `‘REGIST_MATMUL_OBJ’ does not name a type` even though the kernel `#include <kernel_operator.h>`. Adding `#include <matmul_intf.h>` (or letting the per-pass `_aic_intf_pub` / `_aiv_intf_pub` interface libraries propagate it) resolves it.
- **Root cause**: `kernel_operator.h` does NOT transitively include `matmul_intf.h`. The matmul interface is a separate header that ships with the AscendC SDK under `tikcpp/tikcfw/interface/matmul_intf.h`. ascendc.cmake's `legacy_modules/device_preprocess_project/CMakeLists.txt` exposes it via `${BUILD_MODE}_aic_intf_pub` / `${BUILD_MODE}_aiv_intf_pub` INTERFACE libraries, which `target_link_libraries(<sub-target> PUBLIC <intf_pub>)` propagates only when the sub-target opts in.
- **Fix** — pick ONE of:
  1. **Explicit `#include "matmul_intf.h"` in every kernel file using `MatmulImpl<>` / `REGIST_MATMUL_OBJ`** (simplest, no CMake changes; recommended for project-local kernels).
  2. **Per-pass interface-library propagation** (cleaner, only used when ascendc.cmake's full DYNAMIC_MODE pipeline is active): rely on the `_aic_intf_pub` / `_aiv_intf_pub` link chain. Caveat: if `build_ascendc.py` bypasses the legacy_modules path (e.g., NPUKernelBench's slim build), the interface libraries aren't created and Option 1 is mandatory.
- **Detection** (build-fail signature): exact error `error: ‘REGIST_MATMUL_OBJ’ does not name a type` OR `error: ‘MatmulImpl’ does not name a type` — search the file for `kernel_operator.h` include without adjacent `matmul_intf.h` include.
- **Evidence**:
  - 3_FusionAttention 2026-05-21: independent Pattern B build attempts failed with this exact error before adding explicit `#include "matmul_intf.h"`.
  - probe_a5_v300_fa_sync 2026-05-23: main's A5 Pattern B probe also tripped on this until explicit include was added.
- **Cross-ref**: EC-57 (sibling KFC sync activation issue — both share DEBT-20/20.1 family root cause), DEBT-20 follow-up notes in `66a4d985`.

<!-- 迁移自 porter kb/target/ascendc/（EC-58，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
