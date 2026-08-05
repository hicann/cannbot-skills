---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`EVENT_ID4..7` / `PIPE_FIX` undefined identifier in host_bisheng_obj preview compile — wrap kernel-side includes with `#if defined(__CCE_AICORE__)` guard"
description: "applies_to: soc=Ascend950PR_9579 (V351); cann=9.0.0; bisheng=15.0.5; op_class=port_a3_to_a5 (V220-pure entry wrapping kernels using 8-event double-buffer pipelines)"
phenomenon: build_failure
signal:
  - "(verbatim from build log):"
confidence: single_run
original_id: EC-64
timestamp_inferred: true
tags: [pipe_fix, ascendc, ec-64]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR_9579 (V351); cann=9.0.0; bisheng=15.0.5; op_class=port_a3_to_a5 (V220-pure entry wrapping kernels using 8-event double-buffer pipelines)`
`verified_on: flat_quant 2026-05-23 kw-1 — wrapping `op_kernel/flat_quant.cpp` via `kernel/flat_quant_kernels.cpp` thin TU. host_bisheng_obj pass failed; `#if defined(__CCE_AICORE__)` guard around the algorithm-header `#include` + `extern "C" __global__ __aicore__` body fixed it; device aic_obj / aiv_obj passes were unaffected.`

- **Severity**: BUILD-BREAK (build aborts at host preview stage; device passes succeed independently — easy to misdiagnose as "kernel is broken")
- **Status**: CONFIRMED 2026-05-23 (flat_quant kw-1)
- **Symptom** (verbatim from build log):
  ```
  error: use of undeclared identifier 'EVENT_ID4'
  error: use of undeclared identifier 'EVENT_ID5'
  error: use of undeclared identifier 'EVENT_ID6'
  error: use of undeclared identifier 'EVENT_ID7'
  error: use of undeclared identifier 'PIPE_FIX'
  ```
  Source location: a kernel-side `#include "<staged_algorithm>.h"` line in the worker-authored `kernels.cpp` TU.
- **Root cause**: NPUKernelBench's `ascendc_library` build runs a `host_bisheng_obj` preview compile pass on `kernels.cpp` to validate it parses in host-preview mode. That pass does NOT define `__NPU_ARCH__=3510` (only device passes do). CANN 9.0.0 `tools/tikicpulib/lib/include/stub_fun.h` only declares the extended `event_t` enumerators (`EVENT_ID4..7`) and the `PIPE_FIX` pipe-id under one of the device `__NPU_ARCH__` macros:
  ```c
  } event_t;
  #if defined(__NPU_ARCH__) && ((__NPU_ARCH__ == 2002) || (__NPU_ARCH__ == 2201)
    || (__NPU_ARCH__ == 3002) || (__NPU_ARCH__ == 3102) || (__NPU_ARCH__ == 3510)
    || (__NPU_ARCH__ == 5102))
      EVENT_ID4, EVENT_ID5, EVENT_ID6, EVENT_ID7,
  #endif
  ```
  Worker `.cpp` whose `#include`d kernel header references `EVENT_ID4..7` / `PIPE_FIX` in class member-default-initializers (typical of the V351-style 8-deep `DEvent<Pipe1,Pipe2>{EVENT_ID4, EVENT_ID5}` double-buffer template) → preview pass fails.
- **Fix**: wrap kernel-side `#include`s AND each `extern "C" __global__ __aicore__` function body with `#if defined(__CCE_AICORE__) … #endif`. The host preview pass doesn't define `__CCE_AICORE__` either, so the guarded region drops out of preview while remaining identical in the device aic_obj / aiv_obj compile.
  ```cpp
  #if defined(__CCE_AICORE__)
  #include "flat_quant_vec.h"
  #include "flat_quant_cube.h"
  // ... staged algorithm headers
  extern "C" __global__ __aicore__ void flat_quant(GM_ADDR x, GM_ADDR scale, ...) {
      // kernel body
  }
  #endif
  ```
- **Anti-pattern**: defining `EVENT_ID4` / `PIPE_FIX` yourself as a workaround. They ARE defined in device passes; the issue is host preview strictness. Guards are the load-bearing fix.
- **Scope note**: this fires specifically on V220+/V351 kernels using `EVENT_ID4..7` (8-event pipelines). Older V220 kernels using only `EVENT_ID0..3` don't hit this — they remain identifier-clean under host preview.
- **Detection** (grep signature):
  ```
  [host_bisheng_obj] use of undeclared identifier 'EVENT_ID4'
  [host_bisheng_obj] use of undeclared identifier 'PIPE_FIX'
  ```
  Build aborts at host_bisheng stage; device aic_obj / aiv_obj succeed independently.
- **Evidence**: flat_quant 2026-05-23 kw-1 (Ascend950PR V351): wrapping `op_kernel/flat_quant.cpp` via thin worker TU triggered all 5 error variants above; single `#if defined(__CCE_AICORE__)` guard around `#include`+body resolved on 2nd build, then PASS 8/8 T1 BIT_EXACT + 2.24× perf.
- **Extension — same guard rule applies to the regbase MicroAPI surface (2026-06-23, selective_scan_source_a5 perf-loop iter-2, Ascend950PR_957b, CANN 9.1.0.B060)**: a regbase `__simd_vf__` body + `using namespace AscendC::MicroAPI` ALSO requires the `#if defined(__CCE_AICORE__)` guard — same root cause (host preview pass lacks the device-only symbol surface), different symptom. On this CANN the host_bisheng preview pass does NOT expose `AscendC::MicroAPI` (`RegTensor`/`LoadAlign`/`StoreAlign` are absent from `include/ascendc/basic_api`; provided ONLY on the device pass). Unguarded → `error: expected namespace name` / `'MicroAPI' is not a namespace-name` on the host pass; guard wrap fixes it (device aic/aiv passes unaffected, identical to the EVENT_ID4 case above). Three companion build facts on this CANN, recorded for any regbase author: (1) `kernel_basic_intf.h` that the FA wholeport VF `#include`s is NOT shipped here — the FA wholeport VF would not build as-is; the MicroAPI surface is device-pass-only. (2) fp32 `VL = 64` (production `floatRepSize=64`), NOT the migration-doc's 128 — a regbase mask/tile loop sized for 128 mis-tiles fp32. (3) A regbase broadcast SCRATCH buffer must be sized to the full element count being broadcast (`lnElems`/CN), NOT a smaller chunk buffer (`LBUF_`/CH) — too-small → silent UB overflow → NaN (a control test with membase ops on the same rewiring also NaN'd, isolating the bug to the BUFFER, not the VF).
- **Cross-ref**: OL-132 (Mode A vs Mode B), OL-185 (V220→V351 port calibration anchor), OL-245 (regbase-default + its amortization boundary; the regbase VF this extension guards), `.upstream_prestaged.json` worker-authored dispatcher TU pattern.

<!-- 迁移自 porter kb/target/ascendc/（EC-64，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
