---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Ascend950PR int8 `Cast` silently drops tail elements when count not multiple of 32 (VEC 32B width)"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=quantization; dtype=int8"
phenomenon: build_failure
signal:
  - "Cast(dst_int8, src, RoundMode::CAST_TRUNC, count) where count is not a multiple of 32 produces zero-valued int8 elements at tail positions [floor(N/32)32, N). T"
confidence: single_run
original_id: PB-50
timestamp_inferred: true
tags: [507035, cast, ascendc, pb-50]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=quantization; dtype=int8`
`verified_on: soc=Ascend950PR_9579; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 — Cast int8 tail behavior on V220 unconfirmed; VEC width may differ)`

- **Severity**: HIGH — silent data loss: tail elements zeroed without compile error, runtime error, or NaN/Inf signal.
- **Symptom**: `Cast(dst_int8, src, RoundMode::CAST_TRUNC, count)` where `count` is not a multiple of 32 produces zero-valued int8 elements at tail positions `[floor(N/32)*32, N)`. The kernel compiles and runs cleanly — no 507035, no error code. Only precision verification reveals the zeroed tail. Observed at N=129, 130, 131, 144, 257 (all N where N % 32 ≠ 0).
- **Root cause**: The VEC `Cast` to int8 operates at 32B granularity (32 int8 elements per VEC operation). When `count` is not a multiple of 32, the hardware writes only `floor(count/32)*32` complete VEC blocks; the partial tail block is silently dropped (set to zero) rather than written with valid elements.
- **Workaround**: align the quantization count to 32 before `Cast`:
  ```cpp
  int32_t n_al_quant = ((N + 31) / 32) * 32;  // round up to 32B boundary
  // Size all quant-path buffers (i32Buf_, fp16Buf_, y1Buf_) for n_al_quant, not N
  Cast(y1Buf_, fp16Buf_, RoundMode::CAST_TRUNC, n_al_quant);
  ```
  Pass both `valid_count=N` and `aligned_count=n_al_quant` to the kernel so the output path knows where valid data ends. The pybind layer allocates int8 output with `AlignInt8(N_padded)` = N rounded up to 32, so the aligned tail elements land in the padded output region (discarded by pybind post-kernel narrow).
- **Detection**: int8 output shows zero values at tail positions for any N where `N % 32 ≠ 0`. If per-element `max_abs_diff` is non-zero ONLY at indices `[floor(N/32)*32, N)` and the diff magnitude equals the reference value (kernel=0, ref=non-zero), suspect this bug. Systematic sweep across N=128..257 will expose it.
- **Status**: OPEN (VEC hardware constraint; VEC int8 Cast width is 32B = 32 elements).
- **Evidence**: add_rms_norm_quant (2026-06-23, Ascend950PR_9579, CANN 9.0.0): aog-precision-probe iter 1 identified the tail-zero signature across 12 previously-failing N values. Align-to-32 fix resolved all 12.
- **Cross-reference**: PB-22 (similar 32B-alignment truncation for plain DataCopy on V351 — same hardware width class, different primitive), EC-23 (DataCopyPad UB->GM crash — adjacent alignment surface).

<!-- 迁移自 porter kb/target/ascendc/（PB-50，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
