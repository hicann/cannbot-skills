---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "DataCopy with raw `__gm__` pointer silently returns zeros on Ascend950PR pure-AIV kernels"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all (any pure-AIV SIMD kernel using DataCopy for GM->UB reads)"
phenomenon: build_failure
signal:
  - "DataCopy(local, reinterpret_cast<__gm__ float>(gm_addr), byte_count) compiles cleanly (no warning, no error) but at runtime the destination local tensor contain"
confidence: single_run
original_id: EC-79
timestamp_inferred: true
tags: [507035, __gm__, ascendc, ec-79]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all (any pure-AIV SIMD kernel using DataCopy for GM->UB reads)`
`verified_on: soc=Ascend950PR_9579; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 — GM->UB DataCopy with raw pointers on V220 unconfirmed)`

- **Error pattern**: `DataCopy(local, reinterpret_cast<__gm__ float*>(gm_addr), byte_count)` compiles cleanly (no warning, no error) but at runtime the destination `local` tensor contains **all zeros** — the DataCopy reads zero from GM instead of the actual tensor data. All downstream computation produces zero/near-zero output. No runtime error, no 507035 — silent data corruption.
- **Root cause**: On Ascend950PR, `DataCopy` (the high-level AscendC API) expects a `GlobalTensor<T>` (or `LocalTensor<T>` for UB->UB) as the source/destination argument, not a raw `__gm__ T*` pointer. When passed a `reinterpret_cast<__gm__ T*>(gm_addr)`, the template resolves to a path that does not actually perform the GM->UB DMA — it produces zero-filled output without any diagnostic. The raw `__gm__` pointer path works in SIMT VF kernels (scalar pipe) but fails silently in pure-AIV SIMD class kernels.
- **Fix**: Use `GlobalTensor<T>` with `SetGlobalBuffer` + `operator[]` for all GM addressing in pure-AIV kernels:
  ```cpp
  // BEFORE (silently returns zeros):
  __gm__ float* gm_ptr = reinterpret_cast<__gm__ float*>(gm_addr);
  DataCopy(local, gm_ptr, N);  // compiles, reads zeros

  // AFTER (works correctly):
  GlobalTensor<float> gm_tensor;
  gm_tensor.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gm_addr));
  DataCopy(local, gm_tensor[offset], N);  // correct GM->UB DMA
  ```
  This is the proven pattern from FusedAddRmsnorm and other pure-AIV class kernels.
- **Distinct from PB-20**: PB-20 covers GM WRITE paths (`GlobalTensor::SetValue` silent no-op + raw `__gm__` pointer writes failing in pure-AIV). EC-79 covers the GM->UB READ path via `DataCopy` with a raw pointer — different API call, different direction, same root class (raw pointer vs GlobalTensor).
- **Detection**: kernel compiles and produces **all-zero output across ALL cases** (not just edge cases). If the entire output tensor is zero/epsilon and the kernel uses `DataCopy` with `reinterpret_cast<__gm__ T*>` arguments, suspect this first. A one-line probe: replace one `DataCopy(local, raw_gm_ptr, N)` with `DataCopy(local, gm_tensor[0], N)` and check if output becomes non-zero.
- **Evidence**: add_rms_norm_quant (2026-06-23, Ascend950PR_9579, CANN 9.0.0): aog-precision-probe iter 0 — kernel produced all-zero x_out and y1 across all 196 cases. Switching to GlobalTensor + SetGlobalBuffer + operator[] fixed immediately. FusedAddRmsnorm (earlier) used the GlobalTensor pattern from the start — no such issue.
- **Cross-reference**: PB-20 (GlobalTensor::SetValue silent no-op — same raw-pointer-vs-GlobalTensor class but WRITE direction), OL-77 (byte-copy loop workaround for reinterpret_cast GM->non-GM — adjacent API-surface mismatch).

<!-- 迁移自 porter kb/target/ascendc/（EC-79，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
