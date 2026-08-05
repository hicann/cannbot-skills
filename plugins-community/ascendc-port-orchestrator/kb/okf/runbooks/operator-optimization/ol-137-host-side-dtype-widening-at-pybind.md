---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Host-side dtype-widening at pybind sidesteps unsupported in-kernel Cast source dtypes"
description: "For low-precision inputs (int8/uint8/bool) that need fp32 in compute, widening on the host at the pybind layer and passing an fp32 GM buffer avoids the AscendC Cast dtype-pair holes; complementary to the PB-26 in-kernel half-lift, chosen by a perf/complexity tradeoff per input."
original_id: OL-137
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-137, dtype-cast, pybind, quantization]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

AscendC `Cast(dst, src, mode, count)` only supports a finite dtype-pair table (see PB-26 for
the documented holes — `uint8→float`, `uint8→bfloat16_t`, `uint8→int32`, etc., are silent
miscompile / silent garbage, not compile errors). When a kernel reads a low-precision input
(int8 / uint8 / bool mask) and the compute path requires fp32, the in-kernel mitigation is a
two-step lift through `half` (PB-26 §Workaround). An equally valid — often simpler —
alternative is to **do the widening on the host side at the pybind layer** and pass an fp32 GM
buffer to the kernel; the kernel then reads `__gm__ float*` and emits no `Cast` at all.

**Applies to** `soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=all` (any
kernel reading low-precision integer or sub-fp32 input that needs fp32 in compute). Verified
on Ascend950PR / cann 9.0.0. Unverified on Ascend910_V220 (same Cast surface exists; transfer
likely but not probed).

### Choose host-side widen when

- The widened input is small relative to total HBM traffic (per-channel scale/offset vectors
  of length D ≤ a few thousand fp32 elements = ≤ tens of KB extra HBM).
- The kernel's compute is the perf bottleneck, NOT the host→device transfer.
- Simplicity of the kernel (no `Cast` chain, no scratch `tmp_half` TBuf) matters more than the
  marginal HBM cost.

### Choose in-kernel half-lift (PB-26 workaround) when

- The low-precision input is large (a full per-element mask the size of `x`) — host-side widen
  would multiply HBM read by 4× and become bandwidth-dominated.
- Re-running the cast on every kernel launch is cheaper than maintaining a separate fp32
  mirror tensor on host.

The two paths are **complementary**, not exclusive alternatives — the choice is a
perf/complexity tradeoff per input.

### Concrete anchor (rms_norm_quant pybind, 2026-05-13)

```cpp
// pybind11.cpp — int8 offset (per-channel, length D ≤ 4096) widened to fp32 on host.
torch::Tensor offset_fp32_full = offset_full.to(at::kFloat).contiguous();
// Kernel reads gmOffsetF32_ as __gm__ float*; no in-kernel Cast(float, int8) needed.
```

Cost analysis for the rms_norm_quant case: D=512 × 4 bytes = 2KB extra HBM per launch,
negligible against the [N=4, D=512] fp32 mainline traffic (~32KB). The `Cast(float, int8)`
ambiguity (PB-26 documents the unsigned variant; signed `int8→float` had similar suspicion)
is avoided entirely.

### Evidence

- rms_norm_quant kw-1 (2026-05-13): host-side int8→fp32 widen for the per-channel quant
  offset. 8/8 Pass A + 8/8 Pass B PASS first try; no in-kernel Cast chain ambiguity.
  Performance: 6.60× over the Path-A reference (the host-side widen cost is invisible in
  end-to-end measurement).

**Predicted other instances**: any quant kernel that reads `scale=int8/uint8` or
`offset=int8/uint8` per-channel vectors; any kernel reading boolean masks (uint8/bool) when
the mask is short-lived metadata. (Source text truncated at this point.)
