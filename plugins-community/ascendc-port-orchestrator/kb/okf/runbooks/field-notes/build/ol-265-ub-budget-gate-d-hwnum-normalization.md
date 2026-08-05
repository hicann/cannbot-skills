---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "UB budget gate for D×hwNum normalization/elementwise ops (error 271)"
description: "When per-slice UB scales as D × hwNumAligned × sizeof(fp32) × num_buffers, large (D, hwNum) combos exceed UB capacity and crash with error 271; gate on the total UB allocation in Init()."
phenomenon: build_failure
signal:
  - "Runtime crash error 271 (\"MPU address access is invalid\") on a D>1 normalization/elementwise op when (3·D+1) × hwNumAligned × sizeof(fp32) + margin exceeds the UB physical capacity."
confidence: single_run
original_id: OL-265
classified_by: llm-assisted
timestamp_inferred: true
tags: [ub-budget, build, ol-265, normalization, error-271, group-norm-silu]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

For normalization/elementwise ops where the per-slice UB working set scales as `D × hwNumAligned_ × sizeof(float) × num_buffers`, large `(D, hwNum)` combinations exceed the UB physical capacity and crash at runtime with **error 271 ("MPU address access is invalid")**.

Observed on group_norm_silu (2026-06-26, A5/Ascend950PR, CANN 9.0.0): `D=16 × hwNum=1024 × 3 buffers × 4 bytes = 196608 B` for the fp32 tensors alone, plus gamma/beta (~8 KB) and tiling scratch → ~272 KB total → exceeds the 192 KB V220 limit → error 271.

## 根因 / 教训

The per-slice buffers that scale with the D dimension (not shared across channels) blow the UB budget. Model the total allocation and gate on it.

**UB budget model** for a D>1 fullCache normalization kernel (fp32):
- `xfp32`: D × hwNumAligned_ × 4  (input)
- `tmpFp32`: D × hwNumAligned_ × 4  (normalized)
- `workFp32`: D × hwNumAligned_ × 4  (activation scratch, e.g. SiLU)
- `gammaFp32` / `betaFp32`: hwNumAligned_ × 4 each  (shared across D, **not** ×D)
- Stack + tiling scratch: ~10 KB safety margin
- **Total**: `(3·D + 1) × hwNumAligned_ × 4 + 10240` bytes

**Capacities:**
- V220 (A3/A2, Ascend910): 192 KB (196608 B)
- V351 (A5, Ascend950PR): 248 KB usable (253952 B), 256 KB physical

### Decision rule

Add a UB budget check in `Init()` and refuse configurations that exceed capacity:

```cpp
void Init(...) {
    // ... compute hwNumAligned_ ...
    uint32_t ubNeeded   = (3 * shapeD_ + 1) * hwNumAligned_ * sizeof(float) + 10240;
    uint32_t ubCapacity = 192 * 1024;  // conservative: V220 floor for portability
    if (ubNeeded > ubCapacity) {
        // REFUSE_LAUNCH with a clear diagnostic (SetStatus / poison output), then return
        return;  // skip kernel execution
    }
}
```

Mirror the gate host-side so the test harness skips cases that would crash:

```python
ub_needed   = (3 * D + 1) * hwNumAligned * 4 + 10240
ub_capacity = 192 * 1024  # V220 floor
if ub_needed > ub_capacity:
    skip(f"UB budget exceeded: {ub_needed}B > {ub_capacity}B")
```

### Evidence

group_norm_silu (2026-06-26, A5/Ascend950PR, CANN 9.0.0): D=16 × hwNum=1024 × 3-buffer fp32 = 196608 B + gamma/beta + scratch → ~272 KB → error 271. Same D×hwNum combos also fail on V351 if the 3-buffer model exceeds 248 KB. Mitigation: `bench_speedup.py` filters cases where `(3D+1) · hwNumAligned · sizeof_T + 10240 > 192*1024`.

### Other instances (predicted)

Any multi-channel normalization/elementwise op where per-slice buffers are not shared across D: GroupNorm with D>1 (per-channel affine + activation), InstanceNorm / LayerNorm with batched affine params, batched SiLU/GELU/Swish activation (D copies of the work buffer), and fused norm+activation kernels.
