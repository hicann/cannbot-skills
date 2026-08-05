---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Lift loop-invariant scalar instructions out of per-tile loops"
description: "Scalar mode-set instructions (SetDeqScale, SetMaskNorm, SetVectorMask) with constant args inside a per-tile loop waste scalar-pipe cycles per iter; hoist to once-per-row/launch. Verifiable as aiv_scalar_time reduction."
confidence: single_run
original_id: OL-116
classified_by: llm-assisted
timestamp_inferred: true
tags: [scalar-pipe, optimization, ol-116, loop-invariant, tile-loop]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**When to use:** a per-tile loop contains a scalar-config instruction (`SetDeqScale`, `SetMaskNorm`, `SetVectorMask`, `SetSysWorkspace`, or a similar mode-set op) whose argument is loop-invariant. Loaded by aog-kernel-worker (writing a tile-loop kernel) and aog-kernel-optimizer (when msprof shows `aiv_scalar_ratio > 0.20` on a multi-tile path).

**Pattern.** Scalar mode-set instructions emitted inside a per-tile loop with constant arguments are wasted scalar-pipe cycles per iter. Lift them to once-per-row (or once-per-launch) before the loop. Each instruction is cheap (~5-10 cycles) but the cost accumulates as `tile_count × N_rows`; on multi-tile cases (tile_count 3-7) with thousands of rows this can shave 5-15% scalar overhead.

### Concrete anchor
```cpp
// BEFORE (per-tile re-issue)
for (int32_t t = 0; t < tile_count; ++t) {
    SetDeqScale(1.0f);   // CONSTANT — wasted re-issue
    Cast(int8Buf, fp16Buf, RoundMode::CAST_TRUNC, cnt);
    DataCopy(gm_out, int8Buf, cnt);
}

// AFTER (lifted to once-per-row, or higher if applicable)
SetDeqScale(1.0f);   // ONCE
for (int32_t t = 0; t < tile_count; ++t) {
    Cast(int8Buf, fp16Buf, RoundMode::CAST_TRUNC, cnt);
    DataCopy(gm_out, int8Buf, cnt);
}
```
Verifiable in msprof as an `aiv_scalar_time` reduction. This is a generalization of OL-32 (avoid CPU-side fills) to "any loop-invariant instruction".

### Evidence
- 29_DynamicQuant ko-2 (2026-05-02): `SetDeqScale(1.0f)` was lifted out of the per-tile loop in `QuantizeRow`; combined with manual prefetch (OL-115), the iter contributed to `aiv_scalar_ratio` 0.354 → ~0.22 (closer to CANN's 0.224).

### Other instances (predicted)
- Any kernel with `SetDeqScale` / `SetMaskNorm` inside a per-tile or per-row loop with constant args
- `SetVectorMask` config issued repeatedly when a single set would suffice
- Tiling-config instructions (rare in VEC kernels, appear cube-side)

### Related
- OL-32 (avoid CPU-side fills — same anti-pattern at a different layer)
- OL-115 (manual prefetch — both target scalar/MTE2 overhead in tile loops; commonly applied together)
