---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "TQue QBUF_DEPTH=2 is the default for SIMD kernels with ≥2 GM reads per row/tile — hides MTE latency"
description: "TBuf + synchronous DataCopy/SetFlag/WaitFlag stalls VEC on every GM→UB copy; TQue depth-2 prefetches the next tile while VEC works — contributed ~15% of add_rms_norm_quant's 1.94× speedup."
confidence: single_run
original_id: OL-258
classified_by: llm-assisted
timestamp_inferred: true
tags: [double-buffering, optimization, ol-258, tque, mte-latency]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**When to use:** a SIMD kernel with ≥2 distinct GM input tensors per row/tile.

**Principle.** TBuf + synchronous `DataCopy`/`SetFlag`/`WaitFlag` makes VEC stall on every GM→UB transfer. TQue with `QBUF_DEPTH=2` lets the MTE **prefetch the next tile** while VEC processes the current one, overlapping MTE and VEC.

**Decision table:**

| Condition | Use |
|---|---|
| ≥2 GM reads per row/tile AND `(QBUF_DEPTH=2 × tile_buffer) < UB/4` | **TQue QBUF_DEPTH=2** (MUST) |
| 1 GM read (single input) | TBuf OK |
| UB budget too tight (justify in analysis.md) | TBuf with explicit justification |

### Concrete anchor
```cpp
// V1 (TBuf + sync — VEC idle):
DataCopy(x1Fp32, x1_gm_[off], N);
SetFlag<HardEvent::MTE2_V>(evM2V);
WaitFlag<HardEvent::MTE2_V>(evM2V);   // VEC STALLS

// V2 (TQue QBUF_DEPTH=2 — MTE and VEC overlap):
pipe_.InitBuffer(inQueueX1_, 2, tile_size * sizeof(T_X));
CopyInTile(inQueueX1_, x1Gm_, off, N);           // EnQue: MTE starts, VEC continues
LocalTensor<T_X> x1 = inQueueX1_.DeQue<T_X>();   // non-blocking
```

### Evidence
add_rms_norm_quant V1→V2 (Ascend950PR_957b, CANN 9.0.0, 2026-06-24): TQue double-buffering contributed ~15% of the 1.94× speedup. The existing FusedAddRmsnorm archive also uses TQue.

### Related
- OL-94 (TQue vs TBuf sync decision table), OL-256 (Divs→Muls — the VEC-side optimization TQue amplifies), OL-247 (pipe_barrier between VEC and MTE2 — TQue avoids this hazard).
