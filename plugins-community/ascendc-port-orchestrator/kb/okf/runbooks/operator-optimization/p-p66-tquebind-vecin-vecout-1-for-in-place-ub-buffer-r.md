---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "`TQueBind<VECIN, VECOUT, 1>` for in-place UB buffer reuse"
description: "Trigger: a compute pass reads a buffer, mutates in-place, writes the same buffer back to GM. Separate TQue<VECIN> + TQue<VECOUT> would consume 2× the UB. Pattern: cpp TQueBind<QuePosition::VECIN, QueP"
confidence: single_run
original_id: P-P66
timestamp_inferred: true
tags: [memory_access, optimization, tquebind, p-p66, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Trigger**: a compute pass reads a buffer, mutates in-place, writes the same buffer back to GM. Separate `TQue<VECIN>` + `TQue<VECOUT>` would consume 2× the UB.

**Pattern**:
```cpp
TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> buffer1;

pipe_->InitBuffer(buffer1, /*depth=*/1, /*bytes=*/tileSize);

// CopyIn:
LocalTensor<T> local = buffer1.AllocTensor<T>();
DataCopyPad(local, srcGm[offset], params, padParams);
buffer1.EnQue(local);

// Compute (mutate in place):
LocalTensor<T> local = buffer1.DeQue<T>();
Cast(local, local, RoundMode::CAST_NONE, count);
Adds(local, local, 1, count);
buffer1.EnQue(local);

// CopyOut:
LocalTensor<T> local = buffer1.DeQue<T>();
DataCopyPad(dstGm[offset], local, params);
```

**Trade-off**:
- `TQueBind<VECIN, VECOUT, 1>` = 1 × tileSize UB, forced serial CopyIn → Compute → CopyOut on each tile.
- `TQue<VECIN, depth=2>` + `TQue<VECOUT, depth=2>` = 4 × tileSize UB, but CopyIn(iter+1) overlaps Compute(iter).

**When to use**:
- UB budget is tight (multi-tensor fused ops where every KB matters; cf. op#11 DequantSwigluQuant peak 194/192 KB).
- Op is memory-bound on small tiles (pipeline depth doesn't help — bandwidth saturates anyway).
- In-place compute fits in one VEC sequence (no need to keep src + dst alive simultaneously).

**When NOT to use**:
- Op is VEC-bound and benefits from depth ≥ 2 pipelining (don't trade throughput for UB).
- Input and output dtypes/sizes differ (would need 2 separate physical slots regardless).

**Related to PB-17 (P-P65)**: PB-17 is the aggressive form — manually aliasing two named TBufs onto the same physical UB slot, which the framework does not manage. `TQueBind` is the framework-managed version with proper EnQue/DeQue protocol; **use it first**, only escalate to manual aliasing if `TQueBind` doesn't fit the access pattern.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P66，convert_patterns_to_okf.py）。confidence 未升格。 -->
