---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Partial-vector alignment — use AlignUp for VEC accumulation while restricting GM access to the actual element count for sub-vector-width dimensions"
description: "For sub-vector-width dimensions, use AlignUp(count) for VEC accumulation (alignment to issue) but restrict GM read/write to the actual element count to avoid adjacent-data bleed."
confidence: single_run
original_id: OL-221
classified_by: llm-assisted
timestamp_inferred: true
tags: [alignment, optimization, ol-221, vec-pipe, sub-vector-width, pooling]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Applies to**: Ascend950PR / CANN 9.0.0 / all op classes (ascendc backend). Verified 2026-06-16.

**Principle**: AscendC VEC ops (Duplicate, Add, Muls, Cast) on fp32 require element counts aligned to the hardware vector width (`numPerBlock`, typically 8 for fp32 on Ascend950PR). When a dimension (e.g. channel count C) is smaller than the vector width, a naive VEC op on `AlignUp(count, numPerBlock)` reads/writes beyond the valid data range — loading adjacent spatial positions' channels during input and over-writing adjacent output positions.

**The fix**: use `AlignUp(count, numPerBlock)` for the ACCUMULATION ops (which need alignment to issue) while RESTRICTING GM reads and writes to the ACTUAL element count. This decouples the VEC pipe's alignment requirement from the memory-access footprint — the VEC pipe sees an aligned count for correct instruction issue, but the MTE pipe only touches valid data.

**Concrete anchor** (adaptive_avg_pool3d pooling, V220→A5 port, fp32 small-C):
```cpp
// C dimension may be smaller than numPerBlock (8 for fp32)
constexpr uint32_t numPerBlock = 8;
uint32_t cAligned = AlignUp(td.cDim, numPerBlock);
// Accumulate over aligned count (VEC pipe requires aligned element count)
Duplicate(bufAccum, scalarVal, cAligned);
// But GM read/write ONLY the actual C elements
DataCopy(bufAccum, gmIn + spatialOffset, td.cDim);   // restricted to actual count
// ... compute on bufAccum (cAligned-wide, last (cAligned-cDim) elements are padding) ...
DataCopy(gmOut + spatialOffset, bufAccum, td.cDim);  // restricted write
```

**Evidence**: adaptive_avg_pool3d L1 V220→A5 port (2026-06-16): fp32 small-C pooling kernel. Without the alignment fix, small-C cases (C < 8) produced wrong outputs due to adjacent-channel bleed from AlignUp reads over-running the valid C range. With AlignUp accumulation + restricted GM access, 30/30 cases bit-exact vs CPU truth.

**Other instances (predicted)**: any reduction/pooling/normalization op whose inner dimension can be smaller than vector width (group_norm with small groups, layer_norm with small hidden_dim, small-channel convolutions); gather/scatter with per-element counts below numPerBlock; any VEC op on a dimension whose size is not compile-time-guaranteed to be a multiple of numPerBlock.

**Cross-ref**: OL-124 (TQue<VECOUT> constraint — related output-queue alignment); PB-22 (DataCopy 32B alignment restriction — the GM-side complement of this UB-side rule).
