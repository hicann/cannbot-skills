---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "UB-to-UB DataCopy Silent Data Corruption"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "DataCopy(localDst, localSrc, count) between two LocalTensors (both in UB) silently produces garbage data. No compile error, no runtime error — just wrong values"
confidence: single_run
original_id: PB-9
timestamp_inferred: true
tags: [ascendc, pb-9]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass ops (17_EmbeddingWithInitialLayernormBackward, 20_FusedRopeWithQkNormAndKvCacheUpdate). Do not downgrade.
- **Symptom**: `DataCopy(localDst, localSrc, count)` between two LocalTensors (both in UB) silently produces garbage data. No compile error, no runtime error — just wrong values. Discovered when LayerNorm V2 passed for norm_size ≤ 4096 (single tile) but produced ~20% mismatch with mean_abs_diff ~1.14 for norm_size > 4096 (multi-tile). Removing the UB-to-UB DataCopy and operating directly on the dequeued tensor fixed it completely.
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Never copy between LocalTensors using DataCopy. Instead:
  - Operate directly on the source tensor (e.g., run BinaryFoldReduceSum on the dequeued xd tensor)
  - Use VEC ops as a "copy": `Adds(dst, src, 0.0f, count)` if you must copy
  - Or use `Duplicate` to zero a buffer, then `Add(dst, dst, src, count)`
- **Status**: OPEN
- **Evidence**: LayerNorm V2 debugging session 2026-04-09; kernel/layernorm_kernel.h Pass 1 fix
  - op#30 NMS a3 ds kw-1 (2026-05-07): Used `Adds(dst, src, 0.0f, count)` identity copy from TQue<VECIN,1> dequeued tensors to persistent UB compute buffers, avoiding UB→UB DataCopy entirely. 31/31 bit-exact vs Python CPU reference. Confirms Adds-identity as canonical V220 workaround for VECIN→VECCALC data movement.

<!-- 迁移自 porter kb/target/ascendc/（PB-9，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
