---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "DataCopy to TBuf<VECCALC> — silent corruption on multi-iteration loops"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "DataCopy(TBuf<VECCALC>::Get<T>(), GM_tensor, count) with manual SetFlag<MTE2_V>/WaitFlag<MTE2_V> sync produces correct data on the first loop iteration but stal"
confidence: single_run
original_id: PB-11
timestamp_inferred: true
tags: [ascendc, pb-11]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (17_EmbeddingWithInitialLayernormBackward). Do not downgrade.
- **Symptom**: `DataCopy(TBuf<VECCALC>::Get<T>(), GM_tensor, count)` with manual `SetFlag<MTE2_V>/WaitFlag<MTE2_V>` sync produces correct data on the first loop iteration but stale/corrupt data on subsequent iterations. Related to PB-9 (both are DataCopy corruption) but distinct mechanism: PB-9 is UB→UB; PB-11 is GM→VECCALC with manual sync in a loop.
- **Affected**: Ascend950PR, CANN 9.0.0
- **Workaround**: Use `TQue<VECIN, 1>` instead of `TBuf<VECCALC>` for any buffer that receives DataCopy from GM in a loop. The TQue's `AllocTensor/EnQue/DeQue/FreeTensor` pattern provides reliable MTE2→VEC synchronization. Single-iteration usage of TBuf<VECCALC> with DataCopy appears safe.
- **Status**: OPEN
- **Evidence**: DynamicQuant (#29) smooth_scales — 3 cases with row_size > TILE_SIZE failed (2.7%-22.6% mismatch, max_abs_diff=252) due to stale smooth_scales data on 2nd+ tile. First mismatch always at exact TILE_SIZE boundary. Fixed by switching to TQue<VECIN,1>. 42/42 PASS after fix.

## How to Add New Bugs

Append to the appropriate section with:
```
### PB-N: Short Description
- **Symptom**: What you observe
- **Affected**: Platform/version
- **Workaround**: How to work around it
- **Status**: OPEN/FIXED(version)/BY_DESIGN
- **Evidence**: Link to test/doc
```

<!-- 迁移自 porter kb/target/ascendc/（PB-11，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
