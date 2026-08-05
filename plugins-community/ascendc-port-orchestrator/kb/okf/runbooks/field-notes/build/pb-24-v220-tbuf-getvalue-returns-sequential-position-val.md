---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220 TBuf `GetValue` returns sequential-position values (not stored data) when interleaved with TQue `CopyTile` operations [V220]"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "When scalar values are read from a TBuf<VECCALC> via GetValue() AND the same kernel also uses TQue EnQue/DeQue (CopyTile) in interleaved fashion, the TBuf GetVa"
confidence: single_run
original_id: PB-24
timestamp_inferred: true
tags: [getvalue, copytile, ascendc, pb-24]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Severity**: CRITICAL (silent data corruption — kernel produces wrong output with no error)
- **Status**: CONFIRMED 2026-05-02 op#18 Index pp-1 (12 bisection iterations, V220 CANN 9.0.0)
- **Symptom**: When scalar values are read from a `TBuf<VECCALC>` via `GetValue()` AND the same kernel also uses TQue `EnQue/DeQue` (CopyTile) in interleaved fashion, the TBuf GetValue returns the **sequential position index** (0, 1, 2, ...) instead of the actual stored value. All GM scalar read paths are affected: `GlobalTensor::GetValue`, raw `__gm__` pointer dereference, TBuf `DataCopy` with MTE2_V sync.
- **Fix**: Read ALL TBuf values into a local C++ stack array BEFORE any TQue CopyTile operations begin. This eliminates the interleaving that triggers the corruption.
  ```cpp
  // Read all indices before any TQue operations
  int32_t idx_buf[MAX_INDICES];
  for (int i = 0; i < n_indices; ++i)
      idx_buf[i] = idxBuf_.GetValue<int32_t>(i);
  // NOW start TQue CopyTile operations using idx_buf[]
  ```
- **Evidence**: op#18 Index DS kw-1 (2026-05-02): 9 iterations tested all GM read paths — all returned sequential positions. pp-1 (2026-05-02): 12 bisection iterations confirmed the interleaving root cause. kw-2 applied the "read-all-before-TQue" fix → 41/41 PASS (bit-exact, MERE=0, MARE=0), perf 10.78x vs CANN.
- **Other instances (predicted)**: any V220 kernel that mixes TBuf scalar reads with TQue pipeline operations, especially index/gather/scatter ops where indices are loaded via TBuf.
- **Cross-ref**: PB-22 (MTE2 DataCopy 32B limit), OL-124 (TBuf→MTE3 coherence), OL-123 (V220 API gaps).

<!-- 迁移自 porter kb/target/ascendc/（PB-24，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
