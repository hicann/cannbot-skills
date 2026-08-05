---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "VEC↔Scalar interleaved compute with paired V_S / S_V flags"
description: "Trigger: a compute pass needs to (a) produce a VEC result, (b) extract per-element scalars from that result for indirect-index lookup or per-element conditional, (c) feed scalars back into a VEC opera"
severity: critical
confidence: single_run
original_id: P-P64
timestamp_inferred: true
tags: [cooperative, optimization, v_s, s_v, p-p64, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Trigger**: a compute pass needs to (a) produce a VEC result, (b) extract per-element scalars from that result for indirect-index lookup or per-element conditional, (c) feed scalars back into a VEC operation.

**Pattern** (canonical 5-step):
```cpp
// 1. VEC produces result
Cast(bufferLocal, srcInt64, RoundMode::CAST_NONE, count);   // VEC: T -> int32

// 2. Wait for VEC writes to drain before scalar reads them
SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);

// 3. Scalar pipe loop reads + mutates + writes back
for (i = 0; i < count; i++) {
    int32_t v = bufferLocal.GetValue(i);
    bufferLocal.SetValue(i, transform(v));    // or SetValue elsewhere
}

// 4. Wait for scalar writes to drain before VEC reads them
SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);

// 5. Next VEC op
Add(out, bufferLocal, other, count);
```

**Why both flags are required**:
- `V_S` (VEC → Scalar fence): guarantees all VEC writes have committed to UB before scalar reads start. Without it, scalar may read stale data.
- `S_V` (Scalar → VEC fence): guarantees all scalar writes have committed before VEC reads start. Without it, VEC may read stale data.
- AscendC's `EnQue/DeQue` only syncs MTE2↔VEC and VEC↔MTE3 — it does NOT cover VEC↔Scalar. You MUST place these flags manually for the V↔S boundary.

**Failure mode without flags**: precision mismatch on cases where VEC and Scalar compute the same UB region in different orders. Often manifests as "passes when run alone, fails in batched run" — the most insidious symptom.

**Applicability**: any indirect-indexing op (gather/scatter where index needs scalar inspect), any per-row condition evaluation (e.g. `if (mask[i]) v += x`), any quantization step that reads scale-per-row scalars and applies them via VEC.

**Reference the helper template**: P-P64 cohabits naturally with the `SetWaitFlag<HardEvent>` template helper from CANN `advance_step_common.h`:
```cpp
template <HardEvent event>
__aicore__ inline void SetWaitFlag(HardEvent evt) {
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
    SetFlag<event>(eventId);
    WaitFlag<event>(eventId);
}
```
This collapses 3 lines (FetchEventID + SetFlag + WaitFlag) to one call. Worth pasting into per-op kernel headers.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/cooperative.md（P-P64，convert_patterns_to_okf.py）。confidence 未升格。 -->
