---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "VEC pipeline sync choice — TQue auto-rotation vs manual TBuf event sync"
description: "TQue<VECIN/VECOUT> carries implicit MTE2_V / V_MTE3 barriers (default, fewer footguns); bare TBuf<VECCALC> has NO implicit sync and needs explicit SetFlag/WaitFlag — PipeBarrier<PIPE_ALL> is not a safe substitute (PB-21 507015 crash)."
confidence: single_run
original_id: OL-94
classified_by: llm-assisted
timestamp_inferred: true
tags: [platform-compat, optimization, ol-94, tque, tbuf, sync, pipeline]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**When to consult:** designing or refactoring a VEC pipeline (DataCopy in → VEC compute → DataCopy out, repeated per row/tile), and choosing between `TQue<VECIN/VECOUT, depth>` queues and bare `TBuf<VECCALC>` buffers as the UB-resident pipeline primitive — or evaluating an existing kernel for whether it should switch.

### Principle — the two UB primitives have different sync contracts

- **`TQue<VECIN/VECOUT, depth=N>`**: hardware-managed queue rotation between MTE2/V/MTE3 stages. Each `EnQue/DeQue` pair carries an implicit `MTE2_V` or `V_MTE3` barrier — no explicit `SetFlag/WaitFlag` needed. Auto-rotation also enables overlap when `depth ≥ 2`.
- **`TBuf<VECCALC>`**: bare UB allocation, NO implicit sync. Sync between MTE2 (DataCopy in) and V (compute), and between V and MTE3 (DataCopy out), MUST be explicit `SetFlag<HardEvent::MTE2_V>` / `WaitFlag<HardEvent::MTE2_V>` (and the `V_MTE3` analogue). `PipeBarrier<PIPE_ALL>()` is NOT a safe substitute — it fires the 507015 silent crash on V220 with this pattern (PB-21).

### Decision table

| Condition | Pick TQue auto-rotation | Pick TBuf + manual event sync |
|-----------|-------------------------|-------------------------------|
| Kernel has a clear in→compute→out pipeline shape | default | only if other constraints force it |
| Inputs/outputs split across multiple buffers needing distinct lifetimes | hard to express in TQue | preferred |
| Need to alias UB buffers across phases (P-P65 cross-phase liveness) | TQue does not allow alias | preferred |
| Buffer reused across many iters with internal sub-loops | TQue depth is fixed at construction | flexible |
| Worker unsure / first cold-start of an op | default — fewer footguns | only after profiling shows TQue can't express the dataflow |
| Code uses `PipeBarrier<PIPE_ALL>()` between MTE2 and V | TQue handles this implicitly | MUST replace with `SetFlag/WaitFlag` (PB-21) |

### Concrete anchor — manual TBuf pipeline (V220-confirmed; A5 likely-applicable)

```cpp
// Per-kernel one-time event ID fetch
uint16_t evMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
uint16_t evVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);

TBuf<VECCALC> bufA_;
// ... bufA_.Init / pipe.InitBuffer(bufA_, ...) ...

for (int r = 0; r < numRows; ++r) {
    LocalTensor<T> a = bufA_.Get<T>();
    DataCopyPad(a, gmX_[r * H], cp, padParams);
    SetFlag<HardEvent::MTE2_V>(evMte2V);          // mark MTE2 done
    WaitFlag<HardEvent::MTE2_V>(evMte2V);         // V waits for MTE2

    // ... VEC compute on `a` ...

    SetFlag<HardEvent::V_MTE3>(evVMte3);          // mark V done
    WaitFlag<HardEvent::V_MTE3>(evVMte3);         // MTE3 waits for V
    DataCopy(gmOut_[r * H], a, H);
}
```

Rule of thumb: default to TQue for a plain in→compute→out pipeline; reach for TBuf + manual event sync only when buffer lifetimes / aliasing / sub-loop reuse cannot be expressed in a fixed-depth queue — and when you do, never leave a `PipeBarrier<PIPE_ALL>()` standing in for the MTE2↔V / V↔MTE3 barrier.
