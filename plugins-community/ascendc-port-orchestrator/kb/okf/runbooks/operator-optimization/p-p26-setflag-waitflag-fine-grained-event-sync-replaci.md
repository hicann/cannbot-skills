---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SetFlag/WaitFlag fine-grained event sync (replacing PipeBarrier<PIPE_ALL>)"
description: "Problem: PipeBarrier<PIPE_ALL>() blocks all pipes, preventing MTE2/VEC/MTE3 parallelism. Correct pattern: use SetFlag/WaitFlag + HardEvent to specify cross-pipe dependencies precisely. cpp // MTE2→Sca"
severity: high
confidence: single_run
original_id: P-P26
timestamp_inferred: true
tags: [memory_access, optimization, setflag, waitflag, hardevent, p-p26, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: `PipeBarrier<PIPE_ALL>()` blocks all pipes, preventing MTE2/VEC/MTE3 parallelism.

**Correct pattern**: use `SetFlag`/`WaitFlag` + `HardEvent` to specify cross-pipe dependencies precisely.

```cpp
// MTE2→Scalar: can only GetValue from UB after DataCopy completes
event_t id = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
SetFlag<HardEvent::MTE2_S>(id);
WaitFlag<HardEvent::MTE2_S>(id);

// VEC→Scalar: can only GetValue the result after ReduceSum completes
event_t id2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(id2);
WaitFlag<HardEvent::V_S>(id2);

// Scalar→VEC: can only run Muls after SetValue completes
event_t id3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
SetFlag<HardEvent::S_V>(id3);
WaitFlag<HardEvent::S_V>(id3);

// Scalar→MTE3: can only DataCopyPad back after SetValue completes
event_t id4 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
SetFlag<HardEvent::S_MTE3>(id4);
WaitFlag<HardEvent::S_MTE3>(id4);
```

**Interaction with TQue**: TQue<VECIN,2>'s AllocTensor/EnQue/DeQue/FreeTensor manages double-buffering automatically. SetFlag/WaitFlag is for scalar↔vector sync outside of TQue.

**Note**: OL-4 (TQue data corruption) may be a specific buffer-size / configuration issue. E12 expert code uses TQue<VECIN,2> + SetFlag and works correctly.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P26，convert_patterns_to_okf.py）。confidence 未升格。 -->
