---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`PipeBarrier<PIPE_S>` not valid on Ascend950PR"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - ""
confidence: single_run
original_id: EC-15
timestamp_inferred: true
tags: [ascendc, ec-15]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**:
  ```
  error: the range of 1st parameter must be [4, 6]
  ```
  (from `kernel_reg.h`, triggered by `PipeBarrier<PIPE_S>()`)
- **Root cause**: On Ascend950PR, `pipe_barrier()` only accepts pipe values 4 (PIPE_MTE2), 5 (PIPE_V), 6 (PIPE_MTE3). The scalar pipe (PIPE_S) is not supported for PipeBarrier. To synchronize the scalar pipe, use `SetFlag`/`WaitFlag` with appropriate event types.
- **Fix**:
  ```cpp
  // BEFORE (fails):
  AscendC::PipeBarrier<PIPE_S>();

  // AFTER (S→MTE3 sync for scalar writes visible to MTE3 output):
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_MTE3));
  AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(ev);
  AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(ev);

  // For S→V sync (scalar writes visible to VEC):
  event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::S_V));
  AscendC::SetFlag<AscendC::HardEvent::S_V>(ev);
  AscendC::WaitFlag<AscendC::HardEvent::S_V>(ev);
  ```
- **Valid PipeBarrier pipes**: PIPE_MTE2 (4), PIPE_V (5), PIPE_MTE3 (6) — that's it.
- **Evidence**:
  - Sort V1 build failure (2026-04-09)
  - clipped_swiglu port_a3_to_a5 kw-1 (2026-05-17): scalar→vector gather pattern (interleaved A/B half extraction via `LocalTensor::SetValue(i, …)` feeding subsequent `Cast` / `Mins` / `Maxs`) tripped `kernel_reg.h:85` "range of 1st parameter must be [2,6],[10,10]" on AIC and "[4,6]" on AIV. Fixed in iter 2 by replacing `PipeBarrier<PIPE_S>()` with `SetFlag/WaitFlag<HardEvent::S_V>` at two sync sites (F32 interleaved branch + Half interleaved branch). 8/8 cases PASS post-fix. General trigger class: any kernel that scalar-gathers a strided/masked pattern (even/odd interleave, group indexing) into UB that subsequent VEC ops read — common in SwiGLU-family fused activations and small-shape Gather helpers.

<!-- 迁移自 porter kb/target/ascendc/（EC-15，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
