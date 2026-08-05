---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220/CANN 8.5.1 `TPipe::InitBuffer(TQue&, depth)` 2-arg form does not compile — use the 3-arg `(TQue&, depth, buf_size_bytes)` form"
description: "applies_to: soc=Ascend910_V220; cann=8.5.1; bisheng=n/a; op_class=all"
phenomenon: build_failure
signal:
  - "build error no matching member function for call to 'InitBuffer' on a pipe->InitBuffer(que, depth) (2-arg) call that compiles fine on newer CANN."
confidence: single_run
original_id: EC-82
timestamp_inferred: true
tags: [507035, ascendc, ec-82]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_V220; cann=8.5.1; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend910B2C (V220); cann=8.5.1`

- **Symptom**: build error `no matching member function for call to 'InitBuffer'` on a `pipe->InitBuffer(que, depth)` (2-arg) call that compiles fine on newer CANN.
- **Fix**: pass the explicit buffer size as the third argument:
  ```cpp
  // BAD on V220/CANN 8.5.1 — 2-arg overload absent
  pipe->InitBuffer(inQueue_, PIPE_DEPTH);
  // GOOD — 3-arg form
  pipe->InitBuffer(inQueue_, PIPE_DEPTH, TILE_ELEMS * sizeof(float));
  ```
- **Scope**: distinct from EC-62 (TBuf with NO `InitBuffer` at all → 507035 vector core exception). Here the call IS present; only the arg count is wrong. The 2-arg overload may exist on newer CANN — this is the V220/8.5.1 fallback. Applies to `TQue` buffers; `TBuf` workspace uses the 2-arg `(name, size_bytes)` form as usual (EC-62).
- **Detection**: build log shows `no matching member function for call to 'InitBuffer'` pointing at a `InitBuffer(<TQue>, <int>)` line. Fix by adding the byte-size third arg.
- **Evidence**: 1_GELU kw-1 re-spawn (2026-06-23, Ascend910B2C V220, CANN 8.5.1): the 2-arg form was rejected by the compiler; switching to the 3-arg form cleared the build.
- **Cross-ref**: EC-62 (missing InitBuffer → 507035), OL-63 (TQue depth=4 for elementwise bandwidth), P-P28 (TQue depth + explicit buf-size pairing).

<!-- 迁移自 porter kb/target/ascendc/（EC-82，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
