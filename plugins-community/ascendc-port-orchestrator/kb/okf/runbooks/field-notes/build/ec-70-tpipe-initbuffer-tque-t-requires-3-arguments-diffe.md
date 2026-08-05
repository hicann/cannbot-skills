---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`TPipe::InitBuffer(TQue<T>&)` requires 3 arguments — different from TBuf InitBuffer which takes 2"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all"
phenomenon: build_failure
signal:
  - "compile error when calling pipe_.InitBuffer(tQue) with only 2 arguments. Error message: too few arguments to function call, expected 3 or similar."
confidence: single_run
original_id: EC-70
timestamp_inferred: true
tags: [ascendc, ec-70]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.1.T500`
`unverified_on: soc=Ascend910_V220 (A3 chip family — CANN version may differ)`

- **Symptom**: compile error when calling `pipe_.InitBuffer(tQue)` with only 2 arguments. Error message: `too few arguments to function call, expected 3` or similar.

- **Root cause**: `TPipe::InitBuffer` has different signatures for TQue vs TBuf:
  - `InitBuffer(TBuf<T>&, uint8_t num_buffers)` — 2 args
  - `InitBuffer(TQue<T>&, uint8_t num_buffers, uint32_t len_per_buffer)` — 3 args

  The TQue variant requires an explicit per-buffer element length (in bytes) as the third argument. Workers familiar with the TBuf 2-arg form naturally write the TQue form with 2 args → compile fails.

- **Fix**: always provide the third argument for TQue InitBuffer. The `len_per_buffer` is the byte size of one buffer slot: typically `tileLength * sizeof(dtype)`.

```cpp
// TBuf — 2 args (OK)
pipe_.InitBuffer(tBuf, depth);

// TQue — 3 args REQUIRED
pipe_.InitBuffer(tQue, depth, tileLength * sizeof(half));
```

- **Detection**: grep for `InitBuffer(` calls near TQue variable declarations. If the call has exactly 2 args and the first arg is a `TQue<...>`, the third arg is missing.

- **Evidence**:
  - fused_quant_mat_mul kw-1 (2026-06-15): `kernel/fused_quant_mat_mul_kernels.cpp:69-70` — TQue InitBuffer called with 2 args. Added `BUF_LEN * sizeof(half)` as third arg → compile passed.

<!-- 迁移自 porter kb/target/ascendc/（EC-70，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
