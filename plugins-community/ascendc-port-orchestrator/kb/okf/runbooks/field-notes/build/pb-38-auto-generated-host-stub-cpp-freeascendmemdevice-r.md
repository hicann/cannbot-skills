---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Auto-generated host_stub.cpp `FreeAscendMemDevice` races with async kernel execution on V220 [V220]"
description: "applies_to: soc=Ascend910_9382; cann=9.0.0; bisheng=n/a; op_class=all"
phenomenon: build_failure
signal:
  - "First kernel launch produces bit-exact output; subsequent launches show max_abs_diff 5-7 for fp32 tensors with random values."
confidence: single_run
original_id: PB-38
timestamp_inferred: true
tags: [freeascendmemdevice, ascendc, pb-38]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382; cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend910_9382 (V220); cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5 V351 — LaunchAscendKernel may be synchronous on V351)`

- **Severity**: HIGH (silent data corruption on multi-launch)
- **Status**: CONFIRMED 2026-05-21 12_Permute a3-ds kw-3
- **Symptom**: First kernel launch produces bit-exact output; subsequent launches show max_abs_diff 5-7 for fp32 tensors with random values.
- **Root cause**: Auto-generated `host_stub.cpp` calls `FreeAscendMemDevice(overflow_buf)` immediately after `LaunchAscendKernel`, but kernel executes asynchronously. On subsequent launches the same physical memory may be reused while previous kernel still references it.
- **Fix**: Call `torch.npu.synchronize()` after each kernel launch in pybind wrapper.
- **Cross-ref**: OL-66 (torch::zeros not stream-ordered — same class of stream-safety issue).

<!-- 迁移自 porter kb/target/ascendc/（PB-38，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
