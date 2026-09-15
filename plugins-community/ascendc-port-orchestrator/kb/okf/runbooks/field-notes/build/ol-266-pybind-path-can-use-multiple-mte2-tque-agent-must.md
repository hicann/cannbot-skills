---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "pybind path CAN use multiple MTE2 TQue — agent must not blame pybind structure"
description: "applies_to: soc=Ascend950PR (V351/arch35); cann=9.0.0; paradigm=pybind (build_ascendc.py / aclruntime CMake)"
phenomenon: build_failure
signal:
  - "when a worker claims \"NPUKernelBench pybind path doesn't support multiple VECIN/VECOUT TQue\" and uses this as justification to abandon a correct TQue-based desi"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-266
timestamp_inferred: true
tags: [ascendc, ol-266]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR (V351/arch35); cann=9.0.0; paradigm=pybind (build_ascendc.py / aclruntime CMake)`
`verified_on: soc=Ascend950PR; cann=9.0.0`

- **The fact**: NPUKernelBench pybind build path fully supports multiple VECIN/VECOUT TQue on MTE2. When a kernel using two MTE2 TQue fails, the pybind structure is NOT the root cause.
- **Counter-example**: `workspace/group_norm_silu/kernel/group_norm_silu_kernels.cpp` — **6 TQue** (3 VECIN + 3 VECOUT) working correctly on A5 via the same pybind build path. This proves it is possible.
- **Incident**: masked_select_v3 kw-1 (2026-06-25) — worker used two VECIN TQue (data + mask), hit error 95, and concluded "pybind path cannot support multiple VECIN TQue." This conclusion was wrong. kw-2 fixed the issue without changing the build path. The real cause may have been NPU0 hardware state at the time, not a structural pybind limitation.
- **Trigger**: when a worker claims "NPUKernelBench pybind path doesn't support multiple VECIN/VECOUT TQue" and uses this as justification to abandon a correct TQue-based design or fall back to PipeBarrier workarounds — STOP. The claim is false. Point the worker at the group_norm_silu counter-example.
- **Related**: OL-4

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-266（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
