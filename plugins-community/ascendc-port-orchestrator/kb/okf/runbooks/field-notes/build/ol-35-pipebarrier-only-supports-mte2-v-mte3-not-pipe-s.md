---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "PipeBarrier only supports MTE2/V/MTE3 — not PIPE_S"
description: "On Ascend950PR, pipe_barrier() accepts values [4,6] = PIPE_MTE2, PIPE_V, PIPE_MTE3 only. PIPE_S is NOT valid. For scalar pipe synchronization, use SetFlag/WaitFlag with S_MTE3, S_V, MTE2_S, V_S event"
phenomenon: build_failure
signal:
  - "when using PipeBarrier in SIMD kernels with scalar operations"
confidence: single_run
original_id: OL-35
timestamp_inferred: true
tags: [ascendc, platform_bug, ol-35]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
when using PipeBarrier in SIMD kernels with scalar operations

## 教训 / 根因
On Ascend950PR, pipe_barrier() accepts values [4,6] = PIPE_MTE2, PIPE_V, PIPE_MTE3 only. PIPE_S is NOT valid. For scalar pipe synchronization, use SetFlag/WaitFlag with S_MTE3, S_V, MTE2_S, V_S event types.

## 证据
- Sort V1 compile error (2026-04-09), EC-15
  - clipped_swiglu port_a3_to_a5 kw-1 (2026-05-17): scalar-gather of interleaved A/B halves via `SetValue(i, …)` followed by `Cast`/`Mins`/`Maxs` required `HardEvent::S_V`. Replacing the natural `PipeBarrier<PIPE_S>` with `SetFlag/WaitFlag<HardEvent::S_V>` at two sync sites resolved the `kernel_reg.h:85` range error and produced 8/8 cases PASS.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-35（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
