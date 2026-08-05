---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "TQue bug RESOLVED in CANN 9.0.0 — prefer TQue over PipeBarrier"
description: "TQue<VECIN,2> corruption was a CANN 9.0.T501 bug, RESOLVED in CANN 9.0.0. Prefer TQue over PipeBarrier<PIPE_ALL> for SIMD pipeline overlap; TQue<VECIN,4>+TQue<VECOUT,2> gave 1.6-2.3x speedup."
confidence: single_run
original_id: OL-4
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-4, tque, pipebarrier, cann-9.0.0, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**场景 / Trigger**: choosing between TQue and PipeBarrier for SIMD pipelines.

**Status**: `TQue<VECIN,2>` data corruption was a **CANN 9.0.T501 bug**, now **RESOLVED in CANN 9.0.0**. SG backward has used `TQue<VECIN,2>` successfully since E12. SG forward switched from PipeBarrier to `TQue<VECIN,4>` in E13 — **1.6–2.3x speedup** on all cases.

**选型 / Rule**: Always prefer TQue over `PipeBarrier<PIPE_ALL>` for SIMD pipeline overlap. `PipeBarrier<PIPE_ALL>` serializes ALL pipes (MTE2 + VEC + MTE3 + Scalar); TQue only syncs the necessary MTE2→VEC transition.

**Action**: For new SIMD kernels use the `TQue<VECIN,4>` + `TQue<VECOUT,2>` pattern. Never use `PipeBarrier<PIPE_ALL>` in hot loops.

**Evidence**: E13-P1 benchmark data (2026-04-01), EXPERT_FEEDBACK.md E13 section.
