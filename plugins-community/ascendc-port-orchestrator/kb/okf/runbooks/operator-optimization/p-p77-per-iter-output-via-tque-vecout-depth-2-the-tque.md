---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Per-iter output via `TQue<VECOUT, depth=2>` (the TQue side of the OL-94 decision)"
description: "The TQue counterpart to P-P75: when a per-iter loop emits via Cast(out_ub, work_ub, ...) ; DataCopy(gm_out, out_ub, count) and the output UB region does NOT need cross-phase buffer-liveness aliasing ("
severity: high
confidence: single_run
original_id: P-P77
timestamp_inferred: true
tags: [memory_access, optimization, alloctensor, enque, deque, freetensor, p-p77, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

The TQue counterpart to P-P75: when a per-iter loop emits via `Cast(out_ub, work_ub, ...) ; DataCopy(gm_out, out_ub, count)` and the output UB region does NOT need cross-phase buffer-liveness aliasing (P-P65) or persistent-buffer semantics, **prefer `TQue<QuePosition::VECOUT, depth=2>` over a bare `TBuf<VECCALC>`** for the output buffer. The depth=2 queue's `AllocTensor`/`EnQue`/`DeQue`/`FreeTensor` rotation provides automatic MTE3↔V sync via slot rotation: slot N+1's `AllocTensor` blocks until slot N's prior MTE3 retires, removing the need for explicit `SetFlag<HardEvent::MTE3_V>+WaitFlag<...>` flags between the per-iter `DataCopy` and the next iter's V write to the same UB region. **Anti-patterns observed when worker leaves a TBuf in this slot**: (a) `PipeBarrier<PIPE_ALL>()` at iter top — disrupts TPipe's queue scheduling, op#27 Phase D iter-5 regressed to 9/10 wrong-output runs; (b) extra `PipeBarrier<PIPE_V>` between unrelated V ops — drains the V pipe prematurely, op#27 6/10 wrong. Always rotate the output buffer through a queue first; only escalate to the manual-event-sync path (P-P75) if the TQue refactor cannot apply for structural reasons. **Cross-ref**: OL-94 (decision table TQue vs TBuf), P-P75 (TBuf manual-event side), PB-21 (PipeBarrier silent crash on V220), A-P61 (det anti-patterns; TBuf-output-race is the practitioner-side downstream of A-P61's atomicAdd-upstream). Validated op#27 27_MultiMaskAttentionAggregation a3 V220 2026-04-28.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P77，convert_patterns_to_okf.py）。confidence 未升格。 -->
