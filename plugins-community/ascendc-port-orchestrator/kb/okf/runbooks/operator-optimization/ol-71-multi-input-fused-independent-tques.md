---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Multi-input fused ops use independent TQues to avoid PB-9"
description: "Give each independent GM input its own TQue<VECIN,depth> in a fused multi-input op rather than sharing one queue or in-place concatenating, avoiding PB-9 UB-to-UB corruption; depths can differ per input."
confidence: single_run
original_id: OL-71
classified_by: llm-assisted
timestamp_inferred: true
tags: [pipeline-design, optimization, ol-71, fused-op, tque, pb-9]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Trigger**: a fused op must consume 2+ independent GM input tensors in one kernel (e.g.
add+norm, scatter+gather). Loaded by Generator (when fusing ops with 2+ input tensors).

Give each independent GM input its own **independent `TQue<VECIN, depth>`**, instead of sharing
one TQue or in-place concatenating inputs onto a single LocalTensor. Each input flows through
its own EnQue/DeQue channel, which avoids:
- **PB-9** (UB-to-UB DataCopy silent corruption), and
- implicit cross-input synchronization issues.

**Depths can differ per input**: the primary input being reduced uses depth-2 (for pipeline
overlap); an auxiliary input that is merely scaled uses depth-1 (reloading each tile is fast
enough, so no need to double-buffer it).

**Evidence**: 18_FusedAddRmsnorm — `hidden_states` uses `TQue<VECIN,2>`, `residual` uses
`TQue<VECIN,1>`. Two independent DataCopies → separate EnQue/DeQue → both DeQued together for
the fp32 sum. No PB-9 triggered; 50/50 PASS, 4.07x median. E3 level (measured).
