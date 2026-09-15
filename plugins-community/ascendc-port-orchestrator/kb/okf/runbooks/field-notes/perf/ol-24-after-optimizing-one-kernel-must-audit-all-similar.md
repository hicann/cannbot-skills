---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "After optimizing one kernel, must audit all similar kernels"
description: "paradigm: ascendc"
phenomenon: perf_regression
signal:
  - "any optimization applied to one kernel variant (forward/backward, fp32/fp16/bf16)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-24
timestamp_inferred: true
tags: [ascendc, ol-24]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: process
- **Loaded by**: Builder, Optimizer, Lead
- **Trigger**: any optimization applied to one kernel variant (forward/backward, fp32/fp16/bf16)
- **Lesson**: E13 switching Forward PipeBarrier→TQue gave 1.6-2.3x speedup, but we did not check whether Backward Sorted had the same PipeBarrier anti-pattern (it did — 7 PIPE_ALL). Only discovered when the expert pointed it out at E14.
  After applying an optimization, must:
  1. grep all kernel files for similar anti-patterns
  2. Evaluate whether the same optimization applies to each match
  3. Distinguish "production path" from "non-production path" — only optimize the production path
- **Evidence**: E14 feedback (2026-04-06), sparse_gather_simd.h BackwardSimdSortedF32 had 7 unoptimized PIPE_ALL

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-24（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
