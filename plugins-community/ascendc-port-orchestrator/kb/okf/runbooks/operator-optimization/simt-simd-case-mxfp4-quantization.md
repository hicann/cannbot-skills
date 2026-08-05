---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Case: MXFP4 quantization — SIMT wins 6x over SIMD"
description: "Per-element log2/floor/pow2/round with a per-32 shared exponent: SIMT is 6x over SIMD V3 (20x over V1) — group=32 is too small for SIMD per-group loops, while 128 SIMT threads run 128 groups at once."
confidence: single_run
original_id: SIMT_VS_SIMD_DECISION.md#case-3-mxfp4-quantization
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, simt, mxfp4, case-study]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

Verified data point where **SIMT wins decisively** — canonical example of "small group-local
dependency + per-element heterogeneous compute" from P-P9.

| Trait | Value |
|---|---|
| Compute | per-element log2 + floor + pow2 + round (heterogeneous) |
| Group dependency | per-32-element shared exponent |
| Data access | contiguous but group-local |
| Result | SIMT is **6x** faster than SIMD V3, **20x** faster than SIMD V1 |

Why SIMT wins:
1. `group_size = 32` is too small — SIMD is forced into a serial per-group loop.
2. SIMT runs 128 threads over 128 groups (4096 elements) per dispatch — parallelism is native.
3. Eliminating the per-group loop (SIMD V4 "fast") does speed up, but the result no longer matches the
   MXFP4 spec, so it is not a valid production path (see OL-30).
4. Per-element `x_exp` forces Compare+Select overhead in SIMD V3; SIMT threads each compute
   independently with no such overhead.

Generalization: a per-element variable shift plus per-element exponent makes SIMT 6-20x faster here —
when group is tiny and control flow diverges per element, do not chase a SIMD path.
