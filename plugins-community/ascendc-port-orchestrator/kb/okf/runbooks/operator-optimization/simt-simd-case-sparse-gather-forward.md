---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Case: Sparse-Gather forward — SIMD TQue wins 1.6-2.3x"
description: "Fully-vectorized DataCopy+Muls+Add, no group dependency, contiguous reads: SIMD TQue<4> is 1.6-2.3x faster than PipeBarrier because every element runs the same op and MTE2 prefetch overlaps."
confidence: single_run
original_id: SIMT_VS_SIMD_DECISION.md#case-1-sparse-gather-forward
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, simd, sparse-gather, case-study]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

Verified data point where **SIMD wins** — canonical example of the "contiguous + fully vectorizable +
no group dependency" branch of P-P9.

| Trait | Value |
|---|---|
| Compute | DataCopy + Muls + Add (fully vectorized) |
| Group dependency | none (per-token independent) |
| Data access | contiguous (expert embedding) |
| Result | SIMD `TQue<4>` is **1.6-2.3x** faster than PipeBarrier |

Why SIMD wins: every element runs the identical Muls+Add, there is no group-local dependency, and the
contiguous access lets MTE2 prefetch overlap the VEC compute effectively. This is the textbook shape
where switching a PipeBarrier kernel to a `TQue<VECIN, 4>` pipeline pays off.
