---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Case: Sparse-Gather backward (sorted) — SIMD wins only 1-10%"
description: "Sorted backward accumulate (DataCopy+Muls+Add into TQue<VECOUT>), long per-expert runs: SIMD removes 7 PipeBarriers but is only 1-10% faster because the scalar pipe (0.648) is the real bottleneck."
confidence: single_run
original_id: SIMT_VS_SIMD_DECISION.md#case-2-sparse-gather-backward-sorted
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, simd, sparse-gather, case-study]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

Verified data point where **SIMD wins but the margin is small** — a caution that a clean pipeline
rewrite does not always translate into a large speedup.

| Trait | Value |
|---|---|
| Compute | DataCopy + Muls + Add (accumulated into `TQue<VECOUT>`) |
| Group dependency | per-expert accumulation, but each expert run is long enough |
| Data access | contiguous |
| Result | SIMD `TQue` is **1-10%** faster; PipeBarriers cut from 7 → 0 |

Why the gain is small: even after removing all 7 PipeBarriers, the scalar pipe is the true bottleneck
(0.648), so the MTE2/VEC overlap that SIMD unlocks has limited headroom. Lesson: quantify the actual
bottleneck (scalar vs MTE2/VEC) before expecting a SIMD pipeline rewrite to deliver a big win.
