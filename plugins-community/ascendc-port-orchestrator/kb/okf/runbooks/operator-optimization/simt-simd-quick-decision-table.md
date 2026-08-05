---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMT vs SIMD quick decision table for AscendC operators"
description: "Map operator features to SIMT or SIMD at a glance: atomicAdd, indirect addressing, small group-local compute force SIMT; contiguous read plus pure vector compute favors SIMD (MTE2/VEC overlap)."
confidence: single_run
original_id: SIMT_VS_SIMD_DECISION.md#quick-decision-table
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, simt, simd, algorithm-selection]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

One-glance lookup for picking SIMT vs SIMD on an AscendC (V351) kernel. This is the authoritative
quick reference for pattern P-P9. Grounded in production experience with Sparse-Gather (E8-E14) and
MXFP4 quantization. For borderline cases walk the full decision tree instead; always confirm the
pick with an A/B benchmark.

| Operator feature | Pick | Why |
|---|:---:|---|
| atomicAdd / scatter-write | **SIMT** | SIMD has no atomicAdd |
| Indirect addressing `arr[index[i]]` | **SIMT** | SIMD `DataCopy` cannot address indirectly (needs contiguous addresses) |
| Contiguous read + vector compute (Add / Muls / Cast) | **SIMD** | MTE2/VEC pipeline overlap, 1.6-2.3x |
| Group-local compute, `group_size < tile_size` | **SIMT** | SIMD per-group loop is slower than SIMT parallelism |
| Per-element heterogeneous ops (different shift / branch) | **SIMT** | SIMD can emulate with Compare+Select but the overhead is high |
| Pure bit ops (reinterpret float↔int) | either | Both work; decide by available parallelism |
| Bulk data movement + simple compute | **SIMD** | MTE2 DMA bandwidth ≫ dcache |

Rule of thumb: SIMD wins when the same vector instruction applies to every element on contiguous
data; SIMT wins when access is index-driven, work is atomic, or per-element control flow diverges.
