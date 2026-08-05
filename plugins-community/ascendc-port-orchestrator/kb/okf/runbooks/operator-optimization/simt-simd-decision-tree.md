---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMT vs SIMD decision tree (Step 1-4 with group_size threshold)"
description: "Ordered SIMT/SIMD procedure: atomicAdd or indirect address forces SIMT; group-local dep splits at group_size 256 (>=256 SIMD, else default SIMT); pure vectorizable is SIMD, per-element branch is SIMT."
confidence: single_run
original_id: SIMT_VS_SIMD_DECISION.md#decision-tree
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, simt, simd, decision-tree]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

Walk this ordered procedure top-down; the first matching step decides the pick (P-P9).

- **Step 1 — atomicAdd / scatter-write?** YES → **SIMT** (SIMD cannot do atomic ops). NO → Step 2.
- **Step 2 — indirect addressing `arr[index[i]]`?** YES → **SIMT** (SIMD `DataCopy` needs contiguous
  addresses). Exception: if `index` is ordered and batchable, consider sort-to-reuse (P-P24) and a
  SIMD path. NO → Step 3.
- **Step 3 — is there a group-local dependency?** (every N elements share a parameter, `N < tile_size`)
  YES → Step 3a. NO (all elements independent / pure element-wise) → Step 4.
- **Step 3a — `group_size >= 256`?** YES → **SIMD** (group large enough that per-group loop overhead
  is acceptable; use `TQue<VECIN, 4>` pipeline + per-group VEC ops). NO → Step 3b.
- **Step 3b — is the per-group compute fully vectorizable?** (every element in the group runs an
  identical instruction sequence) YES → SIMD *might* win but must be proven by profiling — e.g.
  `group=32, tile=1024` needs 32 per-group loop passes, while 128 SIMT threads process 128 groups
  (4096 elements) per dispatch; empirically MXFP4 (group=32) SIMD V3 was 6x slower, so **default
  SIMT unless profiling proves SIMD faster**. NO → **SIMT** (per-element heterogeneous compute is
  best served by parallel SIMT threads).
- **Step 4 — can the compute be expressed with SIMD vector instructions?** (Abs, Add, Muls, Cast,
  Compare+Select — one operation applied to all elements) YES → **SIMD** (MTE2/VEC pipeline overlap;
  e.g. SG forward DataCopy+Muls+Add gave 1.6-2.3x). NO → **SIMT** (needs per-element branch / bit ops).

Key thresholds to remember: the group-local split is at `group_size = 256`, and a "too small" group
(≤32) essentially always favors SIMT because SIMD is forced into a serial per-group loop.
