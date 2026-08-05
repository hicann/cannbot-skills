---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "2D Tiling for Large Embedding Scatter"
description: "Problem: UB 256KB must simultaneously hold: index array + sort buffer + accumulation buffer + update data. When embDim is very large (e.g., embDim=16384, fp32 = 64KB/row), a single row occupies a larg"
confidence: single_run
original_id: P-P41
timestamp_inferred: true
tags: [scatter_add, optimization, p-p41, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: UB 256KB must simultaneously hold: index array + sort buffer + accumulation buffer + update data. When embDim is very large (e.g., embDim=16384, fp32 = 64KB/row), a single row occupies a large UB footprint.

**Pattern**: Split [indicesNum, embDim] work into a 2D grid of rowTileNum × colTileNum:
1. Row partitioning: different cores handle different index ranges
2. Column partitioning: different embDim positions of the same index are handled by different cores
3. UB budget: `ubFactorRow * ubFactorCol * sizeof(T)` + index + sort buffer must be < UB size
4. Column-split cores finally merge results via atomicAdd (different columns of the same index are written to different GM addresses by different cores — no contention)

**Column-split threshold**: Only consider column splitting when `embDim * dtype_bytes > 4096`. Below that, fit the whole row into UB.

**Evidence**: CANN scatter_add_tiling_base.cpp:290-393 (FindUniqueCut + SimdTiling + DoBlockTiling). E1 level.

**Stop condition**: When embDim * dtype_bytes ≤ 4096, column splitting is not needed. Column splitting adds tiling complexity and should not be used for small-dim cases.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P41，convert_patterns_to_okf.py）。confidence 未升格。 -->
