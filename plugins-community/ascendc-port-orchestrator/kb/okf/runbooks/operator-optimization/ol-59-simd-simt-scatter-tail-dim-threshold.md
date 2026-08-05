---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMD/SIMT mode selection — scatter ops tail-dimension threshold"
description: "Pick SIMT vs SIMD for scatter ops by tail (embedding) dim: scatter_add <128B→SIMT, ≥128B→SIMD; the sort path uses a higher 512B threshold. SIMD is efficient only at ≥1 vector-register width."
confidence: single_run
original_id: OL-59
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-59, scatter-add, simd-simt, tail-dim-threshold]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Trigger**: SIMT vs SIMD decision for scatter-add-class ops. Loaded by Generator and Analyzer.

CANN `scatter_add` selects its execution mode based on the size of the **tail dimension**
(the embedding dim):

- `embDim * dtype_bytes < 128B` → SIMT
- `embDim * dtype_bytes ≥ 128B` → SIMD

The sort path uses a higher threshold: `< 512B` → SIMT sort; `≥ 512B` → SIMD sort.

**Reason**: SIMD vector ops are only efficient when the operand is ≥ 128 bytes — one full
vector-register width. Below that, the per-element SIMT path wins.

**Evidence**: CANN `scatter_add_tiling_base.cpp:199` (`VAR_TAIL_DIM_SIZE = 128`),
`:208-209` (`VAR_TAIL_DIM_SIZE_SORT = 512`). E1 level (source analysis).
