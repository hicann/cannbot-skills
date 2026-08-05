---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "N-dim concat decomposes to 3D (outer × cat_dim × inner)"
description: "Decompose cat(tensors, dim=d) into flat 3D: outer=prod(shape[:d]), cat_dim=shape[d], inner=prod(shape[d+1:]). outer==1 is contiguous copy; outer>1 is per-outer strided copy; one kernel per input tensor."
confidence: single_run
original_id: OL-38
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-38, concat, cat, tensor-decomposition, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**场景 / Trigger**: implementing `torch.cat` along an arbitrary dim.

**选型 / Decomposition**: Decompose `cat(tensors, dim=d)` into a flat 3D shape:
- `outer = prod(shape[:d])`
- `cat_dim = shape[d]`
- `inner = prod(shape[d+1:])`

**Cases**:
- `outer == 1` → flat contiguous copy.
- `outer > 1` → per-outer chunked copy with stride.
- Launch one kernel per input tensor.
- Non-aligned chunks need an overlapping tail write (EC-16).

**Evidence**: Cat V2 kernel (2026-04-09). The same 3D decomposition is applicable to Split and Permute.
