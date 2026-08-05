---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Pooling multi-strategy priority dispatch + UB shrink loop"
description: "Pooling dispatches multiple tiling strategies by priority; a UB shrink loop cuts the UB budget 4KB per iteration until tile count ≥ core count, trading UB utilization for parallelism."
confidence: single_run
original_id: OL-61
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-61, pooling, ub-shrink-loop, tiling-dispatch]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Trigger**: generating pooling ops (AvgPool / MaxPool, etc.). Loaded by Generator and Analyzer.

CANN registers **multiple tiling strategies** for pooling, dispatched by priority:
- (0) NCHW small-kernel
- (1) NHWC small-kernel
- (2) large-kernel
- (19) SIMT fallback

**Key optimization — UB shrink loop**: when the initial tiling produces fewer tiles than
there are cores, reduce the UB budget by 4KB each iteration, forcing progressively smaller
tiles until `tile_count ≥ core_count`. This deliberately trades UB utilization for parallelism
(idle cores are worse than smaller tiles).

**Second key — channel-size branching in NHWC small-kernel**: when `C × dtype < 64B`
(small channel), use gather instructions to pack multiple spatial positions into one vector
register; when `C × dtype ≥ 64B` (large channel), vectorize along C directly.

**Evidence**: CANN `avg_pool_common_nhwc_small_kernel_tiling.cpp:425-436` (shrink loop),
`:355-371` (gather mode). E1 level (source analysis).
