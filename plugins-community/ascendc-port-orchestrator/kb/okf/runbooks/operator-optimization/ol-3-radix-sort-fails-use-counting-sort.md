---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Radix sort fails on AscendC; use counting sort"
description: "Multi-pass radix sort with global atomicAdd scatter fails on AscendC (0/61); single-pass counting sort (histogram->prefix_sum->scatter) gives 61/61 correct and 2.84x over host std::sort."
confidence: single_run
original_id: OL-3
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-3, sort, counting-sort, radix-sort, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**场景 / Trigger**: implementing an on-device sort for AscendC SIMT kernels.

**选型**: A 4-pass byte-wise radix sort using global `atomicAdd` scatter was implemented first and failed completely (0/61 clusters correct). Multi-pass scatter is unstable on AscendC — each pass's global `atomicAdd` destroys the ordering established by the previous pass. Switching to **counting sort** (histogram -> prefix_sum -> scatter, single-pass) achieved **61/61 correct and 2.84x speedup** over host `std::sort`.

**Why counting sort works here**: same-key ordering does not affect the downstream register accumulation, so the sort does not need to preserve stability across passes.

**Evidence**: OPTIMIZATION_PLAN.md Batch 7 ("radix sort 0/61 correct"), `output/src/pooling/radix_sort_kernel.h`.
