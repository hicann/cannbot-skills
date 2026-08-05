---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Sort algorithm selection — prefer the hardware Sort API"
description: "Prefer AscendC's built-in hardware bitonic Sort API over a hand-written sort; dispatch by N: ≤4096 hardware Merge Sort, larger and fitting UB single-core Radix, otherwise multi-core Radix."
confidence: single_run
original_id: OL-57
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-57, sort, hardware-sort-api, bitonic]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Trigger**: generating sort-class ops (sort, argsort, topk). Loaded by Generator and Analyzer.

CANN chooses among 5 sort algorithms based on N and dtype. Key finding: **AscendC exposes a
built-in `Sort` API backed by a hardware bitonic sort network** — a 3-stage pipeline of
Concat + Sort + Extract. For small N (≤4096 fp32, ≤1024 fp16) this hardware sort is an order
of magnitude faster than any software implementation.

The current Sort op uses a scalar merge sort and only reaches 0.31x; switching to the hardware
Sort API is expected to yield a large improvement.

**Decision (dispatch by N):**
- N ≤ 4096 → hardware Merge Sort (`AscendC::Sort<T, true>()`)
- N > 4096 and fits in UB → single-core Radix Sort
- otherwise → multi-core Radix Sort

**Action**: when generating Sort kernels, default to the `AscendC::Sort<T, true>()` hardware
sort pipeline (see P-P42) instead of a hand-written sort algorithm.

**Evidence**: CANN `sort_tiling_arch35.cpp:746-761`, `sort_merge_sort.h:198-243`. See P-P42,
P-P43 for details. E1 level (source analysis).
