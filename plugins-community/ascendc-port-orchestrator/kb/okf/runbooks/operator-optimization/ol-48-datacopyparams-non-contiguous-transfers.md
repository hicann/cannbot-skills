---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "DataCopyParams replaces for-loops for non-contiguous transfers"
description: "Use DataCopy's DataCopyParams (srcStride/dstStride/blockLen/blockCount) instead of block-by-block loops for non-contiguous transfers. Gain depends on large per-call payload; small metadata transfers may not benefit."
confidence: single_run
original_id: OL-48
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-48, datacopyparams, non-contiguous, strided, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**场景 / Trigger**: transferring non-contiguous memory (matrix columns, strided data).

**选型**: Use `DataCopy`'s `DataCopyParams` (`srcStride` / `dstStride` / `blockLen` / `blockCount`) instead of looping block-by-block. Example: transferring the first 2KB of each image row — a for-loop does N row-by-row transfers, while `DataCopyParams` does it in one call.

**Applicability caveat (measured)**: 22_Nonzero kw-3 iter 1 (2026-05-02) applied a `DataCopyParams` strided-buffer-enlargement variant on a SIMD multi-core `CompareScalar` + popcount + `GatherMask` scan workspace → **Δ−1.4% (REVERTED, hypothesis invalidated)**. That transfer surface is small per-block index/count metadata, not large strided rows, so the strided-fanout API does not amortize over that regime. **The gain depends on per-call payload being large enough to dominate DataCopy launch overhead — small-payload metadata transfers may not benefit.**

**Source**: hiascend.com best practices (2026-04).
