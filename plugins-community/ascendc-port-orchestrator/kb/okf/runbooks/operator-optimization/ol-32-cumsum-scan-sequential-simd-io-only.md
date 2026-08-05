---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Cumsum/Scan is inherently sequential — SIMD for I/O only"
description: "Cumsum/scan is not vectorizable within a scan line; use SIMD DataCopyPad for bulk I/O and a scalar GetValue/SetValue loop for the prefix sum. Large scan_len (>4K) is slow (~0.02x at 16K)."
confidence: single_run
original_id: OL-32
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-32, cumsum, scan, prefix-sum, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**场景 / Trigger**: implementing prefix sum / cumulative ops (cumsum, scan).

**选型**: Cumsum cannot be vectorized within a single scan line — it is inherently sequential. Use SIMD `DataCopyPad` for bulk I/O, but a scalar `GetValue` / `SetValue` loop for the actual prefix sum.

**Tradeoff**: For large `scan_len` (>4K) perf is poor (~0.02x for 16K elements) because the serial loop dominates. SIMD accelerates only the I/O, not the scan itself.

**Evidence**: Cumsum V1 (2026-04-09), 51/51 precision PASS, 0.49x mean.
