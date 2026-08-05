---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Avoid UB bank conflicts — consecutive VEC instructions should hit different banks"
description: "UB is composed of multiple banks. When multiple VEC instructions concurrently read/write different addresses in the same bank, a bank conflict occurs and instructions must queue. Solution: ensure the"
phenomenon: build_failure
signal:
  - "SIMD kernel perf is below expectation and msprof shows low VEC pipe utilization"
confidence: single_run
original_id: OL-50
timestamp_inferred: true
tags: [ascendc, platform_bug, ol-50]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
SIMD kernel perf is below expectation and msprof shows low VEC pipe utilization

## 教训 / 根因
UB is composed of multiple banks. When multiple VEC instructions concurrently read/write different addresses in the same bank, a bank conflict occurs and instructions must queue. Solution: ensure the source and destination tensors of consecutive VEC ops are in different banks (adjust buffer allocation offsets). UB bank count and size vary by chip version.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-50（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
