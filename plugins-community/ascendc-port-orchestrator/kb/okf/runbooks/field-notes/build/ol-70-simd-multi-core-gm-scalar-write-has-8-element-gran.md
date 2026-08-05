---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "SIMD multi-core GM scalar write has 8-element granularity contention"
description: "On A5, the minimum granularity of `DataCopy(GM, UB, count)` is 32B (fp32: 8 elements; fp16/bf16: 16 elements). If multiple cores write adjacent positions in a GM array (uid → array[uid]), DataCopy act"
phenomenon: build_failure
signal:
  - "SIMD kernel where multiple AIV cores each write a scalar (mean/rstd/sum etc.) to a shared GM array"
confidence: single_run
original_id: OL-70
timestamp_inferred: true
tags: [ascendc, platform_compat, ol-70]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
SIMD kernel where multiple AIV cores each write a scalar (mean/rstd/sum etc.) to a shared GM array

## 教训 / 根因
On A5, the minimum granularity of `DataCopy(GM, UB, count)` is 32B (fp32: 8 elements; fp16/bf16: 16 elements). If multiple cores write adjacent positions in a GM array (uid → array[uid]), DataCopy actually writes that 8/16-element block, which **overwrites neighboring cores' data** → non-deterministic / all-zero output. **Fix (see P-P49)**: allocate an isolated 8-aligned slot per uid (uid * 8), and have the host (pybind) extract the real value via stride select.

## 证据
2_GroupNormSwish: mean/rstd is one fp32 scalar per (N, group). Direct `out[uid] = value` then DataCopy cross-core overwrote each other intermittently. Changed to `out[uid * 8] = value` + pybind extraction → PASS. E3 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-70（category=platform_compat，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
