---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Case: MXFP4 tile-wide approximate — SIMD 1.08x but wrong precision"
description: "Going tile-wide (Abs+Muls+Cast FLOOR+Select) makes all ops vectorizable and SIMD 1.08x faster than SIMT on 4K, but it breaks the MXFP4 per-32 shared-exponent spec, so it is not production-usable."
confidence: single_run
original_id: SIMT_VS_SIMD_DECISION.md#case-4-mxfp4-tile-wide-approximate
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, simd, mxfp4, case-study]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

Verified cautionary data point: **SIMD "wins" only by cheating on precision.** Read this alongside
OL-30 — it is the concrete failure the precision constraint forbids.

| Trait | Value |
|---|---|
| Compute | tile-wide Abs + Muls + Cast(FLOOR) + Select |
| Group dependency | none (per-group was dropped in favor of tile-wide) |
| Precision | does NOT match the MXFP4 spec |
| Result | SIMD is **1.08x** faster than SIMT on 4K |

Why SIMD wins here: once the per-group loop is removed, every operation becomes a tile-wide SIMD
vector op and the MTE2 pipeline overlaps — hence the small speedup. But dropping the per-32 shared
exponent changes the algorithm's precision semantics, so this kernel is **not usable in production**.
Takeaway: a SIMD margin this thin is not worth a precision regression; keep SIMT and the correct
per-group semantics.
