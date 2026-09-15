---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Row-parallel data unfolding to expose AIV parallelism upstream of single-AIC cube call"
description: "When a workload decomposes into \"unfold input → cube GEMM → reshape output\" and the cube call is single-AIC-per-tile (P-P68), the unfold stage is the parallelism floor: dispatching the unfold by (b, g"
severity: high
confidence: single_run
original_id: P-P78
timestamp_inferred: true
tags: [platform_compat, optimization, k_total, hw_pad, p-p78, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

When a workload decomposes into "unfold input → cube GEMM → reshape output" and the cube call is single-AIC-per-tile (P-P68), the unfold stage is the parallelism floor: dispatching the unfold by `(b, g)` alone leaves 50+ AIVs idle when `B·G ≪ 56`. Pattern: dispatch `blockDim_unfold = B · G · K_total` where `K_total` is the per-`(b, g)` row count of the unfolded matrix (e.g. for conv `K_total = Cin_per_g · K_h · K_w`); decode `bid → (b, g, k_idx) → (ic, kh, kw, ...)`; each AIV builds exactly one row of `HW_pad` fp32 elements via `Duplicate(0) → fill-from-source → DataCopy` (no cross-AIV communication). Determinism by-construction (each row owned by exactly one AIV). 3-kernel pipeline shape: AIV unfold (row-parallel) + AIC cube (per-(b,g)) + AIV reshape/bias (row-parallel). Combine with EC-42 (split AIV vs AIC into separate .cpp). **Generalizes** to: 2D/3D conv via im2col, attention K/V re-assembly before flash-attn cube, group-conv unfolding, segmented-batch attention. **Anti-pattern**: dispatching the unfold by `(b, g)` only — leaves 90%+ of AIVs idle when `B·G ≪ AIV_count`. Validated op#7 ConvStandard2d ko-1 (2026-04-29): direct-VEC 0.087× → cube `(b,g)`-only unfold 0.155× → row-parallel unfold **0.705× median** (4.55× over per-(b,g), 8.10× over Opt0). Slow case `[1,32,64,64] k=7` 91.3 ms → 2.31 ms (39× speedup) when row-parallel exposes K_total=1568 to the dispatcher.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P78，convert_patterns_to_okf.py）。confidence 未升格。 -->
