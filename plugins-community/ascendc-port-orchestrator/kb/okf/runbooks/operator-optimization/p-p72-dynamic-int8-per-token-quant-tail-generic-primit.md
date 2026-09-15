---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Dynamic int8 per-token quant tail — generic primitive chain"
description: "Standard tail of dequant→activation→quant ops: per-row dynamic int8 quant. Inputs: fp32 tensor out [N, H]. Outputs: int8 q [N, H] + fp32 quant_scales [N] (BEFORE clamp — this is what CANN returns). Se"
severity: high
confidence: single_run
original_id: P-P72
timestamp_inferred: true
tags: [reduction_quant, optimization, reducemax, quant_scales, p-p72, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

Standard tail of dequant→activation→quant ops: per-row dynamic int8 quant. Inputs: fp32 tensor `out [N, H]`. Outputs: int8 `q [N, H]` + fp32 `quant_scales [N]` (BEFORE clamp — this is what CANN returns). **Sequence**: (1) `Abs(abs_buf, out)` per row, (2) `ReduceMax<float>(amax_scalar, abs_buf, count=H)` per row → scalar, (3) emit `quant_scales[i] = amax/127.0` BEFORE clamping (this is the returned scale; clamp is internal-only div-guard), (4) `clamped_scale = max(amax/127.0, 1e-10)` via `Maxs(scale, scale, 1e-10)` to avoid div-by-zero on all-zero rows, (5) `Muls(out, out, 1/clamped_scale)` per row (literal-first per OL-82 if precision-sensitive), (6) `Mins(out, out, 127.0)` then `Maxs(out, out, -128.0)` for symmetric saturation, (7) `Cast(q_int32, out, RoundMode::CAST_RINT)` IEEE-RNE per OL-81, (8) `Cast(q_int8, q_int32, RoundMode::CAST_NONE)` per P-P46. **Critical contract**: `ReduceMax` is hardware-deterministic per fixed input (by-construction det). The `quant_scales` returned MUST be the pre-clamp `amax/127.0` value — the clamped variant is kernel-internal guard only. Op#10/op#11 use this exact chain bit-exact vs CANN reference. **Common bug**: returning the clamped scale instead of pre-clamp causes 1-ULP residuals on near-zero rows.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P72，convert_patterns_to_okf.py）。confidence 未升格。 -->
