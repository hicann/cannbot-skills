---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Vendor adv_api regbase primitive substitution requires per-call batch axis A>1 to amortize internal scalar-broadcast cost"
description: "When considering substituting hand-rolled per-row code with a vendor AscendC::<Primitive><U,T> adv_api call (Normalize, LayerNorm, RowMuls, Softmax, Logit, etc.), the perf advantage materializes ONLY"
severity: high
confidence: single_run
original_id: P-P87
timestamp_inferred: true
tags: [platform_compat, optimization, normalize, k_rows_per_aiv, p-p87, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

### Trigger
When considering substituting hand-rolled per-row code with a vendor `AscendC::<Primitive><U,T>` adv_api call (Normalize, LayerNorm, RowMuls, Softmax, Logit, etc.), the perf advantage materializes ONLY when the dispatch shape supplies multiple rows (A>1) per primitive call. Vendor c310 regbase impls (e.g. `normalize_c310_impl.h`) use `Reg::LoadAlign<DIST_BRC_B32>` to broadcast scalar inputs (mean, rstd, gamma, beta, scale, etc.) into vector registers across A rows in ONE call — the broadcast-load cost is fixed per call, so per-row cost scales as `(broadcast_cost / A) + per_row_compute`. At A=1 the broadcast happens once per row → no amortization → vendor primitive performs same as (or worse than) hand-rolled code. **Decision rule (mandatory pre-substitution check)**: (a) for **pure precision improvement** (e.g. CPU-truth alignment via vendor's bit-canonical regbase output), A=1 substitution is net-positive — the regbase impl produces bit-identical results and never regresses precision; (b) for **perf improvement**, FIRST verify the call-site dispatch can supply A>1. If outer loop is per-row (A=1, each `Normalize` call processes 1 row of K elements), the substitution is NOT a perf lever — needs upstream restructure (Kind-2 batch rewrite: load K rows into one buffer of shape [A, K], single batched call, scatter results back). **Snippet**: vendor signature is typically `<Primitive><U,T>(dst[A][K], src[A][K], scalars[A], ...)` — the leading A axis IS the amortization axis; never call with A=1 expecting perf gain. **Anti-pattern (op#10 LayerNorm kw-2-this-session 2026-05-05)**: substituted `AscendC::Normalize<{half|float|bfloat16_t}, ...>` at 3 single-pass sites within a `K_ROWS_PER_AIV` outer loop dispatching A=1 per call → Pass A 60/60 preserved + Pass B 10/16 → 16/16 BIT-EXACT (precision improved via regbase output) + Det 60/60 preserved + **Perf 0.19× = baseline (no improvement)**. Diagnosis: outer loop dispatches A=1 per Normalize, so `Reg::LoadAlign<DIST_BRC_B32>` of mean+rstd+gamma+beta runs once per row with no amortization. Vendor LayerNormV4 perf advantage requires batched A=K rewrite, not exposed by single-row substitution. **Generalizes** to: any vendor adv_api regbase primitive (Normalize, LayerNorm, Softmax, RowMuls/RowAdds, RmsNorm, GroupNorm, etc.) where the impl uses `Reg::LoadAlign<DIST_BRC_*>` for scalar broadcast — these all have the same A>1 amortization gate. **Cross-ref**: OL-54 (Reg-based SIMD overview + adv_api impl-header path caveat + docstring-vs-static_assert caveat); P-P62 (Row-Scalar VEC Multiply via Brcb — same amortization principle for hand-rolled multi-row scalar Mul; precondition R≥8 rows batched); OL-89 (vendor primitive substitution opportunities in the analyzer phase).

<!-- 迁移自 porter patterns/PATTERN_INDEX.md P-P87 索引行（该条目无 domains 正文，索引行即全部内容；B2 手工补卡，format 对齐 convert_patterns_to_okf.py 输出）。confidence/severity 未升格。 -->
