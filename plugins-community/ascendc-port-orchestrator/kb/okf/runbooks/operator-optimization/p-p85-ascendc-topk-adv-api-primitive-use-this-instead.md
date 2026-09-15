---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "`AscendC::TopK` adv_api primitive — use this instead of hand-rolled chunked-Sort + scalar 2-pointer merge for k-selection on arch 3510/5102/3003/3113"
description: "When implementing TopK / TopKTopP / quantile / threshold-mask kernels on Ascend950PR (arch 3510) and similar, the public AscendC::TopK<T, isInitIndex, isHasfinish, isReuseSrc, topkMode, config> from a"
severity: critical
confidence: single_run
original_id: P-P85
timestamp_inferred: true
tags: [sort, optimization, gettopkmaxmintmpsize, topktilingfunc, kernelvbsmergesort, radix_select, merge_sort, p-p85, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

When implementing TopK / TopKTopP / quantile / threshold-mask kernels on Ascend950PR (arch 3510) and similar, the public `AscendC::TopK<T, isInitIndex, isHasfinish, isReuseSrc, topkMode, config>` from `adv_api/topk/topk.h` (host helpers `GetTopKMaxMinTmpSize` + `TopKTilingFunc` from `adv_api/sort/topk_tiling.h`) delegates to vec-pipe-bound primitives (likely the private `KernelVbsMergeSort` family in `opp/built-in/op_impl/.../arch35/merge_sort_simd.h`). **dtype matrix** — `TopKConfig::algo`: `RADIX_SELECT` supports ALL int + half + float + **bf16**; `MERGE_SORT` supports half + float ONLY (no bf16). For bf16 hot-path ops RADIX_SELECT is mandatory; for fp16/fp32 benchmark both. **Snippet**: `TopKConfig cfg{ TopKAlgo::RADIX_SELECT, TopKOrder::LARGEST, /*sorted=*/true }; TopKTilingFunc(platform, /*inner=*/N, /*outter=*/B_per_AIV, k_runtime, sizeof(T), false, TopKMode::TOPK_NORMAL, true, cfg, tiling);` host-side, then `AscendC::TopK<T, false, false, false, TopKMode::TOPK_NORMAL, cfg>(sortedVal, sortedIdx, srcVal, dummyIdx, finishLocal, tmpLocal, k_runtime, tiling, info, true);` kernel-side. **Anti-pattern (current op#9 9_TopKTopP Phase 1 pre-kw-4)**: chunked `AscendC::Sort<>` over CHUNK_LEN=2048 per chunk + scalar 2-pointer merge into TOPK_CAP buffer → `aiv_scalar_ratio=0.898` (scalar-pipe bound) → 1024 us/row for [N=65536, k=1024] bf16. **CANN's own `npu_top_k_top_p` internal Sort kernel** (measured via msprof on a single fused call): `aiv_vec_ratio=0.721`, 204 us/row for the same shape — **5× faster**, vec-pipe-bound, almost certainly using the adv_api or equivalent vec-merge primitive. **Determinism risk (medium)**: RADIX_SELECT may have its own tie-break ordering — characterize with a 5-rep det-check before adopting on DET_POLICY=required ops. **Generalizes** to any kernel currently doing chunked-Sort + scalar-merge for k-selection: rejection-sampling top-K, beam-search prefix, threshold-mask + scatter, sparse-select. **Cross-ref**: P-P81 (runtime-bounded loop cap — orthogonal optimization once the merge primitive is replaced); EC-33 (RADIX_SORT in `Sort<>` chunked context still defensive-MERGE per EC-33; AscendC::TopK::RADIX_SELECT is a DIFFERENT API path from `Sort<>::RADIX_SORT` — does NOT trip EC-33 per pp-2 measurement); P-P84 (analytical decomposition methodology — anti-pattern fixed by P-P85's "use the adv_api primitive" approach). Validated empirically op#9 9_TopKTopP pp-2 (2026-05-03 Ascend950PR_9579, CANN 9.0.0 b103) — measured 3.35× total gap is recoverable, NOT a structural ceiling. kw-4 spawn pending.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P85，convert_patterns_to_okf.py）。confidence 未升格。 -->
