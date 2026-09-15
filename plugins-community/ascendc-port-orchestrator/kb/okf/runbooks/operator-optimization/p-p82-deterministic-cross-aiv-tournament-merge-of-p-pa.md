---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Deterministic cross-AIV tournament merge of P partial top-K buffers via composite key (value DESC, orig_idx ASC)"
description: "When partitioning a wide row across P AIVs (each AIV produces a per-partition top-K_partial buffer sorted DESC by value), the final K outputs need merging into a globally-sorted top-K. Cross-partition"
severity: high
confidence: single_run
original_id: P-P82
timestamp_inferred: true
tags: [sort, optimization, orig_idx, p-p82, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

When partitioning a wide row across P AIVs (each AIV produces a per-partition top-K_partial buffer sorted DESC by value), the final K outputs need merging into a globally-sorted top-K. **Cross-partition ties** (positions in different partitions with the same value) must resolve by global `orig_idx` to match `torch.sort(stable=True)` semantics, regardless of which AIV finished first. A naive tournament using only value comparison is non-deterministic when ties exist. **Solution**: composite key `(value DESC, orig_idx ASC)` — strict total order on outputs because `orig_idx` is unique per row by construction. Snippet: `bool wins = (v > best_v) | (v == best_v && i < best_i);` inside the per-output-position scan over P partial heads. **Determinism guarantee**: composite key gives strict total order on `(value, orig_idx)` pairs; the merge output is bit-exact regardless of AIV scheduling order. Combined with P-P61 4-prong (single-AIV-per-partition Phase 1 + `SyncAll<true>()` Phase 1.5 + no atomicAdd), the entire kernel remains deterministic by construction. **Novel coverage** vs siblings: P-P61 covers single-AIV-per-row det; P-P79 covers load-reverse tie-direction within a single AIV; **P-P82 is the bridge** that allows multi-AIV reductions to remain deterministic AND match PyTorch stable-sort. **Generalizes**: any cross-AIV merge of K-sorted streams that must match PyTorch stable-sort tie semantics (multi-core top-K, histogram quantile reductions, segmented sort merges, beam-search aggregations). **Activation gate**: only fires when `B < TOTAL_AIV` so multi-AIV partition is profitable (see OL-124). For `B ≥ TOTAL_AIV` the partition path falls back to single-AIV-per-row P-P61. Validated op#9 9_TopKTopP kw-2 (2026-05-03 Ascend950PR_9579): Pass A 16/16 bit-exact + det 50/50 PRESERVED; tournament merge correctly implemented. Perf neutral on B≥56 harness (architectural fallback fires); architecturally forward-compatible for any future B<56 wide-N case. **Cross-ref**: P-P61 (single-AIV-per-row determinism — the per-partition prerequisite), P-P79 (load-reverse intra-AIV tie-break), P-P81 (runtime-bounded loop cap — orthogonal cap optimization), OL-124 (multi-AIV-per-row activation gate B<TOTAL_AIV).

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P82，convert_patterns_to_okf.py）。confidence 未升格。 -->
