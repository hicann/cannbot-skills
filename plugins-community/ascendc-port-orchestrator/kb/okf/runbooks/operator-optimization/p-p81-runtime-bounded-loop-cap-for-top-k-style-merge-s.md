---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Runtime-bounded loop cap for top-K-style merge / scan with constexpr K_MAX buffer"
description: "When a kernel uses a constexpr K_MAX to size a top-K-style buffer (e.g. TOPK_CAP = max_benchmark_k + tie_margin), and the per-row k is a runtime scalar that varies across the batch, the for-loop bound"
severity: high
confidence: single_run
original_id: P-P81
timestamp_inferred: true
tags: [sort, optimization, k_max, loop_cap, p-p81, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

When a kernel uses a `constexpr K_MAX` to size a top-K-style buffer (e.g. `TOPK_CAP = max_benchmark_k + tie_margin`), and the per-row `k` is a runtime scalar that varies across the batch, the for-loop bound for the merge/scan that fills this buffer should be `Align8(k_runtime + tie_margin)`, NOT `K_MAX`. The buffer itself stays sized at `K_MAX` (initialized once per row to a sentinel value such as `-inf`), but the work-loop terminates at the per-row bound. Snippet: `int32_t loop_cap = Align8(k_runtime + tie_margin); if (loop_cap > K_MAX) loop_cap = K_MAX; for (int32_t k = 0; k < loop_cap; ++k) { /* merge/scan */ }`. Buffer init MUST cover the full K_MAX so positions `[loop_cap..K_MAX)` read as sentinel (preserves downstream cumsum/threshold-walk invariants). Downstream copy-back uses `loop_cap` as count. **Why it works**: saves `(K_MAX − loop_cap) × ops_per_iter` of scalar-pipe work per chunk per row; for top-K-then-top-P sampling where `k` is typically 64–128 and `K_MAX` is 1088, this is an 8–15× work reduction in the merge stage. **Determinism (P-P61-class)**: `loop_cap` is a pure function of the runtime `k_runtime` scalar, identical across repeat runs; per-row path width varies but each row's path is deterministic. P-P61 4-prong (single-AIV-per-row, no atomic, hardware Sort + scalar merge, queue-rotated output) is unaffected. **Anti-pattern boundary**: if `k_runtime ≥ K_MAX − tie_margin` for ALL rows (i.e. everyone uses `k = K_MAX − tie_margin` so `loop_cap` saturates at `K_MAX`), the optimization yields zero gain. Useful only when the `k`-distribution has variance. **Generalizes** to any constexpr-capped iterative reduction whose true work is per-row variable: top-K, top-P, beam search prefix, segment-sum cap, masked nucleus. Validated op#9 9_TopKTopP ko-1 (2026-05-02 Ascend950PR_9579): scalar_ratio 0.94→0.78 (fp32 small k=64), 0.91→0.72 (fp16 mid k=128); wall-clock −75% / −70% on small/mid; bf16 large k=1024 unchanged (loop_cap saturates at K_MAX). Pass A 16/16 bit-exact + canonical det 50/50 preserved. Median ratio 0.271×→0.397× (+47%). **Cross-ref**: P-P59 (TOPK_CAP sizing for ties — sets the K_MAX), P-P60 (Sort tie direction), P-P79 (load-reverse for ReduceMax-based top-k tie).

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P81，convert_patterns_to_okf.py）。confidence 未升格。 -->
