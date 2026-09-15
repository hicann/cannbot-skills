---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Empirical-measurement-required for fused-op gap analysis — analytical-only decomposition is reward-hacking on the diagnostic side"
description: "When estimating \"CANN's per-sub-op cost\" for a fused op, analytical-only decomposition (estimate \"CANN's Phase-X equivalent\" by selecting a standalone CANN op that LOOKS LIKE Phase X, then compute ana"
severity: critical
confidence: single_run
original_id: P-P86
timestamp_inferred: true
tags: [platform_compat, optimization, npu_top_k_top_p, p-p86, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

When estimating "CANN's per-sub-op cost" for a fused op, analytical-only decomposition (estimate "CANN's Phase-X equivalent" by selecting a standalone CANN op that LOOKS LIKE Phase X, then compute analytical gap) is unreliable AND can be wildly wrong. **Op#9 9_TopKTopP fo-1 vs pp-2 evidence (2026-05-03 Ascend950PR_9579)**: fo-1 estimated `npu_top_k(k=1024)` standalone at ~7 us/row → claimed "146× gap on Phase 1" → declared structural ceiling. pp-2 directly msprof'd `torch_npu.npu_top_k_top_p` and found CANN's internal Sort kernel runs at **204 us/row** (full-row sort, not k=1024-only). Real gap = **5× on Phase 1, 3.35× overall** — **fo-1's claim was off by ~30×**. The analytical estimate was wrong because fused ops decompose internally into sub-ops with DIFFERENT shapes than the user-facing standalone equivalents (CANN's `npu_top_k_top_p` does full-row sort then apply-on-sorted; standalone `npu_top_k(k=K)` does only top-K — different cost model). **Methodology rule (mandatory for fused-op gap analysis)**: profile BOTH (1) `torch_npu.<fused_op>(...)` directly with msprof to observe internal kernel decomposition (kernel names, BlockDim, dur, pipe ratios per internal kernel), AND (2) each kernel name from (1) — that's the actual sub-op-level reference cost. If only fused-op msprof is available, per-row dur is still tighter than analytical estimates: divide by BlockDim × per-AIV row count. **Sustained-call EC-33 mitigation for the standalone-ref measurement step**: each measurement should run in a FRESH Python process (multi-process msprof), not back-to-back in one process — pp-2 measured this directly (M2 process aborted after 3 sub-ops succeeded clean within one process; M1+M1b in fresh processes were stable). **Block conditions (when this rule applies)**: any time aog-fused-optimizer / aog-kernel-optimizer / orchestrator is about to declare "structural ceiling" or "perf plateau" on a fused op. Analytical-only verdict is INSUFFICIENT — must include measured per-internal-kernel msprof of the fused reference op. **Generalizes**: any multi-stage fused op where the user-facing standalone equivalents may use different internal algorithms (Softmax+Mul, RmsNorm+Cast, RoPE+Cache, etc.). Validated op#9 fo-1 vs pp-2 cross-comparison 2026-05-03. **Cross-ref**: P-P84 (the analytical methodology — now downgraded as fallback-only when measurement is impossible); aog-self-critic C27 + new C28 (this rule is C28's empirical-evidence requirement).

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P86，convert_patterns_to_okf.py）。confidence 未升格。 -->
