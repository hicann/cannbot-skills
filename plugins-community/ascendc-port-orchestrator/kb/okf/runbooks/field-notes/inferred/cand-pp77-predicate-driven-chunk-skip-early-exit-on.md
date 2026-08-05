---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Predicate-driven chunk-skip early-exit on iterative merge regresses on Gaussian / well-spread input distributions (ANTI-PATTERN candidate)"
description: "Source: op9 9_TopKTopP ko-1 Opt1 (2026-05-02 Ascend950PR_9579) — REVERT. Validation status: 1 op observed (REVERT evidence). Anti-pattern candidate, not promoted. Hypothesis tested: For a chunked Sort"
phenomenon: build_failure
signal:
  - "Source: op#9 9_TopKTopP ko-1 Opt1 (2026-05-02 Ascend950PR_9579) — REVERT."
confidence: inferred
status: stub
original_id: CAND-PP77
timestamp_inferred: true
tags: [candidate, inferred, k_max, getvalue, cand-pp77]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Source**: op#9 9_TopKTopP ko-1 Opt1 (2026-05-02 Ascend950PR_9579) — REVERT.

**Validation status**: 1 op observed (REVERT evidence). Anti-pattern candidate, not promoted.

**Hypothesis tested**: For a chunked Sort + 2-pointer merge into a top-K buffer of capacity `K_MAX`, add a per-chunk early-exit `if (chunk_max < top_min) skip_merge` once the buffer is full. The intuition: if the new chunk's maximum is below the current top-K floor, no element in this chunk can survive the merge, so the entire `K_MAX`-iter merge loop is wasted work.

**Why it failed (Gaussian-distribution case)**:
- For random-normal input at N = 65536, the top-1088 floor stabilizes at ~2σ after the first 4–8 chunks
- New chunk maxes are routinely 3–4σ → the early-exit predicate `chunk_max < top_min` almost never fires
- Net cost added per chunk: 2× scalar `GetValue` + 1× explicit V→S sync, with zero merge-loop savings
- Wall-clock impact: bf16 N=65536 k=1024 +5.0 % regression; fp16 mid +3.9 % regression; fp32 small −0.4 % (noise)

**When the pattern *would* work** (predicted, not yet observed):
- Long-tailed distributions (real attention scores: a few large positions, many tiny tail)
- Inputs with provable structure (already partially sorted, sparse logits with many filler positions)
- Distributions where the top-K floor saturates fast and most subsequent chunks are below it

**Generalization**: any iterative merge / scan / accumulate with a "skip if this iter cannot contribute" predicate is data-distribution-dependent. The optimization is NOT a free win — it adds fixed predicate-test cost in exchange for a probabilistic savings that depends on input distribution.

**Promotion criteria**:
1. Demonstrate +5 % or better speedup on a real-workload distribution (e.g. captured attention scores or production logits) on ≥2 ops
2. Document a distribution test the kernel can apply at runtime to gate the early-exit (e.g. estimate input variance from the first chunk's max-to-mean ratio; only enable skip when the estimated tail is heavy)
3. Show that the predicate cost (`GetValue` + sync) is amortized across enough merges to net positive

**Until promotion**: aog-kernel-optimizer should NOT propose chunk-skip / iter-skip early-exits on merge-style hot loops without empirical evidence on the target workload's distribution. If proposed, treat as anti-pattern unless input distribution is documented to be long-tailed.

**Related**: P-P81 (runtime-bounded loop cap — the validated alternative for iter-count reduction; works on per-row k variance, NOT per-chunk data variance), OL-85 (anti-overfitting — distribution-specific kernel paths require the distribution probe in the kernel), failures_ledger.md tagging convention (REVERT entries become anti-pattern evidence).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP77，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
