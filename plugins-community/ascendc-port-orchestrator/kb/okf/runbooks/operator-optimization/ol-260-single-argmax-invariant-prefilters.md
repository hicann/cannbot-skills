---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Single-argmax output is invariant to order-preserving pre-filters — collapse the sampling pipeline to one ReduceMax"
description: "When every step between raw logits and a final argmax is order-preserving (softmax/top-K/top-P), collapse the pipeline to one ReduceMax(calcIndex=true). Break: per-element non-uniform divide."
confidence: single_run
original_id: OL-260
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-260, argmax, sampling, reducemax]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**When to use:** any kernel whose final output is the **index of the maximum** (argmax / greedy-decode token id) over a per-row score vector derived from raw logits through a chain of operations.

**Principle (decision rule).** If every operation between the raw logits and the final argmax is **order-preserving on the argmax element** — the element with the largest logit stays the element with the largest score — then the argmax output is **invariant**, and every intermediate order-preserving selector or normalization is **redundant** and can be deleted. The kernel collapses to a single `ReduceMax(logits, calcIndex=true)` (one O(V) pass — no sort, no softmax materialization).

These three operations are order-preserving on the argmax element and therefore delete-safe in front of an argmax:
1. **Softmax / exp / any strictly-monotonic per-element transform applied uniformly** — the max-logit element stays the max-probability element.
2. **Top-K filter** — the top-K set by definition contains the global maximum, so the argmax survives.
3. **Top-P (nucleus) filter** — tokens are retained in descending-probability order until cumulative mass ≥ p; the max-probability token is retained first, so it survives.

**Break condition (do NOT simplify).** The chain becomes rank-altering (the argmax element can be dethroned) when a transform depends on something other than the element's own logit, applied non-uniformly across elements. Canonical case: **per-element division by a non-constant score** — e.g. Q-sampling `softmax(logits) / (|q| + eps)` where `q` varies per element — can promote a non-max-logit element to the top, so top-K/top-P filtering then genuinely changes the result and the full pipeline is required.

### Concrete anchor
```cpp
// No Q tensor provided -> argmax over RAW logits; top-K / top-P / softmax are all redundant
AscendC::ReduceMax<float>(rout, logits, rwork, /*count=*/V, /*calcIndex=*/true);
int32_t argmax_idx = *reinterpret_cast<int32_t*>(&rout.GetValue(1));  // bits of the index
// (Q-path branches off here via a `has_q` tiling flag; it still needs full sort + softmax.)
```

**Why this matters.** Sampling-class kernels (LLM greedy/nucleus decoding, MoE top-1 routing, any "pick the best token" op) are often specced with elaborate top-K + top-P + softmax pipelines. When the output is a single argmax index and no rank-altering Q/weight intervenes, that entire pipeline is dead compute — one `ReduceMax` replaces it. Detect this structurally at design time by tracing whether the max-logit element is guaranteed to survive every stage.

### Evidence
Ascend950PR, CANN 9.0.0. Applies to all backends.
