---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Multi-branch sampling: match downstream softmax count to each branch's selection basis"
description: "In multi-branch sampling, the downstream normalization count must match the selection basis: logit-selection → one softmax; probability-selection is pre-softmaxed, so re-applying double-softmaxes."
confidence: single_run
original_id: OL-263
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-263, sampling, softmax, double-softmax]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**When to use.** Any composite sampling / selection kernel (top-k / top-p, nucleus sampling, gated multi-branch selection) that supports more than one per-branch selection strategy. The rule is chip-independent — it is a pure algorithm-level composition contract, not a hardware constraint.

**Rule — the downstream normalization count must match the selection basis:**

- **Logit-based selection** (select top-K from un-normalized logits) → apply softmax **once** after selection. 1 normalization total.
- **Probability-based selection** (select top-K from values that were *already* softmaxed) → the softmax was applied **before** selection. Applying softmax again produces a **double-softmax** (softmax of a softmax), which is a mathematically different, semantically wrong distribution.

**Why it hides.** The individual primitives (Sort, ReduceMax, Exp, Div, CumSum) are each locally correct. The defect lives in the **composer**: the pipeline wires correct primitives into the wrong algorithm because the selection basis does not match the branch's semantic contract. Every line reads correct, so it passes cursory review and only fails precision validation on the branch that used the wrong basis.

### Concrete anchor (top_k_top_p_sample, A3→A5 port, `top_k_top_p_sample_kernel.h`, 2026-06-25)

The op has two branches that each select top-K via iterative ReduceMax, but from different source buffers:

- **Branch A** — selects top-K **logit** values from the raw-logit GM, then applies a **single softmax** to the selected logits. Must read its source via `TiledCastToGM` (raw copy, no normalization).
- **Branch C** — selects top-K **softmax probabilities** from the GM-resident softmax workspace (produced by the 3-pass tiled softmax), then applies softmax again (its contract expects the already-softmaxed values). Reads its source via `TiledSoftmaxToGM`.

```cpp
// Branch A: logit-based selection — use TiledCastToGM (raw copy, no normalization)
FindGlobalMaxFromGM(rawLogitGm, ...);   // selects from raw logits → single softmax downstream

// Branch C: probability-based selection — use TiledSoftmaxToGM (3-pass result)
FindGlobalMaxFromGM(softmaxWkGm, ...);  // selects from already-softmaxed values
```

**Anti-pattern caught** (kw-5 iter-1, precision FAIL 25/32): Branch A was accidentally routed through `TiledSoftmaxToGM` (3-pass softmax) instead of `TiledCastToGM` (raw copy), so Branch A produced a double-softmax instead of a single softmax. The output diverged from the single-softmax reference across all V sizes. **Fix:** route each branch through its correct selection-source function so the normalization count matches the selection basis.

### Evidence

- top_k_top_p_sample A3→A5 kw-5 (2026-06-25, Ascend950PR, CANN 9.0.0): iter-1 precision failure (25/32 PASS). Root cause: Branch A (logit selection) mis-routed through the softmax source, giving double-softmax. Fixed by correcting the per-branch selection source.

### Other instances (predicted)

Any multi-strategy sampling/selection composer: nucleus (top-p) with a logit vs probability threshold, temperature-scaled top-k where scaling order matters, or any pipeline that reuses one `select top-K` primitive across branches with different upstream normalization state. Audit that each branch's normalization count equals its selection basis.
