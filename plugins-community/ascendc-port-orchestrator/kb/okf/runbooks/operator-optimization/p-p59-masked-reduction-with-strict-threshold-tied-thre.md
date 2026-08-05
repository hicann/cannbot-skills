---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Masked reduction with strict-`<` threshold — tied-threshold buffer truncation"
description: "General criterion: if the reference logic contains python threshold = ... # e.g. kth-largest, p-quantile, score cutoff mask = values < threshold # STRICT \"<\" values = masked_fill(values, mask, fill_va"
severity: critical
confidence: single_run
original_id: P-P59
timestamp_inferred: true
tags: [precision, optimization, topk_cap, exp_denom, threshold, gmax, sum_before_block, p-p59, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**General criterion**: if the reference logic contains

```python
threshold = ...                       # e.g. kth-largest, p-quantile, score cutoff
mask = values < threshold              # STRICT "<"
values = masked_fill(values, mask, fill_value)  # -inf / 0 / sentinel
reduced = some_reduction(values)       # softmax / sum / cumsum / norm
```

then ties at threshold will trigger this problem. Examples of specific ops (**not limited to**):
- Top-K + Top-P sampling (`torch.top_k_top_p`)
- Nucleus sampling / top-p sampling
- Attention tail-drop / sparse attention score threshold
- Sparse gather with score filter
- Quantile-based masking + subsequent normalization

**Abstract trap**: the implementation thinks "kept count = k" and so uses a buffer sized `TOPK_CAP = k_max`; actually kept count = `count(v ≥ threshold) ≥ k` (strict `<` keeps all v ≥ threshold, not strict `>`). Ties at the threshold push effective kept up to `k + (T - 1)` (T = number of ties at the threshold).

If `T > (TOPK_CAP - k)`, **some ties are in the row but outside the buffer**. Consequences:
- The denominator/sum inside the buffer is short by `(T_miss) × reduce_weight(threshold)`
- All normalized values inside the buffer are proportionally too large (if reduction = softmax) or too small (if it's a ratio)
- Downstream cumsum / norm flips kept↔dropped at the boundary rank

**Why bf16-large-N exposes this bug first**: bf16 mantissa 7 bits → in [1, 4) the value spacing is 2^-6 = 0.015625; for N=65536 `torch.randn`, statistically each bf16 bin contains ~50 ties. fp16 (10 bit) and fp32 (23 bit) have exponentially fewer ties.

**Anti-pattern**: implementing with a "fixed-N set" mindset (top-K buffer / top-N compact / preallocate k max), then reducing inside the buffer (softmax, normalization, cumsum). As long as the reference uses a strict-`<` threshold mask, the latent bug exists.

**Key semantics**: `< threshold` STRICT inequality → kept = `count(v ≥ threshold)`, **not** equal to the nominal `k`. If there are T ties at threshold, `effective_kept = k + (T - 1)` (the kth_value itself is already counted in k).

**Concrete example (excerpt from 9_TopKTopP reference implementation, for structural comparison)**:

```python
# Any "sort + strict-< mask + softmax/reduce" reference has this structure
logits_sort, logits_idx = logits.sort(descending=False, stable=True)
threshold = logits_sort[N - k]                 # can be kth, quantile, score cutoff, etc.
mask = logits_sort < threshold                  # STAR STRICT "<" — this keeps all tied values
logits_sort.masked_fill_(mask, fill_value)      # -inf / 0 / sentinel
reduced = some_reduction(logits_sort.to(fp32))  # softmax / sum / cumsum / norm
# then cumsum/normalize/threshold decision...
```

Counter-example (from 9_TopKTopP bf16 N=65536 case 8 row 497 — concrete data point proving the pattern):
- k = 993, threshold = 2.140625
- The row contains 48 values == 2.140625 (ties)
- ref kept (nonzero after mask) = 48 ties + all strict-above ≈ 1031 (including all tied)
- If the kernel uses a top-K buffer with `TOPK_CAP = 1024`, it can only capture 34/48 ties → 14 ties are in the row but outside the buffer
- Consequence A: kernel's denominator is missing 14 × reduce_weight(threshold) (here = exp(threshold - gmax)) → all normalized values inside the buffer are proportionally too large → cumsum globally shifted → boundary decision flips (case 8 row 497 rank 110: ref cumsum 0.8128 kept, kernel 0.8119 dropped, differing by 0.0009)
- Consequence B: for those 14 outside-buffer tied-threshold positions, the kernel writes sentinel by default but ref may write the real value (if the downstream cutoff lands in the middle of the tied block)

**Correct pattern (three fallback layers, increasing complexity)**:

### Layer 1 — Buffer enlarge + global denominator (simplest, sufficient for most scenarios)

- Enlarge `TOPK_CAP` to `k_max + max_expected_ties` (bf16 randn N=65536 empirical value: 50-100 ties; TOPK_CAP=2048 is usually enough)
- Second pass over the whole row to compute the **global** softmax denominator:
  ```cpp
  float exp_denom = 0.f;
  for each chunk in row:
      DataCopy(chunk); Cast fp32;
      mask = (v >= threshold) ? exp(v - gmax) : 0;   // Compare + Select + Exp
      exp_denom += ReduceSum(mask);
  ```
- Softmax normalize uses `exp_denom` (not the partial sum inside the top-K buffer)
- Keep the existing tail cumsum logic unchanged

**Closure effect**: all layer-A "rank flip" errors disappear. **Precondition**: actual ties do not exceed (TOPK_CAP - k). If they do, some tied-threshold columns are still outside the buffer (residual layer-B error).

### Layer 2 — Three-way classification + explicit tie-at-threshold handling (general bit-exact approach)

When the buffer upper bound of Layer 1 is not enough (adversarial inputs or tie count with no statistical bound), use per-column three-way classification instead of a fixed-size buffer:

1. **Phase 1 — find threshold**: use chunked merge / partial sort / selection algorithm to obtain `threshold`, `gmax` (or any global constants needed by the subsequent reduce).
2. **Phase 2 — classify full row**: second pass over the whole row, classify each column into three groups:
   - **v > threshold** → always kept, add to the "strict-above set" (record column index + accumulate reduce_weight)
   - **v == threshold** → conditionally kept, according to the reference's tie-break rule. General pattern: if the reference does `stable-sort(asc)` on the original data then reduce, within a tied cluster the smaller column index is "consumed" first (because asc-stable sorts small idx first, and subsequent cumsum/reduce accumulates in this order). The concrete cutoff position is decided by the reference's cutoff rule (top-p cumsum cutoff / quantile / score norm threshold, etc.) — implementation needs per-tied-block `sum_before_block` and per-tie `weight` to solve for the kept subset.
   - **v < threshold** → always dropped
3. **Phase 3 — Emit**: for kept positions, write the reduce result mapped back to native dtype; for the rest, write the mask sentinel.

**Closure effect**: bit-match ref output (if cumsum / reduction order matches).

### Layer 3 — Full-row sort (when k and ties are both large enough that Layer 1/2 is no cheaper)

If `k` or `effective_kept` is close to `N/2`, chunked merge is no longer cheaper than a full-row sort. Directly do full-row hardware Sort + full reduce, streaming by UB chunks. See P-P43 decision tree.

**P-P59 selection criteria (Layer 1 vs Layer 2)**:

| Condition | Choice |
|-----------|--------|
| `k_max + max_expected_ties ≤ UB_available / (per-entry-bytes)` and tie count has a statistical bound (non-adversarial) | **Layer 1** (buffer bump + global denom second pass) |
| The above does not hold, or inputs may be adversarial / tie count has no worst-case bound | **Layer 2** (three-way classification) |

**Anti-pattern details (do not do)**:
- FORBIDDEN: only change comparator order hoping to match tie convention — even with the right order it cannot fix denominator truncation
- FORBIDDEN: use a "convention waiver" to hide bf16 mismatch — forbidden by CLAUDE.md §No Workarounds
- FORBIDDEN: statically enlarge buffer to N — returns to the full-row UB upper-bound problem
- FORBIDDEN: assume "ties are few in fp32/fp16 so a small buffer is enough" — bf16 dtype will expose it first; fp16/fp32 are only temporarily invisible

**Evidence**:
- 9_TopKTopP V2→V3 (2026-04-17/18) hit this exactly. V1 full-row-sort exceeded UB → 34/50. V2 chunked top-K (`TOPK_CAP=1024`) → 45/50 — 5 bf16 N=65536 cases fail at 1-14 elements per case due to tie-at-threshold buffer truncation. V3 Layer 1 fix (TOPK_CAP 2048 + global denom second pass) → **50/50 PASS**. This is the canonical case for the pattern. Other candidate ops (unverified): nucleus sampling variants, attention tail-drop-by-threshold, sparse gather with score filter.
- **9_TopKTopP cold-run (2026-04-18 round 2)**: independent cold-run verifying the worker+probe pipeline; worker implemented Layer 1 to 29/50 stuck; probe found Layer 1 **necessary but not sufficient** — also requires pairing with P-P60 (AscendC Sort ASC tie-break reverse) fixing cutoff_orig_idx reselection, EC-31 (Select mask polarity), and EC-32 (effective_kept vs buffer_len). With those added, 49/50 (residual 1 case is OL-83 torch_npu drift, not a kernel bug). **Pattern extension**: the full implementation of the P-P59 schema needs the canonical implementation sketch; see the combination of P-P60 + EC-31 + EC-32 + OL-83 — all four together guarantee bit-match against the PyTorch stable-sort reference.

**Canonical implementation sketch (for P-P59 Layer 1 + P-P60 combination)**:
```
Phase 0: gmax = ReduceMax(row)
Phase 1: chunked top-K merge → top_val[TOPK_CAP], top_orig_idx[TOPK_CAP]
Phase 2: global softmax denom (scan whole row once, sum exp(v - gmax) for v >= threshold)
Phase 3: effective_kept = count(top_val[i] >= threshold)     # EC-32: not TOPK_CAP!
         ASC-sort top buffer by val (if using ASC walk)
         Cumsum walk ASC positions [topk_len-effective_kept .. topk_len-2]
         identify cutoff_val + n_drop_tied
         **post-walk re-select cutoff_orig_idx**:             # P-P60: critical!
           tied_idxs = [idx for idx in top_orig_idx if top_val == cutoff_val]
           sort tied_idxs ascending
           cutoff_orig_idx = tied_idxs[n_drop_tied - 1]
Phase 4: emit per-column: scalar SetValue(col, kept_val) if (v > cutoff) OR
         (v == cutoff AND orig_idx > cutoff_orig_idx)         # EC-31: use scalar
         (prefer scalar SetValue over VEC Select to avoid mask polarity bugs)
```

**Branchless merge optimization (2026-04-19, new)**:

Phase 1's 2-way merge is conditional by default (has an `if va >= vb` branch). To eliminate the scalar bottleneck, R3b optimizer adopted a branchless merge: with sentinel-padded inputs, each iter does 1 compare + 2 GetValue + 2 SetValue, no branch.

**Hard preconditions**:
1. The two merge inputs (top buffer and new chunk sortValOut) must have at least `TOPK_CAP` slots
2. All unused slots must be pre-filled with the -inf sentinel
3. **`CHUNK >= TOPK_CAP` must be guaranteed by `static_assert`** — otherwise out-of-bounds read, precision blows up (see PB-14)
4. Copy-back stage: VEC `Adds<float>(top_val, merge_val, 0, TOPK_CAP)` for values (direct copy); idx copy depends on the CANN version:
   - **CANN 9.0.0 on Ascend950PR (as of 2026-04-19)**: `Adds<int32_t>` buffer-to-buffer has a corruption bug (PB-13), so idx copy-back must use a scalar loop
   - **Future CANN versions**: re-verify PB-13; if fixed, switch to VEC `Adds<int32_t>` for a unified path

**Edge cases (must handle, 2026-04-18 added from V3.2 test 2 Phase D iter 1 regression)**:
- **`effective_kept == 0`**: all v < threshold in the row → all positions -inf. Early exit.
- **`effective_kept == 1`**: only 1 kept element remains; reference `top_p_mask[:, -1] = False` forces the max to be kept.
  **Set `cutoff_orig_idx = -1` sentinel** so that the Phase 4 emit `(v == cutoff AND col > cutoff_orig_idx)` branch holds `col > -1` for that element, so it is not falsely dropped.
  **Anti-pattern**: `cutoff_orig_idx = topIdx[0]` (the kept element's own col index). Makes strict `col > cutoff_orig_idx` fail → that kept element is falsely dropped → precision failing_cases (V3.2 test 2 iter 1: 7/50 cases each fail 1 element with max_abs_diff=3.4e38).
- **`effective_kept >= 2`**: standard cumsum walk + P-P60 post-walk re-select.

**Related**:
- P-P42 Hardware Sort pipeline (used in Phase 1 to find threshold)
- P-P43 Sort decision tree (when Layer 3 needed)
- P-P52 fp32 promotion (always for softmax)
- EC-28 fp32 -inf sentinel must be true IEEE -inf (precondition for correct mask output)
- EC-29 SortConfig device-side 2-field schema

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（P-P59，convert_patterns_to_okf.py）。confidence 未升格。 -->
