---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "AscendC Sort ASC tie-break direction is REVERSED from PyTorch stable ASC (ANTI-PATTERN)"
description: "Anti-pattern (NAIVE assumption — wrong): cpp // Assumption: ASC sort + stable → smaller original idx comes first among ties (matches PyTorch) constexpr SortConfig CFG_ASC = {SortType::RADIX_SORT, fals"
confidence: single_run
original_id: P-P60
timestamp_inferred: true
tags: [sort, optimization, cutoff_val, n_drop_tied, cutoff_orig_idx, p-p60, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Anti-pattern** (NAIVE assumption — wrong):

```cpp
// Assumption: ASC sort + stable → smaller original idx comes first among ties (matches PyTorch)
constexpr SortConfig CFG_ASC = {SortType::RADIX_SORT, false};  // isDescend = false
Sort<float, false, CFG_ASC>(dstVal, dstIdx, srcVal, tmp, count);
// WRONG: AscendC Sort ASC actually puts LARGER original idx first in ties
//     — exactly opposite to PyTorch stable ASC
```

**Correct semantics (measured)**:

| Sort config | Tie order (when values are equal) |
|-------------|---------------------|
| `AscendC Sort<DESC>` (documented: "i>j, score[j] is selected first") | smaller original idx appears first |
| `AscendC Sort<ASC>` (undocumented) | **larger** original idx appears first ← reversed |
| `PyTorch .sort(dim, descending=False, stable=True)` (ASC) | smaller original idx appears first |
| `PyTorch .sort(dim, descending=True, stable=True)` (DESC) | smaller original idx appears first |

In other words: PyTorch keeps smaller-idx-first-in-ties in both ASC and DESC (stable semantics); AscendC Sort ASC **flips** this direction.

**Detection**: signature `max_abs_diff = 3.4e38` (-inf vs finite swap) at a few tied-boundary positions per row. fp16/bf16 fail more than fp32 (low-mantissa dtypes have more ties). Only appears on the ASC path — DESC-only sorts (e.g. topk) do not trigger this.

**Fix (two equivalent options)**:

1. **Post-walk reselect cutoff_orig_idx** (validated via probe): after the cumsum walk identifies `cutoff_val`, compute `n_drop_tied` (how many in the tied group to drop), then scan the top-K buffer linearly for all idx with `val == cutoff_val`, and pick the n_drop_tied-th **smallest** as `cutoff_orig_idx`. The Phase 4 mask uses the standard `(v == cutoff AND idx > cutoff_orig_idx)` condition.
   ```cpp
   // After cumsum walk identifies cutoff_val and n_drop_tied:
   int32_t tied_idxs[MAX_TIED];
   int32_t n_tied = 0;
   for (int32_t i = 0; i < effective_kept; i++) {
       if (top_val[i] == cutoff_val) tied_idxs[n_tied++] = top_orig_idx[i];
   }
   // Sort tied_idxs ascending (small n_tied, simple scalar insertion)
   // Pick the n_drop_tied-th smallest as cutoff_orig_idx
   cutoff_orig_idx = tied_idxs[n_drop_tied - 1];
   ```
2. **Secondary key sort by `-original_idx` within ties** (more expensive): do a secondary sort on the top-K buffer. Usually not worth it.

**NOT a fix (time-wasters)**:
- Flipping `SortConfig.isDescend` — both directions are "larger-idx-first-in-ties"; only the value direction differs, tie order still does not match PyTorch
- Per-chunk bubble reorder — only canonicalizes within-chunk; the cross-chunk tie order is determined by the merge logic, not by each chunk's sort tie behavior

**Evidence**: 9_TopKTopP cold-run 2026-04-18, probe iter 4 via `probes/p3_cutoff_boundary_analysis.py` (Python walk-simulation compares AscendC ASC sort output against PyTorch stable ASC output and shows the tied-group idx order is reversed). After the fix: 29/50 → 49/50. The remaining 1/50 is torch_npu vs pytorch-native 1-ULP cumsum drift (see OL-83), not a kernel bug.

**Related**:
- P-P42 (Hardware Sort pipeline): P-P42 describes `Sort<DESC>` tie behavior (smaller-idx-first, per CANN docs). P-P42 makes **no** claim about `Sort<ASC>` — if you use the ASC path, you must additionally handle this anti-pattern
- P-P59 (tied-threshold buffer truncation): P-P59 requires the buffer to hold all tied-threshold values; P-P60 requires correct tie order. **Both are required conditions for a PyTorch-stable-sort-compatible kernel** — neither alone is sufficient
- EC-31 (Select mask polarity): a common parallel bug when building the Phase 4 mask
- EC-32 (effective_kept vs buffer_len): post-walk processing must distinguish the two

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/sort.md（P-P60，convert_patterns_to_okf.py）。confidence 未升格。 -->
