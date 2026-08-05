---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "PyTorch torch.linspace enforces endpoint-exact — step*(n-1) does NOT equal end in fp32"
description: "torch.linspace(start, end, n) guarantees result[n-1] == end exactly; a naive step*(n-1) undershoots by ~1 ULP in fp32, and when floored to an integer gather index it rounds DOWN → off-by-one wrong gather row, not a 1-ULP error."
phenomenon: precision_issue
signal:
  - "kernel emits a linspace-like sequence in SIMT as step*i (step = (end-start)/(n-1)) and the result is floored/converted to integer indices for a downstream gather"
confidence: single_run
original_id: OL-98
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, precision, ol-98, linspace, fp32, indexing]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

Kernel computes element `i` of a linspace-like sequence as `step * i` where `step = (end - start) / (n - 1)`, AND the output is then floored / converted to integer indices for a downstream gather.

`linspace(0, 34, 4)` in PyTorch returns `[0, 11.3333, 22.6667, 34.0]`. Naive `(34/3)*3 = 33.999996…` → `.long()` = 33 (NOT 34). A 1-position error in the integer floor → a completely wrong gather row downstream.

## 根因 / 教训

`torch.linspace(start, end, n)` enforces `result[n-1] == end` **exactly**. The naive `step * (n-1)` formulation does not reproduce `end` in fp32 for many inputs — it lands ~1 ULP below.

### Why this is catastrophic, not 1-ULP

When the linspace output feeds `.long()` (truncate-to-integer), even a 1-ULP undershoot below an integer rounds DOWN, not to nearest:
- PyTorch's exact 34.0 → floor = 34
- Naive 33.999996 → floor = 33

If the gather is `pos_embed_weight[floor * num_grid_per_side + …]`, the wrong `floor` reads from a row 35 positions away — outputs differ by the magnitude of `pos_embed_weight[wrong_row]`, not by a ULP.

### Fix — special-case the endpoint

```cpp
__aicore__ inline float my_linspace(int64_t i, int64_t n, float end) {
    if (n <= 1) return 0.0f;
    if (i == n - 1) return end;     // PyTorch endpoint guarantee
    return (end / float(n - 1)) * float(i);
}
```

For a non-zero `start`, also special-case `i == 0` to return `start` exactly.

### Anti-pattern

```cpp
// WRONG: omits endpoint special case → off-by-one floor on edge cases
__aicore__ inline float my_linspace(int64_t i, int64_t n, float end) {
    if (n <= 1) return 0.0f;
    return (end / float(n - 1)) * float(i);
}
```

### Applies to

Any kernel that (1) emits `linspace`-equivalent in SIMT rather than using AscendC `Arange()` or a precomputed host input, (2) then truncates/floors to integer for gather indexing, (3) where the truncation feeds a gather that depends on integer alignment. MUST endpoint-special-case to match `torch.linspace` semantics.

Precision-audit (CPU-truth, 2026-04-29): VALIDATED-CPU — derived from op#28 MultimodalRopePosComputationWithGridBasedIndexing kw-fix iter 1. Note: the endpoint fix alone does not save op#28 from MARE failure under eps=1e-7 (see OL-99 / OL-101).
