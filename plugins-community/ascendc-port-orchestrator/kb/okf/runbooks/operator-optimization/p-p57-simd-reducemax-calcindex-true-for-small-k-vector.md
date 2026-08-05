---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMD ReduceMax(calcIndex=true) for small-k vectorized topk"
description: "Problem: Scalar insertion sort / selection-by-scan is extremely slow for topk on N ≥ 2048. Per-row for i in N: if buf.GetValue(i) > topk_min: insert(i, v) runs on the S pipe, and msprof shows scalar_r"
confidence: single_run
original_id: P-P57
timestamp_inferred: true
tags: [sort, optimization, scalar_ratio, p-p57, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: Scalar insertion sort / selection-by-scan is extremely slow for topk on N ≥ 2048. Per-row `for i in N: if buf.GetValue(i) > topk_min: insert(i, v)` runs on the S pipe, and msprof shows `scalar_ratio` easily hitting 0.97+. The S pipe does 1 element/cycle — 2-3 orders of magnitude slower than the VEC pipe's 64-256 elements/cycle.

**Pattern** (k SIMD ReduceMax calls, each returning val+idx):

```cpp
// Setup:
// - valBuf: LocalTensor<float> containing all N candidate values (fp32 for precision)
// - reduceWorkBuf: temporary buffer, ≈ ceil(N/16) × 4 bytes
// - reduceOutBuf: 64 bytes (stores val + idx pair)

for (int kk = 0; kk < k; ++kk) {
    // 1) SIMD ReduceMax with index: find val and its position in buf simultaneously
    AscendC::ReduceMax<float>(
        reduceOutBuf,     // dst: [val_fp32 | idx_bits_as_fp32]
        valBuf,           // src
        reduceWorkBuf,    // work
        /*count=*/N,
        /*calcIndex=*/true
    );

    // 2) Scalar read of val and idx
    float  val = reduceOutBuf.GetValue(0);
    int32_t idx = *reinterpret_cast<int32_t*>(&reduceOutBuf.GetValue(1));  // reinterpret bits

    // 3) Output topk_vals[kk] = val, topk_indices[kk] = idx

    // 4) Mask out the selected slot so next ReduceMax picks the next-best
    valBuf.SetValue(idx, -INFINITY);

    // 5) S_V sync: SetValue is on the S pipe; next ReduceMax is on the V pipe
    AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
}
```

**Applicability**:
- k ≤ 16 (k ReduceMax calls are only worth it for small k; switch to P-P42 hardware Sort for large k)
- N ≥ 256 (for small N, ReduceMax setup overhead exceeds the scalar loop cost)
- Small-k large-N topk / selection (MoE gating, beam search step, attention head-k, etc.)

**Not applicable** (use P-P42 hardware Sort):
- k close to N (essentially a full sort)
- Need the complete sorted output order (P-P57 only guarantees the top-k values; tie-break order depends on -INFINITY mask hit ordering)
- int dtype (hardware ReduceMax only supports fp32/fp16/bf16)

**Evidence**:
- 7_MoeGatingTopKSoftmax (2026-04-17): k ∈ [1,10], N ∈ [2048, 7168], dtypes fp16/fp32/bf16.
  - **Before** (scalar insertion topk): msprof `scalar_ratio=0.975, vec_ratio=0.017` on worst case 17 [512,1024,2048] bf16, end-to-end sum-ratio **0.142x**
  - **After** (P-P57): msprof `scalar_ratio=0.271, vec_ratio=0.679` on same case, end-to-end sum-ratio **1.097x** (+673% single iter). case 47 [3584,7168] bf16: 8.0ms → 0.41ms (19.5× speedup on this case alone).
- Precision: 50/50 PASS held (all of fp16/fp32/bf16 pass). Tie-break matches PyTorch topk first-occurrence (when multiple values are equal, ReduceMax returns the smallest index — matching torch.topk's default behavior).
- top_k_top_p_sample A5 (2026-06-24, Ascend950PR CANN 9.0.0): iterative ReduceMax top-K extraction used for the Q-path (top-K over V ≤ 2048). K ≤ 100 completes in ~0.1 ms with precision PASS — see the practical-ceiling note below; the `k ≤ 16` ceiling is a conservative per-call-cost heuristic, NOT a hard limit.

**Practical K ceiling (decision rule)** — the `k ≤ 16` guidance above is conservative. P-P57's total cost is `k × cost(one ReduceMax over V)`; per-call cost scales with V, so the viable k grows as V shrinks. Measured: K ≤ 100 over V ≤ 2048 (~0.1 ms on A5) is practical. Treat `k > 16` as "evaluate the k×V product — prefer P-P42 only when the iteration count dominates", not as an automatic switch trigger. (See OL-252 for the no-Q case where even this iterative top-K is unnecessary — output is a single argmax.)

**Stop condition**: Switch to P-P42 when k > 16 (conservative), when the k×V product makes iterative ReduceMax dominate, or when full sorted output is required.

**Related**: OL-82 (scalar_ratio > 0.9 signature), P-P42 (hardware Sort for larger k)

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/sort.md（P-P57，convert_patterns_to_okf.py）。confidence 未升格。 -->
