---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "K_ROWS_PER_AIV outer-loop fusion to amortize aclrtLaunchKernel overhead — gated by AIV utilization (1-op evidence, gate-aware)"
description: "Source: 10_LayerNorm kw-2 (2026-05-03 Ascend950PR_9579) — IMPLEMENTED + VERIFIED INERT on this op, but well-formed mechanism for genuinely under-utilized cases. Validation status: 1 op implemented (Pa"
phenomenon: build_failure
signal:
  - "Source: 10_LayerNorm kw-2 (2026-05-03 Ascend950PR_9579) — IMPLEMENTED + VERIFIED INERT on this op, but well-formed mechanism for genuinely under-utilized cases."
confidence: inferred
status: stub
original_id: CAND-PP78
timestamp_inferred: true
tags: [candidate, inferred, aclrtlaunchkernel, k_rows_per_aiv, cand-pp78]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Source**: 10_LayerNorm kw-2 (2026-05-03 Ascend950PR_9579) — IMPLEMENTED + VERIFIED INERT on this op, but well-formed mechanism for genuinely under-utilized cases.

**Validation status**: 1 op implemented (Pass A 60/60 + Pass B 16/16 + Det 60/60 PRESERVED bit-exact), perf flat (mechanism inert per OL-124 gate). Promotion blocked until validated on a 2nd op where the gate ACTUALLY fires (B < TOTAL_AIV).

**Pattern**:

For per-row kernels (LayerNorm, RMSNorm, Softmax, GroupNorm, etc.) where the inner per-row computation is bounded and the launch overhead `aclrtLaunchKernel` ≈ 20-25 µs per call dominates small cases, parameterize the per-AIV row loop with a `K_ROWS_PER_AIV` stride:

```cpp
// Inner per-row code unchanged from kw-1/ko-1 baseline:
template <typename T>
__aicore__ inline void ProcessRow(int32_t row) { /* ... */ }

// Outer K-fusion wrapper:
__aicore__ inline void Process() {
    int32_t bid = GetBlockIdx();
    int32_t my_rows = (B + blockDim_ - 1) / blockDim_;
    int32_t row_base = bid * my_rows;
    for (int32_t r = 0; r < my_rows; r += K_ROWS_PER_AIV) {
        for (int32_t k = 0; k < K_ROWS_PER_AIV && (r + k) < my_rows; k++) {
            ProcessRow<T>(row_base + r + k);
        }
    }
}
```

Pybind selects K adaptively based on UB budget:
- `K = 4` if `H × sizeof(T) ≤ 2 KB`
- `K = 2` if `H × sizeof(T) ≤ 8 KB`
- `K = 1` (baseline path identical) otherwise

**Activation gate (PREREQUISITE per OL-124)**: this mechanism only delivers measurable speedup when `min(rows / TOTAL_AIV) < 1` across the test set — i.e. some cases have AIVs idle at baseline. If every benchmark case has `rows ≥ TOTAL_AIV`, the existing inner per-row `for r=0..my_rows_` loop already amortizes K rows per launch and the K-fusion wrapper is algebraically inert.

**Pre-implementation check** (do this BEFORE writing any code):

```python
import json
cases = [json.loads(l) for l in open("vendor/.../<op>.json")]
ratios = [num_rows(c) / 56 for c in cases]   # 56 AIVs on a5 V220
if min(ratios) >= 1.0:
    # GATE FAILS — K_ROWS_PER_AIV is decorative on this op
    # Document in analysis.md and skip; pursue different optimization axis.
```

**When the mechanism IS applicable** (predicted, awaiting 2nd op evidence):
- Tiny-batch decode-style ops (single-token attention B=1..32 vs TOTAL_AIV=56)
- Fan-out scatter where producer count < AIV count
- Per-token RMSNorm/Softmax during LLM decode (B=1 incremental)

**Promotion criteria**:
1. Validated on ≥2 ops where the gate ACTUALLY fires (`min(rows/TOTAL_AIV) < 1`) AND mechanism delivers ≥+30% perf gain
2. UB budget analysis shows K=4 path doesn't bust 192 KB on the target dtype
3. Adaptive-K selection logic generalizes (not op-specific hardcoded thresholds)

**Trap to avoid**: implementing K_ROWS_PER_AIV "preventively" on ops where the gate fails — wastes iter budget, adds code complexity, delivers zero perf. Always probe the gate first.

**Related**: OL-124 (the activation gate principle — Mechanism B is gated by this rule), P-P82 (Mechanism A counterpart — multi-AIV-per-row partition, also gated by OL-124), OL-27 (perf re-measurement of byte-identical kernels — needed to verify "inert" vs "regressed").

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP78，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
