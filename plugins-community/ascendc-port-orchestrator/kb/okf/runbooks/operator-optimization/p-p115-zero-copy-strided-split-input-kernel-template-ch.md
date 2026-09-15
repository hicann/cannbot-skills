---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Zero-copy strided split-input kernel template (chunk-then-compute)"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; op_class=fused-elementwise,split-input verified_on: soc=Ascend950PR_957b; cann=9.0.0 status: canonical Template for ops that split input x into (A, B) along a"
confidence: single_run
original_id: P-P115
timestamp_inferred: true
tags: [patterns-index, optimization, p-p115, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=fused-elementwise,split-input`
`verified_on: soc=Ascend950PR_957b; cann=9.0.0`
`status: canonical`

Template for ops that split input x into (A, B) along a dimension, then compute
`op(A) ⊙ B`. Instead of pybind-side `narrow().contiguous()` (which copies),
pass x directly and let the kernel compute the split internally via offset math.

**Pattern detection** (Phase A analysis):
  Reference contains: `a, b = chunk/split/narrow(x, 2, dim)` followed by `op(a) * b`
  -> Apply this template.

**Pybind side** — NO narrow, NO contiguous A/B copies:

```cpp
torch::Tensor xc = x.contiguous();  // ensure contiguous (no-op if already)
int64_t half = xc.size(dim) / 2;
int64_t stride_dim = xc.stride(dim);
// Launch kernel with (xc, N, half, stride_dim) — no separate A/B tensors
```

**Kernel side** — internal offset computation:

```cpp
uint64_t raw_block = h * s;  // elements per half per outer prefix
uint64_t outer_blocks = (s > 0 && h > 0) ? (N / (2 * raw_block)) : 0;
// In tile loop:
uint64_t xb = ob * 2 * block_size;
DataCopy(a_tile, xGm_[xb + offset], c);     // A = x[ob*2*bs + off]
DataCopy(b_tile, xGm_[xb + block_size + offset], c); // B = x[ob*2*bs+bs + off]
```

**Savings**: Eliminates 2 x (N/2) NPU memory copies. Example [4096,8192] fp32: saves 134MB.

**Anti-pattern**:
```cpp
// DON'T: Copies A and B separately
auto a = x.narrow(dim, 0, half).contiguous();  // NPU copy
auto b = x.narrow(dim, half, half).contiguous(); // another NPU copy
```

**Cross-ref**: OL-255 (decision rule + when NOT to apply), OL-254 (multi-core — combine both), P-P114 (multi-core outer_blocks template — use the same partition structure).

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P115，convert_patterns_to_okf.py）。confidence 未升格。 -->
