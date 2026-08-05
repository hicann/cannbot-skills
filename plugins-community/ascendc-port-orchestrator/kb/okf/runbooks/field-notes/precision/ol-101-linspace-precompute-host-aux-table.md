---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "PyTorch fp32 linspace is not closed-form replicable in kernel — precompute on CPU host and ship as input"
description: "torch.linspace(fp32) on CPU is a per-element HYBRID rounding contract that no single closed-form kernel formula reproduces; when values feed integer indexing or bit-exact arithmetic, precompute the table on host (at::linspace/at::arange) and ship it as a GM input tensor."
phenomenon: precision_issue
signal:
  - "kernel emits linspace/arange-equivalent values in SIMT and downstream arithmetic must bit-match a CPU-truth PyTorch reference"
  - "endpoint-special-case (OL-98) and per-op rounding chain (OL-102) still leave the kernel below the fp32 MARE threshold"
confidence: single_run
original_id: OL-101
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, precision, ol-101, linspace, host-precompute, aux-table]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

Kernel computes `linspace`-equivalent values and downstream uses them for arithmetic that must bit-match a CPU-truth PyTorch reference. No single closed-form fp32 formula matches PyTorch's output.

## 根因 / 教训

`torch.linspace(start, end, n, dtype=fp32)` on CPU produces values that **do not match** any of these closed-form fp32 formulas:
- `(end / (n-1)) * i` — gives `0x41b55555` at i=2,n=4,end=34, but PyTorch gives `0x41b55556` (1 ULP higher)
- `(i / (n-1)) * end` — gives `0x41355556` at i=1, but PyTorch gives `0x41355555`
- `start + step * i` (with step in fp32 OR fp64 then cast)
- FMA `idx * step + start`
- `weight*end + (1-weight)*start`

Empirical brute-force showed PyTorch's output is a HYBRID: index 1 matches "fp64 then cast at end" while index 2 matches "weight × end fp32 chain". Different elements use different rounding contracts — not replicable from kernel SIMT code without porting PyTorch's exact internal vectorized impl.

### Generalizable fix — precompute on host, ship as GM input

Precompute auxiliary tables on host using PyTorch (`at::linspace`, `at::arange`, etc.) and ship them as kernel input GM tensors.

```cpp
// In pybind11.cpp (NPUKernelBench allows pybind to use torch for tensor metadata
// AND for precision-critical aux tables the kernel cannot replicate bit-exactly):
auto h_idxs = torch::linspace(0.0f, end, h,
    torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
auto h_idxs_dev = h_idxs.to(pos_embed_weight.device()).contiguous();
// pass h_idxs_dev.data_ptr() to the kernel launch
```

Kernel reads `coords[h_off + h_pos]` instead of computing the value.

### Empirical impact (op#28)

| Iteration | Approach | Pass rate |
|-----------|----------|-----------|
| baseline | inline `step * i` | 4 / 50 (8%) |
| kw-fix-1 | + endpoint special-case | 4 / 50 (no help) |
| kw-fix-2 | + per-op rounding chain | 7 / 50 |
| kw-fix-3 | + precompute linspace+floor+dh+dw+weights on host | **37 / 50 (74%)** — 9.25× improvement |

Remaining 13 failures: mostly fp32 cases with MARE just above threshold (1e-3 vs 1.22e-3) — borderline rounding-chain differences needing deeper investigation.

### When this rule applies

- Kernel needs `linspace`, `arange`, `range`, or any small math-function-derived auxiliary table.
- The table values feed downstream integer indexing OR bit-exact precision-critical arithmetic.
- Table size is small relative to the kernel's compute (sum of aux tables ≪ total_tokens × H heavy lift).
- Verifier is CPU-truth. (Source text truncated in the batch excerpt at this condition.)

Precision-audit (CPU-truth, 2026-04-29): VALIDATED-CPU — op#28 MultimodalRopePos kw-fix, multi-iteration empirical discovery.
