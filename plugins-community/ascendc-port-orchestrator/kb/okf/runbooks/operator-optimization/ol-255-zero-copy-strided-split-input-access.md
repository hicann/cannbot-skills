---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Zero-copy strided split-input access — offset into the contiguous input in-kernel instead of pybind narrow+contiguous"
description: "For chunk-then-compute ops (a,b=split(x); op(a)⊙b), pybind narrow().contiguous() copies each half on NPU; pass contiguous x and offset in-kernel instead — saved 134MB for [4096,8192] fp32."
confidence: single_run
original_id: OL-255
classified_by: llm-assisted
timestamp_inferred: true
tags: [data-movement, optimization, ol-255, zero-copy, split-input, glu]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**Category:** algorithm_selection / data-movement. Loaded by the Generator (fused / split-input ops) at Phase A pattern detection.

**Pattern detection — when to apply.** Scan the reference `forward()` for:
```
a, b = chunk/split/narrow(x, 2, dim)
result = op(a) ⊙ b        # ⊙ = *, +, or any elementwise combine
```
If detected → apply zero-copy strided access (P-P108). Do NOT generate pybind that calls `narrow+contiguous`.

**Why.** Pybind-side `narrow().contiguous()` creates a full NPU-side memory copy of the split tensor. For a `[4096, 8192]` fp32 input this is 2 × 67 MB = 134 MB of unnecessary NPU memory traffic **before the kernel even starts**.

**Template (P-P115).** Pybind side — one `contiguous()` call, NO narrow:
```cpp
torch::Tensor xc = x.contiguous();  // ensure contiguous; no-op if already
// Pass xc directly to the kernel — NO a_orig/b_orig split
```
Kernel side — offset-based access:
```cpp
uint64_t xb = ob * 2 * block_size;
DataCopy(a_tile, xGm_[xb + offset], count);                 // first half
DataCopy(b_tile, xGm_[xb + block_size + offset], count);    // second half
DataCopy(yGm_[ob * block_size + offset], y_tile, count);
```

**Savings.** Eliminates 2 × input_half_size NPU memory copies and reduces HBM allocation pressure.

**When NOT to apply:**
- Single-input ops (no split) — the standard single-input kernel is correct.
- Split with a non-trivial reshape/permute before compute — evaluate case-by-case.
- port_a3 mode — the transform may be out of scope for an initial port.

### Evidence
swi_glu V1 (2026-06-23): pybind narrow+contiguous → 134 MB extra NPU copies for `[4096,8192]` fp32; perf 0.72× geo_mean. swi_glu V5 (2026-06-24): kernel-side stride access → zero extra copies; combined with OL-254 (multi-core) + OL-63 (tile=8192) → perf 1.52× geo_mean (2.11× vs V1). Ascend950PR_957b, CANN 9.0.0. Unverified on Ascend910_V220 (A3 — strided GM access works on V220; verify on first A3 cold-start). Pattern also applies to gate_proj+up_proj MLP fusion and any GLU-family op (SwiGLU, ReGLU, GeGLU).

### Related
- OL-254 (multi-core outer_blocks — apply together), OL-63 (tile-first UB allocation), P-P115 (zero-copy strided split-input code template).
