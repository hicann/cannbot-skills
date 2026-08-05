---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "torch_npu fused-op path divergence at non-grid-aligned (N, H) shapes — `npu_dequant_swiglu_quant` mode=1 evidence"
description: "applies_to: soc=Ascend910_9382 (V220, A3); cann=9.0.0; bisheng=15.0.5+2026-01-28; op_class=quant-fused (DequantSwigluQuant family) verified_on: soc=Ascend910_9382; cann=9.0.0 unverified_on: soc=Ascend"
phenomenon: build_failure
signal:
  - "torch_npu reference produces a different scale / quantized output for certain (N, H) shapes vs the docstring's manual computation. Independent reproduction via"
confidence: inferred
status: stub
original_id: CAND-PP89
timestamp_inferred: true
tags: [candidate, inferred, npu_dequant_swiglu_quant, torch_npu, manual_sc, torch_npu_sc, cand-pp89]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_9382 (V220, A3); cann=9.0.0; bisheng=15.0.5+2026-01-28; op_class=quant-fused (DequantSwigluQuant family)`
`verified_on: soc=Ascend910_9382; cann=9.0.0`
`unverified_on: soc=Ascend950PR (A5 — same fused op may exist with different tile geometry; A5-side probe needed before generalizing)`

**Trigger**: kernel reference is a CANN fused op of the form `torch_npu.npu_<fused>_quant(...)` AND benchmark Pass A shows shape-conditional precision divergence on a small subset of cases that does NOT correlate with H-alignment, dtype, or scalar-input combination.

**Symptom**: torch_npu reference produces a different scale / quantized output for certain (N, H) shapes vs the docstring's manual computation. Independent reproduction via `torch_npu` ops in the docstring's exact compute path produces THE SAME divergence — proving the divergence is upstream in CANN's internal kernel, NOT in the user's port.

**Empirical shape-sensitivity table** (op#11 DequantSwigluQuant a3 mode=1, swiglu = `(x_glu × sigmoid(α·x_glu)) × (x_linear + β)`, then per-row dynamic int8 quantization):

| (seed, N, 2H)      | scale_max_diff | result |
|--------------------|----------------|--------|
| (0, 32, 64)        | 3.26e-1        | FAILS — small (N divides AIV, 2H divides 32) but tile path differs |
| (0, 32, 128)       | 0.0            | PASSES |
| (0, 64, 1024)      | 0.0            | PASSES |
| (0, 216, 5056)     | 4.03e-1        | FAILS — 216 % 48 ≠ 0 AND 5056 % 128 ≠ 0 |
| (0, 256, 8192)     | 0.0            | PASSES |
| (1, 216, 5056)     | 4.83e-1        | FAILS (consistent across seeds) |
| (42, 256, 8192)    | 0.0            | PASSES |

**Hypothesis**: torch_npu's internal CANN op uses a tile-boundary "fast path" gated on `(N % aiv_count == 0) AND (H % tile_h == 0)` (likely `tile_h ≈ 128 fp32`). When BOTH hold → "pure" path; otherwise → fallback path with different rounding semantics.

**Diagnostic recipe**:
```python
# Same diff signature regardless of whether compute is on CPU or NPU torch:
manual_sc    = (x_fp32_compute(...).abs().amax(dim=-1)) / 127.0
torch_npu_sc = torch_npu.npu_dequant_swiglu_quant(...)[1]
(manual_sc - torch_npu_sc).abs().max() == 0.241  # all paths converge on this divergence
```
If `manual_sc` (computed via NPU torch ops along the docstring's path) shows the SAME diff vs `torch_npu_sc`, the divergence is entirely in the fused CANN op — NOT a port bug.

**Action when this signature applies**:
1. msprof the failing vs passing shape to compare CANN op kernel-name / tile params (different sub-op sequences = different rounding paths).
2. If two paths confirmed: compare both paths with fp64 CPU truth and keep the result unresolved if neither path is justified.
3. Do NOT case-specifically patch (OL-85 violation). Either accept residual or extend verifier to skip the fallback-path subset.

**Promote when**: a second torch_npu fused-quant op (e.g. `npu_swiglu_quant`, `npu_add_rms_norm_dynamic_quant`) shows the same `(N, H)`-grid-conditional fast/fallback divergence. Likely co-promotes with CAND-PP85 + P-P58.X into a unified "CANN fused-op grid-path divergence" family.

**Source**: op#11 DequantSwigluQuant a3 kw-2 (2026-04-28). 1-op evidence. Cross-arch isolation note: A5-side probe of same op family pending; if A5 shows identical (N, H)-grid sensitivity, applies_to broadens to `all`.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP89，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
