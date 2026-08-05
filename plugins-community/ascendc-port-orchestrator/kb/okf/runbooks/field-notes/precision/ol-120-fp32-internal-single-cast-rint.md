---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "fp32-internal compute + single CAST_RINT end-cast is the canonical fp16/bf16 path when the reference is torch_npu.<op> (CANN fused)"
description: "When the verification reference is a CANN-fused torch_npu.<op>, match its fp32-promote / fp32-compute / single-CAST_RINT-end-cast pattern; native-T per-op compute diverges by 1 ULP and fails the tightened verifier on many fp16/bf16 cases."
phenomenon: precision_issue
signal:
  - "Reference is torch_npu.npu_<...> (CANN fused); fp16/bf16 cases fail with max_abs_diff = 1 ULP (0.0078125 fp16 / 0.00390625 bf16) while fp32 cases PASS bit-exact; MARE just over threshold, worse after verifier commit 93c5cf6"
confidence: single_run
original_id: OL-120
classified_by: llm-assisted
timestamp_inferred: true
tags: [precision, cast-strategy, ol-120, cast-rint, cann-fused]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发
The verification reference is a `torch_npu.<op>` CANN fused op (`npu_rotary_mul`, `npu_apply_rotary_pos_emb`, `npu_swiglu`, ...). fp16/bf16 cases fail with `max_abs_diff = 1 ULP` (0.0078125 for fp16, 0.00390625 for bf16) while fp32 cases PASS bit-exact under the same kernel structure. MARE ≈ 1 LSB / output_magnitude, just over threshold — especially after the 2026-04-30 verifier commit `93c5cf6` tightened the MARE filter from `threshold` to `threshold × 1e-3`. Loaded by aog-kernel-worker (Phase A when the reference is `torch_npu.<...>`) and aog-precision-probe.

## 根因 / 教训
CANN fused ops internally promote fp16/bf16 inputs to fp32, run the entire compute chain in fp32, then cast the result back to T with IEEE RNE. To match this bit-exactly under the tightened thresholds, the kernel must follow the SAME pattern: fp32-promote-all, fp32-compute, single CAST_RINT end-cast.

- **Native-T per-op compute** (each Mul/Add directly on fp16/bf16) matches CPU PyTorch but diverges from torch_npu by 1 ULP at catastrophic-cancellation elements — fails the tightened verifier on ~30-60% of fp16/bf16 cases with `max_abs_diff = 1 ULP, MARE >> threshold`.
- **Per-Mul cast-down + cast-up roundtrip** between every fp32 op is ALSO wrong — it accumulates rounding error and fails too.

### Concrete anchor
```cpp
// CANONICAL fp16/bf16 path when ref is torch_npu.<op>:
// 1. Cast all T inputs to fp32 (lossless CAST_NONE).
Cast(xF, x, RoundMode::CAST_NONE, count);
Cast(r1F, r1, RoundMode::CAST_NONE, count);
Cast(r2F, r2, RoundMode::CAST_NONE, count);
// 2. fp32 compute chain (op-order MUST match the reference formula — OL-112).
Mul(tmpF, xF, r1F, count);
// ... rotate-and-mul + sub/add chain ...
Add(outF, tmpF, otherF, count);
// 3. SINGLE CAST_RINT end-cast (IEEE RNE, OL-81).
Cast(out, outF, RoundMode::CAST_RINT, count);
```

### Decision rule (when to apply / when NOT)
| Reference is | fp16/bf16 path |
|---|---|
| `torch_npu.<fused>` (CANN op) | **fp32-promote + fp32-compute + single CAST_RINT end-cast** (this OL) |
| `torch.nn.functional.<...>` (CPU PyTorch) | Native-T per-op (matches CPU semantics) |
| Mixed (CANN preferred, CPU fallback per OL-68 case A) | fp32-promote (worker may receive a REFERENCE_SOURCE flag from Phase O2 step 2) |

### Evidence
- 1_RotaryMul cold-start V3.7.6 kw-1 (2026-05-02): post-2026-04-30 verifier commit `93c5cf6` tightened the MARE filter from `threshold` to `threshold × 1e-3`; the prior-art native-T compute kernel FAILed on ~32/50 fp16/bf16 cases with 1-ULP diffs. Phase D iter 2 rewrote to the fp32-promote + single CAST_RINT pattern [source truncated] → recovered.

### Related
- OL-112 (fp32 op-order is load-bearing — the fp32 chain must match the reference formula's order)
- OL-81 (CAST_RINT vs CAST_ROUND — the end-cast must be RNE)
