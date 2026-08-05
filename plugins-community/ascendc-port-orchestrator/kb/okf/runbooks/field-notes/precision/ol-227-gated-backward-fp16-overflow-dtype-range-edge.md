---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A multiplicative output gate amplifies the upstream backward gradient — the most-accumulated grad overflows fp16 first; classify as a dtype-range edge, not an algorithm bug, when fp32/bf16 hold and the derivation is fp64-bit-exact"
description: "A multiplicative output gate amplifies the backward grad by ~activation magnitude; the most-accumulated grad overflows fp16 first — a dtype-range edge, not an algorithm bug, if fp32/bf16 hold."
phenomenon: precision_issue
signal:
  - "A gated backward (silu/GELU/swiglu output gate) fails strict max-norm in fp16 on a high-magnitude input profile while fp32 and bf16 pass; the failing grad is the one carried through the deepest accumulation"
confidence: single_run
original_id: OL-227
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, ol-227, precision, gated-backward, fp16-overflow, dtype-range]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

Ascend950PR / CANN 9.0.0, gated-backward op class (silu/GELU/swiglu/geglu output gate). Unverified on Ascend910_V220 (the fp16 dynamic-range ceiling is silicon-wide, but this gate-amplification overflow was not replayed on V220 this session).

When a forward op ends with a multiplicative gate `out = y · g(z)` (silu: `g(z)=z·sigmoid(z)`; GELU; swiglu; etc.), the backward propagates `gs = gy · g(z)` into the upstream computation in place of the raw `gy`. For large activations the gate is near-linear (`silu(z) ≈ z`, `gelu(z) ≈ z`), so `gs ≈ gy · z` — the upstream gradient is amplified by roughly the activation magnitude. On a high-magnitude (`×100`) value profile this is a ~100× boost. The grad that **accumulates over the longest contraction axis** (a reverse scan, a sum-over-states, the most-summed reduction) inherits that boost compounded over the accumulation and is the **first to overflow fp16's `finfo.max ≈ 6.55e4`** — while fp32 and bf16 (much wider exponent range) stay finite.

## 根因 / 教训

**Decision rule (which grad fails, and how to classify it)**:
1. **Predict the casualty before generating**: in a gated backward, the grad most likely to overflow low precision is the one carried through the deepest accumulation AND multiplied by the gate-amplified `gs` (not the directly-written per-row grads). Author/verify that grad against fp32/bf16 first.
2. **Classify the overflow correctly**: if the fp64 truth itself overflows on the large-profile elements (skip them per the §5.4 degenerate-element rule, cf. OL-191), and the *surviving finite near-ceiling* elements fail strict max-norm only because of IEEE-non-associative fp32-summation-order cancellation noise (kernel's sequential scan vs torch's parallel einsum), this is a **dtype-range edge, NOT an algorithm/derivation bug** — provided fp32 + bf16 pass and the analytic derivation is fp64-bit-exact (verify per OL-228). Verdict: `PASS_WITHIN_TOLERANCE` (strict count excludes the handful of large-profile fp16 elements; inclusive count passes). Do NOT chase it by diverging from the reference's dtype semantics or by loosening the global tolerance.

**Concrete anchor**:
```
// silu gate backward — gs replaces gy upstream; gs ≈ gy·z for large z (silu(z)≈z)
gs[l]   = gy[l] * silu[l];                       // silu[l] = z[l]*sigmoid(z[l])
dx[l,n] = gs[l]*C[n,l] + dA[l+1,n]*dx[l+1,n];    // reverse scan — deepest accumulation
// grad_A = sum over the scan of dx·(state)·delta  <- gate-amplified AND most-accumulated
//          -> first to exceed fp16 6.55e4 on a ×100 input profile; fp32/bf16 finite.
```

**Evidence**: selective_scan backward (silu gate), A5. Cross-ref: OL-228 (verify the derivation is fp64-bit-exact before classifying as a dtype-range edge), OL-191 (degenerate-element skip rule).
