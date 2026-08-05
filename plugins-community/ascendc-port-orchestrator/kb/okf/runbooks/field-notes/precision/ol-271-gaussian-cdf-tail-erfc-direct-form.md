---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Compute a Gaussian-CDF / probit tail from the erfc direct form, not 1+erf (fp32 cancels where erf→−1)"
description: "Φ(x)=0.5·(1+erf(x/√2)) catastrophically cancels in the fp32 negative tail where erf→−1; use the identity 1+erf(z)≡erfc(−z) and call Erfc directly to get the tiny tail value at full fp32 precision."
phenomenon: precision_issue
signal:
  - "Kernel computes a Gaussian CDF / probit / Gaussian-tail expression of the form 0.5·(1±erf(...)) in fp32 and precision fails in the negative tail (large-magnitude negative x)."
confidence: single_run
original_id: OL-271
classified_by: llm-assisted
timestamp_inferred: true
tags: [precision, numerical-stability, ol-271, erfc, gaussian-cdf, gelu]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

Any quantity of the form `Φ(x) = 0.5·(1 + erf(x/√2))` (the standard-normal CDF, and by extension probit / Gaussian-tail expressions) loses catastrophic precision in fp32 in the **negative tail**: as `x → −∞`, `erf(x/√2) → −1`, so `1 + erf(...)` is a difference of two nearly-equal magnitudes and the tiny true tail value is swamped by the fp32 rounding of `1.0`.

Applies on fp32 specifically. Verified on Ascend950PR / CANN 9.0.0; **unverified on Ascend910_V220 (A3)** — erfc/erf primitive residuals may differ there, so re-check the tail before relying on this on A3.

## 根因 / 教训

Use the algebraic identity **`1 + erf(z) ≡ erfc(−z)`** and call `Erfc` directly. `erfc(−x/√2)` computes the small tail value at full fp32 relative precision **without any subtraction**. This is a general numerical-stability rule for tail-dominated transcendental sums, not a GELU-specific trick: reach for the library "complementary" primitive whenever a `1±erf`, `1−exp`, or `log(1+x)`-class near-cancellation has one (`erfc`, `expm1`, `log1p`).

### Concrete anchor (GELU exact/erf backward, Φ(x) term)

```cpp
// grad_x = Φ(x) + x·φ(x)   with grad_output = ones
// Φ(x): use the erfc DIRECT form, NOT 0.5·(1 + Erf(x·INV_SQRT2))
Erfc(phi, neg_x_over_sqrt2, tmp, n);   // phi = erfc(-x/√2)  ← no cancellation
Muls(phi, phi, 0.5f, n);               // Φ(x) = 0.5·erfc(-x/√2)
// x·φ(x) = x · (1/√(2π)) · exp(-x²/2)  (amplitude-suppressed → 0 in the tail; hw Exp is fine)
```

Both `Erfc` and `Exp` are bit-exact-grade on the aclnn path for this op. With the erfc form, all 6 representative cases pass compare.py Stage-1 outright (`small_value_total_count=0`, `cancel_total_count=0`) — with **no** reliance on the small-value / cancellation carve-out that the `1+erf` form would have forced.

### Evidence

- gelu_spec_grad kw-1 (2026-07-02, A5 Ascend950PR_957b, CANN 9.0.0, backward/gradient elementwise): erfc DIRECT form for the Φ(x) term → fp32 mare 2.7e-5, fp16 mare 9.5e-4, bf16 mare 0.0; 6/6 representative PASS, native 30/30. The `1+erf` sum was empirically shown to cancel in the negative tail; erfc avoided it with zero carve-out reliance. This is the numerical-form refinement of the OL-103 erf-backward precision evidence (which states the same result in `1+erf` form).

### Other instances (predicted)

Any op computing a Gaussian CDF / probit / error-function tail: GELU (fwd + bwd), Gaussian-error-linear variants, normal-CDF-based losses, truncated-Gaussian sampling, any `0.5·(1±erf)` factor. More generally the "reach for the complementary primitive to skip the near-cancellation subtraction" rule: `erfc`/`erf`, `log1p`/`log(1+x)`, `expm1`/`exp(x)−1`.
