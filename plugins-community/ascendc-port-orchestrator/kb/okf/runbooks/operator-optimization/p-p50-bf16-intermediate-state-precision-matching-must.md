---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "bf16 intermediate-state precision matching (must-read for backward ops)"
description: "Trigger: backward operator where the reference code uses an intermediate multiplication like (a_bf16 b_bf16).to(fp32) instead of a.to(fp32) b.to(fp32). The difference looks tiny but has a huge impact"
severity: critical
confidence: single_run
original_id: P-P50
timestamp_inferred: true
tags: [precision, optimization, mul, add, p-p50, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Trigger**: backward operator where the reference code uses an intermediate multiplication like `(a_bf16 * b_bf16).to(fp32)` instead of `a.to(fp32) * b.to(fp32)`. The difference looks tiny but has a huge impact on accumulated sums.

**Problem**: If the kernel casts all inputs to fp32 up front and then multiplies, each element's product has more significant bits than the reference (bf16 mantissa 7 bits vs fp32 mantissa 23 bits). **Per-element difference is tiny**, but a reduction sum over millions of elements amplifies it significantly; output fields sensitive to the sum (e.g., `grad_gate = sum * sech²`) end up 100% mismatch.

**Fix template**:
```cpp
// Wrong: direct fp32 multiply
Cast(hid_fp32, hid_bf16, CAST_NONE, cur);     // bf16 → fp32
Cast(mask_fp32, mask_bf16, CAST_NONE, cur);   // bf16 → fp32
Mul(prod_fp32, hid_fp32, mask_fp32, cur);     // fp32 * fp32 (high precision)
Mul(weighted_fp32, go_fp32, prod_fp32, cur);  // accumulate this
WholeReduceSum(...);                           // → sum larger than reference

// Correct: replicate the reference's bf16 intermediate state
Mul(mh_bf16, hid_bf16, mask_bf16, cur);       // bf16 * bf16 → bf16 (low precision, matches ref)
Cast(mh_fp32, mh_bf16, CAST_NONE, cur);        // then cast to fp32
Mul(weighted_fp32, go_fp32, mh_fp32, cur);    // accumulate this instead
WholeReduceSum(...);                           // → matches reference sum
```

**Detection**: carefully read the call order of `.to(dtype)` in the reference.
- `(a * b).to(fp32)` ← bf16 intermediate state, must replicate
- `a.to(fp32) * b.to(fp32)` ← fp32 intermediate state, direct fp32 is fine

**Applicability**:
- Backward ops involving sum-over-products (grad computation)
- Reference explicitly chains `bf16_op → .to(fp32) → reduce`
- Forward ops usually don't hit this (single-element output, no accumulation)
- Small tile (< 256 elements) — minor impact

**Why fp32 intermediate is "over-precise"**: fp32 is of course more accurate computationally, but reference behavior is the contract. Benchmarks compare **bit-level** closeness to the reference, not closeness to "theoretically correct".

**Related**:
- PB-4 (bf16 scalar cast): explains the hardware limit on bf16 arithmetic
- OL-21 (bf16 SIMD Cast pattern): safe API for bf16↔fp32 conversion
- F-P1 (bf16 precision handling): general bf16 numerical boundary

**Evidence**: 29_TanhGatedResidualAddBackward V2: using fp32 intermediate → 14/50 cases grad_gate FAIL
(up to 100% mismatch). Switched to bf16 intermediate + cast → 50/50 PASS, 1.81x mean speedup.
- 20_FusedRopeWithQkNormAndKvCacheUpdate Phase D iter 1: forward fused-rope op, reference `apply_rope = (x_bf16 * cos_bf16) + (rh_bf16 * sin_bf16)` — products implicitly rounded to bf16 before sum. Initial kernel kept products in fp32 → 17/58 cases FAIL with max_abs_diff 0.03125-0.0625 (1-2 bf16 ULP). Fix: bf16 round-trip on each product before Add → 58/58 PASS. Confirms P-P50 applies to forward ops too, not just backward (anywhere reference chains `bf16 OP1 bf16 OP2 bf16 → bf16` with intermediate dtype = bf16).
- ada_layer_norm Path A port kw-4 (2026-05-13, A3→A5): forward norm + affine post-modulation, reference `F.layer_norm(x, ...).to(native_dtype) * (1 + scale_native) + shift_native` — the `.to(native_dtype)` between normalize and post-mod is the load-bearing intermediate cast. Iter-1 kept everything in fp32 until final write (mathematically more accurate) → bf16 case 5 MARE=1240, 5 of 7 small-value outputs diverged because PyTorch's bf16 intermediate quantized them to zero. Iter-2 attempted intermediate cast bf16→fp32→continue-in-fp32 → made bf16 case 2 regress from PASS to FAIL (the extra round-trip rounds without fixing the post-mod dtype mismatch). Iter-4 fix: cast `ln` result to native, pre-compute `(1+scale_native)` once via `Adds(LocalTensor<native>, scale_native, native(1.0f), count)`, then native `Mul`/`Add` for the entire post-mod → 8/8 PASS, 4 cases bit-exact (MARE=0). Confirms P-P50 extends to **forward norm + affine post-modulation** patterns and to the **fp32 → native intermediate-cast** boundary (not just bf16↔bf16 mul-then-cast). Generalization: when reference inserts a `.to(native)` between two compute stages, replicate that exact cast in the kernel — do NOT keep the higher-precision intermediate.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（P-P50，convert_patterns_to_okf.py）。confidence 未升格。 -->
