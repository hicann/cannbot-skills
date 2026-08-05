---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "fp32 op-order is load-bearing — literal translation beats algebraic-equivalent rewrite for cancellation-prone chains"
description: "For cancellation-prone fp32 chains (EMA/Welford/running-variance), the kernel must reproduce the reference's exact left-to-right evaluation order; associativity swaps and host-side fp64 strength reduction change the bit pattern and fail precision."
phenomenon: precision_issue
signal:
  - "Kernel implements a reference fp32 chain with EMA-style updates (α·a + (1-α)·b·b), running variance, Welford, or any expression where transient cancellation between addends is plausible"
confidence: single_run
original_id: OL-112
classified_by: llm-assisted
timestamp_inferred: true
tags: [precision, kernel-design, ol-112, op-order, cancellation]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发
Kernel implements a reference fp32 chain such as `out = a*b + (1-a)*c*c` (EMA update, running variance, Welford), where transient cancellation between addends is plausible. The math is algebraically correct but precision fails.

## 根因 / 教训
Python evaluates `out = a*b + (1-a)*c*c` left-to-right per operator precedence. The kernel's instruction sequence must reproduce the **same evaluation order**, not just an algebraically equivalent one. Two failure modes:

**FM1 — kernel-side associativity swap.** Multiplication is commutative and (overflow-free) algebraically associative for fp32, but NOT bit-equivalent under rounding when intermediates differ in magnitude. Swapping `((1-α)·c)·c` for `(c·c)·(1-α)` yields a different bit pattern; feeding that into an EMA addition amplifies the divergence by cancellation.

**FM2 — host-side fp64 strength reduction.** A perf-naïve "optimization" pre-computes a reciprocal once per launch on the host (Python float = fp64) then casts to fp32 for a kernel parameter. The reference does the division in-line in fp32, which rounds differently from `host_fp64_recip → fp32_param → kernel Muls`.

### Concrete anchors
**FM1** — 17_AdamW kw-1 (2026-05-01):
```cpp
// REFERENCE: v_new = beta2 * v + (1-beta2) * grad * grad     // left-to-right
// WRONG kernel order (catastrophic when β2·v ≈ -(1-β2)·grad²):
Mul(t1, grad, grad);          // t1 = grad²
Muls(t1, t1, 1.0f - beta2);   // t1 = (1-β2)·grad²
// CORRECT kernel order (literal translation):
Muls(t1, grad, 1.0f - beta2); // t1 = (1-β2)·grad
Mul(t1, t1, grad);            // t1 = ((1-β2)·grad)·grad
```
Result: prior 0/18 PASS → kw-1 9/9 T1 PASS (5 bit-exact).

**FM2** — 17_AdamW kw-1 (2026-05-01):
```python
# BUGGY: host fp64 reciprocal cast to fp32 param
inv_one_minus_b1p = 1.0 / (1.0 - beta1_power)   # fp64 division
m_hat = m_new * inv_one_minus_b1p               # kernel Muls
# CORRECT: in-kernel fp32 division matches the reference
# kernel: Divs(t, m_new, 1.0f - beta1_power)
```

### Recommended action when this signature applies
1. Translate the reference op-by-op into kernel ops, preserving the expression-tree shape.
2. Refuse "performance" rewrites that reorder commutative operands or pre-compute reciprocals on host until precision is locked in.
3. After precision PASS, perf can recover the reciprocal via in-kernel `Reciprocal+Mul` (algebraically equivalent on the same hardware) — a fundamentally different transformation from host-side strength reduction.
4. When debugging cancellation-prone chains, compute an fp64 ground-truth to bisect.
