---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Mask-free additive piecewise decomposition — replace MicroAPI Select-based activation forms with `Maxs/Mins` branch gates so each domain reduces to 0 outside its active range, then sum"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=elementwise-activation, port_a3_to_a5 verified_on: soc=Ascend950PR; cann=9.0.0 unverified_on: soc=Ascend910_V220 (A3 family — pattern is"
confidence: single_run
original_id: P-P97
timestamp_inferred: true
tags: [patterns-index, optimization, maxs, mins, add, adds, muls, p-p97, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=elementwise-activation, port_a3_to_a5`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 family — pattern is dtype-arithmetic and should transfer, but no cross-arch witness yet)`

**Principle**: Activation functions defined as `out = mask ? f_pos(x) : f_neg(x)` (one branch per sign of `x`, or per side of a threshold `c`) can be rewritten without any mask register as `out = f_pos(Maxs(x, c)) + f_neg(Mins(x, c))` IF AND ONLY IF each branch evaluates to `0` outside its active range — i.e., `f_pos(c) = 0` AND `f_neg(c) = 0`. The additive form uses only universal SIMD primitives (`Maxs`, `Mins`, `Add`, `Adds`, `Muls`, plus whatever transcendental each branch needs), avoiding the MicroAPI Select / `MaskReg` machinery. Bit-equivalence to the masked form follows from `f_pos(Maxs(x,c)) ≡ 0` whenever `x ≤ c` and `f_neg(Mins(x,c)) ≡ 0` whenever `x ≥ c`, so the sum collapses to exactly one branch at any input.

**When this applies**:
- Activation has a single threshold `c` (typically 0) separating two analytic branches
- One branch reduces to 0 at the threshold (and saturation: `f_pos(x ≤ c) = 0` or `f_neg(x ≥ c) = 0` after the clamp)
- Both branches are expressible in SIMD primitives present in `API_CATALOG.md` (so neither requires the Select / Mask form)

**Canonical anchor (ELU, op#elu kw-1 2026-05-17)**:

```cpp
// MicroAPI Select form (what upstream arch35/elu.h uses inside ElementwiseSch<EluDag>):
//   out = (x >= 0) ? scale * x
//                  : scale * alpha * (exp(input_scale * x) - 1)
//
// Mask-free additive equivalent (bare AscendC SIMD):
LocalTensor<float> pos = ...;   // clamps to x for x >= 0, else 0
LocalTensor<float> neg = ...;   // clamps to x for x <= 0, else 0
LocalTensor<float> exp_in = ...;

Maxs(pos, x, 0.0f, count);                              // pos = max(x, 0)
Mins(neg, x, 0.0f, count);                              // neg = min(x, 0)
Muls(exp_in, neg, input_scale, count);                  // input_scale * min(x, 0)
Exp(exp_in, exp_in, count);                             // exp(...)
Adds(exp_in, exp_in, -1.0f, count);                     // ... - 1
Muls(exp_in, exp_in, alpha, count);                     // alpha * (...)
Add(out, pos, exp_in, count);                           // pos + neg-branch
Muls(out, out, scale, count);                           // outer scale
```

**Bit-equivalence proof sketch**: `pos = Maxs(x, 0)` is `x` for `x ≥ 0` and `0` otherwise. `Mins(x, 0)` is `x` for `x ≤ 0` and `0` otherwise. The neg-branch chain has `exp(0) - 1 = 0`, so for `x ≥ 0` the neg-branch contribution is exactly `0` and `out = scale * pos = scale * x`. For `x < 0` the pos contribution is `0` and `out = scale * alpha * (exp(input_scale * x) - 1)`. `Add(x, 0) = x` is exact in fp32 (no precision loss). Confirmed bit-equivalent for all 8 elu cases on Ascend950PR (3 bit-exact + 5 within T2 ULP tolerance vs CPU truth) with the same Iron-law §5 literal-first ordering as the masked reference.

**Why use this rewrite**:
- The verify-artifact path (OL-164) requires bare AscendC primitives bound via `extern "C" __global__ __aicore__` — MicroAPI Select needs the L2 `__VEC_SCOPE__` + `MaskReg` machinery (P-P96), which is heavier to wire and ties the verify kernel to a specific bisheng codegen path
- The additive form composes with `LAUNCH_BOUND` / TQue depth-4 (OL-63) without extra register-pressure analysis
- The five SIMD primitives used (`Maxs`, `Mins`, `Muls`, `Adds`, `Exp`, `Add`) all live in `API_CATALOG.md` — no missing-primitive risk, no version pinning

**Anti-pattern (don't apply when)**:
- Branches don't reduce to 0 at the threshold — e.g. `f_pos(x) = a*x + b1`, `f_neg(x) = c*x + b2` where `b1 ≠ 0 ≠ b2`: the additive form double-counts the offsets. Either pre-subtract the offsets (so the rewritten branches DO reduce to 0) or keep the masked form.
- The activation requires more than two branches (e.g., a 3-piece function like ReLU6 with clamp ceiling). Recursive application is possible but the `Mins/Maxs` chain grows; consider a `Maxs(Mins(x, hi), lo)` clamp + single-branch instead.
- One branch is transcendental and the other diverges at the threshold (e.g., `1/x` for `x > 0`, `0` for `x ≤ 0`): the `Mins(x, ε)` clamp shifts the divergence to a fixed point but the precision behavior near the seam needs separate verification.

**Other instances (predicted)**:
- ReLU: `out = Maxs(x, 0)` — already this form, trivially
- LeakyReLU: `out = Maxs(x, 0) + alpha * Mins(x, 0)` — direct application
- Softplus: branchless via `log(1 + exp(-|x|)) + Maxs(x, 0)` (different rewrite — Maxs handles the linear tail, log1pexp handles the curved tail)
- SELU: `out = scale * (Maxs(x, 0) + alpha * (exp(Mins(x, 0)) - 1))` — same pattern as ELU with `input_scale=1`
- HardSwish below threshold: piecewise-clamp branches that reduce to 0 outside active range
- Any future elewise activation whose upstream `op_kernel/arch35/<op>.h` uses `ElementwiseSch<<Op>Dag>` and whose `<Op>Dag::Compute` uses a `Select`/`Mask`-based branch — applying this rewrite gives the verify-artifact a SIMD-only implementation

**Cross-reference**:
- OL-164 — the dual-output rule that motivates this rewrite (verify-artifact requires bare SIMD primitives, not MicroAPI Select)
- OL-63 — TQue depth=4 for elementwise (composed alongside this pattern in the verify kernel template)
- OL-81 — CAST_RINT for narrow-dtype cast-back at chain end (composed for half/bf16 paths)
- OL-82 / Iron law §5 — literal-first VEC ordering (preserved across the rewrite; no fusion, no strength reduction)
- P-P96 — the L2 `__VEC_SCOPE__` MicroAPI form (the form being avoided in the verify-artifact)
- `output/npukernelbench/src/kernels/1_GELU/` — companion verify-artifact template that this pattern slots into

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P97，convert_patterns_to_okf.py）。confidence 未升格。 -->
