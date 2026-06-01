# Precision and Cast Constraints

Read this file when a kernel changes dtype across cube, vec, micro, or output stages.

## Goal

Make dtype boundaries explicit enough that:
- the kernel matches the exact PyTorch contract
- the right stage owns the cast
- reductions happen in the intended dtype
- later comparisons use the correct tolerance

## 1. Start from the exact cast order

Write the full PyTorch formula first.
Keep cast order exact.
A mathematically equivalent formula with a different cast placement is often not equivalent here.

If the reference is ambiguous, fix that first.
Do not start moving casts around in DSL code blindly.

## 2. Repository default

Keep matmul accumulation in `float`.
Downcast later unless the design has a strong reason not to.

This is the default expectation for mixed-precision kernels in this repository.

## 3. Usual cast boundary ownership

Common ownership pattern:
- source tensors keep their authoring dtype
- cube matmul accumulates in `float` inside `L0C`
- vec or micro stage performs later downcast or quantization
- final output store uses the required target dtype

Typical examples:
- float -> half downcast during vec postprocess
- float -> fp8 cast in a later vec or micro stage
- half staging before a later cube consume path when the formula requires it
- `uint8` compare flags steering float quantization branches through `select(...)`, not through a later `uint8 -> float` cast

## 4. Do not cast too early

Common failure modes:
- downcasting before a reduction that should remain in `float`
- packing fp8 too early and then expecting a float-like downstream contract
- comparing against a reference whose cast order is different from the kernel

When in doubt, preserve higher precision until the stage boundary that really needs the cast.

## 5. Finite sentinels instead of literal infinities

When a fill value is only meant to stand in for `-inf` or `+inf` semantics:
- do not fill with literal `float("inf")` or `float("-inf")`
- fill with a sufficiently large finite positive/negative sentinel instead
- choose the magnitude from the real numeric range and dtype path of that stage

Keep the semantic rule in mind:
- the sentinel must stay beyond every valid value that the later max/min/reduction logic should ever see
- document the math as "`-inf` / `+inf` behavior" if that is the intent, but keep the concrete fill finite

## 6. Special repository patterns

Stable patterns already present in this repository:
- float-output baseline matmul
- vec-side float -> half assignment for postprocess output
- fp8 output conversion with optional `pack4()` fallback
- micro fp8 cast paths that require `pack4()` before UB writeback
- scalar-broadcast multiply that reads `[*,1]` inputs via `single()` without host-side expand
- for native `e4m3` cube matmul on default L0A/L0B DBuffs, reinterpret the L0 slot to `DT.e4m3` immediately before `mmad` instead of adding a vec-side cast to half

Scalar-broadcast multiply pattern:
- keep source shape as `[*,1]` (no host-side expand)
- load scalar lane into float UB (`[*,8]` padded), then read by `single()`
- multiply full row `RegList(float)` and cast on store if destination is half
- if a previous stage must round the scalar source to half before later consumption, materialize that half tensor first, then widen to float scalars for the final multiply

Repeat-along-new-axis pattern:
- preserve the host contract with `reshape` only
- flatten source/output so the repeated axis becomes a stride in the destination row index
- tile the contiguous inner rows (`L`) into UB once, then write that same UB tile to each repeated slice (`C`)

Alignment rule for half reads:
- for large non-contiguous half reads over `K`, `TILE_K=256` (512B) is robust

## 7. Layout-sensitive precision rules

Precision is often tied to layout and staging decisions.
Examples:
- keep reshape-only contracts when a scalar or gathered value must preserve exact semantics
- do not republish freshly packed fp8 UB data to a later stage if the downstream consumer expects ND semantics instead of the packed view
- on a2, treat `uint8` compare outputs as control values only; if float math depends on them, choose the float branches with `select(...)`

## 8. Validation rule

When testing a precision-sensitive kernel:
- compare against the exact reference formula
- use tolerances appropriate for the final dtype
- test at least one case where the cast boundary actually matters

A passing aligned float case is not enough to validate a mixed-precision design.

## Files to study

- `agent/example/kernels/a5/matmul_float_mmad.py`
- `agent/example/kernels/a5/matmul_half_splitn_bias10p2_vf.py`
- `agent/example/kernels/a5/matmul_kmkn_blockwise_quant128.py`
- `agent/example/kernels/a5/micro_cast_fp8_pack4_dual.py`
- `agent/example/kernels/a5/recompute_wu_cube_vec.py`
- `agent/example/kernels/a5/test_mla_entire.py`
