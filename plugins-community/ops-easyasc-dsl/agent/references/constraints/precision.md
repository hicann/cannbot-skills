# Precision and Cast Constraints

Read this file when a kernel changes dtype across cube, vec, micro, or output stages.

## 1. Start from the exact cast order

Write the full PyTorch formula first. Keep cast order exact.
A mathematically equivalent formula with a different cast placement is often not equivalent here.
If the reference is ambiguous, fix that first. Do not move casts around in DSL code blindly.

Preserve parenthesization around mixed-precision products too. For example,
`(key * beta) * exp(g)` and `key * (beta * exp(g))` are algebraically equivalent
over real numbers, but they can produce different rounded results once the
kernel or reference inserts bf16/fp32 boundaries. Match the reference operation
order before tuning the expression for fewer instructions.

## 2. Repository default: float accumulation

`mmad` requires `dst` at `L0C`. For float/half/bf16/fp8/hif8/MX paths, the
repository default is `DT.float` L0C accumulation; int8 and int4 paths are the
int32 exceptions. Downcast later unless the design has a strong reason not to.

Matmul BT/C2 bias (`matmul(..., bias=)`, `Position.BT`) follows the same split:
`int8 @ int8 -> int32` bias and dst; every other supported input dtype -> `float`
bias and dst (a5 additionally accepts fp8 `e4m3` / `e5m2` / hif8 -> float; a2
rejects fp8). Bias is fp32/int32 only and applies on the `is_init` tile only
(rejected on accumulate-only `mmad`).

Source: `easyasc/stub_functions/cube.py:1014` (`mmad` signature), `:1018`
(L0C position check)

Kernel examples:
- `agent/example/kernels/a5/matmul/matmul_float_mmad.py:8` — `l0c = Tensor(DT.float, [M, N], Position.L0C)`
- `agent/example/kernels/a5/matmul/matmul_half_splitn_bias10p2_vf.py:27` — half inputs, `DT.float` L0C, half output
- `agent/example/kernels/a5/matmul/matmul_rowwise_norm.py:35` — half inputs, `DT.float` L0C, float-normalized output

## 3. Cast boundary ownership

Pattern:
- source tensors keep their authoring dtype
- non-int cube matmul paths accumulate in `float` inside L0C
- vec or vf stage downcasts or normalizes
- output store uses the required target dtype

Examples:
- `agent/example/kernels/a5/matmul/matmul_half_splitn_bias10p2_vf.py:18` — vf reads float UB, casts to half on store
- `agent/example/kernels/a5/matmul/matmul_rowwise_norm.py:14` — normalize_rows_vf uses `RegList(DT.float)` throughout

## 4. Do not cast too early

Common failure modes:
- downcasting before a reduction that should remain in `float`
- packing fp8 before the downstream consumer expects float-like semantics
- comparing against a reference whose cast order differs from the kernel

When in doubt, preserve higher precision until the stage boundary that actually needs the cast.

When a value crosses a typed workspace or an intermediate UB cast before a later
cube stage, the PyTorch reference must model that boundary exactly. For example,
an A2 attention path that stores `qm @ k_smooth` through a `DT.half` GM
workspace should compare against `qm_ksmooth.to(torch.float16).float()`, not a
pure fp32 matmul. Likewise, DSL `cast(float -> half)` followed by
`cast(half -> int8)` uses the declared round modes; PyTorch
`.to(torch.int8)` truncates and is not a substitute for the default
away-from-zero int cast path.

## 5. fp8 dtype names

`DT.e4m3` maps to `"float8_e4m3_t"` and `DT.e5m2` maps to `"float8_e5m2_t"`.

Source: `easyasc/utils/datatype.py:119-120`

On A5, public micro cast pairs include `DT.float -> DT.e4m3`,
`DT.float -> DT.e5m2`, `DT.float -> DT.hif8`, `DT.half -> DT.hif8`,
`DT.e4m3 -> DT.float`, `DT.e5m2 -> DT.float`, `DT.hif8 -> DT.float`,
and `DT.hif8 -> DT.half`.
The wide-to-narrow `float -> fp8` path follows the ratio-4 micro register
layout rule, so it normally needs `pack4()` before UB writeback. The `fp8 ->
float` path decodes to float lanes and should not receive an extra
`float -> float` rounding pass in the simulator.

Source: `easyasc/stub_functions/cast_rules.py`, `easyasc/simulator/pipe_micro.py`

For native e4m3/e5m2 cube matmul, keep L1 operands in the FP8 dtype and let the
matmul shortcut reinterpret the L0 slots to matching FP8 views immediately before
`mmad`, instead of adding a vec-side cast. Source:
`easyasc/shortcuts/matmul.py:85-86`, `easyasc/stub_functions/cube.py:1036`.

For complete A5 public-carrier conversion chains, use the dataflow Pattern that
matches the ABI:

- native BF16 -> packed FP4 carrier:
  `agent/references/patterns/a5-fp4-cast-pack.md`;
- exact `{0,1,2,3}` BF16 <-> packed uint2 carrier:
  `agent/references/patterns/a5-uint2-pack-unpack.md`.

This file remains the owner of cast order and numeric precision boundaries;
the Pattern owns register placement, compaction, bit spreading, and UB reload
composition.

## 6. True hif8 carrier casts

Torch has no public hif8 dtype in the current workflow. For A5/CANNSIM, use
`DT.uint8` public GM tensors as carriers, copy them to UB, then reinterpret the
UB carrier as `DT.hif8` before micro casts.

Validated pattern:
- decode: `uint8 GM -> UB uint8 -> reinterpret(DT.hif8) -> MicroAPI::Cast<float, hifloat8_t>`
- encode: `float GM -> MicroAPI::Cast<hifloat8_t, float> -> pack4() -> UB uint8 carrier -> uint8 GM`
- half decode: `uint8 GM -> UB uint8 -> reinterpret(DT.hif8) -> MicroAPI::Cast<half, hifloat8_t>`
- half encode: `half GM -> MicroAPI::Cast<hifloat8_t, half> -> downsample() -> UB uint8 carrier -> uint8 GM`

Source: `agent/example/kernels/a5/utility/hif8_carrier_reinterpret_cast.py`.
VM-local CANNSIM passed float/half decode with `max_abs_diff=0` and
float/half encode with `mismatches=0`; `half -> hif8 CAST_HYBRID` was also
cross-checked over all 65536 half bit patterns with `mismatch_count=0`.
The host-side reference codec lives in `easyasc/dtypehelper/hif8_codec.py`
and is exposed through `easyasc/a5.py`.
`fp32_to_hif8(..., round_mode=RoundMode.HYBRID)` models the `CAST_HYBRID`
path: TA conversion for exponents with `abs(e) < 4`, SSR conversion elsewhere.
`fp16_to_hif8(..., round_mode=RoundMode.HYBRID)` uses the half-specific SSR
threshold observed in CANNSIM: the SSR threshold bit comes from the original
half mantissa LSB, not the normalized fraction LSB used by the draft script.
`float -> hif8` and `half -> hif8` support only `CAST_ROUND` and
`CAST_HYBRID`. `RoundMode.HYBRID` is valid only for those A5 hif8 encode
micro casts. `hif8 -> float` and `hif8 -> half` accept any round mode value,
but simulator execution ignores it because the cast is a hif8 decode into a
higher-precision representation.

For A5 native FP4/MX helper data, `easyasc/dtypehelper/fp4_fp32.py` exposes
`fp32 <-> uint8 carrier` helpers for `fp4_e2m1` and `fp4_e1m2`, and
`easyasc/dtypehelper/e8m0_fp32.py` exposes `fp32 <-> e8m0 byte` scale helpers.
These helpers are also exported by `easyasc/a5.py`.

Native MXFP4 is a carrier path, not a vec cast path: L1/L0 operands are
`DT.uint8` carrier tensors reinterpreted as `DT.fp4_e2m1` / `DT.fp4_e1m2` before
`l1_to_l0_mx` / `matmul_mx`. FP4 has `C0 == 64`; transposed MX loads
(`src.T` / NZ2ZN) therefore need 64-element physical source alignment. Sources:
`easyasc/shortcuts/matmul_mx.py:119-123`,
`easyasc/stub_functions/cube.py:230-244`, `easyasc/utils/datatype.py:33-34`.

## 7. Hif8 emulated rounding

The `to_hif8_torch_kernel` emulates rounding-away-from-zero:
1. derive scale from float exponent bits
2. add ±0.5 bias based on sign
3. truncate via `cast(roundint, …, round_mode=RoundMode.TRUNC)`
4. cast back to float and multiply by scale

Source: `agent/example/kernels/a2/vec_only/to_hif8_torch.py:104–106`

`uint8` compare flags (`finiteflag`, `keepflag`, …) steer float branches with `select(…)`,
not via a `uint8 → float` cast — `agent/example/kernels/a2/vec_only/to_hif8_torch.py:72`.

See `agent/references/constraints/vec.md` for `uint8` flag semantics on a2.

## 8. Finite sentinels

Fill with a large finite value, not literal `float("inf")`. The sentinel must exceed every
valid value the reduction can see. Example: `FLOAT32_FINITE_MAX` used as saturation —
`agent/example/kernels/a2/vec_only/to_hif8_torch.py:11`.

## 9. A2 non-finite branch classification

Do not rely on `compare(x, x, CompareMode.NE)` as an A2 NaN detector in public
vec kernels. A2 comparison uses unordered-false semantics in the current
simulator and tests, matching the observed 910B3 path: a comparison with a NaN
operand does not set the predicate, including `NaN != NaN`.

For mixed finite / overflow / NaN handling, derive the `nanflag` from packed
compare flags instead:

1. `finiteflag = (abs(x) <= FLOAT32_FINITE_MAX)`
2. `overflowflag = (abs(x) > FLOAT32_FINITE_MAX)`
3. reinterpret the `uint8` flag buffers to `uint16`
4. compute `nanflag = (~finiteflag) & (~overflowflag)` with `vnot` / `vand`

Source pattern: `agent/example/kernels/a2/vec_only/to_hif8_torch.py:123-129`.

## 10. Validation

Compare against the exact reference formula using tolerances appropriate for the final dtype.
Test at least one case where the cast boundary actually matters.
A passing aligned float case is not enough to validate a mixed-precision design.

Default kernel validation tolerances:
- low-precision output or mixed low-precision path: `rtol=2e-3, atol=2e-3`
- high-precision output path: `rtol=1e-4, atol=1e-4`

Do not use `rtol=0.0` or `atol=0.0` for kernel numerical validation.
Exact equality belongs only in focused non-numerical invariants where bitwise identity is the actual contract.

### A5 micro division modes

The default float `micro.div` lowers to A5's intrinsic vector division.  It is
fast, but a real A5 result can differ by one ULP from the correctly rounded
CPU division used by the simulator.  The simulator emits
`MicroDivPrecisionWarning` for this case instead of silently implying last-bit
agreement.  It omits the advisory for a normal-domain power-of-two divisor,
where the operation is only an exact exponent shift.

When the contract requires correctly rounded fp32 division, pass an explicit
`DivConfig`.  For normal operands/results, the board-validated fast exact mode
is:

```python
cfg = DivConfig(
    algo=DivAlgo.PRECISION_0ULP_FTZ_TRUE,
    precision_mode=True,
    name="exact_normal_div",
)
div(dst, numerator, denominator, config=cfg)
```

`PRECISION_0ULP_FTZ_TRUE` does not promise gradual-underflow behavior.  Use
`PRECISION_0ULP_FTZ_FALSE` when subnormal inputs or results are part of the
contract, and measure it on hardware: the C310 implementation has a much
larger normalization/special-value sequence.  The simulator's arithmetic
value model is correctly rounded CPU division; it is not a cycle model for
either precision path.

## 10b. A NaN guard that lets Inf through turns Inf into NaN

A compensated accumulator has to drop its residual when the residual is not
finite, or the correction poisons an already-correct Inf or NaN. The obvious
test is `x == x`, which rejects NaN - **and passes for `+-Inf`**. A sum that
*overflows* to Inf leaves an Inf residual, so `total - residual` is
`Inf - Inf = NaN` where the reference is `Inf`.

CANN Bench compares NaN and Inf *positions* before it looks at any relative
error, so this fails a case outright with every finite element exact
(`MERE=0.000000, MARE=0.000000 ... NaN位置不匹配`). It cost one hidden case on
`adaptive_avg_pool_3d`.

Scale by zero before the test - it maps every finite value to a zero and both
Inf and NaN to a NaN, so the same equality test then rejects all three:

```python
muls(probe, residual, 0.0)
compare(finite, probe, probe, CompareMode.EQ)
select(residual, residual, zero, mask=finite)
sub(total, total, residual)
```

`torch.randn` inputs never reach this, so it needs its own sweep - see
`easyasc_cannbench_kernels/kernels/a5/vec_only/cann_bench/adaptive_avg_pool_3d/check_nonfinite.py`,
which covers overflow, `+-Inf` samples and NaN samples on every route.

## 11. What the CANN Bench float checker actually measures

The bench does *not* pass or fail a float output on `MERE`/`MARE` alone. Its
`relative_error` checker splits every element into three classes and scores each
separately against the same computation run on **CPU**:

- `normal` - the bulk. `normal_error_count` has to be zero (the CPU count is too).
- `cancel` - elements whose window sum nearly cancels, so the exact result is
  many orders of magnitude smaller than the terms. Scored as
  `cancel_error_count` vs `cancel_cpu_error_count`.
- `small_value` - elements whose magnitude is near the dtype's floor.

A run can report `mare` an order of magnitude over `threshold` and still pass,
because the offenders all land in `cancel` and the rule there is **no worse than
the framework reference**, not an absolute bound. An a5 `adaptive_avg_pool_3d`
case measured `mare = 5.2e-4` against a `1.2e-4` threshold and passed with
`cancel_error_count = 59` against `cancel_cpu_error_count = 95`.

Two consequences when tuning a reduction:

- **Do not delete a compensated accumulator to save vector ops** without
  measuring the cancel class. On that same case a plain fp32 sum moved the count
  from ~59 to ~100 against a CPU reference of 95 - the mean error barely changed
  (1.0e-7 -> 2.2e-7, both three orders under the limit) while the pass/fail
  quantity crossed. Reordering does not help either: a blocked or pairwise sum
  scored bit-identically to the naive one, because cancellation is a
  *conditioning* problem, not a summation-length one.
- The cheap way to check this before touching the board is to replay the
  kernel's exact summation order on CPU in fp32 against an fp64 golden and count
  how many elements exceed the threshold, next to the same count for
  `torch`'s own fp32 result. That ratio is what the bench is going to compare.
