---
applies_to: soc=all
reason: Type conversion / fp16-bf16 rounding behavior is universal. NOTE: FP16/BF16 atomicAdd availability is A5-specific (TBD on a3/a2) — patterns referring to atomicAdd FP16 carry their own per-pattern chip_scope.
---

# Domain: Precision & Functional Correctness
> Patterns for dtype handling, bf16/fp16 precision, type conversion, and functional correctness.
> Load when: Analyzer detects bf16/fp16 dtype, type casting, or dtype dispatch logic.

---

## Patterns

### F-P1: bf16 precision handling (scatter-add waiver)

**Severity**: Medium | **Mode**: SIMT/SIMD

**Anti-pattern**: using fp32 atol/rtol for bf16 tests → many false-positive FAILs

**Correct pattern**:
```cpp
float atol = (dtype == "bf16") ? 2e-2f : 1e-4f;
bool waiver = (dtype == "bf16");
compare_data(npu, cpu_truth, n, dtype_str, atol, rtol, "fwd", waiver);
```

**Note**: bf16 mismatch is expected behavior **only** for scatter-add-class ops (Pooling fwd/bwd). **SG forward is deterministic computation — bf16 must not have any mismatch**; if it does, it is a bug.

---

### F-P2: Multi-dtype support architecture

**Severity**: Low | **Mode**: SIMT/SIMD

Templated kernel + per-type dispatcher:
```cpp
template <typename T>
__simt_vf__ __aicore__ LAUNCH_BOUND(N) inline void kernel_vf(...) { ... }

extern "C" __global__ __aicore__ void kernel_fp32(...) {
    Simt::VF_CALL<kernel_vf<float>>(Simt::Dim3{threads}, ...);
}
// Same for fp16, bf16
```

---

### F-P3: SIMD bf16 mixed precision (MicroAPI)

**Severity**: Medium | **Mode**: SIMD | **Platform**: A5 (bisheng 15.0.5)

High-level `Cast()` does not support bf16↔float. Use register-level MicroAPI Cast:
```cpp
__VEC_SCOPE__ {
    RegTensor<bfloat16_t> vreg_bf16;
    RegTensor<float> vreg_f32;
    MaskReg preg;
    AscendC::MicroAPI::DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(vreg_bf16, ub_addr);
    AscendC::MicroAPI::Cast<float, bfloat16_t, castTrait>(vreg_f32, vreg_bf16, preg);
    // ... float computation ...
}
```

**Simplified alternative**: when precision permits, accumulate directly in bf16 (`Muls` + `Add` natively support bf16).

---

### F-P5: Warp-aligned loop bound (n_align_warp)

**Severity**: High | **Mode**: SIMT | **Source**: HKV hand-written version, verified

**This is a mandatory correctness requirement for cooperative-group programming, not an optional optimization.**

When the loop body contains `__shfl` / `__shfl_xor` / `ThreadBarrier`, the loop bound must be aligned to the group size:
```cpp
uint64_t n_align = ((n + GROUP_SIZE - 1) / GROUP_SIZE) * GROUP_SIZE;
for (uint64_t idx = ...; idx < n_align; idx += stride) {
    if (idx < n) { /* normal processing */ }
    else { result = ILLEGAL; }  // out-of-range thread marked invalid but still participates in __shfl
}
```

**Root cause**: `__shfl` requires all lanes in the group to execute the same instruction simultaneously. If some lanes exit the loop → deadlock.

---

## Anti-Patterns

### F-AP1: dtype string-match substring trap

**Severity**: **Critical**

```cpp
// "bfloat16".find("float16") == 1 → matches!
if (dtype.find("float16") != npos) { ... }      // BUG
else if (dtype.find("bfloat16") != npos) { ... } // never executes
```

**Fix**: the `bfloat16` check must come before `float16`.

---

### F-AP2: `__threadfence` misused as delay/wait

**Severity**: Medium | **Source**: HKV AI version, verified

`__threadfence()` is a memory barrier (ensures writes become visible to other threads), not a delay/wait. The correct approach is cooperative-group `__shfl` sync or a spin-wait.

---

## P-P50: bf16 intermediate-state precision matching (must-read for backward ops)

**Severity**: CRITICAL | **Source**: 29_TanhGatedResidualAddBackward (Level 2, 2026-04-15)

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

---

## P-P51: Native-dtype constraint for scalar-coefficient chains (must-read for fp16 normalization backward)

**Severity**: CRITICAL | **Source**: 14_AdaptiveInstanceNormalization2DBackward (Level 2, 2026-04-16)

**Trigger**: kernel obtains a scalar value from a reduction (e.g., `sum(x)`, `mean(x)`), then performs a chain of scalar arithmetic to compute "coefficients" (e.g., `grad_var = sum * (-0.5) * pow(std,-3)`), which are finally used for elementwise Muls/Adds in Pass 2.

**Problem**: If the reference performs this scalar arithmetic chain in the native dtype (fp16/bf16), each step has rounding loss and potential inf/NaN overflow. If the kernel keeps the reduction sum in fp32 for the scalar chain and only casts to half at the end for Pass 2, the rounding paths at each step differ → bit-level mismatch vs reference. **This difference broadcasts as a scalar across the entire tile, causing large numbers of element mismatches** (unlike sum-accumulation error which only affects one value).

**P-P50 vs P-P51 distinction**:
- P-P50: bf16 **tensor intermediate state** (`(a*b).to(fp32)` replicates the bf16→fp32 cast rounding)
- P-P51: fp16 **scalar coefficient chain** (`grad_var = sum * a * b` — the multi-step chain must stay in fp16)

**Fix template** (fp16 normalization backward):
```cpp
// Wrong: fp32 scalar chain with cast to half at the end
float acc_sum = 0.0f;
// ... ReduceSum accumulating into acc_sum (fp32) ...
float inv_std = 1.0f / std_f;
float m3 = inv_std * inv_std * inv_std;        // fp32 (no overflow)
float grad_var = acc_sum * (-0.5f) * m3;       // fp32
float coeff_xc = 2.0f * grad_var / spatial;    // fp32
half coeff_xc_h = static_cast<half>(coeff_xc); // final cast (wrong rounding path)

// Correct: fp32 accumulate → cast back to half → half chain compute
float acc_sum = 0.0f;
// ... ReduceSum accumulating into acc_sum (fp32) ...
half sum_h = static_cast<half>(acc_sum);       // matches fp16 output of .sum()
half std_h = static_cast<half>(std_f);
half inv_h = static_cast<half>(1.0f) / std_h;
half m3_h = inv_h * inv_h * inv_h;              // fp16 may be inf — matches reference
half grad_var_h = sum_h * static_cast<half>(-0.5f) * m3_h;
half coeff_xc_h = static_cast<half>(2.0f) * grad_var_h * static_cast<half>(inv_spatial);
```

**Key principle**: the dtype of `.sum()` output in PyTorch equals the input dtype (even though it accumulates in fp32 internally). The subsequent scalar chain operates entirely in fp16 in the reference, so the kernel must replicate this dtype boundary.

**Detection**: read the reference and find all scalar arithmetic chains (consecutive operations without `.to()`):
```python
grad_var = sum(...) * (-0.5) * torch.pow(std, -3)  # ← entire chain is fp16 (scalar dtype = input dtype)
grad_mean = sum(...) * (-1.0/std) + grad_var * (-2 * mean(...))  # ← entire chain is fp16 too
```
The corresponding scalars in the kernel must all be in half (or bf16).

**Applicability**:
- Normalization backward with fp16 / bf16 inputs (AdaIN, GroupNorm, LayerNorm backward)
- Reference does multi-step scalar arithmetic from a reduction result, then broadcasts back to tensor
- Ops sensitive to small std / small divisor (may overflow to inf)
- Pure fp32 input does not hit this (already fp32 natively)
- Single-step cast (without a multi-step scalar chain) does not hit this

**Related**:
- P-P50 (bf16 tensor intermediate state): sister pattern at tensor level
- OL-79 (NPU fp16 divide-by-zero = inf matches CPU): proves NPU fp16 hardware behavior matches PyTorch
- F-P1 (bf16 precision handling): overview

**Evidence**: 14_AdaIN2DBackward V7: fp32 scalar chain → 24/50 PASS (18 fp32 + 6 fp16/bf16).
Switching to half scalar chain → target 50/50 (pending verification). Failure symptom: fp16 cases produce `max_abs_diff=inf` mismatch (kernel outputs finite, reference outputs inf, or vice versa).

---

## P-P52: CANN op precision contract — bf16/fp16 reduction must use fp32 promotion

**Severity**: **CRITICAL** | **Source**: CANN `ops-math/math/reduce_sum/op_kernel/arch35/reduce_sum_dag.h` (2026-04-16)

**Trigger**: kernel needs to reduce a bf16/fp16 tensor (sum, mean, max, min, prod, etc.).

**Pattern**: every CANN reduction operator's DAG follows a **fixed 5-step structure**:
```
CopyIn<T>  →  Cast<fp32, T>  →  ReduceXxxOp<fp32>  →  Cast<T, fp32>  →  CopyOut<T>
```
i.e., **bf16/fp16 input must first be cast to fp32, reduction is performed in fp32, and the result is cast back to the original dtype**.

**Source evidence** (`reduce_sum_dag.h:28-38`):
```cpp
template <typename T, typename PromteT>  // T = input dtype, PromteT = fp32
struct ReduceSumDag {
    using Cast0 = Bind<Vec::Cast<PromteT, T, 0>, OpCopyIn0>;    // T → fp32
    using ReduceOp0 = Bind<Vec::ReduceSumOp<PromteT>, Cast0>;    // reduce in fp32
    using Cast1 = Bind<Vec::Cast<T, PromteT, 1>, ReduceOp0>;     // fp32 → T
};
```

**In our kernel writing style**:
```cpp
// bf16/fp16 input
Cast(fp32Buf, srcBf16, RoundMode::CAST_NONE, count);  // T → fp32 (lossless)
PipeBarrier<PIPE_V>();
ReduceSumP47(fp32Buf, count);  // or AscendC::ReduceSum<float, Pattern::Reduce::AR, true>
// result in fp32Buf[0]
// if you need to put the scalar result back into T, then Cast(..., CAST_RINT)
```

**Key**: Cast must use `CAST_RINT` (IEEE RNE), not `CAST_ROUND` (round half up) — see OL-81.

**Applicability**: all ops with bf16/fp16 input + internal reduction (LayerNorm, GroupNorm, AdaIN, Softmax, Sum, Mean, Var, etc., including backward)

---

## P-P53: reduce_mean internal Muls(1/N) — **performance pattern, not a precision requirement**

**Severity**: LOW (perf optimization, not required for bit-exact) | **Source**: CANN `reduce_mean_dag.h` + empirical 2026-04-16
**Major correction (2026-04-16 evening)**: minimal repro confirmed that `tensor / N` and `tensor * (1/N)` are **bit-identical** on NPU (both fp32 and bf16, across all N values). **The original P-P53 inference that "precision differs" is wrong**.

**Trigger**: kernel internally implements a `.mean()` aggregation (sum then divide by count to get the mean).

**Pattern**: CANN's `reduce_mean_dag.h`:
```cpp
CopyIn → Cast<fp32> → ReduceSumOp<fp32> → Muls<fp32>(1/N) → Cast<T> → CopyOut
//                                        ^^^^^^^^^^^^^^^
//                                   Muls with pre-computed reciprocal
```

**Correct interpretation**: CANN uses Muls(1/N) for **performance** (pre-divide once vs divide per element), not for precision — the two are bit-equal on NPU.

**Do NOT conclude that "Muls(1/N) can always replace Div(N)"**:
- In the CANN reduce_mean DAG scenario — "divide by N immediately after sum is produced" — the two are equivalent (bit-identical).
- But in complex chained-division scenarios like op #14, **forcing the substitution can introduce other bugs** (see lesson below).

**Counter-example (misuse)** (the `grad_mean / spatial_size` term in the grad_input formula of op #14):
```python
# reference (PyTorch):
grad_input = ... + grad_mean / spatial_size  # ordinary tensor / int → tensor Div
```
- kernel should use `Divs(grad_mean_tensor, spatial_float)` or `Duplicate(spatial)+Div` → tensor Div path
- should **not** be rewritten as `Muls(grad_mean_tensor, invSpatial)` — that is the internal form of reduce_mean
- Lesson (2026-04-16): generalizing `/spatial` to `*invSpatial` broke fp32-case precision.

**Example (correct use)** (if the kernel implements `.mean()` itself):
```cpp
// simulate tensor.mean() behavior:
ReduceSumP47(src, count);  // sum over all elements
// then:
Muls(src, src, invN, count);  // matches CANN reduce_mean DAG
// not:
Divs(src, src, float(count));  // not the internal form of CANN mean
```

**Detection**: ask yourself — is this division **part of an aggregation** (inside `.mean()`) or an **ordinary tensor/scalar operation**?
- inside aggregation → Muls(1/N)
- general division → Div / Divs (preserve tensor Div semantics)

**Related trap**: P-P55 (Pow) has a similar issue — do not generalize the internal implementation of Pow (`Exp(Ln(x)*y)`) to all pow scenarios; if you need `pow(x, 3)`, `x*x*x` may actually be closer to CANN's Pow implementation.

**Source code path** (`reduce_mean_dag.h:28-39`): `Vec::Muls<PromteT>(reciprocal_constant)`

---

## P-P54: Reciprocal via tensor Div(1, x), not scalar `1/x`

**Severity**: HIGH | **Source**: CANN `ops-math/math/reciprocal/op_kernel/arch35/reciprocal_dag.h`

**Trigger**: kernel needs to compute `1/x` or `c/x` (c is a scalar constant).

**Pattern**: CANN `aclnnReciprocal` does not use Newton-Raphson nor a hardware reciprocal instruction — it uses a direct tensor-level Div:
```cpp
// reciprocal_dag.h core loop
AscendC::MicroAPI::Duplicate(ones, (T)1.0, mask);       // build constant tensor of 1.0
AscendC::MicroAPI::Div<T>(vregOutput, ones, vregInput);  // elementwise 1 / x
```

**Requirements**:
- Do not write `half inv_x = static_cast<half>(1.0f) / x_h;` (scalar FPU path — not guaranteed bit-match with vector Div).
- Broadcast x to a tensor and use VEC `Div(ones_tensor, x_tensor)`.
- Or, if only a single scalar reciprocal is needed, Duplicate to a small tensor, Div, then GetValue.

**When it matters**: when the kernel's internal scalar chain passes through multiple reciprocal/div ops (e.g., `pow(std, -3) = 1/std * 1/std * 1/std`), scalar vs vector precision differences accumulate.

---

## P-P55: Use AscendC `Power` VEC API for Pow; do not decompose manually

**Severity**: HIGH | **Source**: CANN `ops-math/math/pow/op_kernel/arch35/pow_{bf16,f16,f32}_*.h`

**Trigger**: computing `pow(x, y)` or `pow(x, const_int)`.

**Pattern**: CANN has a dedicated `Power<T>` VEC template per dtype:
```cpp
// pow_bf16_nddma_without_loops.h
Power<bfloat16_t, false, pConfig_>(dstBuf, baseBuf, expBuf, count);
```
`Power`'s internal implementation (fused bf16/fp16/fp32 paths) differs completely from a manual `1/x * 1/x * 1/x`, with different rounding paths.

**Counter-example** (my kernel for `pow(std, -3)`):
```cpp
half inv_std = 1.0f / std;
half m3 = inv_std * inv_std * inv_std;  // WRONG: 3 muls, not bit-equal to Power(std, -3)
```

**Correct** (use the public AscendC VEC API):
```cpp
// broadcast std to a tensor
LocalTensor<half> stdT = ...; Duplicate(stdT, std_h, count);
LocalTensor<half> expT = ...; Duplicate(expT, static_cast<half>(-3.0), count);
Pow(resT, stdT, expT, count);  // OK: vector-level pow, matches CANN
// GetValue(resT, 0) if a scalar is needed
```

**When it matters**: whenever the reference contains `torch.pow(x, y)` or `x ** y` and bit-exact matching is required.

---

## P-P56: All "scalar arithmetic" in bf16/fp16 must go through the VEC path

**Severity**: HIGH | **Source**: combined P-P53/54/55 + NPU scalar unit vs vector unit precision difference

**Trigger**: kernel has `half/bf16` scalar arithmetic (not a tensor op).

**Pattern**: NPU scalar FPU and Vector FPU are **different hardware units**. The same `a / b` done via scalar code vs `Div(tensor)` through the VEC pipeline may **produce different bits**.

**What CANN does**: all bf16/fp16 arithmetic goes through the VEC pipeline (tensor op + Duplicate to broadcast scalars).

**Kernel writing advice**:
- Need scalar `r = a op b` where a/b are bf16/fp16? → either cast to fp32 and compute at scalar level (safe), or construct a 1-element tensor and use a VEC op
- Do not assume `half x = half_y op half_z;` bit-matches the vector version
- Especially for `/` and `pow`: **always** go through VEC

**Impact on op #14**: my kernel has lots of half/bf16 scalar arithmetic (`pow_neg3_h = inv_std_h * inv_std_h * inv_std_h`, `grad_var_h = sum * -0.5 * pow`). These are all scalar paths and may not match CANN's vector path.

**Fix template**:
```cpp
// Old: scalar path
half inv_std_h = static_cast<half>(1.0f) / std_h;  // scalar FPU

// New: VEC path (at least for precision-sensitive scalars)
LocalTensor<half> smallBuf = scratchBuf_.Get<half>();  // allocate a small scratch buf
Duplicate(smallBuf, std_h, 16);  // VL-aligned length
Duplicate(onesBuf, static_cast<half>(1.0f), 16);
Div(smallBuf, onesBuf, smallBuf, 16);  // vector Div
half inv_std_h = smallBuf.GetValue(0);  // take the first element
```
Cost: extra VEC scratch buf + sync. Benefit: matches CANN's precision path.

---


## P-P58.X: torch_npu fused-op spec gap — mode-flag dispatch may be no-op

**Severity**: HIGH | **Mode**: any | **Scenario**: porting a `torch_npu.<fused_op>` that advertises multiple "modes" via an integer attribute

**Criterion**: When the `torch_npu` op docstring describes mode-specific semantics (e.g. "mode=1 is variant with params X, Y, Z") but trusting the docstring leads to wrong-formula kernels that pass superficial QA on one mode and fail on the other. The underlying CANN fused op may **ignore** mode-specific parameters and dispatch identically across mode values.

**Detection heuristic**:
1. Implement the kernel per docstring.
2. If ≥50% of cases at one mode value fail while the other mode passes bit-exact, **do not assume the kernel formula is off by a small factor**.
3. Probe instead: test candidate formulas including "mode=1 dispatches identically to mode=0".
4. Small-shape RNG-replayed probes (1-2 rows, default + non-default attrs) × candidate formulas; compute per-row pre-quant amax rel_diff. Unambiguous winner shows ≤1e-7 rel_diff while others ≥1e-3.

**Evidence (op#11 DequantSwigluQuant, 2026-04-21)**:
- `torch_npu.npu_dequant_swiglu_quant` docstring advertises `swiglu_mode=1` as "variant with `clamp_limit`, `glu_alpha`, `glu_bias`".
- Initial kernel implemented mode=1 per docstring: `self·sigmoid(self·α)·(other+β)` → 25/50 PASS (mode=0 all PASS, mode=1 all FAIL with 28% mismatch, max_abs_diff up to 174).
- Precision-probe iter 4 ran 16-candidate formula sweep (K1-K16) per-row on 7 mode=1 cases covering default AND non-default `(clamp_limit, glu_alpha, glu_bias)` attribute combos:
  - K7 = `silu(self)*other` (== mode=0 formula): mean rel_amax_diff = 3.75e-9 (fp32 noise), 408/408 rows match
  - K1 docstring formula: mean rel_amax_diff = 7.81e-3 (orders of magnitude worse)
  - K8 variant with β inside sigmoid: mean = 8.67e-4 (still orders worse)
- Fix: replace mode=1 branch with mode=0 body → 50/50 bit-exact immediately.
- Root cause: underlying `aclnnDequantSwigluQuantV2` on CANN 9.0.0 does NOT branch on `swiglu_mode` for the activation step; `clamp_limit/glu_alpha/glu_bias` are received but unused.

**Mitigation**:
1. Implement the empirically correct formula.
2. Keep docstring-advertised parameters in kernel signature/fields for **ABI compatibility** (callers still pass them; kernel silently ignores).
3. Document the spec gap in the `else` branch inline comment with reference to the probe report.

**Anti-patterns to avoid**:
- ❌ Trusting torch_npu docstring without per-mode probe validation
- ❌ Writing complex "best-effort variant" kernel code for mode=1 when actual implementation is mode=0
- ❌ Claiming mismatch is "1-ULP boundary OL-83" without formula-sweep probe confirmation — OL-83 residuals are max_abs_diff=1 on int8, NOT max_abs_diff 30-174

**Cross-ref**:
- `output/npukernelbench/src/kernels/11_DequantSwigluQuant/probe_report.md` — full iter 1-4 trace
- `output/npukernelbench/src/kernels/11_DequantSwigluQuant/knowledge_update.md` — structured entry
- OL-83 (torch_npu 1-ULP drift) — distinct signature (≤ ±1 int8 diff; this P-P58.X has 30-174 int8 diff)

**Why this matters for future ports**: any CANN fused op with a mode/variant parameter should be treated as suspect until probed. Docstring semantics ≠ implementation semantics. This pattern will likely recur on other CANN fused ops where internal dispatch was consolidated across versions but docs were not updated.

---

## P-P59: Masked reduction with strict-`<` threshold — tied-threshold buffer truncation

**Severity**: CRITICAL | **Mode**: SIMD | **Scenario**: any op whose reference is of the form `mask = (v < threshold); x[mask] = fill; reduce(x)`, when the implementation uses a fixed-size top-K / top-N buffer

**General criterion**: if the reference logic contains

```python
threshold = ...                       # e.g. kth-largest, p-quantile, score cutoff
mask = values < threshold              # STRICT "<"
values = masked_fill(values, mask, fill_value)  # -inf / 0 / sentinel
reduced = some_reduction(values)       # softmax / sum / cumsum / norm
```

then ties at threshold will trigger this problem. Examples of specific ops (**not limited to**):
- Top-K + Top-P sampling (`torch.top_k_top_p`)
- Nucleus sampling / top-p sampling
- Attention tail-drop / sparse attention score threshold
- Sparse gather with score filter
- Quantile-based masking + subsequent normalization

**Abstract trap**: the implementation thinks "kept count = k" and so uses a buffer sized `TOPK_CAP = k_max`; actually kept count = `count(v ≥ threshold) ≥ k` (strict `<` keeps all v ≥ threshold, not strict `>`). Ties at the threshold push effective kept up to `k + (T - 1)` (T = number of ties at the threshold).

If `T > (TOPK_CAP - k)`, **some ties are in the row but outside the buffer**. Consequences:
- The denominator/sum inside the buffer is short by `(T_miss) × reduce_weight(threshold)`
- All normalized values inside the buffer are proportionally too large (if reduction = softmax) or too small (if it's a ratio)
- Downstream cumsum / norm flips kept↔dropped at the boundary rank

**Why bf16-large-N exposes this bug first**: bf16 mantissa 7 bits → in [1, 4) the value spacing is 2^-6 = 0.015625; for N=65536 `torch.randn`, statistically each bf16 bin contains ~50 ties. fp16 (10 bit) and fp32 (23 bit) have exponentially fewer ties.

**Anti-pattern**: implementing with a "fixed-N set" mindset (top-K buffer / top-N compact / preallocate k max), then reducing inside the buffer (softmax, normalization, cumsum). As long as the reference uses a strict-`<` threshold mask, the latent bug exists.

**Key semantics**: `< threshold` STRICT inequality → kept = `count(v ≥ threshold)`, **not** equal to the nominal `k`. If there are T ties at threshold, `effective_kept = k + (T - 1)` (the kth_value itself is already counted in k).

**Concrete example (excerpt from 9_TopKTopP reference implementation, for structural comparison)**:

```python
# Any "sort + strict-< mask + softmax/reduce" reference has this structure
logits_sort, logits_idx = logits.sort(descending=False, stable=True)
threshold = logits_sort[N - k]                 # can be kth, quantile, score cutoff, etc.
mask = logits_sort < threshold                  # STAR STRICT "<" — this keeps all tied values
logits_sort.masked_fill_(mask, fill_value)      # -inf / 0 / sentinel
reduced = some_reduction(logits_sort.to(fp32))  # softmax / sum / cumsum / norm
# then cumsum/normalize/threshold decision...
```

Counter-example (from 9_TopKTopP bf16 N=65536 case 8 row 497 — concrete data point proving the pattern):
- k = 993, threshold = 2.140625
- The row contains 48 values == 2.140625 (ties)
- ref kept (nonzero after mask) = 48 ties + all strict-above ≈ 1031 (including all tied)
- If the kernel uses a top-K buffer with `TOPK_CAP = 1024`, it can only capture 34/48 ties → 14 ties are in the row but outside the buffer
- Consequence A: kernel's denominator is missing 14 × reduce_weight(threshold) (here = exp(threshold - gmax)) → all normalized values inside the buffer are proportionally too large → cumsum globally shifted → boundary decision flips (case 8 row 497 rank 110: ref cumsum 0.8128 kept, kernel 0.8119 dropped, differing by 0.0009)
- Consequence B: for those 14 outside-buffer tied-threshold positions, the kernel writes sentinel by default but ref may write the real value (if the downstream cutoff lands in the middle of the tied block)

**Correct pattern (three fallback layers, increasing complexity)**:

### Layer 1 — Buffer enlarge + global denominator (simplest, sufficient for most scenarios)

- Enlarge `TOPK_CAP` to `k_max + max_expected_ties` (bf16 randn N=65536 empirical value: 50-100 ties; TOPK_CAP=2048 is usually enough)
- Second pass over the whole row to compute the **global** softmax denominator:
  ```cpp
  float exp_denom = 0.f;
  for each chunk in row:
      DataCopy(chunk); Cast fp32;
      mask = (v >= threshold) ? exp(v - gmax) : 0;   // Compare + Select + Exp
      exp_denom += ReduceSum(mask);
  ```
- Softmax normalize uses `exp_denom` (not the partial sum inside the top-K buffer)
- Keep the existing tail cumsum logic unchanged

**Closure effect**: all layer-A "rank flip" errors disappear. **Precondition**: actual ties do not exceed (TOPK_CAP - k). If they do, some tied-threshold columns are still outside the buffer (residual layer-B error).

### Layer 2 — Three-way classification + explicit tie-at-threshold handling (general bit-exact approach)

When the buffer upper bound of Layer 1 is not enough (adversarial inputs or tie count with no statistical bound), use per-column three-way classification instead of a fixed-size buffer:

1. **Phase 1 — find threshold**: use chunked merge / partial sort / selection algorithm to obtain `threshold`, `gmax` (or any global constants needed by the subsequent reduce).
2. **Phase 2 — classify full row**: second pass over the whole row, classify each column into three groups:
   - **v > threshold** → always kept, add to the "strict-above set" (record column index + accumulate reduce_weight)
   - **v == threshold** → conditionally kept, according to the reference's tie-break rule. General pattern: if the reference does `stable-sort(asc)` on the original data then reduce, within a tied cluster the smaller column index is "consumed" first (because asc-stable sorts small idx first, and subsequent cumsum/reduce accumulates in this order). The concrete cutoff position is decided by the reference's cutoff rule (top-p cumsum cutoff / quantile / score norm threshold, etc.) — implementation needs per-tied-block `sum_before_block` and per-tie `weight` to solve for the kept subset.
   - **v < threshold** → always dropped
3. **Phase 3 — Emit**: for kept positions, write the reduce result mapped back to native dtype; for the rest, write the mask sentinel.

**Closure effect**: bit-match ref output (if cumsum / reduction order matches).

### Layer 3 — Full-row sort (when k and ties are both large enough that Layer 1/2 is no cheaper)

If `k` or `effective_kept` is close to `N/2`, chunked merge is no longer cheaper than a full-row sort. Directly do full-row hardware Sort + full reduce, streaming by UB chunks. See P-P43 decision tree.

**P-P59 selection criteria (Layer 1 vs Layer 2)**:

| Condition | Choice |
|-----------|--------|
| `k_max + max_expected_ties ≤ UB_available / (per-entry-bytes)` and tie count has a statistical bound (non-adversarial) | **Layer 1** (buffer bump + global denom second pass) |
| The above does not hold, or inputs may be adversarial / tie count has no worst-case bound | **Layer 2** (three-way classification) |

**Anti-pattern details (do not do)**:
- FORBIDDEN: only change comparator order hoping to match tie convention — even with the right order it cannot fix denominator truncation
- FORBIDDEN: use a "convention waiver" to hide bf16 mismatch — forbidden by CLAUDE.md §No Workarounds
- FORBIDDEN: statically enlarge buffer to N — returns to the full-row UB upper-bound problem
- FORBIDDEN: assume "ties are few in fp32/fp16 so a small buffer is enough" — bf16 dtype will expose it first; fp16/fp32 are only temporarily invisible

**Evidence**:
- 9_TopKTopP V2→V3 (2026-04-17/18) hit this exactly. V1 full-row-sort exceeded UB → 34/50. V2 chunked top-K (`TOPK_CAP=1024`) → 45/50 — 5 bf16 N=65536 cases fail at 1-14 elements per case due to tie-at-threshold buffer truncation. V3 Layer 1 fix (TOPK_CAP 2048 + global denom second pass) → **50/50 PASS**. This is the canonical case for the pattern. Other candidate ops (unverified): nucleus sampling variants, attention tail-drop-by-threshold, sparse gather with score filter.
- **9_TopKTopP cold-run (2026-04-18 round 2)**: independent cold-run verifying the worker+probe pipeline; worker implemented Layer 1 to 29/50 stuck; probe found Layer 1 **necessary but not sufficient** — also requires pairing with P-P60 (AscendC Sort ASC tie-break reverse) fixing cutoff_orig_idx reselection, EC-31 (Select mask polarity), and EC-32 (effective_kept vs buffer_len). With those added, 49/50 (residual 1 case is OL-83 torch_npu drift, not a kernel bug). **Pattern extension**: the full implementation of the P-P59 schema needs the canonical implementation sketch; see the combination of P-P60 + EC-31 + EC-32 + OL-83 — all four together guarantee bit-match against the PyTorch stable-sort reference.

**Canonical implementation sketch (for P-P59 Layer 1 + P-P60 combination)**:
```
Phase 0: gmax = ReduceMax(row)
Phase 1: chunked top-K merge → top_val[TOPK_CAP], top_orig_idx[TOPK_CAP]
Phase 2: global softmax denom (scan whole row once, sum exp(v - gmax) for v >= threshold)
Phase 3: effective_kept = count(top_val[i] >= threshold)     # EC-32: not TOPK_CAP!
         ASC-sort top buffer by val (if using ASC walk)
         Cumsum walk ASC positions [topk_len-effective_kept .. topk_len-2]
         identify cutoff_val + n_drop_tied
         **post-walk re-select cutoff_orig_idx**:             # P-P60: critical!
           tied_idxs = [idx for idx in top_orig_idx if top_val == cutoff_val]
           sort tied_idxs ascending
           cutoff_orig_idx = tied_idxs[n_drop_tied - 1]
Phase 4: emit per-column: scalar SetValue(col, kept_val) if (v > cutoff) OR
         (v == cutoff AND orig_idx > cutoff_orig_idx)         # EC-31: use scalar
         (prefer scalar SetValue over VEC Select to avoid mask polarity bugs)
```

**Branchless merge optimization (2026-04-19, new)**:

Phase 1's 2-way merge is conditional by default (has an `if va >= vb` branch). To eliminate the scalar bottleneck, R3b optimizer adopted a branchless merge: with sentinel-padded inputs, each iter does 1 compare + 2 GetValue + 2 SetValue, no branch.

**Hard preconditions**:
1. The two merge inputs (top buffer and new chunk sortValOut) must have at least `TOPK_CAP` slots
2. All unused slots must be pre-filled with the -inf sentinel
3. **`CHUNK >= TOPK_CAP` must be guaranteed by `static_assert`** — otherwise out-of-bounds read, precision blows up (see PB-14)
4. Copy-back stage: VEC `Adds<float>(top_val, merge_val, 0, TOPK_CAP)` for values (direct copy); idx copy depends on the CANN version:
   - **CANN 9.0.0 on Ascend950PR (as of 2026-04-19)**: `Adds<int32_t>` buffer-to-buffer has a corruption bug (PB-13), so idx copy-back must use a scalar loop
   - **Future CANN versions**: re-verify PB-13; if fixed, switch to VEC `Adds<int32_t>` for a unified path

**Edge cases (must handle, 2026-04-18 added from V3.2 test 2 Phase D iter 1 regression)**:
- **`effective_kept == 0`**: all v < threshold in the row → all positions -inf. Early exit.
- **`effective_kept == 1`**: only 1 kept element remains; reference `top_p_mask[:, -1] = False` forces the max to be kept.
  **Set `cutoff_orig_idx = -1` sentinel** so that the Phase 4 emit `(v == cutoff AND col > cutoff_orig_idx)` branch holds `col > -1` for that element, so it is not falsely dropped.
  **Anti-pattern**: `cutoff_orig_idx = topIdx[0]` (the kept element's own col index). Makes strict `col > cutoff_orig_idx` fail → that kept element is falsely dropped → precision failing_cases (V3.2 test 2 iter 1: 7/50 cases each fail 1 element with max_abs_diff=3.4e38).
- **`effective_kept >= 2`**: standard cumsum walk + P-P60 post-walk re-select.

**Related**:
- P-P42 Hardware Sort pipeline (used in Phase 1 to find threshold)
- P-P43 Sort decision tree (when Layer 3 needed)
- P-P52 fp32 promotion (always for softmax)
- EC-28 fp32 -inf sentinel must be true IEEE -inf (precondition for correct mask output)
- EC-29 SortConfig device-side 2-field schema
