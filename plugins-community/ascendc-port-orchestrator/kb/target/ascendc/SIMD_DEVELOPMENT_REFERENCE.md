# AscendC SIMD Development Reference

> Compiled from CANN source code, A3 MXFP4 human implementation, and official docs.
> Focus: how to do **bit-level manipulation in SIMD mode** without scalar fallback.
> Last updated: 2026-06-23 (added §0 arch35 regbase-default)

## 0. arch35 (A5) SIMD: PREFER Regbase over Membase (DEFAULT for compute chains)

**On A5 / arch35 (Ascend950PR), a SIMD compute CHAIN defaults to REGBASE, not Membase/LocalTensor.**
This is the FIRST decision for any A5 vector op — make it before writing any `Add`/`Mul`/`Cast`
LocalTensor calls.

### Why: Membase round-trips every intermediate through UB

- **Membase** (the LocalTensor `Add(dst,a,b,count)` / `Mul(...)` / `Cast(...)` style): every vector
  op READS its operands from UB, WRITES the intermediate result back to UB, and the NEXT op RELOADS
  that intermediate from UB. Each op-to-op handoff is a redundant UB store + reload, **plus a
  `PipeBarrier<PIPE_V>` per dependent step**. For an N-op compute chain this is ~N redundant UB
  round-trips and N barriers — pure overhead that scales with chain length.
- **Regbase** (MicroAPI): load UB→`RegTensor` ONCE at the head of the chain, keep the WHOLE compute
  chain in registers (`Mul`/`Add`/... operate register→register), store back to UB ONCE at the tail.
  The intermediate round-trips and their per-step barriers disappear.

A SIMD op that is 100% Membase on a multi-op chain is a **major vector-utilization-gap source**:
selective_scan fwd+bwd were 100% Membase → ~15× redundant vector work, measured ~33× off the
roofline floor (2490µs vs ~75µs). See OL-245.

### The MicroAPI shape (regbase)

Wrap the chain in `__VEC_SCOPE__`, declare `RegTensor<float>` for each operand/intermediate, `LoadAlign`
UB→reg at the head, compute reg→reg, `StoreAlign` reg→UB at the tail:

```cpp
__VEC_SCOPE__
{
  AscendC::MicroAPI::MaskReg maskReg =
      AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
  AscendC::MicroAPI::RegTensor<float> vreg0, vreg1, vreg2, outreg0;

  for (uint16_t i = 0; i < repeatTimes; i++) {
    AscendC::MicroAPI::AddrReg addr = AscendC::MicroAPI::CreateAddrReg<float>(/*...*/);
    // load UB -> RegTensor (ONCE per operand, head of chain)
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0, srcAUbAddr, addr);
    AscendC::MicroAPI::LoadAlign<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg1, srcBUbAddr, addr);
    // compute reg -> reg (NO UB round-trip, NO per-op PipeBarrier)
    AscendC::MicroAPI::Mul(vreg2, vreg0, vreg1, maskReg);
    AscendC::MicroAPI::Add(outreg0, vreg2, vreg1, maskReg);
    // store RegTensor -> UB (ONCE, tail of chain)
    AscendC::MicroAPI::StoreAlign<float, AscendC::MicroAPI::StoreDist::DIST_NORM_B32>(dstUbAddr, outreg0, addr, maskReg);
  }
}
```

(Full worked example — fp32/fp16/bf16 AttnCompute Membase-vs-Regbase side-by-side —
in `workspace/ss_perf_loop/expert_regbase_input.txt`, expert input 2026-06-23.)

### When Membase is still fine

Regbase is the default for **compute chains** (≥2 dependent vector ops). Membase/LocalTensor is
acceptable only for the trivial cases where there is no chain to keep in registers:
- a single elementwise op (one `Add`/`Muls`/`Cast` with no dependent successor), or
- bitwise / type-punning ops (§1–§2 below) where the LocalTensor `And`/`Ors`/`ShiftRight` high-level
  API is the documented A5 path.

If your kernel has Add→Mul→Cast→Add style dependent steps, that IS a chain — use regbase.

### Full regbase guides (do not duplicate — cross-reference)

- `migration/reg-base-vector/Reg矢量计算编程.md` — the canonical MicroAPI / RegTensor programming guide
- `migration/l2-register-based-guide.md` — L2 register-based implementation guide
- `migration/l5-register-based-guide.md` — L5 register-based implementation guide

These cover `LoadAlign`/`StoreAlign` variants, `AddrReg`/`CreateAddrReg`, mask patterns, `LoadDist`/
`StoreDist`, and the load-distribution (broadcast / normal) modes in full.

## 1. Core Insight: SIMD Can Do Bitwise Ops on Float Tensors

AscendC SIMD is NOT limited to "same operation per element" float math.
It supports **bitwise integer operations on float tensors** via type-punning:

```cpp
// Cast float tensor pointer to uint16_t* for bitwise ops
vand((__ubuf__ uint16_t*)dst.ptr(), (__ubuf__ uint16_t*)mask.ptr(),
     (__ubuf__ uint16_t*)src.ptr(), repeat, ...);
```

This enables IEEE 754 bit extraction (exponent, mantissa, sign) in full SIMD parallelism.

## 2. Key SIMD APIs for Quantization Kernels

### 2.1 Bitwise Operations

**Platform difference**:
- **A3 (910C)**: Uses low-level `vand`/`vbor` with `__ubuf__ uint16_t*` cast
- **A5 (950PR)**: Uses high-level `And`/`Ands`/`Or`/`Ors` on `LocalTensor<int32_t>`

**Ascend950PR approach** (high-level API):
```cpp
// Step 1: ReinterpretCast float → int32 for bit access
LocalTensor<int32_t> x_int = x_float.ReinterpretCast<int32_t>();

// Step 2: Bitwise AND with scalar mask — extract IEEE exponent
// Ands(dst, src, scalar, count) — tensor AND scalar
Ands(exp_int, x_int, static_cast<int32_t>(0x7F800000), count);

// Step 3: ShiftRight to get exponent value
// ShiftRight(dst, src, shift_amount, count)
ShiftRight(exp_int, exp_int, static_cast<int32_t>(23), count);

// Step 4: ReinterpretCast int32 → float if needed
LocalTensor<float> exp_float = exp_int.ReinterpretCast<float>();
```

**Also available**: `And(dst, src0, src1, count)` for tensor-tensor AND, `ShiftLeft`, `Or`, `Ors`.

**Type restrictions (verified from official docs)**:
| API | int32 support | Workaround |
|-----|:------------:|------------|
| `And`/`Or`/`Not` | ❌ int16 only | ReinterpretCast<int16_t>, count×2 |
| `ShiftRight`/`ShiftLeft` | ✅ | Native int32 support |
| `Compare` | ⚠️ EQ only | Cast to float for LT/GT/GE/LE |
| `Select` | ❌ float/half only | Work in float domain |
| `ReduceMax` | ❌ float/half only | Needs sharedTmpBuffer param |
| `Duplicate` | ✅ | Native int32 support |
| `Add`/`Sub`/`Mul`/`Adds`/`Muls` | ✅ | Native int32 support |
| `Cast(float↔int32)` | ✅ | FLOOR/CEIL/TRUNC/RINT modes |

**A3 approach** (low-level intrinsics, NOT available on 950PR):
```cpp
// ❌ DOES NOT COMPILE on 950PR:
vand((__ubuf__ uint16_t*)dst, (__ubuf__ uint16_t*)mask, (__ubuf__ uint16_t*)src, ...);
```

### 2.2 Vector Reduce-Max

```cpp
// Find max across N elements (N/64 repeats for 64-element blocks)
vcmax(maxbuf.ptr(), srcbuf.ptr(), N/64, 1, 1, 8, ONLY_VALUE);
pipe_barrier(PIPE_V);
// maxbuf[0] now contains the max value
```

### 2.3 Scalar Broadcast (vbrcb)

Two-stage broadcast pattern to replicate a scalar across all elements:

```cpp
// Stage 1: scalar → 64 elements
vbrcb((__ubuf__ uint32_t*)tmp64.ptr(),
      (__ubuf__ uint32_t*)maxbuf.ptr(),
      1, 8, N/64/8);
pipe_barrier(PIPE_V);

// Stage 2: 64 elements → N elements
vbrcb((__ubuf__ uint32_t*)fullbuf.ptr(),
      (__ubuf__ uint32_t*)tmp64.ptr(),
      1, 8, N/64);
```

### 2.4 Rounding Modes (vconv_f322f32*)

```cpp
vconv_f322f32a(dst, src, N/64, ...);  // Round to nearest even
vconv_f322f32f(dst, src, N/64, ...);  // Floor
vconv_f322f32c(dst, src, N/64, ...);  // Ceil
```

These are **vectorized floor/ceil/round** — exactly what's needed for quantization.

### 2.5 Standard Arithmetic (vectorized)

```cpp
vmuls(dst, src, scalar, N/64, ...);  // dst[i] = src[i] * scalar
vadds(dst, src, scalar, N/64, ...);  // dst[i] = src[i] + scalar
vdiv(dst, a, b, N/64, ...);          // dst[i] = a[i] / b[i]
vadd(dst, a, b, N/64, ...);          // dst[i] = a[i] + b[i]
vsub(dst, a, b, N/64, ...);          // dst[i] = a[i] - b[i]
vmul(dst, a, b, N/64, ...);          // dst[i] = a[i] * b[i]
vabs(dst, src, N/64, ...);           // dst[i] = |src[i]|
vmin(dst, a, b, N/64, ...);          // dst[i] = min(a[i], b[i])
vmax(dst, a, b, N/64, ...);          // dst[i] = max(a[i], b[i])
```

### 2.6 UB-to-UB Copy

```cpp
copy_ubuf_to_ubuf(dst, src, 0, repeats, stride, src_stride, dst_stride);
```

Duplicates/reshapes data within UB without going through GM.

## 3. ⚠️ CRITICAL: SIMD Precision vs Performance Trade-off

**Rule**: SIMD tile-wide operations that skip per-group processing will break precision for group-local algorithms.

**Case study (MXFP4, 2026-04-07)**:
- MXFP4 spec: shared exponent per **32 elements** (group_size=32)
- A3 hand-written SIMD: shared exponent per **512 elements** (BATCH=512) → **precision bug**
- Our SIMD V4 "fast": shared exponent per **1024 elements** → 1.08x faster than SIMT on small tensors, **but precision doesn't match spec**
- Our SIMD V3 (correct per-group): matches PyTorch spec exactly, **but 6x slower than SIMT**

**The trade-off is fundamental**:
- Full SIMD vectorization requires processing all elements with the same operation
- Per-group algorithms need a serial loop over groups → destroys SIMD parallelism
- Skipping the per-group loop = wrong precision
- **There is no free lunch**: group-local algorithms with small group_size favor SIMT

**When SIMD wins despite per-group**:
- group_size >= tile_size (every element in the tile shares the same parameters)
- Per-group parameters can be pre-computed and vectorized (e.g., broadcast + Select)
- MTE2 pipeline benefit outweighs per-group loop overhead (not the case for MXFP4)

## 3b. MXFP4 Quantization in SIMD: Proven Pattern (from A3)

The human-written A3 implementation shows how to do **full MXFP4 quantization in SIMD** without any scalar GetValue/SetValue:

```
Algorithm (all vectorized, N elements):
1. vabs → |x|                                    (VEC, N elements)
2. vand(|x|, 0x7F800000) → exponent bits         (VEC, bitwise on float)
3. vcmax → max exponent (scalar)                  (VEC, reduce)
4. vbrcb → broadcast max to all elements          (VEC, broadcast)
5. vdiv(x, shared_exp) → normalized mantissa      (VEC, per-element)
6. vmuls(exp, 1.5) → upper clip bound             (VEC, scalar multiply)
7. vmin/vmax → clip mantissa                      (VEC, clamp)
8. vconv_f322f32a → round to nearest              (VEC, rounding)
9. vmul(rounded_mant, shared_exp) → result        (VEC, per-element)
```

**Zero scalar operations in the hot loop.** All per-element work is SIMD vectorized.

## 4. Shift-via-Multiply Pattern

AscendC SIMD doesn't have ShiftLeft/ShiftRight for floats. Instead, use multiply:
- Left shift by N bits: `vmuls(x, 2^N)` e.g., `vmuls(x, 4.0)` = shift left 2
- Right shift by N bits: `vmuls(x, 2^(-N))` e.g., `vmuls(x, 0.25)` = shift right 2

For **per-element variable shifts**: divide by the per-element exponent value (which is already a power of 2):
```cpp
vdiv(mantissa, input, per_element_exp, N/64, ...);
```

## 5. Buffer Strategy for Quantization SIMD Kernels

### Required Buffers (from A3 implementation):
| Buffer | Size | Purpose |
|--------|------|---------|
| input (double-buffered) | 2 × BATCH × sizeof(T) | MTE2/MTE3 overlap |
| output (double-buffered) | 2 × BATCH × sizeof(T) | MTE2/MTE3 overlap |
| absbuf | BATCH × 4 | |x| values |
| expbuf | BATCH × 4 | Extracted exponents |
| expmask | 64 × 4 | 0x7F800000 constant |
| expmaxbuf | 64 × 4 | Reduced max |
| grpexpbuf | BATCH × 4 | Broadcasted max |
| privexpbuf | BATCH × 4 | Per-element clamped exp |
| upperbuf | BATCH × 4 | Upper clip bound |
| lowerbuf | BATCH × 4 | Lower clip bound |
| mantissa | BATCH × 4 | Normalized mantissa |

### UB Budget
Total ≈ 10 × BATCH × 4 bytes. For BATCH=512: ~20KB. UB=192KB → plenty of room.

### Pipeline Synchronization
Use `DEvent<PIPE_MTE2, PIPE_V>` and `DEvent<PIPE_V, PIPE_MTE3>` for double-buffered event-based sync (finer than TQue, more manual than PipeBarrier).

## 6. Common Mistakes (from A3 precision bug)

### Bug: Incorrect exponent scaling
The A3 implementation had `vmuls(privexpbuf, grpexpbuf, 0.25)` which reduced the shared exponent by 4x, causing systematic underflow. The correct scale should match the MXFP4 format's dynamic range.

**Rule**: After extracting the shared exponent via vand + vcmax, do NOT multiply by fractional constants unless the math requires it. The exponent IS the power-of-2 scale.

### Precision validation
Always validate against the audited CPU PyTorch specification (OL-28). A device-to-device comparison cannot replace CPU truth.

## 7. API Naming: Low-Level vs High-Level

The A3 implementation uses **low-level intrinsics** (`vabs`, `vand`, `vcmax`, `vmuls`, etc.). The high-level AscendC API (`Abs`, `Muls`, `Add`, etc.) wraps these with LocalTensor type safety.

| Low-level (A3 style) | High-level (AscendC class API) | Notes |
|----------------------|-------------------------------|-------|
| `vabs(dst, src, repeat, ...)` | `Abs(dstTensor, srcTensor, count)` | Same HW instruction |
| `vmuls(dst, src, scalar, repeat, ...)` | `Muls(dstTensor, srcTensor, scalar, count)` | Same HW instruction |
| `vand(uint16_t* dst, uint16_t* mask, uint16_t* src, ...)` | **No direct high-level equivalent** | Must use low-level or ReinterpretCast + And |
| `vcmax(dst, src, repeat, ...)` | **ReduceMax** (API varies by CANN version) | Check API compatibility |
| `vbrcb(dst, src, ...)` | **No direct equivalent** | Use Duplicate + manual broadcast |
| `vconv_f322f32a(dst, src, ...)` | **Cast with RoundMode** | High-level may have different signature |

**Recommendation for Ascend950PR**: Try high-level API first. Fall back to low-level intrinsics (`vabs`, `vand`, etc.) if high-level doesn't support the operation.

## 8. Production Usage Statistics (from CANN source, 2026-04-07)

| Operation | Count | Primary Use Case |
|-----------|:-----:|-----------------|
| ReinterpretCast<int32_t/int16_t> | 50+ | Quant/dequant bit access |
| And (bitwise) | 20+ | Mask extraction (e.g., 0x0F0F for int4) |
| Or (bitwise) | 8+ | Bit reconstruction |
| Xor (bitwise) | 10+ | Hashing, sign manipulation |
| ShiftRight | 30+ | Field extraction, division by 2^n |
| ShiftLeft | 15+ | Field packing, multiplication by 2^n |
| Compare | 40+ | NaN detection, boundary check |
| Select | 25+ | Conditional masking |

Key production code references:
- **A8W4 dequant**: `cann/ops-transformer/gmm/grouped_matmul/op_kernel/*_antiquant_a8w4_*.h`
- **MX-format quant**: `cann/ops-transformer/gmm/*/arch35/weight_quant_basic_block/basic_block_vf_mx.h`
- **MOE sort**: `cann/ops-transformer/moe/3rd/moe_inplace_index_add/op_kernel/arch35/indices_sort_utils.h`

## 9. Platform Differences (A3 vs A5)

| Feature | A3 (910C) | A5 (950PR) |
|---------|-----------|------------|
| Low-level `vand`/`vbor` | ✅ | ❌ (target feature unsupported) |
| High-level `And`/`Ands`/`Or`/`Ors` | ✅ | ✅ |
| `ShiftLeft`/`ShiftRight` | ✅ | ✅ |
| `Compare` + `Select` | ✅ | ✅ |
| `vcmax` (low-level reduce) | ✅ | ❌ |
| `ReduceMax` (high-level) | ✅ | ✅ |
| `vbrcb` (low-level broadcast) | ✅ | ❌ |
| `Duplicate` (high-level broadcast) | ✅ | ✅ |
| `vconv_f322f32a` (round) | ✅ | ❌ (use Cast + RoundMode) |
| `Cast(int32, float, FLOOR)` | ✅ | ✅ |
| MicroAPI (register-level) | ✅ (arch35) | needs verification |

**Rule**: Always use high-level API on 950PR. Low-level intrinsics from A3 code will NOT compile.

## 10. Documentation Resources

- **A3 MXFP4 reference** (precision bug, but correct SIMD pattern): `~/workspace/temp/quant_packages/quant_compute_with_issue_only_for_reference/quant_cy_npu/`
- **CANN source**: `~/workspace/cann/` (git fetch first)
- **Official AscendC docs**: https://www.hiascend.com (JS-rendered, use dev-browser)
- **This file**: `src/skills/references/target/ascendc/SIMD_DEVELOPMENT_REFERENCE.md`
