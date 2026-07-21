# Multi-Dtype Support Guide for AscendC Kernels

## Overview

AscendC operators targeting float32, float16, and bfloat16 use a single source file compiled
three times by the build system with `-DDTYPE_X1=<type>`. This guide documents the canonical
pattern, Cast mode rules, and UB budget for multi-dtype kernels on **dav_c220 (Ascend910b)**.

---

## Pattern: DTYPE_X1 Macro + `if constexpr`

### 1. Macro Guard (top of kernel file, before class)

The build system injects **`-DDTYPE_{INPUT_NAME_UPPER}=<type>`** based on the first input
tensor's name in the operator JSON.  This is `DTYPE_X1` **only** when the input is named
`x1`.  For any other name (e.g. `query`, `input`, `x`), you must add an alias **before**
the `#ifndef DTYPE_X1` guard, or the fallback will always use `float` silently.

```cpp
// ── Macro bridge ─────────────────────────────────────────────────────────
// The build system injects -DDTYPE_{INPUT_NAME_UPPER}=<type>.
// If the first input is named "x1"  → -DDTYPE_X1 is injected directly ✓
// If it is named "query"            → -DDTYPE_QUERY is injected; add alias ↓
// If it is named "input"            → -DDTYPE_INPUT is injected; add alias ↓
// Replace DTYPE_QUERY below with the actual upper-cased input name.
#ifndef DTYPE_X1
  #ifdef DTYPE_QUERY          // ← change to match your input name (e.g. DTYPE_INPUT)
    #define DTYPE_X1 DTYPE_QUERY
  #else
    #define DTYPE_X1 float    // fallback for IDE / linters when building without -D flag
  #endif
#endif
```

For the standard case where the input **is** named `x1`, the simpler guard is sufficient:

```cpp
// DTYPE_X1 is injected by the build system per-dtype compilation:
//   float       for float32
//   half        for float16
//   bfloat16_t  for bfloat16
#ifndef DTYPE_X1
#define DTYPE_X1 float   // fallback for IDE / linters
#endif
```

> **Silent-failure diagnostic**: if float16 and bfloat16 variants produce results identical
> to the float32 variant, the macro was never injected.  Verify with:
> ```bash
> xxd build_out/.../float16_variant.o > /tmp/f16.hex
> xxd build_out/.../float32_variant.o > /tmp/f32.hex
> diff /tmp/f16.hex /tmp/f32.hex | head -20
> # Empty diff → macro injection failed; check input name vs DTYPE_* guard
> ```

### 2. Class Member Layout

```cpp
class KernelOp {
private:
    AscendC::TPipe pipe;

    // I/O queues — typed as DTYPE_X1 for all three variants
    AscendC::TQue<AscendC::TPosition::VECIN,  1> x1Queue;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> yQueue;

    // Float32 intermediate compute buffers.
    // Declared unconditionally; pipe.InitBuffer is called ONLY for bfloat16_t variant.
    // Zero UB cost for float / half variants.
    AscendC::TBuf<AscendC::TPosition::VECCALC> x1FloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> yFloatBuf;

    // Global memory tensors in DTYPE_X1
    AscendC::GlobalTensor<DTYPE_X1> x1Gm;
    AscendC::GlobalTensor<DTYPE_X1> yGm;
};
```

> **Note**: Use `TBuf<VECCALC>` (not `TQue`) for float32 intermediates. `TBuf` has no
> queue overhead and supports re-use via `.Get<float>()` across iterations.

### 3. Init() — Conditional Buffer Allocation

```cpp
__aicore__ inline void Init(GM_ADDR x1, GM_ADDR y, uint32_t tileSize, ...)
{
    uint32_t qBytes = tileSize * sizeof(DTYPE_X1);
    pipe.InitBuffer(x1Queue, 1, qBytes);
    pipe.InitBuffer(yQueue,  1, qBytes);

    // Only allocate float32 TBufs for the bfloat16_t compilation variant
    if constexpr (std::is_same<DTYPE_X1, bfloat16_t>::value) {
        uint32_t fBytes = tileSize * sizeof(float);
        pipe.InitBuffer(x1FloatBuf, fBytes);
        pipe.InitBuffer(yFloatBuf,  fBytes);
    }
}
```

### 4. Compute() — `if constexpr` Branching

```cpp
__aicore__ inline void Compute(uint32_t idx, uint32_t cnt)
{
    AscendC::LocalTensor<DTYPE_X1> x1Local = x1Queue.DeQue<DTYPE_X1>();
    AscendC::LocalTensor<DTYPE_X1> yLocal  = yQueue.AllocTensor<DTYPE_X1>();

    if constexpr (std::is_same<DTYPE_X1, bfloat16_t>::value) {
        // ---- bfloat16_t path: upcast → float32 compute → downcast ----
        AscendC::LocalTensor<float> x1F = x1FloatBuf.Get<float>();
        AscendC::LocalTensor<float> yF  = yFloatBuf.Get<float>();

        // Cast in: bf16 → f32 (CAST_NONE is correct; vconv_bf162f32)
        AscendC::Cast(x1F, x1Local, AscendC::RoundMode::CAST_NONE, cnt);

        // ... compute in float32 (Muls, Add, etc.) ...

        // Cast out: f32 → bf16 — MUST use CAST_RINT, NOT CAST_NONE!
        // On dav_c220, CAST_NONE for f32→bf16 triggers ASCENDC_ASSERT(false)
        // and leaves the output buffer uninitialized (silent garbage output).
        AscendC::Cast(yLocal, yF, AscendC::RoundMode::CAST_RINT, cnt);
    } else {
        // ---- float / half path: compute directly in DTYPE_X1 ----
        // Muls and Add are supported for float and half on dav_c220
        // ...
    }

    yQueue.EnQue(yLocal);
    x1Queue.FreeTensor(x1Local);
}
```

---

## Cast Mode Rules (dav_c220 / Ascend910b)

| Conversion direction | Correct `RoundMode`  | What the intrinsic does                                      |
|----------------------|----------------------|--------------------------------------------------------------|
| `bf16 → f32`         | `CAST_NONE`          | `vconv_bf162f32` ✓ — exact, no rounding needed              |
| **`f32 → bf16`**     | **`CAST_RINT`**      | Round-to-nearest-even ✓ — matches PyTorch bfloat16 rounding  |
| `f16 → f32`          | `CAST_NONE`          | `vconv_f162f32` ✓                                           |
| `f32 → f16`          | `CAST_NONE`          | `vconv_f322f16` ✓                                           |
| `f32 → f32`          | — never cast —       | `CAST_NONE` → `ASCENDC_ASSERT(false)` on dav_c220 ❌         |

> **Critical**: `Cast(bfloat16_t_dst, float_src, CAST_NONE)` on dav_c220 contains only
> `ASCENDC_ASSERT(false)` with no `vconv` instruction. The destination buffer is never
> written, retaining uninitialized UB data (silent garbage output). This bug will pass
> shape checks and dtype checks but fail all precision tests.
>
> Confirmed cross-reference: `add_rms_norm_custom.cpp` line 214 uses `CAST_RINT`.

---

## Operations NOT Supported for bfloat16_t on dav_c220

These intrinsics are unsupported for `bfloat16_t` on Ascend910b and must be wrapped in
the `if constexpr (std::is_same<DTYPE_X1, bfloat16_t>::value)` branch (upcast to float32):

- `Muls<bfloat16_t>` ❌
- `Mul<bfloat16_t>` ❌
- `Add<bfloat16_t>` ❌

Supported directly for all dtypes (no upcast needed):
- `Cast` (with correct `RoundMode` — see table above)
- `DataCopyPad` (works with any element type)

---

## UB Budget Estimate

| Variant      | TQueues                           | TBufs                        | Total  |
|--------------|-----------------------------------|------------------------------|--------|
| float32      | N × `tileSize × 4`                | 0                            | N×8KB  |
| float16      | N × `tileSize × 2`                | 0                            | N×4KB  |
| bfloat16_t   | N × `tileSize × 2`                | M × `tileSize × 4`           | N×4KB + M×8KB |

For N=3 queues, M=3 float32 TBufs, tileSize=2048:
- float32: 3 × 2048 × 4 = **24 KB**
- float16: 3 × 2048 × 2 = **12 KB**
- bfloat16_t: 12 KB + 24 KB = **36 KB** (well under 192 KB UB limit)

### ⚠️ Cast Block Alignment for Float32 Buffers

`Cast(fp16/bf16 → fp32)` processes **16 elements per block**, writing **64 bytes per block**.
For non-16-aligned tileSizes, `ceil(tileSize / 16) * 64` exceeds `ALIGN_UP32(tileSize * 4)`:

| tileSize | ALIGN_UP32(tileSize×4) | ceil(tileSize/16)×64 | Overflow |
|----------|------------------------|----------------------|----------|
| 196      | 800                    | 832                  | **32 B** |
| 256      | 1024                   | 1024                 | 0        |
| 784      | 3136                   | 3136                 | 0        |
| 169      | 704                    | 704                  | 0        |

**Fix**: always use `max(ALIGN_UP32(N*4), ceil(N/16)*16*4)` for float32 Cast destination buffers.
Undersized buffers silently corrupt adjacent UB regions, causing NaN or garbage output.

---

## Two Upcast Patterns

### Pattern A — Selective Upcast (preferred for element-wise ops)

Only the bfloat16_t path allocates float32 TBufs and upcasts. float/half compute directly.
This is the **Add pattern** (see `add_custom_reference.cpp` in this directory).

```
float/half → direct compute in DTYPE_X1
bfloat16_t → Cast(CAST_NONE) → f32 compute → Cast(CAST_RINT)
```

### Pattern B — Full Upcast (for ops that must always compute in f32)

Always cast to f32 regardless of input dtype; use `if constexpr` only to select the
downcast `RoundMode` (CAST_RINT for bf16, CAST_NONE for f16).
This is the **LeakyRelu pattern** (see `lowering_examples/leaky_relu/`).

```
bf16 IO queue + f32 compute queue (always)
downcast: CAST_RINT for bf16, CAST_NONE for f16
```

#### ⚠️ Pattern B requires pipeline barriers when using TBuf staging

Pattern A (`add_custom_reference.cpp`) uses TQue for I/O — `EnQue`/`DeQue` provides
implicit MTE↔VEC pipeline synchronization.  Pattern B often uses `TBuf` (not `TQue`) for
the Cast staging buffers because the Cast happens in the same function as computation.

**When TBuf is used for Cast staging, explicit pipeline barriers are MANDATORY:**

```cpp
// CopyIn: DataCopyPad (MTE2 engine) → barrier → Cast (VEC engine)
DataCopyPad(srcLocal, src[offset], ...);
SetFlag<HardEvent::MTE2_V>(0);   // signal: MTE2 write done
WaitFlag<HardEvent::MTE2_V>(0);  // wait:   VEC can read
Cast(dst, srcLocal, RoundMode::CAST_NONE, count);

// CopyOut: Cast (VEC engine) → barrier → DataCopyPad (MTE3 engine)
Cast(dstLocal, src, RoundMode::CAST_RINT, count);  // or CAST_NONE for f16
SetFlag<HardEvent::V_MTE3>(0);   // signal: VEC write done
WaitFlag<HardEvent::V_MTE3>(0);  // wait:   MTE3 can read
DataCopyPad(dst[offset], dstLocal, ...);
```

Without these barriers, the VEC engine reads from or writes to UB while MTE is still
in-flight. The result is **non-deterministic garbage or NaN** — confirmed on GroupNorm
where all fp16/bf16 outputs were NaN until barriers were added.

#### Pattern B — Complete CopyIn/CopyOut Template (copy-paste ready)

This combines all the rules: template wrapper, simple DataCopyPadParams API,
pipeline barriers, correct Cast modes, and Cast-block-aligned buffers.

```cpp
// In Init(): allocate conversion buffers for ALL dtype variants
// Cast block alignment: ceil(N/16)*16 floats may exceed ALIGN_UP32(N*4)
uint32_t castAligned = ((tileSize + 15u) / 16u) * 16u * sizeof(float);
uint32_t padAligned  = ((tileSize * sizeof(float) + 31u) / 32u) * 32u;
uint32_t f32Bytes    = (castAligned > padAligned) ? castAligned : padAligned;
pipe.InitBuffer(x1FloatBuf, f32Bytes);
pipe.InitBuffer(yFloatBuf,  f32Bytes);
// dtype staging buffer
uint32_t dtypeBufBytes = ((tileSize * sizeof(DTYPE_X) + 31u) / 32u) * 32u;
pipe.InitBuffer(ioConvBuf, dtypeBufBytes);

// ---- CopyIn: GM(DTYPE_X) → UB(float32) ----
template <typename T = DTYPE_X>
__aicore__ inline void CopyInTile(LocalTensor<float>& dst,
                                   GlobalTensor<T>& src,
                                   uint32_t offset, uint32_t count) {
    if constexpr (std::is_same<T, float>::value) {
        DataCopyPad(dst, src[offset],
                    {1, static_cast<uint16_t>(count * sizeof(float)), 0, 0, 0},
                    {false, 0, 0, 0.0f});
    } else {
        LocalTensor<T> srcLocal = ioConvBuf.Get<T>();
        uint32_t blockLen  = count * sizeof(T);
        uint32_t paddedLen = ((blockLen + 31u) / 32u) * 32u;
        uint8_t  rightPad  = static_cast<uint8_t>((paddedLen - blockLen) / sizeof(T));
        DataCopyPad(srcLocal, src[offset],
                    {1, static_cast<uint16_t>(blockLen), 0, 0},
                    {true, static_cast<uint8_t>(0), rightPad, static_cast<uint64_t>(0)});
        SetFlag<HardEvent::MTE2_V>(0);
        WaitFlag<HardEvent::MTE2_V>(0);
        Cast(dst, srcLocal, RoundMode::CAST_NONE, count);
    }
}

// ---- CopyOut: UB(float32) → GM(DTYPE_X) ----
template <typename T = DTYPE_X>
__aicore__ inline void CopyOutTile(GlobalTensor<T>& dst,
                                    LocalTensor<float>& src,
                                    uint32_t offset, uint32_t count) {
    if constexpr (std::is_same<T, float>::value) {
        DataCopyPad(dst[offset], src,
                    {1, static_cast<uint16_t>(count * sizeof(float)), 0, 0, 0});
    } else {
        LocalTensor<T> dstLocal = ioConvBuf.Get<T>();
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(dstLocal, src, RoundMode::CAST_RINT, count);
        } else {
            Cast(dstLocal, src, RoundMode::CAST_NONE, count);
        }
        SetFlag<HardEvent::V_MTE3>(0);
        WaitFlag<HardEvent::V_MTE3>(0);
        DataCopyPad(dst[offset], dstLocal,
                    {1, static_cast<uint16_t>(count * sizeof(T)), 0, 0, 0});
    }
}
```

> Replace `DTYPE_X` with whatever your actual DTYPE macro name is (e.g. `DTYPE_X1`,
> `DTYPE_INPUT`, etc.).

---

## `if constexpr` and Type-Dependent Overloads

`if constexpr` in a non-template function works fine when both branches are
independently well-formed — for example, when they call different functions but
all argument types match in both branches:

**Works fine** — both branches use the same tensor type `DTYPE_X1`, so both
calls are valid regardless of which branch is taken:
```cpp
__aicore__ inline void Compute(...) {
    AscendC::LocalTensor<DTYPE_X1> x1Local = x1Queue.DeQue<DTYPE_X1>();
    if constexpr (std::is_same<DTYPE_X1, bfloat16_t>::value) {
        AscendC::Cast(x1F, x1Local, AscendC::RoundMode::CAST_NONE, cnt);  // bf16→f32 ✓
    } else {
        AscendC::Muls(yLocal, x1Local, (DTYPE_X1)alpha, cnt);              // direct ✓
    }
}
```

**Fails to compile** — branches call overloaded functions where **template
argument deduction** produces conflicting types. Since `DTYPE_X1` is a macro
(not a template parameter), the compiler resolves types in both branches even
though only one is taken at runtime:
```cpp
// ❌ error: no matching function for call to 'DataCopy'
__aicore__ inline void LoadRowF32(
    GlobalTensor<DTYPE_X1>& gm, uint32_t offset, LocalTensor<float>& xFloat)
{
    LocalTensor<DTYPE_X1> slot = ...;
    if constexpr (std::is_same<DTYPE_X1, float>::value) {
        AscendC::DataCopy(xFloat, slot, n);   // DataCopy<float>(f32, f32) ✓
    } else {
        AscendC::Cast(xFloat, slot, ...);     // Cast(f32, half) ✓
    }
    // When DTYPE_X1=half: DataCopy deduces T=float from xFloat and T=half from slot
    // → "deduced conflicting types for parameter 'T'" in the discarded branch
}
```

**Fix** — make the helper a `template <typename T>` function so that the
condition depends on a template parameter. During instantiation with a specific
`T`, `if constexpr` genuinely discards the non-taken branch:
```cpp
template <typename T>
__aicore__ inline void LoadRowF32(
    GlobalTensor<T>& gm, uint32_t offset, LocalTensor<float>& xFloat)
{
    LocalTensor<T> slot = inQueue.AllocTensor<T>();
    // ... DataCopyPad GM→VECIN ...
    slot = inQueue.DeQue<T>();
    if constexpr (std::is_same_v<T, float>) {
        AscendC::DataCopy(xFloat, slot, n);       // T=float → DataCopy(f32, f32) ✓
    } else {
        AscendC::Cast(xFloat, slot, AscendC::RoundMode::CAST_NONE, n); // T=half/bf16 ✓
    }
    inQueue.FreeTensor(slot);
}
```

> **When to use template helpers**: any time the two `if constexpr` branches call
> functions that differ in the *type* of their arguments (not just values), lift the
> function into a `template <typename T>` member.  The simple bf16/float dispatch inside
> a single `Compute()` using one consistent `DTYPE_X1`-typed local tensor does not need
> this treatment.

---

## DataCopyPad API for Non-Float Types

**Use `DataCopyPadParams` (simple API), NOT `DataCopyPadExtParams<T>`** for GM→UB transfers
with half or bfloat16_t types.  The `DataCopyPadExtParams<bfloat16_t>` specialization has
an ABI bug that produces incorrect reads (confirmed on CANN 8.5.0).

```cpp
// ✅ Correct: simple API with uint64_t padding value
uint32_t blockLen  = count * sizeof(T);
uint32_t paddedLen = ((blockLen + 31u) / 32u) * 32u;
uint8_t  rightPad  = static_cast<uint8_t>((paddedLen - blockLen) / sizeof(T));
DataCopyPad(dstLocal, srcGlobal[offset],
            {1, static_cast<uint16_t>(blockLen), 0, 0},
            {true, static_cast<uint8_t>(0), rightPad, static_cast<uint64_t>(0)});

// ❌ Wrong for bfloat16_t: DataCopyPadExtParams<T> with typed padding value
DataCopyPad(dstLocal, srcGlobal[offset],
            {1, static_cast<uint16_t>(blockLen), 0, 0, 0},
            {false, 0, 0, static_cast<T>(0)});  // ← incorrect reads for bf16!
```

---

## Rsqrt Hardware Precision

AscendC's hardware `Rsqrt` vector instruction returns only **~10-bit mantissa** precision
(approximately float16 level), not full float32.  This is a property of the polynomial
approximation used in the vector unit and affects ALL dtypes.

**Any kernel that uses Rsqrt must add a Newton-Raphson refinement step:**

```cpp
// Step 1: Hardware Rsqrt (~10 bits)
float x_val = variance + eps;
Duplicate(scalarLocal, x_val, 8);   // count >= 8 for float32 SIMD
Rsqrt(scalarLocal, scalarLocal, 8);
float y0 = scalarLocal.GetValue(0);

// Step 2: Newton-Raphson refinement (~20+ bits, sufficient for float32)
// f(y) = 1/y² - x = 0  →  y_{n+1} = y_n * (3 - x * y_n²) / 2
float invStd = y0 * (3.0f - x_val * y0 * y0) * 0.5f;
```

Without NR refinement, GroupNorm-type operators see ~1e-3 systematic precision drift
in invStd, causing all float32 test cases to fail the three-way precision check.

Common operators affected: **LayerNorm, RMSNorm, GroupNorm, Softmax, BatchNorm** —
any normalization that computes `1/sqrt(variance + eps)`.

---

## Canonical Reference

See `add_custom_reference.cpp` in this directory for a complete, tested multi-dtype
kernel implementing Pattern A (selective upcast) with all the above rules applied.
