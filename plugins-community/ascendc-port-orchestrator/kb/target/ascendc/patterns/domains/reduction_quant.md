---
applies_to: soc=all
reason: ReduceMax/ReduceSum/ReduceMin and quantization fusion are universal SIMD operations across a5/a3/a2. Tile-size constants tied to UB capacity must be parameterized per-target.
---

# Domain: Reduction & Quantization Optimization
> Patterns for kernels with reduction (absmax, sum, norm) + quantization.
> Load when: Analyzer detects dynamic quantization, per-row/per-tensor scaling, or fused norm+quant.
> Source: CANN ops-nn/norm/add_rms_norm_dynamic_quant/ knowledge extraction (2026-04-14)

---

## Patterns

### P-P45: Single-Pass UB-Resident Dynamic Quantization

**Severity**: **CRITICAL** | **Source**: CANN add_rms_norm_dynamic_quant knowledge extraction (2026-04-14) | **Applicability**: per-row dynamic quantization (absmax -> scale -> int8)

**Problem**: Naive 2-pass implementation: Pass1 reads HBM to compute absmax -> Pass2 re-reads HBM to do scale+quantize. HBM bandwidth is used 2x.

**Pattern**: When an entire row of D fits in UB, all operations complete in UB; HBM is only read/written once each:
```cpp
// Load row into UB (1 HBM read)
DataCopy(xLocal, xGm[rowOffset], D);
Cast(xFp32, xLocal, CAST_NONE, D);  // cast to fp32 if needed

// Step 1: Find absmax (ALL IN UB)
Abs(tmpLocal, xFp32, D);
ReduceMaxInplace(tmpLocal, D);  // -> tmpLocal[0] = max
pipe_barrier(PIPE_V);
float maxVal = tmpLocal.GetValue(0);

// Step 2: Compute scale (scalar unit)
float scaleTemp = 127.0f / maxVal;
float outScale = maxVal / 127.0f;  // save to output

// Step 3: Scale + quantize (ALL IN UB)
pipe_barrier(PIPE_S);  // S->V sync: scalar result to vector pipe
Muls(xFp32, xFp32, scaleTemp, D);

// Step 4: Cast chain to int8 (P-P46)
Cast(int32Local, xFp32, CAST_RINT, D);   // round to nearest
SetDeqScale(half(1.0f));
Cast(fp16Local, int32Local, CAST_NONE, D); // int32->fp16
Cast(int8Local, fp16Local, CAST_TRUNC, D); // fp16->int8

// CopyOut (1 HBM write)
DataCopy(yGm[rowOffset], int8Local, D);
```

**Key**: After data is loaded from HBM into UB, absmax, scale, and quantize all complete in UB. **No write-back to HBM followed by re-read is needed.**

**UB space requirement**: D * sizeof(float) * 2 (original data + temporary buffer) + intermediate-type buffer.
For D=8192 fp32: about 64KB * 2 = 128KB; 256KB UB is sufficient.

**Multi-row batching**: If UB space allows, multiRowNum rows can be processed simultaneously, using the Brcb instruction to broadcast per-row scale to full row width.

**Evidence**: CANN add_rms_norm_dynamic_quant_normal_kernel.h:329-353 (ScaleTensor), dynamic_quant_single_row.h:138-187. E1 level.

**Stop condition**: When D * sizeof(float) * 2 > available UB space, the D dimension must be tiled (see OL-60 Slice-D workspace pattern).

---

### P-P46: Quantize Cast Chain (fp32 → int8)

**Severity**: **HIGH** | **Source**: CANN add_rms_norm_dynamic_quant_helper knowledge extraction (2026-04-14) | **Applicability**: any fp32->int8 quantization output

**Problem**: AscendC has no direct fp32->int8 Cast instruction. A multi-step conversion is required.

**Cast chain**:
```cpp
// 1. fp32 -> int32 (round to nearest)
Cast(int32Local, fp32Local, CAST_RINT, count);

// 2. Set dequant scale (hardware requirement)
SetDeqScale(half(1.0f));  // scale=1.0 means pure type conversion

// 3. int32 -> fp16 (use hardware deq path)
Cast(fp16Local, int32Local, CAST_NONE, count);

// 4. fp16 -> int8 (truncate, auto-clip to [-128, 127])
Cast(int8Local, fp16Local, CAST_TRUNC, count);
```

**arch35 VF register variant** (more efficient):
```cpp
Truncate<float, CAST_RINT>(fp32Tmp, fp32Src);  // in-register round
Cast<half, float>(fp16, fp32Tmp);
Cast<int8, half>(int8, fp16);
```

**Precondition**: Input must already be scaled into the [-127, 127] range. Values outside the range are truncated by TRUNC.

**Evidence**: CANN add_rms_norm_dynamic_quant_helper.h:179-190, regbase_common.h:229-232. E1 level.

**Stop condition**: Cast chain differs when outputting int4 or fp8. fp8 (E4M3) uses Cast(RINT) directly from fp16.

---

### P-P47: Half-Interval Tree Reduction (In-Place)

**Severity**: **HIGH** | **Source**: CANN reduce_common.h knowledge extraction (2026-04-14) | **Applicability**: reducing D elements into 1 scalar (sum, max, min)

**Problem**: WholeReduceSum/WholeReduceMax only handle 64 fp32 elements. When D > 64, prior folding is required. Naive loop iteration is slow.

**Pattern**: O(log2 D) vector operations, folding in-place. Pseudocode:
```cpp
void ReduceSumHalfInterval(LocalTensor<float>& src, int count) {
    int bodyCount = findPowerOfTwo(count);  // largest 2^k <= count
    int tailCount = count - bodyCount;

    // Step 1: fold the tail (non-power-of-2 remainder)
    if (tailCount > 0) {
        Add(src[0], src[0], src[bodyCount], tailCount);
        // only bodyCount valid elements remain
    }

    // Step 2: repeated halving
    while (bodyCount > 64) {
        bodyCount /= 2;
        Add(src[0], src[0], src[bodyCount], bodyCount);
    }

    // Step 3: hardware instruction handles the last 64 elements
    WholeReduceSum(src, src, 64);  // or WholeReduceMax
}
// result at src.GetValue(0)
```

**Advantages**:
- O(log2 D) SIMD Add/Max, halving per iteration
- **In-place**: no extra buffer required (but source data is destroyed)
- Last 64 elements handled by a single hardware instruction

**Evidence**: CANN reduce_common.h:106-166 (ReduceSumHalfInterval). E1 level.

**Stop condition**: Source data is destroyed. If the original data is needed later, copy it first. count must be > 0.

#### Reference implementation (verified on V220, 2026-05-22)

The pseudocode above omits five practical details that any production kernel needs. The reference implementation below is V220-empirically-verified (3_FusionAttention 2026-05-22 PR #109/#112/#114, max_abs ≤ 1.5e-5 PASS_T1 across 13/13 cube path shapes including S=64..1024 row-reductions). It captures the practical wrinkles:

1. **Tail-element alignment**: `Max/Add` instructions need 8-fp32 (32-byte) aligned counts. The `tail` from a non-power-of-2 may not be aligned → wrap in `Align8(tail)` (rounds up). Reading `count` valid + (Align8(tail) - tail) garbage doesn't change the rowmax / rowsum because the garbage was previously valid scratch data already folded once (max-with-self or add-into-self idempotent in this range).
2. **`PipeBarrier<PIPE_V>` between fold stages**: each fold writes `src[0..body)` and the next reads `src[0..body/2)`; without barrier, vector pipe may issue next read before write retires → silent data race.
3. **Mask setup before `WholeReduceMax/Sum`**: the hardware instruction reads N lanes where N = current mask. For `count ≤ 64` (skip-fold path) you must `SetMask<float>(count)`, otherwise it consumes 64 lanes including past-end garbage.
4. **`HardEvent::V_S` flag before `GetValue`**: the reduce result is in src[0] after MTE→V pipe; scalar `GetValue` reads on scalar pipe, needs sync.
5. **Returns the scalar value via `GetValue`**, not via `src` (caller doesn't need to know where it lives in UB).

```cpp
// Helper deps (one-line definitions; or copy from FA kernel source):
__aicore__ inline int32_t FloorPow2(int32_t n)  {  // largest 2^k <= n
    int32_t p = 1; while (p * 2 <= n) p *= 2; return p;
}
__aicore__ inline int32_t Align8(int32_t n) { return ((n + 7) / 8) * 8; }

constexpr int32_t VEC_FP32_ELEMS = 64;  // one WholeReduce repeat for fp32

// Returns max-of-first-`count` fp32 elements of src. Result lands at src[0]
// (destructive). Caller passes `count > 0` precondition. Mask is left in
// "VEC_FP32_ELEMS" state — caller may need SetMask<float>(...) before
// downstream VEC if downstream needs a different mask.
__aicore__ inline float BinaryFoldReduceMax(const LocalTensor<float>& src, int32_t count) {
    if (count <= 0) return -3.402823e+38f;  // -FLT_MAX
    if (count > VEC_FP32_ELEMS) {
        int32_t body = FloorPow2(count);
        int32_t tail = count - body;
        if (tail > 0) {
            Max(src, src, src[body], Align8(tail));   // Align8 wraps non-32B-aligned tail
            PipeBarrier<PIPE_V>();
        }
        while (body > VEC_FP32_ELEMS) {
            body /= 2;
            Max(src, src, src[body], body);
            PipeBarrier<PIPE_V>();
        }
        AscendCUtils::SetMask<float>(VEC_FP32_ELEMS);   // full 64-fp32 mask for WholeReduce
    } else {
        AscendCUtils::SetMask<float>(count);              // sub-64 path uses tighter mask
    }
    WholeReduceMax<float, false>(src, src, MASK_PLACEHOLDER, 1, 1, 1, 8);
    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(ev);
    WaitFlag<HardEvent::V_S>(ev);
    return src.GetValue(0);
}

// Same shape for sum; substitute Add↔Max and WholeReduceSum↔WholeReduceMax.
__aicore__ inline float BinaryFoldReduceSum(const LocalTensor<float>& src, int32_t count) {
    if (count <= 0) return 0.0f;
    if (count > VEC_FP32_ELEMS) {
        int32_t body = FloorPow2(count);
        int32_t tail = count - body;
        if (tail > 0) {
            Add(src, src, src[body], Align8(tail));
            PipeBarrier<PIPE_V>();
        }
        while (body > VEC_FP32_ELEMS) {
            body /= 2;
            Add(src, src, src[body], body);
            PipeBarrier<PIPE_V>();
        }
        AscendCUtils::SetMask<float>(VEC_FP32_ELEMS);
    } else {
        AscendCUtils::SetMask<float>(count);
    }
    WholeReduceSum<float, false>(src, src, MASK_PLACEHOLDER, 1, 1, 1, 8);
    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(ev);
    WaitFlag<HardEvent::V_S>(ev);
    return src.GetValue(0);
}
```

**Living source**: `workspace/3_FusionAttention/kernel/fusion_attention_kernel.h:62-100` (FA's helpers). Promoted here so the next op needing fp32 fold-reduce doesn't have to grep through FA's private headers — copy-paste this into your op's kernel.h.

**Anti-pattern this replaces** (CAND-FA-MULTI-LAUNCH-PERF-GAP §2 sub-improvement, observed in FA Δ#2 PR #112):
```cpp
// BAD on AIV: per-row scalar GetValue loop burns scalar pipe.
for (int32_t s = 0; s < S; ++s) {
    float row_max = -1e30f;
    for (int32_t d = 0; d < D; ++d) {
        float v = src.GetValue(s * D + d);
        if (v > row_max) row_max = v;
    }
    // ... use row_max
}
// Cost: S*D scalar GetValue calls, each ~10 cycles on scalar pipe.
```

**Variants & cross-refs**:
- **Bit-exact determinism**: half-interval fold is deterministic by construction (reduction order is fixed by power-of-2 indexing).
- **Multi-row packed reduce** (R parallel rows × W wide fp32 in UB): see CAND-FA4-style chained `BlockReduceMax/Sum` recipe in candidates.md — different pattern when row parallelism matters; P-P47 is for single-row or non-packed cases.
- **Need argmin/argmax**: use `ReduceMin/Max(calc_index=true)` (P-P62) instead; that's hardware support for index-tracking and avoids the scalar finish entirely.

---

## P-P62: ReduceMin/ReduceMax with `calc_index=true` for argmin/argmax in one VEC instruction

**Severity**: Useful (saves ~N scalar GetValue + compares) | **Source**: CANN `ops-nn/optim/advance_step/op_kernel/advance_step_spec.h` lines 343-347 (2026-04-24); op#7 MoeGatingTopKSoftmax uses the same pattern.

**Trigger**: kernel needs argmin/argmax of a vector (e.g. find first `-1` in a row, find best expert score, find first NaN, etc.).

**Pattern**:
```cpp
// Last bool arg = calc_index. Result occupies first 2 elements:
//   dst[0] = min/max value (as float)
//   dst[1] = the INDEX of that value, but stored as a float-bit-pattern
ReduceMin(dst, src, work, count, /*calc_index=*/true);

// Extract the index — the float at dst[1] is actually a uint32 bit-pattern:
SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
float tempIndex = dst.GetValue(1);
uint32_t minIndex = *reinterpret_cast<uint32_t*>(&tempIndex);   // <-- critical
```

**Critical**: the index is **bit-encoded as a float**, NOT a sane fp32-integer-equivalent value. You MUST `reinterpret_cast<uint32_t*>(&temp)` to extract it. If you `static_cast<int>(temp)` you get a corrupted value (whatever the bit pattern happens to look like as fp32).

**Why this trick exists**: AscendC VEC ops produce only homogeneous-typed output. Hardware packs the int index into a float slot by raw bit reinterpretation. The cost is a one-line `reinterpret_cast` to recover.

**When NOT to use**: if all you need is the min/max value (no index), use `calc_index=false` (or the default) — saves the index slot.

**Applicability**: any kernel needing argmin/argmax over a single vector. Combined with stride / row dispatch, can do per-row argmin/argmax in O(rows * log(N)) VEC ops (vs O(rows * N) scalar). Common in topk, dynamic-quant scale finding, masked-position search, MoE expert selection.

**Anti-pattern**: scalar loop `for (i...) if (a[i] < min) { min = a[i]; idx = i; }` — burns scalar pipe + V→S sync, ~10x slower at N=64+.

---

## P-P63: `ArithProgression` for vectorized arange / index-sequence generation

**Severity**: Useful | **Source**: CANN `ops-nn/optim/advance_step/op_kernel/advance_step_spec.h` line 234 (2026-04-24).

**Trigger**: kernel needs a vector of `[start, start+step, start+2*step, ...]` (row offsets for gather, position indices for RoPE, identity permutation, stride vectors).

**Pattern**:
```cpp
// Generates count elements: dst[i] = start + i * step
ArithProgression<int32_t>(dst, /*start=*/0, /*step=*/blockTablesStride, /*count=*/seqsNum);
```

This is one VEC instruction. Replaces:
```cpp
for (int i = 0; i < count; i++) dst.SetValue(i, start + i * step);   // O(N) scalar ops at ~50ns each
```

**Use cases observed**: row offsets for indirect gather, identity permutation for sort initial state, stride vectors for non-contiguous access, position indices for embeddings.

**Type variants**: `ArithProgression<int32_t>`, `<float>`, `<int64_t>` all available.

**Combine with `Add`/`Adds`**: ArithProgression for the base sequence + vector `Add(dst, base_seq, offset_vector, count)` composes complex index patterns in O(1) instructions.


---

### Performance-critical cross-references (2026-06-24, add_rms_norm_quant V1->V2)

When implementing a fused reduction+normalize+quantize kernel on A5/arch35,
the following OL entries MUST be consulted BEFORE writing Phase B code:

- **OL-256** (Divs->Muls for fp32 normalize): Use `Muls(x, 1.0f/rms)`, not `Divs(x, rms)`.
  This is precision-safe in fp32 per ALWAYS_LOADED_RULES §5 fp32 carve-out. Contributed ~45%
  of the 1.94x speedup in add_rms_norm_quant V1->V2.
- **OL-257** (VEC cost model): Divs is Tier 3 (1-2 elem/cyc), Muls is Tier 0 (8-16 elem/cyc).
  Count Tier 3+4 ops and PipeBarriers per row before committing the design.
- **OL-258** (TQue double-buffering): Use TQue QBUF_DEPTH=2 for >=2 GM reads/row.
  Contributed ~15% of the 1.94x speedup.
- **OL-245** (regbase default): A5 SIMD compute chains default to regbase, not Membase.
  Eliminates per-op PipeBarriers in multi-op chains.

The add_rms_norm_quant V1->V2 A/B comparison (2026-06-24, NPU 0, 196 cases) showed that
applying ALL FOUR optimizations together improved geo_mean from 0.47x to 0.91x vs PyTorch NPU,
while improving precision (196/196 vs 194/196 PASS).
