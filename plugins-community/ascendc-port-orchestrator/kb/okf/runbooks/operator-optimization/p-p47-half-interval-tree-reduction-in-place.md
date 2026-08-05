---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Half-Interval Tree Reduction (In-Place)"
description: "Problem: WholeReduceSum/WholeReduceMax only handle 64 fp32 elements. When D > 64, prior folding is required. Naive loop iteration is slow. Pattern: O(log2 D) vector operations, folding in-place. Pseud"
confidence: single_run
original_id: P-P47
timestamp_inferred: true
tags: [reduction_quant, optimization, tail, count, getvalue, p-p47, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

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

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/reduction_quant.md（P-P47，convert_patterns_to_okf.py）。confidence 未升格。 -->
