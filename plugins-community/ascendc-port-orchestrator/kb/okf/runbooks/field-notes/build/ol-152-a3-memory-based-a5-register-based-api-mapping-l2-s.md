---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A3 Memory-based ↔ A5 Register-based API mapping (L2 substitution table) [V351, microapi-l2]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=norm,activation,reduction,elementwise,quant"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=norm,activation,reduction,elementwise,quant"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-152
timestamp_inferred: true
tags: [ascendc, ol-152]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=norm,activation,reduction,elementwise,quant`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 l2-register-based-guide.md §237-260`

**Rule**: When porting A3 (Memory-based) compute kernels to A5 L2 (Register-based MicroAPI), each Memory-based primitive has a canonical Register-based replacement. The mapping is **mechanical** for hot-path inner loops — replace, wrap in `__VEC_SCOPE__`, use `MaskReg` instead of implicit count. This table is the single-source substitution reference.

**Side-by-side mapping** (full 12-row table):

| Operation | A3 — Memory-based (220x) | A5 — Register-based (351x / L2) |
|---|---|---|
| Data declaration | `LocalTensor<float> xLocal` (UB buffer) | `RegTensor<float> reg0` (single vector register) |
| GM → UB load | `DataCopy(xLocal, xGm, count)` | `DataCopy<LoadDist::DIST_UNPACK_B16>(reg0, addr)` |
| Type cast | `Cast<float, half>(dst, src, RoundMode::CAST_RINT, count)` | `Cast<float, half, CAST_B16_TO_B32>(dst, src, maskReg)` |
| Vector binary op | `Mul(dstLocal, src1Local, src2Local, count)` | `MicroAPI::Mul(regDst, regSrc1, regSrc2, maskReg)` |
| Scalar binary op | `Muls(dstLocal, srcLocal, scalar, count)` | `MicroAPI::Muls(regDst, regReg, scalar, maskFull)` |
| Reduction | `ReduceSumCustom(dst, src, work, count)` | `ReduceSum(regDst, regSrc, maskReg)` |
| UB → GM store | `DataCopy(yGm, yLocal, count)` | `DataCopy<StoreDist::DIST_NORM>(addr, reg, maskReg)` |
| Non-aligned store | `DataCopyPad(yGm, yLocal, padParams)` | `DataCopyUnAlign(addr, reg, unalignReg, stride)` |
| Mask | implicit (via `count` parameter) | explicit `MaskReg` + `CreateMask<T, MaskPattern::ALL>()` / `UpdateMask<T>(count)` |
| Memory barrier | `SetFlag<HardEvent::MTE2_V>` / `WaitFlag` | `LocalMemBar<MemType::UB>` |
| Compute scope | (no wrapping) | `__VEC_SCOPE__ { ... }` wraps the compute |
| Quant Pack | (not needed) | `Pack((RegTensor<uint16_t>&)dst, (RegTensor<uint32_t>&)src)` 2 levels |

**The "everything in `__VEC_SCOPE__`" rule**:

```cpp
// Canonical A5 L2 inner loop (RMSNorm fragment):
__VEC_SCOPE__
{
    RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;
    MaskReg pMask = UpdateMask<float>(count);
    MaskReg pFull = CreateMask<float, MaskPattern::ALL>();

    for (uint16_t i = 0; i < loopTimes; ++i) {
        LoadTensorForDtypeT<T_KV>(x_ub, reg0, pMask, i * stride);
        LoadTensorForDtypeT<T_KV>(gamma_ub, reg1, pMask, 0);
        MicroAPI::Mul(reg2, reg0, reg0, pMask);      // x * x
        ReduceSum(reg2, reg2, pMask);                  // ΣΣ
        MicroAPI::Muls(reg3, reg2, reciprocal, pFull); // / N
        MicroAPI::Adds(reg4, reg3, epsilon, pFull);    // + eps
        MicroAPI::Sqrt(reg5, reg4, pFull);
        MicroAPI::Div(reg6, reg0, reg5, pMask);
        MicroAPI::Mul(reg7, reg1, reg6, pMask);
        StoreTensorForDtypeTOut<float>(dst_ub, reg7, pMask, i * stride);
    }
}
```

**Mask helpers** (`MaskReg` is opaque — use these factories):

```cpp
MaskReg pFull = CreateMask<float, MaskPattern::ALL>();       // all-1 mask
MaskReg pMask = UpdateMask<float>(count);                      // first `count` lanes = 1
// (Other patterns: FIRST_HALF, ODD, EVEN — see API catalog)
```

**Helper templates** (live in CANN headers, but worth reproducing as part of the port):

```cpp
// LoadTensorForDtypeT — auto-Cast from narrow to fp32 during load
template <typename T_SRC>
__simd_callee__ inline void LoadTensorForDtypeT(
    __ubuf__ T_SRC* src, RegTensor<float>& reg, MaskReg& mask, int32_t offset);

// StoreTensorForDtypeTOut — auto-Cast from fp32 to narrow during store
template <typename T_DST>
__simd_callee__ inline void StoreTensorForDtypeTOut(
    __ubuf__ T_DST* dst, RegTensor<float>& reg, MaskReg& mask, int32_t offset);
```

**Anti-patterns** (compile-pass but wrong on A5):

```cpp
// ❌ A3-style Cast inside arch35/ — works but uses default SatMode (UNKNOWN)
Cast<float, half>(dst, src, RoundMode::CAST_RINT, count);

// ✅ A5 L2 Cast with explicit CastTrait
Cast<float, half, CAST_B16_TO_B32>(dst, src, maskReg);

// ❌ ReduceSumCustom in arch35/ — slower, no MicroAPI path
ReduceSumCustom(dst, src, work, count);

// ✅ Use ReduceSum (MicroAPI version on A5)
ReduceSum(regDst, regSrc, maskReg);

// ❌ DataCopyPad in arch35/ — A3 non-aligned store API
DataCopyPad(yGm, yLocal, padParams);

// ✅ DataCopyUnAlign (A5)
DataCopyUnAlign(addr, reg, unalignReg, stride);

// ❌ SetFlag/WaitFlag inside __VEC_SCOPE__ — wrong sync layer
SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

// ✅ LocalMemBar at L2 boundaries
LocalMemBar<MemType::UB>();
```

**Detection signature** (post-worker audit):

```bash
# Grep arch35/ for residual A3-style APIs:
grep -nE "(ReduceSumCustom|DataCopyPad|SetFlag<HardEvent::|WaitFlag<HardEvent::)" arch35/*.h
# Each hit is a substitution candidate per the table above.

# Grep arch35/ for Cast without CastTrait (4-arg form):
grep -nE "Cast<.+>\([^,]+,[^,]+,\s*RoundMode::" arch35/*.h
# Hits = A3-style 4-arg Cast that should be 3-arg with CastTrait template param.
```

**When to apply this table**:
- ANY op classified L2 by OL-143 (perf-critical / quant-Cast / overflow-mode / new dtype)
- Targeted hot-loop sections of L1 ops where opportunistic perf gain is large (5-30% per loop iteration)

**When NOT to apply**:
- L1 mechanical ports of non-perf-critical ops — keep Memory-based, it's correct and shorter to author
- Vendor primitives (`AscendC::LayerNorm<T,T>`, `SoftMax`, `RmsNorm` etc.) — already cross-arch compatible per OL-149 §"API tier compatibility"; don't unwind them into MicroAPI manually unless profiling identifies them as the bottleneck

**Evidence**:
- PR 103 l2-guide §237-260 ships this table verbatim
- Our cohort 1 archive perf gap analysis: ada_layer_norm postmortem (0.38× perf) used `AscendC::LayerNorm<T,T>` vendor primitive (correct precision-wise) — staying Memory-based on a perf-critical op was the bug. Applying this table to ada_layer_norm would have routed to L2 MicroAPI rewrite.
- expand_into_jagged_permute 2026-05-17 (port_a3_to_a5, kw-1): §"When NOT to apply" carve-out confirmed load-bearing. Brief's A.1.5 decision tree would have routed this to L2 purely because the A3 source uses `DataCopyPad`. Carve-out overruled — int32 index-arithmetic, non-perf-critical data-movement → L1 mechanical port retaining `DataCopyPad` + `Adds` + `CreateVecIndex`. Result: 8/8 bit-exact PASS, median 3.77× ratio (independent re-measure 3.75×). Counterfactual L2 rewrite would not have improved perf — this op is dominated by launch overhead, not compute.

**Other instances (predicted)**: every L2-classified op in cohort 2 ACTIONABLE (12 of 14 have qualifying L2 triggers — RMSNorm/LayerNorm/SoftMax-shaped + quant-Cast paths).

**Cross-reference**:
- OL-143 (L1/L2/L3 classifier — decides whether to apply this table)
- OL-146 (CastTrait constants — feeds the `Cast<DST, SRC, TRAIT>` substitutions)
- OL-148 (SPR overflow toggle — L2-only optimization)
- P-P95 (`LocalMemBar<MemType::UB>` sub-pattern)
- P-P96 (`__VEC_SCOPE__` inner-loop sub-pattern)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-152（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
