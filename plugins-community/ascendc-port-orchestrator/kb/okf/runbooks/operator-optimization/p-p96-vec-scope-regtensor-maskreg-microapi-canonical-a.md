---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "`__VEC_SCOPE__ { RegTensor / MaskReg / MicroAPI::* }` — canonical A5 L2 inner-loop shape"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=l2-microapi; phase=kernel-author verified_on: soc=Ascend950PR; cann=9.0.0 source: PR 103 l2-register-based-guide.md §114-155 Pattern: A5"
confidence: single_run
original_id: P-P96
timestamp_inferred: true
tags: [patterns-index, optimization, maskreg, count, __vec_scope__, loadtensorfordtypet, p-p96, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=all; op_class=l2-microapi; phase=kernel-author`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 l2-register-based-guide.md §114-155`

**Pattern**: A5 L2 (Register-based) compute MUST be wrapped in `__VEC_SCOPE__ { ... }`. Inside the scope, declare `RegTensor<T>` registers + `MaskReg` masks, use `MicroAPI::Op(...)` calls. The scope marker tells the compiler "lower to register-resident codegen", enabling the perf win that motivates L2.

**Canonical shape**:

```cpp
__aicore__ inline void ComputeL2(__ubuf__ T_KV* x_ub, __ubuf__ T_KV* gamma_ub,
                                  __local_mem__ float* dst_ub,
                                  uint16_t loopTimes, uint32_t stride,
                                  uint16_t count, float reciprocal, float epsilon)
{
    __VEC_SCOPE__
    {
        // 1. Declare all RegTensors at scope head (up to ~8 typical without spill)
        RegTensor<float> reg0, reg1, reg2, reg3, reg4, reg5, reg6, reg7;

        // 2. Build masks once (per-mask helpers)
        MaskReg pMask = UpdateMask<float>(count);                  // partial — first `count` lanes
        MaskReg pFull = CreateMask<float, MaskPattern::ALL>();      // full — all lanes active

        // 3. Inner loop — entire compute in register space
        for (uint16_t i = 0; i < loopTimes; ++i) {
            LoadTensorForDtypeT<T_KV>(x_ub, reg0, pMask, i * stride);
            LoadTensorForDtypeT<T_KV>(gamma_ub, reg1, pMask, 0);
            MicroAPI::Mul(reg2, reg0, reg0, pMask);          // x * x
            ReduceSum(reg2, reg2, pMask);                      // ΣΣ
            MicroAPI::Muls(reg3, reg2, reciprocal, pFull);     // / N
            MicroAPI::Adds(reg4, reg3, epsilon, pFull);        // + eps
            MicroAPI::Sqrt(reg5, reg4, pFull);
            MicroAPI::Div(reg6, reg0, reg5, pMask);
            MicroAPI::Mul(reg7, reg1, reg6, pMask);
            StoreTensorForDtypeTOut<float>(dst_ub, reg7, pMask, i * stride);
        }
    }   // ← __VEC_SCOPE__ end. Registers freed.
}
```

**Five mandatory ingredients**:

1. `__VEC_SCOPE__` — outer brace block enables register codegen
2. `RegTensor<T>` — single vector register typed for `T`
3. `MaskReg` — predicate register controlling per-lane execution
4. `MicroAPI::Op(...)` prefix — namespace-qualified vector ops
5. **`LoadTensorForDtypeT` + `StoreTensorForDtypeTOut`** helpers — handle automatic Cast (half / bf16 → fp32 on load; fp32 → narrow on store)

**Discipline rules**:

- **Declare all RegTensors at scope head** — compiler does register allocation per scope; mid-scope declarations create lifetime bugs
- **Reuse registers across the loop body** — `reg2, reg3, ..., reg7` get reused per iteration (the compiler allocates them to the same physical registers across iterations). DON'T declare `reg8, reg9, ..., regN` for each loop iter
- **Keep scope tight** — only the hot inner loop in `__VEC_SCOPE__`. UB allocation, tiling decode, GM→UB copy stay outside the scope
- **No `SetFlag`/`WaitFlag` inside scope** — use `LocalMemBar<MemType::UB>()` at boundaries (see P-P95)
- **No `LocalTensor` declarations inside scope** — only `RegTensor` lives here; `LocalTensor` (UB tensor) is for outside-scope UB tracking

**Anti-patterns**:

```cpp
// ❌ RegTensor declared mid-loop — confuses register allocator
__VEC_SCOPE__ {
    for (uint16_t i = 0; i < loopTimes; ++i) {
        RegTensor<float> reg;     // ← wrong place
        ...
    }
}

// ❌ A3 LocalTensor inside __VEC_SCOPE__
__VEC_SCOPE__ {
    LocalTensor<float> tmp;       // ← wrong type
    ...
}

// ❌ Whole-kernel __VEC_SCOPE__ wrapping unrelated UB management
__VEC_SCOPE__ {
    DataCopy(srcLocal, srcGm, count);   // ← outside-scope work in scope
    ...vector compute...
    DataCopy(dstGm, dstLocal, count);   // ← outside-scope work in scope
}
```

**Detection signature**:

```bash
# In L2 arch35/ kernels, verify __VEC_SCOPE__ is present in compute paths
grep -nc "__VEC_SCOPE__" arch35/*.h
# Compute path counted >= 1 per kernel function with MicroAPI:: calls

# Conversely, find MicroAPI:: calls NOT wrapped in __VEC_SCOPE__:
awk '/__VEC_SCOPE__/{in_scope=1} /^}/{in_scope=0}
     /MicroAPI::/&&!in_scope{print FILENAME":"NR": MicroAPI outside __VEC_SCOPE__: "$0}' arch35/*.h
```

**Evidence**: PR 103 l2-guide §114-155 codifies as L2 canonical pattern (RMSNorm + Mul example).

**Other instances (predicted)**: every L2 norm / activation / quant kernel in cohort 2.

**Cross-reference**:
- OL-152 (Memory↔Register API map — this pattern uses every right-column entry)
- OL-146 (CastTrait — used in inline Cast calls within the scope)
- OL-148 (SPR overflow toggle — often paired with the scope for bounded-output ops)
- P-P95 (`LocalMemBar` — boundary sync)

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P96，convert_patterns_to_okf.py）。confidence 未升格。 -->
