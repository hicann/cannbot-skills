---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Cube transposed-input via runtime `SetTensor*(_, isTrans=true)` — NOT template ISTRANS"
description: "### Trigger Level-3 cube op with logical transpose: torch.matmul(A.T, B), torch.matmul(A, B.T), torch.matmul(A.T, B.T), nn.Linear with weight transpose, Conv backward weights. Direct extension of P-P6"
severity: critical
confidence: single_run
original_id: P-P69
timestamp_inferred: true
tags: [platform_compat, optimization, p-p69, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

### Trigger
Level-3 cube op with logical transpose: `torch.matmul(A.T, B)`, `torch.matmul(A, B.T)`, `torch.matmul(A.T, B.T)`, `nn.Linear` with weight transpose, Conv backward weights. Direct extension of P-P68.

### Pattern (the runtime bool drives transpose, NOT the template flag)

```cpp
// All MatmulType template flags stay default false:
using AT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
using BT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;
using CT = MatmulType<TPosition::GM, CubeFormat::ND, T, /*ISTRANS=*/false>;

// Runtime bools do the actual transpose:
mm.SetTensorA(aTensor, /*isTransposeA=*/true);   // for A.T @ B  or  A.T @ B.T
mm.SetTensorB(bTensor, /*isTransposeB=*/true);   // for A   @ B.T or  A.T @ B.T
mm.SetSingleShape(/*M=*/output_rows, /*N=*/output_cols, /*K=*/reduction);
```

### Tiling field map — shape-flat regardless of which side transposes

| Op | M (out rows) | N (out cols) | Ka, Kb (reduction) | A bool | B bool |
|----|--------------|--------------|--------------------|--------|--------|
| `A @ B` (no trans) | A.size(0) | B.size(1) | A.size(1)=B.size(0) | false | false |
| `A.T @ B` (TransA) | A.size(1) | B.size(1) | A.size(0)=B.size(0) | **true** | false |
| `A @ B.T` (TransB) | A.size(0) | B.size(0) | A.size(1)=B.size(1) | false | **true** |
| `A.T @ B.T` (BothTrans) | A.size(1) | B.size(0) | A.size(0)=B.size(1) | **true** | **true** |

### Mechanism trace (CANN 9.0.0)
- Static `MatmulType<..., ISTRANS>` flag is stored as `A_TYPE::isTrans` (`asc/impl/adv_api/detail/matmul/utils/matmul_type_def.h:42`) but referenced ONLY in MX-FP8 / scale path (`mx_matmul_utils.h:321`).
- Actual ND→ND transpose driver: runtime member `MatmulShapeInfoBase::isTransposeA_` set by `SetTransposeA(bool)` or `SetTensorA(gm, bool)` 2nd arg. `IsTransposeA()` reads the runtime member.
- A-side and B-side bools are independent and symmetric — no special handling for "both true".

### Anti-pattern (compiles, FAILS precision with garbage output)
```cpp
using AT = MatmulType<..., /*ISTRANS=*/true>;   // template flag — IGNORED in ND→ND
mm.SetTensorA(aTensor, /*isTransposeA=*/false); // runtime — drives transpose
// → cube computes A @ B (no transpose); output has algorithmic noise (max_abs_diff 5-120)
```

### Evidence
- 4_MatmulTransA Phase D iter 1: `ISTRANS=true` template + `SetTensorA(_, false)` → max_abs_diff 5–120, mean_abs_diff 5–18. Iter 2 with runtime-bool fix: 50/50 + 16/16 PASS, fp32 bit-exact, 1.36× median.
- 5_MatmulTransB: `SetTensorB(_, true)` only, 0+0 iters, 1.29× median, fp32 bit-exact.
- 3_MatmulBothTrans: BOTH bools true, 0+0 iters, 1.45× median, fp32 bit-exact. Confirms A/B symmetry.
- The {none, A, B, both} 4-corner lattice is now fully validated.

### Combine with P-P68
P-P69 only specifies the transpose mechanism. constexpr static tiling, on-stack TCubeTiling, AIC scheduling come from P-P68. Use them together for any single-AIC transposed GEMM.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（P-P69，convert_patterns_to_okf.py）。confidence 未升格。 -->
