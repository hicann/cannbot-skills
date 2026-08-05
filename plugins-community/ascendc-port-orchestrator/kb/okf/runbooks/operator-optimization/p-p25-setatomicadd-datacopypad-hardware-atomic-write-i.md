---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SetAtomicAdd + DataCopyPad — hardware atomic write in SIMD mode"
description: "Problem: in SIMD mode, scatter-add (e.g. grad_in[expert] += weight grad_out) needs an atomic write, but SIMT atomicAdd goes through a VEC-pipe CAS loop (slow), and SetValue in AIV mode is unreliable ("
confidence: single_run
original_id: P-P25
timestamp_inferred: true
tags: [memory_access, optimization, atomicadd, setvalue, datacopypad, setatomicadd, p-p25, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: in SIMD mode, scatter-add (e.g. `grad_in[expert] += weight * grad_out`) needs an atomic write, but SIMT `atomicAdd` goes through a VEC-pipe CAS loop (slow), and `SetValue` in AIV mode is unreliable (OL-19).

**Correct pattern**: `SetAtomicAdd<T>()` + `DataCopyPad` — hardware atomic on the MTE3 pipe

```cpp
// SIMD backward: write each token's grad_in contribution back to the expert slot
Muls(gradInLocal, gradOutLocal, expertWeight, hdim);
// EnQue + DeQue for pipeline sync
gradInOutQue_.EnQue(gradInLocal);
LocalTensor<float> gradInOut = gradInOutQue_.DeQue<float>();
// MTE3 atomic add: atomicity guaranteed by hardware, no VEC CAS
SetAtomicAdd<float>();
DataCopyPad(gradInGm_[expertIdx * hdim], gradInOut, copyParams);
SetAtomicNone();
gradInOutQue_.FreeTensor(gradInOut);
```

**Comparison**:
| Method | Pipe | Mechanism | Speed |
|--------|------|-----------|-------|
| SIMT `atomicAdd(ptr, val)` | VEC | CAS loop | Slow (serialized under contention) |
| SIMD `SetAtomicAdd` + `DataCopyPad` | **MTE3** | Hardware atomic DMA | **Fast** (bulk atomic add) |
| SIMD `SetValue` (GM) | Scalar | — | Unreliable (OL-19) |

**Key advantage**: DataCopyPad transfers the whole hdim vector in one atomic add, not per-element CAS.
**No sorting needed**: per-token SIMD + SetAtomicAdd writes back directly; no counting-sort preprocessing required.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P25，convert_patterns_to_okf.py）。confidence 未升格。 -->
