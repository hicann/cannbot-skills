---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "All \"scalar arithmetic\" in bf16/fp16 must go through the VEC path"
description: "Trigger: kernel has half/bf16 scalar arithmetic (not a tensor op). Pattern: NPU scalar FPU and Vector FPU are different hardware units. The same a / b done via scalar code vs Div(tensor) through the V"
severity: high
confidence: single_run
original_id: P-P56
timestamp_inferred: true
tags: [precision, optimization, pow, p-p56, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

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

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（P-P56，convert_patterns_to_okf.py）。confidence 未升格。 -->
