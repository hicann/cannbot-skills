---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Reciprocal via tensor Div(1, x), not scalar `1/x`"
description: "Trigger: kernel needs to compute 1/x or c/x (c is a scalar constant). Pattern: CANN aclnnReciprocal does not use Newton-Raphson nor a hardware reciprocal instruction — it uses a direct tensor-level Di"
severity: high
confidence: single_run
original_id: P-P54
timestamp_inferred: true
tags: [precision, optimization, aclnnreciprocal, p-p54, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

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

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（P-P54，convert_patterns_to_okf.py）。confidence 未升格。 -->
