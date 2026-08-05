---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "bf16 precision handling (scatter-add waiver)"
description: "Anti-pattern: using fp32 atol/rtol for bf16 tests → many false-positive FAILs Correct pattern: cpp float atol = (dtype == \"bf16\") ? 2e-2f : 1e-4f; bool waiver = (dtype == \"bf16\"); compare_data(npu, gp"
severity: medium
confidence: single_run
original_id: F-P1
timestamp_inferred: true
tags: [precision, optimization, f-p1, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Anti-pattern**: using fp32 atol/rtol for bf16 tests → many false-positive FAILs

**Correct pattern**:
```cpp
float atol = (dtype == "bf16") ? 2e-2f : 1e-4f;
bool waiver = (dtype == "bf16");
compare_data(npu, cpu_truth, n, dtype_str, atol, rtol, "fwd", waiver);
```

**Note**: bf16 mismatch is expected behavior **only** for scatter-add-class ops (Pooling fwd/bwd). **SG forward is deterministic computation — bf16 must not have any mismatch**; if it does, it is a bug.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（F-P1，convert_patterns_to_okf.py）。confidence 未升格。 -->
