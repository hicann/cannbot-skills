---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Use AscendC `Power` VEC API for Pow; do not decompose manually"
description: "Trigger: computing pow(x, y) or pow(x, const_int). Pattern: CANN has a dedicated Power<T> VEC template per dtype: cpp // pow_bf16_nddma_without_loops.h Power<bfloat16_t, false, pConfig_>(dstBuf, baseB"
severity: high
confidence: single_run
original_id: P-P55
timestamp_inferred: true
tags: [precision, optimization, power, p-p55, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Trigger**: computing `pow(x, y)` or `pow(x, const_int)`.

**Pattern**: CANN has a dedicated `Power<T>` VEC template per dtype:
```cpp
// pow_bf16_nddma_without_loops.h
Power<bfloat16_t, false, pConfig_>(dstBuf, baseBuf, expBuf, count);
```
`Power`'s internal implementation (fused bf16/fp16/fp32 paths) differs completely from a manual `1/x * 1/x * 1/x`, with different rounding paths.

**Counter-example** (my kernel for `pow(std, -3)`):
```cpp
half inv_std = 1.0f / std;
half m3 = inv_std * inv_std * inv_std;  // WRONG: 3 muls, not bit-equal to Power(std, -3)
```

**Correct** (use the public AscendC VEC API):
```cpp
// broadcast std to a tensor
LocalTensor<half> stdT = ...; Duplicate(stdT, std_h, count);
LocalTensor<half> expT = ...; Duplicate(expT, static_cast<half>(-3.0), count);
Pow(resT, stdT, expT, count);  // OK: vector-level pow, matches CANN
// GetValue(resT, 0) if a scalar is needed
```

**When it matters**: whenever the reference contains `torch.pow(x, y)` or `x ** y` and bit-exact matching is required.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（P-P55，convert_patterns_to_okf.py）。confidence 未升格。 -->
