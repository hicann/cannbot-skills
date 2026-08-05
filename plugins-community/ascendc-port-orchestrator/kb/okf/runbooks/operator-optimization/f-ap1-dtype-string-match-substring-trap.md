---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "dtype string-match substring trap"
description: "cpp // \"bfloat16\".find(\"float16\") == 1 → matches! if (dtype.find(\"float16\") != npos) { ... } // BUG else if (dtype.find(\"bfloat16\") != npos) { ... } // never executes Fix: the bfloat16 check must come"
confidence: single_run
original_id: F-AP1
timestamp_inferred: true
tags: [precision, anti-pattern, bfloat16, float16, f-ap1, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 反模式

```cpp
// "bfloat16".find("float16") == 1 → matches!
if (dtype.find("float16") != npos) { ... }      // BUG
else if (dtype.find("bfloat16") != npos) { ... } // never executes
```

**Fix**: the `bfloat16` check must come before `float16`.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（F-AP1，convert_patterns_to_okf.py）。confidence 未升格。 -->
