---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Adaptive tile size"
description: "dim ≤ 256: <BRE=32, TI=16> 16 edges × 32 emb threads dim ≤ 512: <BRE=64, TI=8> 8 edges × 64 emb threads dim > 512: <BRE=512, TI=1> 1 edge × 512 emb threads NPU 56 blocks × TI=16 = 896 edges/step → TI="
confidence: single_run
original_id: P-P11
timestamp_inferred: true
tags: [memory_access, optimization, p-p11, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

```
dim ≤ 256:  <BRE=32,  TI=16>    16 edges × 32 emb threads
dim ≤ 512:  <BRE=64,  TI=8>     8 edges × 64 emb threads
dim > 512:  <BRE=512, TI=1>     1 edge × 512 emb threads
```

On a 56-AIV target, 56 blocks × TI=16 covers 896 edges per step. Tune TI
from the target workload and UB budget instead of copying a fixed launch shape.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P11，convert_patterns_to_okf.py）。confidence 未升格。 -->
