---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "#pragma unroll scope"
description: "Only use on small loops with a compile-time-inferrable upper bound. Loops containing WarpReduceAddSync + ThreadBarrier must not be unrolled."
severity: low
confidence: single_run
original_id: P-P7
timestamp_inferred: true
tags: [kernel_launch, optimization, p-p7, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

Only use on small loops with a compile-time-inferrable upper bound. Loops containing `WarpReduceAddSync + ThreadBarrier` must not be unrolled.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/kernel_launch.md（P-P7，convert_patterns_to_okf.py）。confidence 未升格。 -->
