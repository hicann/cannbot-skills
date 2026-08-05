---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Host benchmark best practice"
description: "Warm up 3 times, time 10 NPU iterations, compare precision with audited CPU truth, then run boundary tests."
severity: low
confidence: single_run
original_id: P-P8
timestamp_inferred: true
tags: [kernel_launch, optimization, p-p8, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

Warm up 3 times, time 10 NPU iterations and take the mean, compare precision
against audited CPU truth, then run boundary tests (`edges=0`, `dim=1`, `dim=3`).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/kernel_launch.md（P-P8，convert_patterns_to_okf.py）。confidence 未升格。 -->
