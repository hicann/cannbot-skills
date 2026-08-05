---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "LAUNCH_BOUND + LAUNCH_CHECK"
description: "LAUNCH_BOUND(1024) >= the maximum number of threads any dispatcher may launch. Use LAUNCH_CHECK to inspect the return value of each launch."
severity: medium
confidence: single_run
original_id: P-P5
timestamp_inferred: true
tags: [kernel_launch, optimization, launch_check, p-p5, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

`LAUNCH_BOUND(1024)` >= the maximum number of threads any dispatcher may launch. Use `LAUNCH_CHECK` to inspect the return value of each launch.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/kernel_launch.md（P-P5，convert_patterns_to_okf.py）。confidence 未升格。 -->
