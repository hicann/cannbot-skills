---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "grid_y and thread count consistency"
description: "The host-side grid_y computation must match the actual thread count used by the dispatcher. Inconsistency between the two causes incorrect work distribution."
severity: medium
confidence: single_run
original_id: P-P6
timestamp_inferred: true
tags: [kernel_launch, optimization, p-p6, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

The host-side grid_y computation must match the actual thread count used by the dispatcher. Inconsistency between the two causes incorrect work distribution.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/kernel_launch.md（P-P6，convert_patterns_to_okf.py）。confidence 未升格。 -->
