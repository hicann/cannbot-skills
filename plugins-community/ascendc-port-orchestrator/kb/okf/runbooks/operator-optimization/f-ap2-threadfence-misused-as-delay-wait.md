---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "`__threadfence` misused as delay/wait"
description: "__threadfence() is a memory barrier (ensures writes become visible to other threads), not a delay/wait. The correct approach is cooperative-group __shfl sync or a spin-wait."
severity: medium
confidence: single_run
original_id: F-AP2
timestamp_inferred: true
tags: [precision, anti-pattern, __threadfence, __shfl, f-ap2, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 反模式

`__threadfence()` is a memory barrier (ensures writes become visible to other threads), not a delay/wait. The correct approach is cooperative-group `__shfl` sync or a spin-wait.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/precision.md（F-AP2，convert_patterns_to_okf.py）。confidence 未升格。 -->
