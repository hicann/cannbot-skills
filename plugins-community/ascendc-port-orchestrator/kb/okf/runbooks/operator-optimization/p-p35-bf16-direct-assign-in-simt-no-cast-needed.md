---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "bf16 direct assign in SIMT (no Cast needed)"
description: "SIMT kernel with bf16 copy (no arithmetic)"
severity: medium
confidence: single_run
original_id: P-P35
timestamp_inferred: true
tags: [platform_compat, optimization, p-p35, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

SIMT kernel with bf16 copy (no arithmetic)

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P35，convert_patterns_to_okf.py）。confidence 未升格。 -->
