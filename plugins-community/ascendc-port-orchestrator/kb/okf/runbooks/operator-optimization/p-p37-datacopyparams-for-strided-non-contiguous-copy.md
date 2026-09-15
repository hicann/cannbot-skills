---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "DataCopyParams for strided/non-contiguous copy"
description: "non-contiguous memory copy (columns, strided data) — replaces for loop"
severity: high
confidence: single_run
original_id: P-P37
timestamp_inferred: true
tags: [memory_access, optimization, p-p37, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

non-contiguous memory copy (columns, strided data) — replaces for loop

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P37，convert_patterns_to_okf.py）。confidence 未升格。 -->
