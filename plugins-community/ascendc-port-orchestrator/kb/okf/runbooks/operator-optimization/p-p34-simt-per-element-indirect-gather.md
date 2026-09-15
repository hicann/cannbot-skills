---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMT per-element indirect gather"
description: "torch.gather, index_select, per-element indirect addressing"
severity: high
confidence: single_run
original_id: P-P34
timestamp_inferred: true
tags: [memory_access, optimization, p-p34, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

torch.gather, index_select, per-element indirect addressing

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P34，convert_patterns_to_okf.py）。confidence 未升格。 -->
