---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "FlashAttention-class advisory template knowledge — arch35 cube-MIX FA interfaces, recipe, tiling, and sync"
description: "For FA-class arch22→arch35 generation. Target templates preserve block interfaces, phase order, host-tiling categories, and completeness hypotheses; they are advisory only. Emit task-owned code from t"
severity: high
confidence: single_run
original_id: P-P103
timestamp_inferred: true
tags: [flash_attention, optimization, p-p103, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

For FA-class arch22→arch35 generation. Target templates preserve block interfaces, phase order, host-tiling categories, and completeness hypotheses; they are advisory only. Emit task-owned code from the selected arch22 contract and current arch35 public APIs, then validate against source-NPU truth. A copied target body or target output cannot close generation. Full body: `patterns/domains/fa_class_template.md`.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P103，convert_patterns_to_okf.py）。confidence 未升格。 -->
