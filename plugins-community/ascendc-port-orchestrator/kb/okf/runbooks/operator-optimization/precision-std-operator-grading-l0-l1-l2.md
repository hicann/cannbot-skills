---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "算子精度分级 L0/L1/L2:用例规模与浮点阈值逐级加严"
description: "按算子重要性分 L0/L1/L2 三级;用例规模 ≥5k/10k/30k、浮点通过阈值(MARE/MERE/RMSE 比值)逐级收紧,决定验收严苛度。"
confidence: single_run
original_id: PRECISION_STANDARD_v2.1.md#1.2 算子精度分级说明
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, grading, thresholds]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
选择算子精度等级决定后续用例规模和压测次数，三级从 L0→L2 逐级加严：L0 常规算子用例规模 ≥5,000，L1 重要算子 ≥10,000，L2 关键算子 ≥30,000。所有等级均直接与 CPU 真值比较，并使用项目已声明的 dtype 阈值；不得用跨硬件误差比替代绝对精度结论。
