---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "特殊场景测试规则:空/边界/标量 Tensor 与 INF-NAN 覆盖"
description: "必测特殊值场景:空 Tensor、上下边界、标量 Tensor、INF/-INF/NAN 遍历、异常值防护;这些用例不计入用例规模。"
confidence: single_run
original_id: PRECISION_STANDARD_v2.1.md#2.1 特殊场景测试规则
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, edge-cases, boundary]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
除常规用例外须覆盖以下特殊值场景(不计入用例规模):
- 空 Tensor:某维度为 0、其余为非负整数;每种 dtype 每个 dim 遍历得到 dim 个空 Tensor。
- 上下边界:下边界 = 每个维度都为 1 的 Tensor;上边界 = 某维度为 2^31+1、其余维度都为 1。每种 dtype 每个 dim 一个/遍历得到 dim 个标量 Tensor。
- 标量 Tensor:shape 为 [1];每种 dtype 一个。
- INF/-INF/NAN:所有输入 Tensor 元素值遍历 "nan"/"inf"/"-inf"/["-inf","inf"] 之一;每种 dtype 每种元素值生成 4 个不同 shape 的用例。
- 异常值覆盖(Tensor/Attr):边界值外、约束外或不支持场景须有明确防护/拦截(资料说明 + 代码拦截),包括超值域边界测试及算子自身约束/不支持场景测试。
