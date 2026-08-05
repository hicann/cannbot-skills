---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "非浮点算子通过标准:非计算/整数/量化类判定"
description: "非计算类与整数类须二进制一致(整数类 AE=0 也过);量化类浮点输入整型输出 AE≤1,浮点输出参考浮点标准。"
confidence: single_run
original_id: PRECISION_STANDARD_v2.1.md#4.2-4.4 非计算/整数/量化通过标准
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, pass-criteria, quantization]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
非浮点三类算子的通过标准(除量化浮点输出外均为单标杆比对):
- 非计算类(搬移/Cast):与真值二进制一致(Bitwise Match)即通过;单标杆比对。
- 整数计算类:与真值二进制一致即通过;二进制不一致但绝对误差为 0 也视为通过;单标杆比对。
- 量化计算类(两种输出模式,整型输出单标杆 / 浮点输出双标杆):
  - 浮点输入 × 整型输出:绝对误差 ≤ 1。
  - 整型输入 × 整型输出:N/A。
  - 任意输入 × 浮点输出:参考浮点精度标准。
