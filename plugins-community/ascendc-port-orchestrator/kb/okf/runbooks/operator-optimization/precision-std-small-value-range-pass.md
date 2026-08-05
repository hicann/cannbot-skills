---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "小值域通过标准:|golden|≈0 时改用 ErrorCount 判定"
description: "真值接近 0 时相对误差不稳定,当 |golden|<按 dtype 的阈值改用 ErrorCount 判定,ErrorCount_npu/max(ErrorCount_baseline,1)≤2 即通过。"
confidence: single_run
original_id: PRECISION_STANDARD_v2.1.md#4.5.3 小值域通过说明
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, small-value, false-positive]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
当算子真值为极小值(接近 0)时相对误差计算可能不稳定,须改用小值域通过标准(解决 |golden|≈0 时 MARE 假阳性;适用所有数据类型)。当 |golden| < Small Value Threshold 时启用。

按 dtype 的 (小值域阈值 threshold / 小值域 error 指标):
- FLOAT16: 2^-11 / 2^-16
- BFLOAT16: 2^-8 / 2^-16
- FLOAT32: 2^-14 / 2^-30
- HiFLOAT32: 2^-12 / 2^-28
- FLOAT8 E4M3: 2^-4 / 2^-6
- FLOAT8 E5M2: 2^-3 / 2^-5

小值域数值错误数量 ErrorCount = Σ I(|golden|<threshold ∧ |actual−golden|>error),I(·) 为指示函数。
通过标准:ErrorCount_npu / max(ErrorCount_baseline, 1) ≤ 2。
