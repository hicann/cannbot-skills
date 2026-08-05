---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "算子分四类选误差度量:Bitwise/AE/MARE/MERE/RMSE 映射"
description: "先按计算特性把算子分非计算/整数/量化/浮点四类,再选对应误差度量与单/双标杆比对方法;随机数生成类走 kstest。"
confidence: single_run
original_id: PRECISION_STANDARD_v2.1.md#4.1 算子类别与误差度量表
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, error-metrics, classification]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
不同计算类型算子误差特性不同,须先按计算特性分四类,再选对应误差度量与比对方法:
- 非计算类(搬移/Cast):仅 Bitwise Match;单标杆比对。
- 整数计算(INT8/16/32):Bitwise Match + AE;单标杆比对。
- 量化计算(FLOAT4/FLOAT8/INT8):AE + MARE + MERE + RMSE;整型输出用单标杆、浮点输出用双标杆。
- 浮点计算(FLOAT16/FLOAT32):MARE + MERE + RMSE;双标杆比对。
- 随机数生成类为特殊类,不属上述四类,单独用 kstest 分布检验标准。

误差度量公式:AE=abs(actual−golden);MARE=max(abs(actual−golden)/(abs(golden)+1e-7));MERE=avg(abs(actual−golden)/(abs(golden)+1e-7));RMSE=sqrt((1/N)·Σ(actual_i−golden_i)^2)。引入小值 1e-7 避免 golden 除 0 风险。
