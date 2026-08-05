---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "用例生成规则:dtype/格式/维度/值域分布按等级正交覆盖"
description: "按精度等级正交生成用例:覆盖所有 dtype/格式,维度 1-8,值域按均匀/正态/离群点分布配比,输出总元素数须 ≥100 万。"
confidence: single_run
original_id: PRECISION_STANDARD_v2.1.md#2 用例生成规则
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, case-generation, coverage]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
按精度等级生成正交测试用例(数据类型、格式、维度、参数类型须相互正交):
- 数据类型:覆盖所有支持类型(FLOAT16/BFLOAT16/FLOAT32 等);L0 每种 ≥200,L1 每种 ≥700。
- 数据格式:覆盖所有支持格式(ND/NCHW/FRACTAL_NZ 等)。
- 数据维度:维度数 1-8,维度值 1 至 2^31,总元素数 ≤ 2^31;L0 维度值按步长(15,16)泛化,L1 维度值随机。
- 值域分布:L0 = 均匀[-5,5] 50% + 正态(μ∈[-100,100],σ∈[1,25]) 50%;L1 = 均匀[-0.001,0.001] 10% + 均匀[-5,5] 30% + 正态 40% + 离群点分布 20%。
- L2 = 泛化用例基础上新增真实业务训练/推理输入,泛化用例:模型用例 = 2:1。
- Attr:标量覆盖所有等价类,布尔覆盖 True/False,枚举覆盖所有取值,参数类型组合遍历。

约束:输出总元素数(所有用例输出样本点之和)不得低于 1,000,000;训练反向算子须用正反向级联测试;特殊场景用例不计入用例规模;不同随机种子产生的结果记为相同用例。
