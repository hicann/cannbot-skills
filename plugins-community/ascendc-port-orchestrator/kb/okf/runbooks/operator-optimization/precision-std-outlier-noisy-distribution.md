---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "离群点(带噪)分布生成:0.1% 元素放大 1000 倍并校验值域"
description: "模拟真实训练噪声:随机选 k=max(1,floor(n/1000)) 个元素乘 1000,校验仍在有效值域内;多输入各自独立注入。"
confidence: single_run
original_id: PRECISION_STANDARD_v2.1.md#5.1.2 离群点分布生成规则
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, noisy-distribution, robustness]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
离群点分布(带噪分布 Noisy Distribution)模拟真实训练中的数据异常/计算误差/硬件扰动,评估算子在非理想数据下的数值鲁棒性与计算稳定性(在 L1 用例中占 20%)。生成规则:
1. 离群点数量 k = max(1, floor(n/1000)),确保 ≥1 个且约占总数据量 0.1%(n 为总元素数)。
2. 从 [0, n-1] 线性索引中均匀随机、不重复选取 k 个位置,构成离群点索引集合 I。
3. 数值构造:noisy[i] = X[i]×1000 (i∈I),否则保持 X[i]。
4. 值域合规校验(后处理):Xnoisy 所有元素须仍处于算子有效输入值域 [min_val, max_val] 内;若越界,则重新生成基础数据 X 或调整离群点倍率直至满足。
5. 多输入算子:每个参与数值计算的输入独立应用本规则(离群点位置各自独立随机选取);不参与计算的输入(索引、形状参数等)保持原值、不注入噪声。
