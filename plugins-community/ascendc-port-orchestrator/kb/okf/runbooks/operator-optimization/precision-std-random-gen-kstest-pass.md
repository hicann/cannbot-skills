---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "随机数生成类算子通过标准:kstest 分布一致性 + p 值门槛"
description: "rand 类按分布一致性验收:整张量 kstest,p>α(α=0.01)视为分布相同;重复 N 次须至少 ((1-α)+z·sqrt(α(1-α)/N)) 比例用例满足,z=-3.0902。"
confidence: single_run
original_id: PRECISION_STANDARD_v2.1.md#5.3.5 随机数生成算子通过标准
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, random-gen, kstest]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
随机数生成类算子(如 rand)不属四大基础类,按分布一致性单独验收:
- 用 kstest 检验衡量与标杆生成数据的分布差异,整张量做检验;p>α 认为分布相同(显著性水平 α 设定为 0.01)。
- 重复测试 N 次,要求至少 ((1−α)+z·sqrt(α(1−α)/N))×100% 的用例满足 p>α。
- z 取正态分布 99.9% 截尾点 = -3.0902(查表固定值);N 为不同 shape 重复次数,可取 100 便于测试。
