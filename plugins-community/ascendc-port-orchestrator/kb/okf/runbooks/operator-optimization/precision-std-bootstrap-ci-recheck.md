---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "复检流程:Bootstrap 置信区间判定(PARTIAL→PASS 官方路径)"
description: "单用例不达标时换种子重跑 N 次(推荐 1000),Bootstrap 重采样求中位数 95% CI;N<200 直接失败,CI_Lower>1.0 判定系统性恶化。"
confidence: single_run
original_id: PRECISION_STANDARD_v2.1.md#4.5.2 复检说明 / 5.3.4 置信区间计算
classified_by: llm-assisted
timestamp_inferred: true
tags: [ascendc, optimization, precision, bootstrap, confidence-interval]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策
单用例不满足通过标准时启动复检,用基于统计中位数置信区间的方法判定,避免数值不稳定性带来的误判(统计学升级 PARTIAL→PASS 的官方路径):
- 采样:更换不同随机种子重新生成输入并执行 N 次(推荐 1000)。
- Bootstrap 重采样:对原始误差比值数据有放回重采样(§5.3.4 为 2000 次),每次计算中位数(np.median),得到中位数集合。
- 置信区间:取 2.5% 与 97.5% 分位数构成 95% CI = [CI_Lower, CI_Upper]。N=1000 例:CI_Lower = 第 25 小的中位数,CI_Upper = 第 976 小的中位数。
- 小样本熔断:若 N < 200,重采样分布可能严重失真,置信区间不可靠,直接判定不通过。
- 判定规则:若 CI_Lower > 1.0(置信区间完全位于 1.0 右侧),说明有统计学证据表明 NPU 存在系统性精度恶化,判定不通过;否则通过。方法为非参数假设检验,无需对数据分布做特定假设。
