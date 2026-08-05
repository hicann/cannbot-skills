---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "msprof 完整优化 workflow（采集 → 定位 → 深入 → 判断 → 重 profile）"
description: "端到端闭环：采集一轮 → op_statistic 定位瓶颈 kernel(<10 行) → op_summary grep/awk 抽 ratio(<50 行) → 判断优化方向 → 实施 → 重新 profile。"
confidence: single_run
original_id: MSPROF_AGENT_GUIDE.md#5-完整-workflow
classified_by: llm-assisted
timestamp_inferred: true
tags: [optimization, msprof, workflow, iteration-loop]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

把分层读取、metrics 选择、瓶颈判断串成一个可迭代的闭环：

1. **采集**：`msprof --output=/tmp/msprof_X -- ./benchmark_command`（首轮用默认 PipeUtilization）。
2. **快速定位瓶颈（<10 行输出）**：`cat PROF_*/mindstudio_profiler_output/op_statistic_*.csv`，按 Ratio% 找出占时间最多的 kernel。
3. **深入分析（grep 过滤后 <50 行）**：`grep "kernel_name" op_summary_*.csv | awk 提取 vec/scalar/mte ratio`。
4. **判断 → 实施优化 → 重新 profile**：按瓶颈判断速查表选优化方向，改完后回到 Step 1 再跑一轮对比。

每一轮只读聚合 CSV（Level 1/2），二进制 trace 永不读；需要换视角时按 metrics-group 递进流程切 `--aic-metrics` 再跑。
