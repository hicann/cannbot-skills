---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "用 aog-researcher agent 做多轮 msprof 深度分析（保护主 context）"
description: "复杂优化交给 researcher sub-agent 连跑 3-4 轮 msprof（多 metrics group）交叉分析，返回结构化诊断+假设，避免 profiling 数据污染主 agent context。"
confidence: single_run
original_id: MSPROF_AGENT_GUIDE.md#8-aog-researcher-深度分析
classified_by: llm-assisted
timestamp_inferred: true
tags: [optimization, msprof, sub-agent, context-safety]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

**何时用**：当需要跑多轮 msprof（不同 metrics group）并交叉分析时，把活交给 `aog-researcher` sub-agent，而不是在主 agent 里跑。

**收益**：
1. 保护主 agent 的 context 不被 profiling 数据污染。
2. researcher agent 可以连续跑 3-4 轮 msprof 并综合分析。
3. 返回结构化的诊断报告和假设建议。

**Batch 14-6 验证的深度分析 workflow**：
1. PipeUtilization 跑基线 kernel + 待优化 kernel。
2. L2Cache 跑待优化 kernel（看缓存命中）。
3. 对比两个 kernel 的 4 管线利用率差异。
4. 提取 L2 读/写 hit/miss 数据。
5. 综合诊断 → 输出结构化假设。

**关键发现模板（示例）**：
> "Kernel X 的 `mte3_ratio=0.88` 证明瓶颈在 MTE3 原子写。L2 写 miss=8298 说明原子写导致 L2 thrashing。对比 Kernel Y 的 `mte3_ratio=0.35` 和 L2 写 miss=0。结论：X 的原子写模式在 A5 上比 Y 的直接写慢 N 倍。"

**规模阈值**：当前小项目（2 个算子、<10 个 kernel）主 agent 直接 grep 足够。未来 profiling 上千 kernel 的大模型训练时，可用 haiku sub-agent 做 CSV 过滤+聚合（输入 op_summary CSV + 查询条件，输出 <20 行摘要）。
