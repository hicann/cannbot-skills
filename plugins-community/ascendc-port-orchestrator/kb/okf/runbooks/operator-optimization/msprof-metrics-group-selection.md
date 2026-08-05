---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "msprof metrics group 选择与递进采集流程"
description: "PipeUtilization 默认首跑；按发现分支到 Memory / L2Cache / MemoryUB 等；不同 metrics group 不能同时采集。"
confidence: single_run
original_id: MSPROF_AGENT_GUIDE.md#4-metrics-group
classified_by: llm-assisted
timestamp_inferred: true
tags: [optimization, msprof, metrics-group, aic-metrics]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

`--aic-metrics` / `--aiv-metrics` 参数控制采集内容，**不同 group 不能同时采集**——一次只能选一个，所以要按需分轮跑。

**Group 目录（问什么问题选哪个）**：
| Group | 关键指标 | 用途 |
|-------|---------|------|
| **PipeUtilization**（默认，首选） | `aiv_vec_ratio` / `aiv_scalar_ratio` / `aiv_mte2_ratio` / `aiv_mte3_ratio` | 判断瓶颈在计算、标量、DMA 读还是 DMA 写 |
| Memory | HBM 带宽利用率 | 判断是否带宽瓶颈 |
| **L2Cache** | L2 读/写命中率、miss 率、eviction 计数 | 判断 L2 缓存效果（Batch 14 验证有效） |
| MemoryUB | UB 读写带宽 | 判断 UB 利用率 |
| ArithmeticUtilization | MAC 利用率 | 矩阵运算密集型 kernel |
| ResourceConflictRatio | 资源冲突 | 排查 bank conflict |

**递进采集流程（先粗后细，按上一轮发现决定下一轮跑什么）**：
1. 先用默认 **PipeUtilization** 跑一次。
2. 若 `vec_ratio ≈ 1.0` 且性能仍差 → 跑 **Memory** 检查带宽。
3. 若 `scalar_ratio` 高 → 间接寻址/控制流瓶颈（GetValue GM 标量读）。
4. 若 `mte3_ratio` 高 → SetAtomicAdd 写瓶颈（Batch 14-6 确认）。
5. 若需区分"读缓存 vs 写缓存" → 跑 **L2Cache**（Batch 14-6 的关键手段）。
