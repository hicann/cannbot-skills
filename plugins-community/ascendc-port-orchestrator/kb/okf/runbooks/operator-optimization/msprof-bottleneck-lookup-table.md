---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "msprof 瓶颈判断速查表 → 优化方向"
description: "由 vec/scalar/mte2 ratio 与 HBM 带宽读数映射到瓶颈成因与对应优化手段（减计算量 / 常驻核心分发 / 数据复用）。"
confidence: single_run
original_id: MSPROF_AGENT_GUIDE.md#9-瓶颈判断速查表
classified_by: llm-assisted
timestamp_inferred: true
tags: [optimization, msprof, bottleneck, tuning]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

读到某组指标后，如何判断瓶颈成因并选优化方向：

| 现象 | 指标来源 | 含义 | 优化方向 |
|------|---------|------|---------|
| `vec_ratio ≈ 1.0` 但性能差 | PipeUtilization | 计算已满，瓶颈在算法 | 减少计算量（如排序消除 atomicAdd） |
| `scalar_ratio > 0.2` | PipeUtilization | 间接寻址/控制流开销大 | 常驻核心分发（P-P22）、减少分支 |
| `mte2_ratio > 0.5` | PipeUtilization | DMA 搬运占主导 | 数据复用（UB 缓存）、减少搬运次数 |
| HBM 带宽 < 10% | Memory | 带宽未饱和 | 不是带宽瓶颈，转看其他指标 |
| HBM 带宽 > 80% | Memory | 带宽饱和 | 减少数据量、提高计算密度 |

**项目实证（关键数据）**：
- **compute-bound 时"常驻核心分发"无效**：SG forward xlarge `vec_ratio=0.69, scalar_ratio=0.31` → 常驻核心分发得 **1.86x**；SG backward xlarge `vec_ratio=0.989` 属 compute-bound → persistent **无效**。即"常驻分发只在 vec 未满/scalar 偏高时有用"。
- **vec 满 → 从算法端减计算量**：Pooling A baseline backward `vec_ratio=1.0`、atomicAdd 15.9 cycles → sorted-edge 寄存器累加，**-81%**。
- **排序后不需要超订**：Pooling nblk sweep `nblk=56 vs 448` → bwd **-28%**、fwd **+14%**（排序后不需要超订）。
