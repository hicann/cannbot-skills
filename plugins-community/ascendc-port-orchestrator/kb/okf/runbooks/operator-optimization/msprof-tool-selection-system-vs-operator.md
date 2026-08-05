---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "msprof（系统级）vs msopprof（算子级）：先系统后算子"
description: "先用 msprof 找系统级瓶颈；一旦瓶颈定位到某个算子，再用 msopprof / msprof op 展开该单算子内部（指令/pipe 级）。"
confidence: single_run
original_id: MSPROF_AGENT_GUIDE.md#0-msprof-vs-msopprof
classified_by: llm-assisted
timestamp_inferred: true
tags: [optimization, msprof, msopprof, tool-selection]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

两个工具对象不同、功能略有重叠，选哪个由"想看系统级还是单算子内部"决定：

- **`msprof` = 系统级 profiling（最常用，默认先用）** —— 查看整个执行的系统级时间线/瓶颈：host-device 调度、算子间隙、内存搬运、AI Core 全局利用率。
- **`msopprof` = 算子（operator）级 profiling** —— 对单个算子内部展开指令/pipe 级细节。`msopprof` 是 CANN 包里的可执行文件，接口用法与 **`msprof op`** 一致（同一算子级能力的两种调用名）。

**工作流**：先 `msprof` 找系统级瓶颈 → 若瓶颈定位到某个算子 → 再用 `msopprof` 对那个单算子展开看内部。

**context 纪律同样适用于 msopprof**：context-安全分层读取策略主要针对 `msprof` 系统级输出，但 `msopprof` 单算子展开同理——只读聚合 csv，绝不读二进制 trace。

> 校核来源：Owner 2026-06-24 澄清；对照官方 CANN/MindStudio/Huawei 文档多源核实（msprof=系统级 / msopprof=算子级）。
