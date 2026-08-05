---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "msprof 输出的 context-安全分层读取（绝不读二进制 trace）"
description: "msprof 原始输出可达数 GB 二进制 trace；agent 只读 mindstudio_profiler_output 下的聚合 CSV，按 Level 1→3 逐层深入，Level 4 二进制永不读。"
confidence: single_run
original_id: MSPROF_AGENT_GUIDE.md#1-3-分层读取策略
classified_by: llm-assisted
timestamp_inferred: true
tags: [optimization, msprof, context-safety, csv]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

**核心问题**：msprof 原始输出可达数 GB（二进制 trace），而 agent context 有效空间仅 ~200K tokens。直接读原始数据不可行，必须分层过滤。

**输出结构（读哪个、不读哪个）**：
- `PROF_*/device_0/data/*`（`stars_soc_profile.data.*` / `ffts_profile.data.*`）= 二进制 trace，数十 MB~数 GB —— **NEVER READ**。
- `PROF_*/device_0/sample.json` = 小 JSON，设备信息。
- `PROF_*/mindstudio_profiler_output/` = **agent 只读这个目录**（聚合 CSV）。

**分层读取（由浅入深，够用即停）**：
- **Level 1（首选，<10 行）** `op_statistic_*.csv`：聚合统计，格式 `OP Type, Core Type, Count, Total Time(us), Min/Avg/Max Time(us), Ratio(%)`。按 Ratio% 排序一眼看出瓶颈 kernel。命令 `cat PROF_*/mindstudio_profiler_output/op_statistic_*.csv`。
- **Level 2（按需，<50 行）** `op_summary_*.csv`：逐 task，含 `aiv_vec_ratio` / `aiv_scalar_ratio` / `aiv_mte2_ratio` 等。**用 grep 过滤特定 kernel + awk 抽关键列，绝不全读。**
- **Level 3（按需）** `task_time_*.csv`：timeline，用于分析 kernel 间 gap、stream 并发、逐 task 精确 timing。数百~千行 CSV（非二进制，可 grep/awk），先 `head -1` + `wc -l`，不读全部内容。需要 `--task-time=l2` 才生成详细数据（默认 `--task-time=on` 只有 l0/l1 粒度）。
- **Level 4** `device_0/data/*`：二进制 trace，只能用 MindStudio GUI 或 msprof CLI 解析，**agent 绝不直接读取**。

**要点**：真正大的只有二进制 trace（永远不读）；CSV 最多几千行，grep 后很小。
