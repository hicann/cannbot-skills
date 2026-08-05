---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "从 op_summary_*.csv 抽取 per-kernel 管线利用率（列号 + 动态表头）"
description: "PipeUtilization 下 op_summary 关键列号相对固定（Op Name/Task Duration/vec/scalar/mte2/mte3）；L2Cache 等列号随表头变，务必先 grep 表头再定列号。"
confidence: single_run
original_id: MSPROF_AGENT_GUIDE.md#4-op_summary-列号对照
classified_by: llm-assisted
timestamp_inferred: true
tags: [optimization, msprof, op-summary, awk]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

`op_summary_*.csv` 逐 task 数据用 `grep` 过滤 kernel + `awk -F','` 抽列，输出压到 <50 行。

**PipeUtilization metrics 列号对照（1-based，来自 §4 表）**：

| 列号 | 字段 | | 列号 | 字段 |
|------|------|--|------|------|
| 5 | Op Name | | 42 | **aiv_scalar_ratio** |
| 10 | Task Duration(us) | | 43 | aiv_mte2_time(us) |
| 37 | aiv_time(us) | | 44 | **aiv_mte2_ratio** |
| 39 | aiv_vec_time(us) | | 45 | aiv_mte3_time(us) |
| 40 | **aiv_vec_ratio** | | 46 | **aiv_mte3_ratio** |
| 41 | aiv_scalar_time(us) | | | |

**注意列号不一致**：源文档在 §3（Level 2）另给过一套列号（`Op Name=4`, `Task Duration=9`, `aiv_vec_ratio=37`, `scalar=39`, `mte2=41`），与 §4 的 1-based 表在 ratio 列上对不上（相差约 2 列）。**不要硬记列号——先用下面的动态表头技巧核对实际 header 再定列。**

**动态表头技巧（L2Cache 等 group，列号随 header 变，Batch 14-6 验证）**：
```bash
# 先打印带序号的表头，grep 出目标列的列号
head -1 op_summary_*.csv | tr ',' '\n' | cat -n | grep -i 'l2\|cache'
# 再按查到的列号 awk 抽 l2_rd_hit / l2_wr_hit / l2_wr_miss
```

**per-kernel 聚合示例（平均 dur/vec/scl/mte）**：
```bash
grep -v '^Device' PROF_*/mindstudio_profiler_output/op_summary_*.csv | \
  awk -F',' '{
    name=$5; dur=$10; vec=$40; scl=$42; mte=$44;   # 列号以实际 header 为准
    sum[name]+=dur; cnt[name]++; svec[name]+=vec; sscl[name]+=scl; smte[name]+=mte
  } END {
    for(n in sum) printf "%s: avg_dur=%.1fus vec=%.3f scl=%.3f mte=%.3f (n=%d)\n",
      n, sum[n]/cnt[n], svec[n]/cnt[n], sscl[n]/cnt[n], smte[n]/cnt[n], cnt[n]
  }'
```
