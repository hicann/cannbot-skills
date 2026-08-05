---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "msprof 采集命令模板与关键陷阱"
description: "用 --application= 而非 -- command（后者某些 CANN 版本不工作）；LD_LIBRARY_PATH 必须含 kernel .so 路径；--task-time=l2 才出详细 timeline。"
confidence: single_run
original_id: MSPROF_AGENT_GUIDE.md#6-9-运行命令模板
classified_by: llm-assisted
timestamp_inferred: true
tags: [optimization, msprof, command, ld-library-path]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

**基础采集（PipeUtilization 首选）**：
```bash
MSPROF=/usr/local/Ascend/cann-9.0.0/tools/profiler/bin/msprof   # 路径随 CANN 版本，以实际为准
export LD_LIBRARY_PATH=/root/a5_ops/build/lib:/usr/local/Ascend/cann-9.0.0/x86_64-linux/lib64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:$LD_LIBRARY_PATH
$MSPROF --output=/tmp/msprof_out --application='./benchmark' --aic-metrics=PipeUtilization
```

**切换 metrics group**（只改 `--aic-metrics`，见对应 group 卡）：
```bash
$MSPROF --output=/tmp/msprof_l2  --application='./benchmark' --aic-metrics=L2Cache
$MSPROF --output=/tmp/msprof_mem --application='./benchmark' --aic-metrics=Memory
$MSPROF --output=/tmp/msprof_ub  --application='./benchmark' --aic-metrics=MemoryUB
```

**详细 task timing（L2 级 timeline）**：加 `--task-time=l2`（默认 `--task-time=on` 只有 l0/l1）：
```bash
$MSPROF --output=/tmp/msprof_detail --application='./benchmark' --aic-metrics=PipeUtilization --task-time=l2
```

**导出 CSV（若默认未生成）**：
```bash
$MSPROF --export=on --output=/tmp/msprof_out --type=text --summary-format=csv
```

**关键陷阱**：
- 用 **`--application='./cmd'`** 而不是 `-- ./cmd`——后者在某些 CANN 版本不工作。
- **`LD_LIBRARY_PATH` 必须包含 kernel `.so` 路径**（如 `/root/a5_ops/build/lib`），外加 CANN `x86_64-linux/lib64` 与 driver `lib64`，否则采集失败。
- `--analyze` / `--parse` 模式需要 Python 3.7.5+。
