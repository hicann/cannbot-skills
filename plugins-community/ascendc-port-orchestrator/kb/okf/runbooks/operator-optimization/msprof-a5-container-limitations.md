---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "A5 容器内 msprof 已知限制与 workaround"
description: "容器内 msprof 可能报 \"Please check the driver package\"（容器权限不足或 driver 版本不匹配）；重启容器或检查 driver 挂载。"
confidence: single_run
original_id: MSPROF_AGENT_GUIDE.md#7-已知限制
classified_by: llm-assisted
timestamp_inferred: true
tags: [optimization, msprof, a5, container]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

**A5 容器环境的已知限制**：

- 在 `can_torch_cann_device_1` 容器内运行 msprof 可能报：`Running profiling failed. Please check the driver package`。
  - **原因**：容器权限不足，或 driver 版本不匹配。
  - **Workaround**：之前的 session 成功跑过——可能需要重启容器或检查 driver 挂载。
- `--analyze` 和 `--parse` 模式需要 Python 3.7.5+。

遇到 profiling failed 时先排除容器权限 / driver 挂载，而不是先怀疑命令写错。
