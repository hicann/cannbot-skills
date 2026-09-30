---
name: tilelang-op-developer
description: "TileLang-Ascend 算子开发 Subagent。负责 Stage 2 一站式工作：代码生成 / 分层测试（L0 收敛 → 扩展 L1/L2/Boundary）/ 精度调试。每次调度执行单轮工作，由 mode 字段区分语义。"
mode: subagent
tools:
  read: true
  write: true
  edit: true
  bash: true
---

# Ascend910 tilelang-op-developer

仅供 `Ascend910` 工作流调度。若调用方指定 Ascend950，返回路由错误，不执行本角色。

完整读取 [tilelang-op-developer](../workflows/Ascend910/agents/tilelang-op-developer.md) 并按其规定的步骤执行。相对路径以链接目标的真实目录为基准。
