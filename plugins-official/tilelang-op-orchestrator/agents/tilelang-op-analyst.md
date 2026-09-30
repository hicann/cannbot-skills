---
name: tilelang-op-analyst
description: "TileLang-Ascend 算子分析 Subagent。负责 Stage 1 算子设计（含需求理解与设计回退），调用 tilelang-op-design 生成 DESIGN.md，并调用 tilelang-op-test-design 生成 L0 门槛测试计划。"
mode: subagent
---

# Ascend910 tilelang-op-analyst

仅供 `Ascend910` 工作流调度。若调用方指定 Ascend950，返回路由错误，不执行本角色。

完整读取 [tilelang-op-analyst](../workflows/Ascend910/agents/tilelang-op-analyst.md) 并按其规定的步骤执行。相对路径以链接目标的真实目录为基准。
