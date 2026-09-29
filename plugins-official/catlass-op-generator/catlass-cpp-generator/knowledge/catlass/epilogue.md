---
type: "catlass"
title: "Epilogue"
description: "Epilogue 的 CATLASS C++ 组件职责、组合边界和源码核对路径。"
tags: ["catlass-cpp", "epilogue", "component"]
status: "draft"
generated: {"by": "process:catlass-cpp-knowledge-bootstrap", "at": "2026-09-18T00:00:00Z"}
verified: [{"by": "process:repository-source-check", "at": "2026-09-18T00:00:00Z"}]
sources: [{"id": "catlass-component-guidance", "resource": "git:cann/cannbot-skills-dev@f89b066b9ea80c3bd72b071e98a3cfdcf1e371ca:ops/catlass-op-develop/SKILL.md", "title": "CATLASS C++ development skill component assembly guidance", "kind": "repository"}]
---
# 接口与概念

`Epilogue` 在工作流中的职责是：消费主计算结果并完成写回、转换或已设计的后处理。与相邻组件的边界为：不能隐藏算子专用 Stage 依赖；与 Fixpipe/AIV、workspace 和输出 contract 对齐。[^catlass-component-guidance]

# 用法

先在当前工作区记录 CATLASS 提交，再从 `catlass/include/`、`catlass/docs/` 和
`catlass/examples/` 搜索 `Epilogue`。以实际偏特化、Params 和 example 为准选择组件。

# 代码模式

设计文档只记录已在固定源码确认的 namespace、模板实参、Params 构造和 device 调用链。
若无法在当前提交定位定义和可运行 example，停止组装并报告来源缺口。

# 约束

本 concept 不固定跨版本模板签名。不得从名称猜测参数顺序，不得把可编译等同于正确或高性能，
不得绕过设计规定的 CATLASS 组件边界。

# 失败表现

常见表现包括偏特化不匹配、Params 字段或 layout 错误、架构标签不兼容、同步生命周期不闭合，
以及能够编译但运行挂起、结果错误或性能退化。

# 验证方法

记录固定源码位置和 example；完成编译、最小运行、CPU golden 精度比较，并按任务要求采集 profiler。
对实际采用的接口在 `docs/design.md` 标注版本和证据路径。

[^catlass-component-guidance]: git:cann/cannbot-skills-dev@f89b066b9ea80c3bd72b071e98a3cfdcf1e371ca:ops/catlass-op-develop/SKILL.md
