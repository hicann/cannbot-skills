---
name: repo-knowledge
description: 仓库领域知识，提供本仓算子涉及的领域标准、概念与背景。触发：需要算子族设计方法论、目标芯片架构概念、精度领域标准等背景知识时加载。
---

# 仓库领域知识

本 skill 提供本仓算子涉及的领域标准、概念与背景，作为方案设计与开发时的领域依据。

基类默认实现不内置特定算子仓的私有领域知识，而是路由到跨仓通用的领域 skill；各算子仓 override 本 skill 时，在此注入本仓专用的领域标准与背景（引导读取仓内文档原文，不硬编码）。

> 本目录是基类**默认领域知识**（属 virtual），各算子仓可 override 为本仓专用领域知识；override 时保持 `name:` 与逻辑名 `repo-knowledge` 不变。

## 领域知识路由

| 领域 | 加载 | 提供 |
|------|------|------|
| 算子族设计方法论 | `ascendc-tiling-design` | 各算子族（Reduction / Elementwise / Broadcast / Conversion / MatMul 等）的场景路由、Tiling 策略、Buffer 规划、数据流方法论 |
| 目标芯片架构 | `npu-arch` | 芯片型号 / SocVersion / NpuArch 概念、`--npu-arch` 合法值、架构特性 |
| 代码架构选型 | `ascendc-regbase-best-practice`（RegBase 路线适用条件、约束与陷阱）、`ascendc-simt-best-practices`（SIMT 路线编程范式与 API 边界）；MemBase 默认路线的适用性依 `ascendc-tiling-design` 判断，目标芯片对各架构的支持情况依 `npu-arch` 判断 | 各代码架构的适用条件与代价，作为架构选型推荐的判断依据 |
| 精度领域标准 | `ops-precision-standard` | 各 dtype 的精度容差标准与判定口径 |

上述均按逻辑名引用，由 init 从共享 `ops/` 绑定。
