---
name: repo-knowledge
description: 仓库领域知识，提供本仓算子涉及的领域标准、概念与背景。触发：需要算子族设计方法论、目标芯片架构概念、精度领域标准等背景知识时加载。
---

# 仓库领域知识

本 skill 提供本仓算子涉及的领域标准、概念与背景，作为方案设计与开发时的领域依据。

本仓默认采用 direct launch 算子工程架构（bisheng 编译 kernel + g++ 编译 plugin + torch.library 注册），产出可被外部评测集评测的算子工程。direct launch 算子的评测契约（算子原型/golden/用例由评测集提供、schema 对齐、三阶段评测、HAP 性能评分）是本仓领域知识的一部分，详见 references。

## 领域知识路由

| 领域 | 加载 | 提供 |
|------|------|------|
| **direct launch 评测契约** | [references/evaluation-contract.md](references/evaluation-contract.md) | 评测集结构（算子原型 / golden / cases / 性能基线）、提交工程契约（wheel + torch.ops schema 对齐）、三阶段评测流程（编译/精度/性能）、HAP 评分公式与取值边界、提交反作弊红线总览 |
| 算子族设计方法论 | `ascendc-tiling-design` | 各算子族（Reduction / Elementwise / Broadcast / Conversion / MatMul 等）的场景路由、Tiling 策略、Buffer 规划、数据流方法论 |
| 目标芯片架构 | `npu-arch` | 芯片型号 / SocVersion / NpuArch 概念、`--npu-arch` 合法值（ascend910b / ascend910_93 / ascend950）、架构特性 |
| 代码架构选型 | `ascendc-regbase-best-practice`（RegBase 路线适用条件、约束与陷阱）、`ascendc-simt-best-practices`（SIMT 路线编程范式与 API 边界）；MemBase 默认路线的适用性依 `ascendc-tiling-design` 判断，目标芯片对各架构的支持情况依 `npu-arch` 判断 | 各代码架构的适用条件与代价，作为架构选型推荐的判断依据 |
| 精度领域标准 | `ops-precision-standard` | 各 dtype 的精度容差标准与判定口径；**外部评测以评测集内置容差为最终裁定**，本标准作为开发期自检参照 |
