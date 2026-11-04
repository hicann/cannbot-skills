<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式入口（语言中立）

> 本文件是 ControlFlow 范式的**统一入口**：场景判定 + 语言路由 + 消费规则。具体知识分布在两个层次：
>
> - **通用知识层** `generic/` — 语言无关的信号模型、落核决策、验证方法
> - **语言适配层** `adapters/<lang>/` — 强耦合具体开发语言的实现知识（当前提供 `ascendc` 与 `dsl`（CANNBotDSL），实现路线统一为**设备计算核（AICore）**：SIMT 为主，SIMD/混合按信号类型选型；断言失败路径信号化，不依赖宿主上报通道）
>
> 范式路由（`routes.yaml` → 本文件）与语言路由（本文件 → 适配层）分离：范式路由由 `op.paradigms` 决定，语言路由由开发语言决定。

---

## 场景判定

满足以下条件的算子属于 ControlFlow 范式（**状态驱动控制信号算子**）：

> **前置约束（芯片 SIMT 编程能力）**：本范式设备核主体形态为 SIMT——get / sleep / assert / case_condition 型均为 SIMT 单线程退化 kernel，clear 型混合段含 SIMT VF；仅 alloc 型（标量写零 + DCCI，无 SIMT VF）不依赖 SIMT。**目标芯片（`op.platform_constraints.supported_chips`）不支持 SIMT 编程能力时，除 alloc 型外不进入本范式**——信号算子改走通用方法论与宿主侧方案论证，并报告能力缺口。

1. **产出/副作用是控制面信号**——输出值或硬件状态供控制决策消费（训练循环的动态 loss scale / 溢出跳步、宿主报错中止、分支选择、时序控制），而非数据面计算结果
2. **信号语义由消费方定义**——算子只负责信号的生产/读取/清除/求值
3. **数据面退化**——输出固定微小 shape、幂等常量写，或无输出（纯副作用）

五类信号：**数值健康状态**（FloatStatus 家族）、**断言判定**（assert）、**时序控制**（sleep）、**条件求值**（case_condition）、**描述符更新**（update_tensor_desc）。

**边界**：图级编排算子（If / While / Case / For / Switch / Merge / Enter / Exit / partitioned_call 等，`REG_OP + .GRAPH` 子图属性、无 kernel）**不属本范式**（图编译域）；与 DynamicShape / Stateful / MaskPredicate 的交界见 [generic/scene.md](generic/scene.md) §5。

---

## 语言路由

按开发语言选择适配层。语言由调用方显式提供，或读取 `spec.yaml` 的 `op.language`；缺失或未知时先报告输入缺口，不选择语言适配层。

| 开发语言 | 加载内容 | 入口 |
|------|------|------|
| `ascendc` | 通用知识层 + AscendC 适配层（设备核 SIMT 为主，SIMD/混合按信号类型选型） | [adapters/ascendc/index.md](adapters/ascendc/index.md) |
| `dsl`（CANNBotDSL） | 通用知识层 + DSL 适配层（设备核 SIMT VF 为主，SIMD/混合按信号类型选型） | [adapters/dsl/index.md](adapters/dsl/index.md) |
| 其他语言 | 通用知识层；适配层待确认 | 报告当前没有对应的语言适配层，不读取其他语言的 API 或模板 |

**互斥规则**：一次任务只加载命中语言的适配层，其他语言的适配层不加载、不参考其 API 与代码模板。

---

## 通用知识层索引

语言无关的算法与模型，按需展开（完整检索表见 [generic/index.md](generic/index.md)）：

| 文档 | 覆盖 |
|------|------|
| [generic/scene.md](generic/scene.md) | 准入条件、五类信号、状态生命周期、图级编排边界声明 |
| [generic/capability-contract.md](generic/capability-contract.md) | 能力谓词定义（含降级态登记规则） |
| [generic/signal-model.md](generic/signal-model.md) | 信号契约表、幂等不变量 I3、副作用可观测性 I4、降级兼容契约 I5、时序/断言语义 ★核心 |
| [generic/kernel-placement.md](generic/kernel-placement.md) | 设备核形态选型决策树（含无宿主上报通道负知识）、退化 kernel 形态、二进制形态 |
| [generic/verification.md](generic/verification.md) | 信号契约验证（闭环/幂等/降级/失败路径）、用例矩阵、平台迁移义务 |

---

## 消费规则

1. **按需加载**：先读对应语言适配层的 index，再展开当前问题需要的叶子文档。
2. **叶子文档限额**：单个子问题最多展开 5 个叶子文档，仍未闭环则拆分为更小的子问题
3. **形态选型强制登记（CF-R1）**：本范式全部信号类型落设备计算核（AICore），设计必须在首个设计文档（Kernel/DevView 侧）登记**形态选型决策 + 选型依据行**（适配层形态选型表 [adapters/ascendc/index.md](adapters/ascendc/index.md) §3 或落核决策树行）——按信号类型机械匹配（SIMT / SIMD / 混合 + 输出契约），禁止按参考算子熟悉度、既有实现惯性或上游产物（REQUIREMENTS 等）的参照倾向选形态；偏离选型表形态即范式违规
4. **能力真值**：通用层算法中的能力谓词，以命中语言适配层的 capability-mapping 填值为准（ascendc: [adapters/ascendc/capability-mapping.md](adapters/ascendc/capability-mapping.md)）；**降级态真值与"逐平台实测"义务必须原样传递到产出**
5. **负知识**：本范式算子无 tiling 设计空间（退化 kernel）——不套用常规 tiling 方法论；设计重心是信号契约（幂等 I3 / 副作用 I4 / 降级 I5）与落核决策
6. **实现依据**：编码前对照适配层登记的基准算子（FloatStatus 家族 / assert / case_condition / sleep / update_tensor_desc）核实形态；参考实现中的副作用通道参数须按目标平台重新论证
7. **范式优先级**：本范式的专用知识（含路线选型 CF-R1）优先于 agent 通用知识与上游产物参照倾向；同一设计点结论冲突时，以本范式指向的必读文件为准
