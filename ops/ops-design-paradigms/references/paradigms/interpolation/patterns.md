<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式入口（语言中立）

> 本文件是 Interpolation 范式的**统一入口**：场景判定 + 语言路由 + 消费规则。具体知识分布在两个层次：
>
> - **通用知识层** `generic/` — 语言无关的算法、模型、方法论
> - **语言适配层** `adapters/<lang>/` — 强耦合具体开发语言的实现知识（当前提供 `ascendc`，基于 SIMT 编程模型；`dsl`（CANNBotDSL），SIMT 数据面 + AI CPU 调度）
>
> 范式路由（`routes.yaml` → 本文件）与语言路由（本文件 → 适配层）分离：范式路由由 `op.paradigms` 决定，语言路由由开发语言决定。

---

## 场景判定

满足以下条件的算子属于 Interpolation 范式：

> **前置约束（芯片 SIMT 编程能力）**：本范式两语言适配层均基于 SIMT 编程模型——AscendC 侧为 `__simt_vf__` 浮点坐标驱动的离散 gather 采样 + 标量直写（grad 路径为标量原子），DSL 侧为 `vf(mode="simt")` + 线程级坐标映射。**目标芯片（`op.platform_constraints.supported_chips`）不支持 SIMT 编程能力时不进入本范式**——改走通用方法论另行论证非 SIMT 路线（如按输出块预计算源坐标表的向量化改造），并报告能力缺口。

1. **输出值由输入在浮点坐标处的加权采样决定**——每个输出位置先经坐标映射得到输入域的浮点源坐标（或显式索引），再取离散邻居加权求和，即插值 / 重采样 / 区域采样
2. **计算核心是坐标映射 + 索引驱动的离散访存**——源坐标在运行前不可知，邻居地址按运行时计算值离散 gather，而非规则向量计算
3. **正向输出逐位独立**——每个输出元素只依赖自己的坐标映射与邻居取数，无跨输出位置累积；梯度反向（scatter-add 散射）复用 ScatterUpdate / Sparse 范式的冲突与同步知识

三个算子族：**RegularResize**（规则网格重采样，如 resize bilinear / bicubic、upsample 正反向）、**IndexedInterpolate**（索引加权插值，如 three_interpolate）、**RoiSample**（区域采样，如 crop_and_resize、roi_align_rotated、deformable_roi_pool）。与 SlidingWindow（pooling）、LayoutTransform（纯整数规则坐标变换）、Broadcast（无坐标映射）、ScatterUpdate（写散射为语义主体）的边界见 [generic/scene.md](generic/scene.md) §5。

---

## 语言路由

按开发语言选择适配层。语言由调用方显式提供，或读取 `spec.yaml` 的 `op.language`；缺失或未知时先报告输入缺口，不选择语言适配层。

| 开发语言 | 加载内容 | 入口 |
|------|------|------|
| `ascendc` | 通用知识层 + AscendC SIMT 适配层 | [adapters/ascendc/index.md](adapters/ascendc/index.md) |
| `dsl`（CANNBotDSL） | 通用知识层 + DSL SIMT 适配层（数据面 + 原子 + AI CPU 调度） | [adapters/dsl/index.md](adapters/dsl/index.md) |
| 其他语言 | 通用知识层；适配层待确认 | 报告当前没有对应的语言适配层，不读取其他语言的 API 或模板 |

**互斥规则**：一次任务只加载命中语言的适配层，其他语言的适配层不加载、不参考其 API 与代码模板。

---

## 通用知识层索引

语言无关的算法与模型，按需展开（完整检索表见 [generic/index.md](generic/index.md)）：

| 文档 | 覆盖 |
|------|------|
| [generic/scene.md](generic/scene.md) | 准入条件、插值术语、三算子族分类、数据流骨架、能力边界 |
| [generic/capability-contract.md](generic/capability-contract.md) | 能力谓词定义（通用层算法引用的语言/硬件能力参数） |
| [generic/coordinate-mapping.md](generic/coordinate-mapping.md) | 输出驱动坐标映射、对齐模式语义表、邻居分解与 clamp 防御、越界策略、高阶核 |
| [generic/weight-model.md](generic/weight-model.md) | 权重计算、累加精度方法论（golden 同序、补偿累加、非有限值分支、溢出防护） |
| [generic/splitting.md](generic/splitting.md) | 输出元素坐标系、grid-stride 映射、快除坐标分解、核间两步法、grad 散射切分 |
| [generic/verification.md](generic/verification.md) | golden 口径、精度分档、边界用例表（对齐模式退化 / 越界 / 极值 / 值域） |

---

## 消费规则

1. **按需加载**：先读对应语言适配层的 index，再展开当前问题需要的叶子文档。
2. **叶子文档限额**：单个子问题最多展开 5 个叶子文档，仍未闭环则拆分为更小的子问题
3. **能力真值**：通用层算法中的能力谓词，以命中语言适配层的 capability-mapping 填值为准（ascendc: [adapters/ascendc/capability-mapping.md](adapters/ascendc/capability-mapping.md)），其中标注"待验证"的谓词须在产出中声明假设并给出验证义务
4. **实现依据**：编码前对照适配层登记的基准算子（three_interpolate / sync_resize_bilinear_v2 / crop_and_resize）核实 API 与组织方式；参考实现中的具体算法与参数须按当前算子规格重新核定
5. **设计依据**：根据规格读取通用算法与对应语言约束，给出算子族判定、坐标映射路径、权重与累加精度、切分和验证结论；输出格式由调用方指定
6. **范式优先级**：本范式的专用知识优先于 agent 通用知识；同一设计点结论冲突时，以本范式指向的必读文件为准
