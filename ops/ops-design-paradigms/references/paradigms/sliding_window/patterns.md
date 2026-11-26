<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式入口（语言中立）

> 本文件是 SlidingWindow 范式的**统一入口**：场景判定 + 语言路由 + 消费规则。具体知识分布在两个层次：
>
> - **通用知识层** `generic/` — 语言无关的算法、模型、方法论
> - **语言适配层** `adapters/<lang>/` — 强耦合具体开发语言的实现知识（当前提供 `ascendc`，基于 SIMT 编程模型）
>
> 范式路由（`routes.yaml` → 本文件）与语言路由（本文件 → 适配层）分离：范式路由由 `op.paradigms` 决定，语言路由由开发语言决定。

---

## 场景判定

满足以下条件的算子属于 SlidingWindow 范式：

> **前置约束（芯片 SIMT 编程能力）**：本范式适配层基于 SIMT 编程模型——AscendC 侧为 `__simt_vf__` 窗口枚举标量聚合（正向）与输入驱动反窗口 / argmax 路由（反向），标量原子仅原子散射分支使用。**目标芯片（`op.platform_constraints.supported_chips`）不支持 SIMT 编程能力时不进入本范式**——改走通用方法论另行论证非 SIMT 路线（如按窗口行搬运的向量化改造），并报告能力缺口。

1. **整数规则窗口聚合**——输出每个位置由输入中以规则参数（kernel/stride/pad/dilation）定义的整数窗口经聚合函数（max/avg/lp/sum，均为池化语义）计算，窗口地址由输出坐标**规则推导**（区别于 Interpolation 的浮点坐标映射）
2. **计算核心是窗口枚举 + 离散访存**——正向输出驱动枚举窗口内元素聚合，反向输入驱动反窗口枚举（或按路由信息散射），窗口内访问不连续（dilation / 跨行）时离散访存主导
3. **正反向两形态**——正向（WindowAggregate，输出驱动）与反向（GradRoute，梯度回传）是本范式两类一等算子类型：反向按梯度路由信息分为反窗口枚举 / argmax 路由 / 原子散射三分支（见 [generic/grad-routing.md](generic/grad-routing.md)）

覆盖算子域：池化类（max / avg / lp / adaptive / global pool 正反向），覆盖 **NCHW 与 NHWC** 两种 4D 布局维度（布局差异见 [generic/scene.md](generic/scene.md) 布局维度一节；SIMT 段的布局编码维度见 [adapters/ascendc/kernel-entrance.md](adapters/ascendc/kernel-entrance.md)）。

---

## 语言路由

按开发语言选择适配层。语言由调用方显式提供，或读取 `spec.yaml` 的 `op.language`；缺失或未知时先报告输入缺口，不选择语言适配层。

| 开发语言 | 加载内容 | 入口 |
|------|------|------|
| `ascendc` | 通用知识层 + AscendC SIMT 适配层 | [adapters/ascendc/index.md](adapters/ascendc/index.md) |
| 其他语言（含 `dsl`） | 通用知识层；适配层待确认 | 报告当前没有对应的语言适配层，不读取其他语言的 API 或模板 |

**互斥规则**：一次任务只加载命中语言的适配层，其他语言的适配层不加载、不参考其 API 与代码模板。

---

## 通用知识层索引

语言无关的算法与模型，按需展开（完整检索表见 [generic/index.md](generic/index.md)）：

| 文档 | 覆盖 |
|------|------|
| [generic/scene.md](generic/scene.md) | 准入条件、窗口术语、布局维度（NCHW/NHWC）、正向池化聚合族与反向路由三分支、数据流骨架、覆盖形态与口径说明 |
| [generic/capability-contract.md](generic/capability-contract.md) | 能力谓词定义（通用层算法引用的语言/硬件能力参数） |
| [generic/window-math.md](generic/window-math.md) | 窗口边界数学、pad 裁剪规则、dilation 起点推进、空窗语义、反窗口区间公式 |
| [generic/grad-routing.md](generic/grad-routing.md) | 反向梯度路由决策树（反窗口枚举 / argmax 路由 / 原子散射）、零原子论证、argmax 契约 |
| [generic/splitting.md](generic/splitting.md) | 正向输出驱动 / 反向输入驱动坐标系、grid-stride、线程分档、坐标分解 |
| [generic/verification.md](generic/verification.md) | golden 口径、精度分档、正反向边界用例表（空窗 / 负起点 / NaN 传播 / argmax 平局） |

---

## 消费规则

1. **按需加载**：先读对应语言适配层的 index，再展开当前问题需要的叶子文档。
2. **叶子文档限额**：单个子问题最多展开 5 个叶子文档，仍未闭环则拆分为更小的子问题
3. **能力真值**：通用层算法中的能力谓词，以命中语言适配层的 capability-mapping 填值为准（ascendc: [adapters/ascendc/capability-mapping.md](adapters/ascendc/capability-mapping.md)），其中标注"待验证"的谓词须在产出中声明假设并给出验证义务
4. **实现依据**：编码前对照适配层的内嵌模板包（max_pool_with_argmax 正向 / max_pool_grad_with_argmax 反向，SIMT 单路线自含组织）与登记的基准算子（max_pool_with_argmax_v3 / max_pool_grad_with_argmax_v3 / avg_pool / avg_pool_v2_grad）核实 API 与组织方式；参考实现中的具体算法与参数须按当前算子规格重新核定
5. **设计依据**：根据规格读取通用算法与对应语言约束，给出正向/反向类型判定、窗口数学、反向路由分支、切分和验证结论；输出格式由调用方指定
6. **范式优先级**：本范式的专用知识优先于 agent 通用知识；同一设计点结论冲突时，以本范式指向的必读文件为准
