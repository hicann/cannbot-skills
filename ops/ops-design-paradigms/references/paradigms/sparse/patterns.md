<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式入口（语言中立）

> 本文件是 Sparse 范式的**统一入口**：场景判定 + 语言路由 + 消费规则。具体知识分布在两个层次：
>
> - **通用知识层** `generic/` — 语言无关的算法、模型、方法论
> - **语言适配层** `adapters/<lang>/` — 强耦合具体开发语言的实现知识（当前提供 `ascendc`，基于 SIMT 编程模型；`dsl`（CANNBotDSL），SIMT 数据面 + AI CPU 调度）
>
> 范式路由（`routes.yaml` → 本文件）与语言路由（本文件 → 适配层）分离：范式路由由 `op.paradigms` 决定，语言路由由开发语言决定。

---

## 场景判定

满足以下条件的算子属于 Sparse 范式：

> **前置约束（芯片 SIMT 编程能力）**：本范式两语言适配层均基于 SIMT 编程模型——AscendC 侧为 `__simt_vf__` 索引驱动离散访存 + 标量原子（`asc_atomic_add`），DSL 侧为 `vf(mode="simt")` + `cb.simt.atomic_add`；标量原子仅 **Ascend 950PR/950DT（arch35 / DAV_3510）系列及以上**支持，A2/A3 等不支持（capability-mapping 平台边界）。**目标芯片（`op.platform_constraints.supported_chips`）不支持 SIMT 编程能力时不进入本范式**——改走通用方法论另行论证非 SIMT 路线（如属主序扫描向量化改造），并报告能力缺口。

1. **数据面为稀疏表示**——输入/输出以 `(indices, values)`、CSR `(indptr, indices, values)`、IndexedRows 或等价结构组织，数据位置由索引张量的值决定
2. **计算核心是索引驱动的访存**——scatter/scatter-add 散射、gather 取数、计数累积或索引坐标重映射，而非规则向量计算
3. **工作量由 nnz 主导**——切分与循环以索引条目数为坐标系

四个算子族：**Count**（计数累积，如 sparse_bincount）、**ScatterAdd**（散射累积，如 sparse_segment_sum_grad）、**RowApply**（稀疏行更新，如 sparse_apply_adadelta/ftrl）、**IndexRemap**（索引重映射，如 sparse_reshape）。与 IndexGather/ScatterUpdate（dense 数据面）、FA 稀疏 trait（注意力块跳过）、Contraction（稀疏×稠密矩阵乘）的边界见 [generic/scene.md](generic/scene.md) §5。

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
| [generic/scene.md](generic/scene.md) | 准入条件、稀疏表示术语、四算子族分类、数据流骨架、能力边界 |
| [generic/capability-contract.md](generic/capability-contract.md) | 能力谓词定义（通用层算法引用的语言/硬件能力参数） |
| [generic/access-model.md](generic/access-model.md) | 索引驱动访存、唯一性判定、冲突处理决策树、值域策略、值依赖、索引数学 |
| [generic/splitting.md](generic/splitting.md) | 并行粒度选择、grid-stride 映射、核间两步法、PER_CORE_MIN、线程数预算 |
| [generic/sync-model.md](generic/sync-model.md) | 零初始化三段式骨架、多段同步限制、原子确定性、分桶预归约路线 |
| [generic/verification.md](generic/verification.md) | golden 口径、精度分档、边界用例表（zero nnz/全密集/重复索引/越界/-1 维）、值依赖验证 |

---

## 消费规则

1. **按需加载**：先读对应语言适配层的 index，再展开当前问题需要的叶子文档。
2. **叶子文档限额**：单个子问题最多展开 5 个叶子文档，仍未闭环则拆分为更小的子问题
3. **能力真值**：通用层算法中的能力谓词，以命中语言适配层的 capability-mapping 填值为准（ascendc: [adapters/ascendc/capability-mapping.md](adapters/ascendc/capability-mapping.md)），其中标注"待验证"的谓词须在产出中声明假设并给出验证义务
4. **实现依据**：编码前对照适配层登记的基准算子（sparse_bincount / sparse_segment_sum_grad / sparse_apply_adadelta / sparse_apply_ftrl / sparse_reshape）核实 API 与组织方式；参考实现中的具体算法与参数须按当前算子规格重新核定
5. **设计依据**：根据规格读取通用算法与对应语言约束，给出算子族判定、访存/冲突路径、切分、同步和验证结论；输出格式由调用方指定
6. **范式优先级**：本范式的专用知识优先于 agent 通用知识；同一设计点结论冲突时，以本范式指向的必读文件为准
