<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Interpolation 范式通用知识层索引

> 语言中立的算法与模型。语言专用实现见 [../adapters/](../patterns.md#语言路由)。

## 检索表

| 文档 | 覆盖 | 何时读 |
|------|------|--------|
| [scene.md](scene.md) | 准入条件、插值术语、三算子族分类、数据流骨架、能力边界 | 判定算子是否属于本范式、确定算子族 |
| [coordinate-mapping.md](coordinate-mapping.md) | 输出驱动映射、对齐模式语义表、邻居分解与 clamp 防御、越界策略、浮点同序、高阶核 | 设计坐标映射与邻居索引路径 |
| [weight-model.md](weight-model.md) | 权重计算、加权 gather 访存形态、累加精度判定树、补偿累加、非有限值传播 | 设计权重与累加精度路线 |
| [splitting.md](splitting.md) | 并行粒度选择、快除坐标分解、元数据传递、线程数预算、核间两步法、grad 散射切分 | 设计切分与线程排布 |
| [verification.md](verification.md) | golden 口径、精度分档、边界用例表、性能与逐位验证义务 | 制定验证方案 |
| [capability-contract.md](capability-contract.md) | 能力谓词定义与使用规则 | 引用任何硬件/语言能力前 |

## 跨范式公共知识

| 主题 | 位置 |
|------|------|
| 快除（magic/shift）参数与布局 | [../../common/uintdiv-rules.md](../../common/uintdiv-rules.md) |
| SIMT 硬件模型（线程档位/缓存/UB） | [../../common/simt-hardware-model.md](../../common/simt-hardware-model.md) |
| 同步与一致性（栅栏/调度模式） | [../../common/sync-and-consistency.md](../../common/sync-and-consistency.md) |
| 值依赖三处声明 | [../../common/value-depend-process.md](../../common/value-depend-process.md) |

## 相邻范式互引

| 主题 | 位置 |
|------|------|
| grad 散射的原子与冲突模型（散射累积族） | [../../sparse/generic/access-model.md](../../sparse/generic/access-model.md) |
| grad 零初始化两段同步骨架 | [../../sparse/generic/sync-model.md](../../sparse/generic/sync-model.md) |
| dense 写散射为语义主体时的设计 | [../../scatterupdate/patterns.md](../../scatterupdate/patterns.md) |
