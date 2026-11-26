<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式通用知识层索引

> 语言中立的算法与模型。语言专用实现见 [../adapters/](../patterns.md#语言路由)。

## 检索表

| 文档 | 覆盖 | 何时读 |
|------|------|--------|
| [scene.md](scene.md) | 准入条件、窗口术语、布局维度（NCHW/NHWC）、正向池化聚合族与反向路由三分支、数据流骨架、覆盖形态与口径说明 | 判定算子是否属于本范式、确定正反向类型与族 |
| [window-math.md](window-math.md) | 窗口边界数学、pad 裁剪、dilation 起点推进、空窗语义、聚合初始化与 NaN/平局契约、反窗口区间公式 | 设计正向窗口枚举或反向区间推导 |
| [grad-routing.md](grad-routing.md) | 反向路由决策树、反窗口枚举 / argmax 路由 / 原子散射三分支、零原子论证、mask 契约 | 设计反向算子 |
| [splitting.md](splitting.md) | 方向坐标系、坐标分解（快除/直除）、核间两步法、线程分档、元数据传递 | 设计切分与线程排布 |
| [verification.md](verification.md) | golden 口径（tie-break 契约 / 原子容差）、精度分档、正反向边界用例、梯度守恒 | 制定验证方案 |
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
| 分支 C 原子与冲突模型（散射累积族） | [../../sparse/generic/access-model.md](../../sparse/generic/access-model.md) |
| 分支 C 零初始化两段同步骨架 | [../../sparse/generic/sync-model.md](../../sparse/generic/sync-model.md) |
| 卷积主体设计（滑窗 × 权重内积） | [../../contraction/patterns.md](../../contraction/patterns.md) |
