<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式通用知识层索引

> 语言中立的算法与模型。语言专用实现见 [../adapters/](../patterns.md#语言路由)。

## 检索表

| 文档 | 覆盖 | 何时读 |
|------|------|--------|
| [scene.md](scene.md) | 准入条件、稀疏表示术语、四算子族分类、数据流骨架、能力边界 | 判定算子是否属于本范式、确定算子族 |
| [access-model.md](access-model.md) | 索引驱动访存、唯一性判定、冲突处理决策树、值域策略、值依赖、索引数学 | 设计访存路径与冲突处理 |
| [splitting.md](splitting.md) | 并行粒度选择、grid-stride 映射、核间两步法、线程数与资源预算 | 设计切分与线程排布 |
| [sync-model.md](sync-model.md) | 零初始化三段式骨架、多段同步限制、原子确定性与分桶路线 | 设计输出初始化与同步点 |
| [verification.md](verification.md) | golden 口径、精度分档、边界用例表、性能分布敏感性、值依赖验证 | 制定验证方案 |
| [capability-contract.md](capability-contract.md) | 能力谓词定义与使用规则 | 引用任何硬件/语言能力前 |

## 跨范式公共知识

| 主题 | 位置 |
|------|------|
| 值依赖三处声明 | [../../common/value-depend-process.md](../../common/value-depend-process.md) |
| 同步与一致性（通用） | [../../common/sync-and-consistency.md](../../common/sync-and-consistency.md) |
