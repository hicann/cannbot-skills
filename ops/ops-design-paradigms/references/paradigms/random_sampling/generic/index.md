<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式通用知识层索引

> 语言中立的算法与模型。语言专用实现见 [../patterns.md](../patterns.md#语言路由)。

## 检索表

| 文档 | 覆盖 | 何时读 |
|------|------|--------|
| [scene.md](scene.md) | 准入条件、随机流三要素、四算子族（分布生成/掩码生成/离散采样/排列生成）、数据流骨架、能力边界 | 判定算子是否属于本范式、确定算子族 |
| [prng-model.md](prng-model.md) | 流代数、Skip 跳越、多单元不重叠不变量 I1/I2、预留律、块间流分割协议 ★核心 | 设计流切分与正确性论证 |
| [state-management.md](state-management.md) | seed 语义、接口形态分类（stateless/带状态）、offset 递进协议、shape 值依赖 | 设计状态管理与回写 |
| [distribution-transform.md](distribution-transform.md) | 变换目录（uniform/normal/exponential/truncated/bernoulli/multinomial/permutation）、每输出消耗数、精度边界 | 设计分布变换与消耗推导 |
| [splitting.md](splitting.md) | 输出空间均分、连续/离散布局与展开因子、大输出块协议 | 设计切分与线程映射 |
| [verification.md](verification.md) | 双轨验证（逐位/统计）、golden 同源性、边界用例表、offset 链专项 | 制定验证方案 |
| [capability-contract.md](capability-contract.md) | 能力谓词定义与使用规则 | 引用任何硬件/语言能力前 |

## 跨范式公共知识

| 主题 | 位置 |
|------|------|
| 值依赖三处声明 | [../../common/value-depend-process.md](../../common/value-depend-process.md) |
| 同步与一致性（通用） | [../../common/sync-and-consistency.md](../../common/sync-and-consistency.md) |
