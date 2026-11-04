<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式通用知识层索引

> 语言中立的算法与模型。语言专用实现见 [../patterns.md](../patterns.md#语言路由)。

## 检索表

| 文档 | 覆盖 | 何时读 |
|------|------|--------|
| [scene.md](scene.md) | 准入条件（控制面信号）、五类信号、状态生命周期、**图级编排不属本范式的边界声明** | 判定算子是否属于本范式 |
| [signal-model.md](signal-model.md) | 信号契约表、幂等不变量 I3、副作用可观测性 I4、降级兼容契约 I5、时序/断言语义 ★核心 | 设计信号操作与正确性论证 |
| [kernel-placement.md](kernel-placement.md) | 设备核形态选型决策树（含无宿主上报通道负知识）、退化 kernel 形态、二进制形态 | 第一设计决策（形态选型） |
| [verification.md](verification.md) | 信号契约验证（闭环/幂等/降级/失败路径）、用例矩阵、平台迁移义务 | 制定验证方案 |
| [capability-contract.md](capability-contract.md) | 能力谓词定义（含降级态登记规则） | 引用任何硬件/语言能力前 |

## 跨范式公共知识

| 主题 | 位置 |
|------|------|
| 值依赖三处声明 | [../../common/value-depend-process.md](../../common/value-depend-process.md) |
| 同步与一致性（通用） | [../../common/sync-and-consistency.md](../../common/sync-and-consistency.md) |
