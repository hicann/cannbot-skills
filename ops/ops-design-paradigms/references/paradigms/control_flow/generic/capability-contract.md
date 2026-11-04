<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式能力谓词契约

> **层级**：通用知识层（generic/），语言中立。
>
> 通用层算法引用以下能力参数；真值由命中语言的适配层 capability-mapping 填写。标注"待验证"的谓词须在产出中声明假设并给出验证义务。

## 1 谓词定义

| 谓词 | 类型 | 含义 | 影响的通用层结论 |
|------|------|------|----------------|
| `CAP_FP_STATUS_READ` | bool | 提供硬件 FP 异常状态读取 intrinsic | get 的真实/空实现分叉（signal-model §1） |
| `CAP_FP_STATUS_CLEAR_API` | bool | 提供标志复位**接口** | clear 走接口 or 副作用通道（signal-model §3） |
| `CAP_VECTOR_SIDE_EFFECT_CLEAR` | bool | 计算单元活动可触发标志复位 | 副作用通道可用性（signal-model §3） |
| `CAP_SCALAR_SCOPE_INTRINSIC` | bool | 状态 intrinsic 仅标量作用域可用（禁入线程函数） | 调用位置约束（取值→传参模式） |
| `CAP_DEVICE_CLOCK` | bool | 设备时钟读取 | 时序算子成立（signal-model §5） |
| `CAP_ASSERT_SIGNAL_LAYOUT` | spec | 断言失败路径的标志输出契约（布局/通过-失败码取值） | 断言类输出契约（signal-model §6） |
| `CAP_PREBUILT_BINARY` | bool | 预编译二进制注册（shape 无关单实例） | 二进制形态选型（kernel-placement §4） |
| `CAP_STATUS_LAYOUT` | spec | 状态张量布局（元素数/dtype） | alloc/get/clear 三算子的输出契约 |

## 2 使用规则

1. 通用层文档中的具体取值（`[8]` fp32、最小线程数等）只作示例；设计产出必须引用适配层真值
2. **降级态也是真值**：`CAP_FP_STATUS_READ=false`（或"未启用"）必须如实登记，空实现据此设计（signal-model §5 的 I5）
3. 副作用通道类谓词（`CAP_VECTOR_SIDE_EFFECT_CLEAR`）无接口保证，逐平台实测登记，禁止外推
4. 谓词以平台为单位登记，不同 SoC/架构禁止外推
