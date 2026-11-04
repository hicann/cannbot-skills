<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式场景模型

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：ops-nn/control/（FloatStatus 家族、assert、sleep、update_tensor_desc）与 ops-math/conversion/case_condition 实测蒸馏（ascend950 / arch35）。
>
> **范围声明**：本范式只覆盖**状态驱动控制信号算子**（B 类）。图级编排算子（If / While / Case / For / Switch / Merge / Enter / Exit / partitioned_call 等，`REG_OP + .GRAPH` 子图属性、无 kernel）属图编译域，不属本范式（见 §5 边界）。

## 1 准入条件

本范式覆盖的算子需同时满足：

1. **产出/副作用是控制面信号**——输出值或硬件状态供**控制决策**消费（训练循环的动态 loss scale / 溢出跳步、宿主的报错中止、分支选择、时序控制），而非数据面计算结果
2. **信号语义由消费方定义**——算子本身不解释信号，只负责信号的生产/读取/清除/求值
3. **数据面退化**——输出固定微小 shape（如 `[8]` fp32）、幂等常量写，或无输出（纯副作用）

## 2 控制信号类型

| 信号类型 | 语义 | 实例 |
|---------|------|------|
| **数值健康状态** | 硬件 FP 异常（溢出等）累积标志，`[8]` fp32 | npu_alloc / get / clear_float_status |
| **断言判定** | 标量条件为假时产出失败标志（供消费方中止） | assert |
| **时序控制** | 按周期数延迟（测试/同步辅助） | sleep |
| **条件求值** | 输入值 → 离散 case 编号（供分支选择） | case_condition |
| **描述符更新** | 运行时更新 tensor 的 shape/format 描述 | update_tensor_desc |

## 3 状态生命周期（以数值健康状态为例）

```
alloc（分配清零）→ [计算阶段：硬件累积异常标志] → get（读取快照）→ clear（复位）
```

- 三操作闭环是标准形态：alloc 产出初始状态张量、get 读取当前状态、clear 复位硬件标志
- 状态本体在**硬件标志寄存器**（设备全局），不在算子输出——输出只是快照
- 消费方（宿主训练循环）按 get 结果决定跳步/缩放，再 clear 进入下一轮

## 4 数据流骨架

```
[段 1: 信号源]   硬件状态 / 输入条件值 / 属性参数
[段 2: 信号操作] 读快照 / 清复位 / 求值 / 延迟
[段 3: 信号落地] 固定 shape 输出（幂等写）/ 错误上报 / 无输出副作用
```

- 无 tiling 设计空间：输出固定或直通，切分/缓冲规划退化为常量（这是本范式对 tiling 方法论的**负知识**——设计重心在信号契约与副作用，不在数据面切分）

## 5 能力边界（互斥声明）

1. **图级编排不属本范式**：If / While / Case / For / partitioned_call / Switch / Merge / Enter / Exit / NextIteration / ConditionCalc 等（`canndev/ops/control/`）无 kernel，设计物是子图接线契约，由图引擎执行
2. **描述符更新与 DynamicShape 交界**：update_tensor_desc 本质是运行时 shape 控制手段，shape 推导规则归 DynamicShape，本范式只覆盖其"运行时控制语义"面
3. **断言与 MaskPredicate 交界**：逐元素掩码比较（isnan/isfinite 等）是数据面谓词；本范式的断言是**控制语义**（条件假 → 报错中止），输出无数据面意义
4. **与 Stateful 的边界**：本范式状态在硬件标志寄存器，无跨调用持久化张量状态（clear 即归零，非 checkpoint 语义）

## 6 依赖谓词

| 谓词 | 用途 |
|------|------|
| `CAP_FP_STATUS_READ` / `CAP_FP_STATUS_CLEAR` | 数值健康状态的读取/复位能力 |
| `CAP_SCALAR_SCOPE_INTRINSIC` | 状态类 intrinsic 的作用域约束（决定调用位置） |
| `CAP_VECTOR_SIDE_EFFECT_CLEAR` | 向量单元活动触发标志复位的副作用通道 |
| `CAP_DEVICE_CLOCK` | 设备时钟读取（时序控制） |
| `CAP_ASSERT_SIGNAL_LAYOUT` | 断言失败路径的标志输出契约（布局/通过-失败码取值） | 断言类输出契约（signal-model §6） |

谓词定义见 [capability-contract.md](capability-contract.md)；各语言真值见对应适配层 capability-mapping。
