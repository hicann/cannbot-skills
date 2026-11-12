<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式算子档案

ControlFlow（控制流信号）范式覆盖状态驱动控制信号算子：产出或副作用是控制面信号（供训练循环、宿主或分支决策消费），而非数据面计算结果；数据面退化为固定微小 shape、幂等常量写或无输出。

## 典型算子

- npu_get_float_status：数值健康状态读取（FloatStatus 家族），范式最小样例
- assert：断言判定，失败路径信号化
- sleep：时序控制（周期延迟）
- case_condition：条件求值（输入值到离散 case 编号的映射）
- update_tensor_desc：运行时描述符更新

## 典型生成耗时与性能

| 算子名称 | 模型类型 | 平均性能 | 最差性能 | 算子生成时长 |
|---------|---------|---------|---------|-------------|
| npu_get_float_status | GLM-5.3 | —（数据面退化，无性能对标） | — | 约 6 小时（不含性能优化阶段） |

总体情况：控制面信号算子数据面退化（输出固定 `[8]` fp32 或无输出），性能对标意义有限；生成耗时以接口契约、降级态论证与副作用闭环验证为主，约 6 小时。FloatStatus 家族在 950PR/DT 为降级态空实现（恒写规范零值，溢出探测待芯片能力补齐）；clear 型混合段（SIMD 副作用 + SIMT 数据面）的副作用通道有效性需逐平台实测闭环。

## 支持范围

- **开发语言**：AscendC（设备核 SIMT 为主，SIMD / 混合按信号类型选型）；CANNBotDSL（同构路线，SIMT VF 为主）
- **芯片**：Ascend 950PR / 950DT（arch35）及以上；alloc 型（标量写零 + DCCI）不依赖 SIMT，其余信号类型依赖芯片 SIMT 能力
- **版本**：DSL 0.8.0；SIMT VF 需 CANN ≥ 9.2.0（cannbot-dsl 仓配套 9.2.1）
- **覆盖信号**：数值健康状态（FloatStatus 家族）/ 断言判定 / 时序控制 / 条件求值 / 描述符更新
- **能力边界**：图级编排算子（If / While / Case / For / Switch / Merge / Enter / Exit 等无 kernel 子图）属图编译域，不属本范式；update_tensor_desc 的 shape 推导规则归 DynamicShape，本范式只覆盖其运行时控制语义面

## 未来支持计划

暂无
