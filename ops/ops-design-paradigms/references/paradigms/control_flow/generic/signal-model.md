<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式控制信号模型

> **层级**：通用知识层（generic/），语言中立。
>
> 核心问题：信号的正确性语义、幂等性与副作用可观测性——本范式替代"数据正确性"的核心不变量。

## 1 信号契约

| 操作 | 前置 | 效果 | 输出 |
|------|------|------|------|
| **alloc** | 无 | 分配并清零状态载体 | `[8]` 全零张量 |
| **get（读）** | 状态载体存在 | 读取硬件状态快照 | 当前状态值（能力未启用时为规范零值） |
| **clear（清）** | 状态载体存在 | **复位硬件标志** | `[8]` 全零张量（副作用载体） |
| **eval（求值）** | 输入条件值 | 离散化（输入 → case 编号） | int32 编号 |
| **assert** | 标量条件 | 真：写通过标志（常量）；假：写失败标志（常量） | 固定微小 shape 状态标志 |

契约要点：
- **读不改状态**：get 是快照语义，重复调用同结果（在无新异常累积的前提下）
- **清必全局生效**：状态是设备全局的，clear 必须覆盖全部执行单元的累积效应
- **求值无副作用**：case 编号是纯函数

## 2 幂等性不变量（I3）

> 全部执行单元对同一输出写**相同常量值**是合法且预期的；不同值即错误。

- 多核/多线程同时写 `[8]` 零值不构成竞争（写合并语义）
- 设计含义：kernel 不需要"唯一写者"选择（与状态回写类协议相反）；但**禁止**任何执行单元写非约定值

## 3 副作用语义与可观测性（I4）

> clear 类操作的真实效果不在输出数据，而在硬件标志——**输出正确 ≠ 语义达成**。

- 验证必须闭环：`clear → get == 0`（消费可观测路径）
- 副作用通道本身可能是**工程性**的（借道某类执行单元的活动触发硬件行为），无接口级保证——设计文档必须声明该通道的依据与脆弱性（见适配层 status-kernel）
- 副作用的时序：clear 必须发生在消费方读取之后才有意义（由消费方的图依赖保证，非算子内同步）

## 4 降级兼容契约（I5）

> 平台能力缺失时，接口契约保持：调用不报错、输出规范零值、语义留待能力补齐。

- 空实现是**一等设计**而非偷懒：调用方代码无需感知平台差异
- 必须预留能力补齐接入点：真实实现切换的位置（如"Process 内取值 → 传标量给执行体"）在设计文档中写明
- 降级期间消费方的行为退化路径（如恒 0 → 永不触发跳步）要向消费方声明

## 5 时序控制语义

- 周期数语义：延迟量以设备时钟周期计，**时钟粒度决定精度下界**——验证只做区间断言（`实际延迟 ≥ 请求`），不做等值
- 单执行体语义：时序操作由单一执行单元承担（多单元重复延迟无意义且浪费）

## 6 断言语义

- 条件为**标量 bool**（非标量为契约违规，宿主侧校验）
- **失败路径信号化**：kernel 写失败标志输出 → 消费方（宿主/下游）读取标志后中止调用链；条件真写通过标志（短路零开销）
- 设备核无宿主级日志/中止通道（[kernel-placement.md](kernel-placement.md) §1 负知识）——摘要打印在**宿主侧**完成（读输入张量生成，条数受 summarize 属性约束），kernel 不承担上报
- 验证闭环（I4）：条件假 → 标志 == 失败码；条件真 → 标志 == 通过码；消费方行为向调用方声明

## 7 检查清单

- [ ] 信号类型与操作族归类（scene §2/§3）
- [ ] 幂等性设计（全执行单元同值）或唯一写者（如需）
- [ ] 副作用通道声明 + 闭环验证方案
- [ ] 降级路径与能力补齐接入点
- [ ] 消费方退化行为已声明
