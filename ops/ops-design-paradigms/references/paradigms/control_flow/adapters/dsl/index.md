<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式 CANNBotDSL 语言适配层

> **层级**：CANNBotDSL 语言适配层（adapters/dsl/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。
>
> 本范式全部信号类型落**设备核（AI Core）**：`@kernel` + SIMT VF 为主，SIMD/混合按信号类型选型；断言失败路径信号化（标志输出 + 宿主消费闭环），kernel 不依赖宿主上报通道。权威依据：cannbot-dsl 仓 docs/zh/guide/（开源：https://gitcode.com/cann/cannbot-dsl ；compute / hardware-model / synchronization），DSL 0.8.0，Ascend 950PR/950DT。

## 1 定位

本目录承载 ControlFlow 范式中**强耦合 CANNBotDSL（下文简称 DSL）**的知识：状态信号算子的设备核落地（SIMT/SIMD/混合，覆盖断言/条件求值/时序控制）、能力真值与陷阱。

```
通用算法/模型（generic/）× DSL 实现细节（本目录）→ DSL 设计与代码产出
```

与 AscendC 适配层的对齐关系：

| AscendC 路线（adapters/ascendc/） | DSL 对等物 |
|------|------|
| 设备核（status-kernel.md：SIMT 写规范值 / SIMD 副作用段） | [status-kernel.md](status-kernel.md)：`@kernel` + SIMT VF / SIMD VF |
| 断言/条件求值（ascendc assert-case-kernel.md：标志化 + 离散求值） | [assert-case-kernel.md](assert-case-kernel.md)：`@kernel` + SIMT VF 单线程写标志/编号 |
| 预编译二进制（binary.json shape[-2] 单实例） | AOT / Native 打包（`@export` + `cannbotdsl.aot.build`，Dim 动态维度泛化） |
| 退化 tiling（空 TilingData + 常量 TilingFunc） | **无 tiling 概念**——host 构图期直接启动常量几何（正对应退化形态） |

## 2 能力映射

本语言对通用层能力谓词的真值见 [capability-mapping.md](capability-mapping.md)。**降级态也是真值**：DSL API 面未暴露 FP 异常状态读取——get 的空实现路线在 DSL 同样适用且是当前唯一形态。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| 设备核路线 | [status-kernel.md](status-kernel.md) | alloc/get/clear/sleep 的 DSL 形态：幂等写、副作用段、`get_system_cycle` 自旋、AOT 发布 |
| 断言/条件求值路线 | [assert-case-kernel.md](assert-case-kernel.md) | assert 失败路径信号化（标志 + 宿主消费闭环）、case_condition 离散求值、与状态类形态对齐 |
| 陷阱与定位 | [pitfalls.md](pitfalls.md) | 断言语义误植、VF 纪律、launch 形态、UB 预算等错题集 |

## 4 形态选型表（DSL 版，CF-R1 机械匹配）

信号类型判定见 [../../generic/signal-model.md](../../generic/signal-model.md) 与落核决策树 [../../generic/kernel-placement.md](../../generic/kernel-placement.md) §2；本表给出 DSL 执行载体：

| 信号特征 | DSL 形态 | 理由 |
|------|---------|------|
| 状态快照/写规范值（FloatStatus 家族型） | **设备核**：`@kernel` + SIMT VF 幂等写 | 无宿主需求，最小 launch |
| 副作用通道（标志复位类） | 设备核：SIMD VF 活动段 + SIMT 写值段 | 副作用无接口保证（同 AscendC 负知识，逐平台实测） |
| 标量条件的断言判定（assert 型） | **设备核**：`@kernel` + SIMT VF 单线程写通过/失败标志 | 失败路径信号化 + 宿主消费闭环；DSL kernel 无宿主上报对应物（capability-mapping） |
| 小数据条件求值（case_condition 型） | **设备核**：SIMT VF 单线程写离散编号 | 输入 `[3]` 微量，单线程足够；运行期读真值 |
| 时序控制（sleep 型） | 设备核：`get_system_cycle()` 自旋（单线程） | 周期读取 API 存在；位置约束待验证 |

**形态选型强制登记（CF-R1）**：设计必须在首个设计文档登记形态决策 + 本表依据行，义务同 ascendc 版。

**内嵌样例**：设备核路线结构参考见 [example/npu_get_float_status_simt_draft/](example/npu_get_float_status_simt_draft/)（**DRAFT 未验证**——与 AscendC 侧同算子（npu_get_float_status）对齐：纯 SIMT 幂等写、混合形态副作用段、退化 launch 契约的 DSL 骨架；真实样例包在 [../ascendc/example/npu_get_float_status_package/](../ascendc/example/npu_get_float_status_package/)）。

## 5 公共知识依赖

| 主题 | 位置 |
|------|------|
| 同步与一致性 | [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md)（DSL 差异：stream 顺序 launch 替代 SyncAll 两段式） |
| 值依赖处理 | [../../../common/value-depend-process.md](../../../common/value-depend-process.md)（DSL：运行期值不可构图期读，经设备核 SIMT 读真值） |

## 6 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载 ascendc 适配层、不参考其 API 与代码模板
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 能力谓词以 [capability-mapping.md](capability-mapping.md) 为真值；**降级态真值与"逐平台实测"义务必须原样传递到产出**
4. 负知识传递：本范式算子无数据面自由度（退化 kernel）——DSL 侧无 tiling 设计空间，设计重心是信号契约（幂等 I3 / 副作用 I4 / 降级 I5）与落核决策
5. 副作用通道参数（触发形态）来自 V1 行为对齐——无接口保证，目标平台重新论证（同 ascendc 义务）
