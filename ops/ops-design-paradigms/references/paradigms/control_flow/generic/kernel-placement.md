<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ControlFlow 范式落核决策

> **层级**：通用知识层（generic/），语言中立。
>
> 核心问题：控制信号算子统一落在**设备计算核（AICore）**上，第一设计决策是**形态选型**（SIMT / SIMD / 混合 + 输出契约），先于一切实现细节。

## 1 设备核的能力面与边界

| 能力 | 设备计算核（AICore） | 设计含义 |
|------|--------------------|---------|
| 标量条件分支与小数据访问 | ✅ SIMT 单线程 | assert / case_condition 型的可承载形态 |
| 低时延、设备侧闭环 | ✅ | 状态读/清、时序控制 |
| 触发向量/计算单元活动（副作用通道） | ✅（混合形态） | clear 类借道副作用 |
| 设备时钟读取（时序） | ✅ | sleep 自旋 |
| 宿主级日志/字符串/错误上报 | ❌（仅调试 printf，生产禁用） | **负知识**：断言失败路径必须信号化（写标志输出），上报/中止由消费方读取标志后执行，摘要打印在宿主侧 |

> 设备核无宿主上报通道是本范式最重要的负知识：任何"失败要报错/中止"的语义都不得依赖 kernel 内打印或错误返回，必须落成**标志输出 + 消费方闭环**（I4）。

## 2 形态选型判据（决策树）

```
信号操作是什么？
├─ 硬件状态的设备侧读/清（低时延）   → SIMT 幂等写 / 混合（FloatStatus get/clear 类）
├─ 标量条件的断言判定               → SIMT 单线程写通过/失败标志（assert 类）——消费方读标志闭环
├─ 小规模输入的条件求值             → SIMT 单线程写离散编号（case_condition 类）
├─ 状态分配清零                    → SIMD/标量写零 + DCCI（alloc 类）
└─ 设备时钟（时序控制）             → SIMT 单线程自旋（sleep 类）
```

- 判据本质：**输出契约**（状态快照 / 通过-失败标志 / 离散编号 / 无输出纯副作用）与**副作用触达性**（要触达的硬件行为需哪类执行单元活动）
- 混合形态合法：clear 在设备核（副作用 + SIMT 写零），其中"触发计算单元活动"段用向量指令、"写输出"段用标量线程

## 3 退化 kernel 形态

本范式的信号算子不是常规计算 kernel，形态退化为：

| 特征 | 常规计算 kernel | 信号退化 kernel |
|------|---------------|----------------|
| 输出 | 由数据推导 | 固定微小 shape（`[8]`/`[1]`）/ 常量 |
| 切分/缓冲 | tiling 设计 | 常量（无设计空间）；**交付契约**（空 TilingData + 四步 TilingFunc + 单 key 集）见适配层 [status-kernel.md](../adapters/ascendc/status-kernel.md) §6 |
| 线程规模 | 按负载分档 | 最小有效（单元素量级；断言/求值/时序类为单线程） |
| 写语义 | 唯一归属 | 幂等（全单元同值） |
| 正确性 | 数据逐位 | 副作用闭环 + 幂等性（断言类为标志取值 + 消费闭环） |

## 4 二进制形态

部分信号算子以**预编译二进制**交付（源码仅空实现/占位）：注册时声明 shape 无关（任意 shape 匹配）的单一实例。选型依据：算子逻辑极简且稳定、无需按 shape 编译多实例。

## 5 检查清单

- [ ] 输出契约已定（快照 / 标志 / 编号 / 纯副作用）且不依赖设备核宿主上报通道（§1 负知识）
- [ ] 形态选型已过（§2 决策树：SIMT / SIMD / 混合）
- [ ] 退化形态确认（固定输出/幂等/最小线程）
- [ ] 二进制 vs 源码编译形态选型
- [ ] 断言类失败路径的标志取值与消费方闭环方案已声明（I4）
