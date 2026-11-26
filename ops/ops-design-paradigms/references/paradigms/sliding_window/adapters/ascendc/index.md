<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# SlidingWindow 范式 AscendC 语言适配层

> **层级**：AscendC 语言适配层（adapters/ascendc/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。
>
> 本适配层基于 **Ascend C SIMT 编程模型**（整数规则窗口聚合与反向梯度路由的标准选择）。regbase（SIMD 向量）路线的通用开发方法见 [../../../general-methodology.md](../../../general-methodology.md)；成熟池化包内的 SIMD 小核/大核路线不在本范式消费范围（[simt-path.md](simt-path.md) §4）。

## 1 定位

本目录承载 SlidingWindow 范式中**强耦合 AscendC SIMT** 的知识：正向窗口聚合与反向路由的落地、宿主类与 TilingKey 组织、能力真值与陷阱。开发语言为 ascendc（由调用方明确提供）时，与通用知识层组合消费：

```
通用算法/模型（generic/）× AscendC SIMT 实现细节（本目录）→ SIMT 设计与代码产出
```

SIMT 基础 API（线程模型、访存函数、数学函数、调试接口）的官方文档位于 `$ASC_DEVKIT_DIR/docs/zh/api/SIMT-API/`；`ascendc-simt-best-practices` skill 提供 API 分类导航与实践总结，本适配层只展开 SlidingWindow 范式特有的内容。

## 2 能力映射

本语言对通用层能力谓词的真值见 [capability-mapping.md](capability-mapping.md)。通用层算法中引用的谓词（−Inf 哨兵、NaN 传播、直除/快除双档、线程上限等）一律以该文件填值为准。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| Kernel 入口与 tiling | [kernel-entrance.md](kernel-entrance.md) | `__global__` 入口、宿主类形态、TilingData/TilingKey 段组织（多路线包内 SIMT 段定位）、属性归一化、mask 契约 |
| SIMT 实现路线 | [simt-path.md](simt-path.md) | 正向窗口聚合、反向三分支（反窗口枚举 / argmax 路由 / 原子散射）、快除参数表、正反向线程分档 |
| 陷阱与定位 | [pitfalls.md](pitfalls.md) | dilation 负起点、NaN 漏传播、反窗口不同源、mask 口径漂移、除子不一致、SyncAll/ATTR 漏配等错题集 |

## 4 实测基准算子

本适配层结论蒸馏自以下 ascend950（arch35）实测算子（ops-nn 仓），编码前可对照源码组织方式：

| 算子 | 方向 / 分支 | 关键特征 |
|------|-----------|---------|
| max_pool_with_argmax_v3 | 正向（max 聚合） | 输出驱动 + −Inf 哨兵 + CycleUpdate NaN 优先 + argmax mask 双输出、dilation 窗口、直除分解、256 线程 |
| max_pool_grad_with_argmax_v3 | 反向（B argmax 路由） | 输入驱动反窗口 + PStart/PEnd 快除 + mask 命中检查、零原子直写、1024 线程 |
| avg_pool | 正向（avg 聚合） | 输出驱动 + 除子三分档 + countIncludePad、UintDiv 分解、NCHW/NHWC 双布局 |
| avg_pool_v2_grad | 反向（A 反窗口枚举） | 输入驱动反窗口 + 均匀贡献累加、零原子、1024 线程 |

**内嵌样例包**：max_pool_with_argmax（正向）与 max_pool_grad_with_argmax（反向 B 分支）两个**SIMT 单路线模板包**（DRAFT，自含组织、无 common 依赖）见 [example/](example/)；算法语义蒸馏自 ops-nn 实测 SIMT kernel，对照关系与验证义务见 [example/README.md](example/README.md)。avg_pool / avg_pool_v2_grad（A 分支正向/反向配对）按算子名在 [ops-nn](https://gitcode.com/cann/ops-nn) 仓 `pooling/` 检索阅读。

## 5 公共知识依赖

| 主题 | 位置 |
|------|------|
| 快除（magic/shift）参数与布局 | [../../../common/uintdiv-rules.md](../../../common/uintdiv-rules.md) |
| SIMT 硬件模型 | [../../../common/simt-hardware-model.md](../../../common/simt-hardware-model.md) |
| 同步与一致性 | [../../../common/sync-and-consistency.md](../../../common/sync-and-consistency.md) |

## 6 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载其他语言适配层
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 能力谓词以 [capability-mapping.md](capability-mapping.md) 为真值；标注"待验证"的项作为假设输出并附验证义务
4. 编码前对照 §4 基准算子核实 API 用法；参考实现的具体参数（线程分档、TilingKey 段）须按当前算子规格重新核定
