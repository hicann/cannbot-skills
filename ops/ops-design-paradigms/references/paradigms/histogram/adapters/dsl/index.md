<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 范式 CANNBotDSL 语言适配层

> **层级**：CANNBotDSL（下文简称 DSL）语言适配层（adapters/dsl/）。语言中立的算法与模型见 [../../generic/index.md](../../generic/index.md)；范式入口与语言路由见 [../../patterns.md](../../patterns.md)。

## 1 定位

本目录承载 Histogram 范式中**强耦合 DSL** 的知识：DSL 实体映射、能力谓词真值、代码组织与实现约束。开发语言为 `dsl` 时，与通用知识层组合消费：

```
通用算法/模型（generic/）× DSL 实现细节（本目录）→ CANNBotDSL 设计与代码产出
```

Histogram 的核心是「分桶计数 + 冲突累加收敛」：前两模板（局部直方图 / 无局部直方图）在 DSL 下的差异落到**片上局部直方图 Buffer/RingBuffer 与原子/归并能力**的选择，确定性模板（第三模板）无原子依赖、走 GM 点赋值串行累加；三者落地见 [capability-mapping.md](capability-mapping.md) 与 [implementation-guide.md](implementation-guide.md)。

## 2 能力映射

本语言对通用层能力谓词的真值与 DSL 实体见 [capability-mapping.md](capability-mapping.md)。通用层算法中引用的谓词（片上原子累加、DMA 归并、GM 原子、片上容量、对齐粒度、前置 min/max 归约、展平 index 等）一律以该文件填值为准；标注**待验证**的谓词不得当作已验证真值使用，须在产出中显式声明假设。

## 3 阅读路径

| 任务阶段 | 入口 | 内容 |
|------|------|------|
| 方案设计 | [../../generic/index.md](../../generic/index.md)（通用层算法）→ [capability-mapping.md](capability-mapping.md)（谓词真值） | bin 模型 / bin index / 累加策略 / 线程切分 / 同步等语言中立算法，叠加 DSL 能力真值与落地差异 |
| DSL 代码组织与实现 | [implementation-guide.md](implementation-guide.md) | 三模板数据流骨架到 DSL 入口/Host/Kernel 三层的映射、Buffer/RingBuffer 规划、分批 bins、原子累加与归并、确定性模板、同步 |

## 4 补充输入

设计需要公开 DSL API 资料、目标设备能力和已有实现参考时，由调用方提供相应资料。只依据实际提供的证据选择 API、资源和同步写法；缺少证据时列出待确认事实。DSL 无 SIMT VF / warp / TQue / TilingKey / OpDef 概念，通用层中与这些强耦合的描述按 [capability-mapping.md](capability-mapping.md) §2 的实体对照转译，不得照搬 ascendc 适配层的 API。

## 5 消费规则

1. 语言路由命中本适配层后，**只读本适配层 + 通用层**，不加载 ascendc 适配层、不参考其 API 与代码模板
2. 单个子问题最多展开 5 个叶子文档；仍未闭环则拆分子问题
3. 能力谓词以 [capability-mapping.md](capability-mapping.md) 为准；**待验证**谓词支撑的设计结论必须在产出中声明假设并给出验证义务，禁止写成已验证事实
4. **设计文档生成**：使用 `references/paradigms/general-methodology-dsl.md` 通用方法与通用层知识，并按 [capability-mapping.md](capability-mapping.md) §4 和 [implementation-guide.md](implementation-guide.md) 补充 DSL 实现约束。
