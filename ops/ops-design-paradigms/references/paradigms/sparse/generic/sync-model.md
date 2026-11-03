<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式同步模型

> **层级**：通用知识层（generic/），语言中立。
>
> 核心问题：零初始化与散射两阶段的跨核有序、原子语义的确定性问题。

## 1 三段式同步骨架（散射累积族）

```
[阶段 1: 零初始化]   全部输出元素清零，按输出元素数 grid-stride 均分
[同步点: 全核栅栏]   所有核确认初始化完成
[阶段 2: 散射]       各核独立做原子/直写散射，无阶段间依赖
```

- 依据：框架/调用方**不保证**输出初始为零；跳过阶段 1 会读到未定义初始值（sparse_segment_sum_grad 实测采用此骨架）
- 同步点必须覆盖全部参与核：阶段 1 与阶段 2 的职责划分不同（按输出元素 vs 按索引条目），栅栏后各核重新计算自己的职责区间
- 阶段 2 内部无任何跨核依赖——原子操作自解决竞争，不需要二级同步

## 2 输出初始化的两条路线

| 路线 | 条件 | 风险 |
|------|------|------|
| **kernel 内两阶段**（零初始化 + 全核栅栏 + 散射） | 无外部清零契约时的默认 | 多一次全核同步开销 |
| **host/框架侧清零**，kernel 只散射 | 调用链明确保证输出缓冲已清零 | 契约未验证时是隐患：任何未清零路径都是随机初始值污染（须以"待验证"标注并给出验证义务） |

设计检查：规格中没有明确清零契约时选路线 1。

## 3 多段 VF 之间的同步限制

- 同一 kernel 内多次启动的执行段之间**只保证核内串行，不保证全核同步**
- 段间存在"全核生产 → 全核消费"依赖（如零初始化段 → 散射段）时，必须在段间插入全核栅栏，并在 host 侧启用同步调度模式（语言落地见适配层）
- 只读元数据（stride 表、shape 表）的生成若在 host 侧完成，则无此问题

## 4 原子语义与确定性

| 输出 dtype | 原子累加结果 | 验证口径 |
|-----------|-------------|---------|
| 整型 | 精确，顺序无关 | 逐位一致 |
| 浮点 | 舍入路径取决于竞争顺序，**非确定** | 与竞品 GPU atomicAdd 对齐；容差比对 |
| 幂等直写（binary_output） | 确定 | 逐位一致 |

- 浮点非确定性是**语义允许的实现自由度**：竞品 GPU 同为非确定实现，golden 应以 GPU 结果或容差判定，不以 CPU 顺序累加为逐位基准
- 需要确定性浮点结果时，退化为"分桶预归约 + 单写者"路线：桶数 = 输出分组数，桶内顺序累加后一次写出；代价是额外一趟与 workspace
- 特殊值遵循 IEEE 754 传播：`inf + (-inf) = nan`、NaN 污染；`±0.0` 的合并顺序可能影响符号位

## 5 同步模型检查清单

- [ ] 输出初始值契约已确认（kernel 两阶段 or host 清零）
- [ ] 全核栅栏覆盖阶段 1 的全部参与核
- [ ] 栅栏对应 host 侧调度模式已启用（见适配层 capability-mapping）
- [ ] 浮点散射的验证基准声明为非确定口径 + 容差
- [ ] 确定性需求算子已改走分桶预归约路线
