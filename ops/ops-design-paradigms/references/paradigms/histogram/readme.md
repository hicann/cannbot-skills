<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 范式算子档案

Histogram（索引化分桶计数）范式覆盖输出为固定大小直方图的算子：每个输入元素经「bin index 计算」映射到 `[0, nbins)` 的一个桶并累加计数。覆盖等宽/自定义边界分桶、多维直方图、离散类标轴，以及 weight/density 加权变体与可选 bin_edges 双输出。核心是元素→桶的数据相关映射与桶间冲突累加（原子累加或分批 DMA 归并），唯一形态为 SIMT 混合 VF。

## 典型算子

- HistogramV2：int/自定义边界分桶，可选 bin_edges 双输出，范式最小样例
- HistogramFixedWidth：显式 value_range 等宽分桶

## 典型生成耗时与性能

| 算子名称 | 模型类型 | 平均性能 | 最差性能 | 算子生成时长 |
|---------|---------|---------|---------|-------------|
| HistogramV2 | GLM-5.3 | 0.87倍库上已有算子 | 0.2倍库上已有算子 | 约9小时（不包含性能优化阶段） |

总体情况：Histogram 类算子统一走 SIMT 混合 VF 路线，实现收敛到三个模板——局部直方图模板（片上 UB 分批 bins 计数 + DMA 原子归并）与无局部直方图模板（GM 直接原子累加）由边界条件 `bins/inputsize < GM_ATOMIC_ADD_FACTOR` 分派，确定性模板（浮点累加需确定时选用）由确定性开关选择、不参与该边界条件分派，生成耗时约 9 小时。SIMT索引驱动分桶计数的性能受原子冲突域影响，生成算子平均性能约为库上已有算子的 0.87 倍，原子冲突大的最差场景性能下降明显（0.2 倍）。

## 支持范围

- **开发语言**：AscendC（SIMT 混合 VF：regbase 框架 + `__simt_vf__` 内核 + `asc_vf_call`）；CANNBotDSL（SIMT VF `cb.vf(mode="simt")`）
- **芯片**：Ascend 950PR / 950DT（arch35）及以上；依赖芯片 SIMT 编程能力，目标芯片不支持 SIMT 时不进入本范式
- **版本**：DSL 0.8.0；SIMT（含混合）需 CANN ≥ 9.2.0（cannbot-dsl 仓配套 9.2.1）
- **覆盖算子族**：等宽（EqualWidth）/ 自定义边界（CustomEdges）分桶、多维直方图（D≥2，行优先展平）、离散类标轴、weight/density 加权、可选 bin_edges 双输出
- **能力边界**：模板由 `bins/inputsize < GM_ATOMIC_ADD_FACTOR` 分派（局部直方图/无局部直方图），局部直方图单批 bins 受片上容量 `ONCHIP_HIST_CAPACITY` 约束、超限分批；计数段为 SIMT、局部直方图归并段为 DMA 原子归并；数据规模极小时应回退串行或低并行度；需 MTE2/Scalar/Vector 三级全重叠等更精细多级流水时本范式不覆盖。

## 未来支持计划

暂无