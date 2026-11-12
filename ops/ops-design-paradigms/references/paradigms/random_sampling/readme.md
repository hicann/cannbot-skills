<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# RandomSampling 范式算子档案

RandomSampling（随机数生成）范式覆盖输出值由伪随机数驱动的算子：均匀、正态、指数、截断正态、伯努利、多项、排列等分布生成，随机掩码生成，离散采样，以及随机量与逐元素计算的复合（如带噪声的激活）。

## 典型算子

- random_uniform_v2：区间均匀分布生成，范式最小样例
- stateless_random_normal_v2：正态分布生成（无状态接口）
- stateless_bernoulli：伯努利采样
- truncated_normal_v2：截断正态（拒绝采样，消耗数不定）
- drop_out_v3：随机丢弃
- rrelu_with_noise：随机噪声 + 逐元素复合

## 典型生成耗时与性能

| 算子名称 | 模型类型 | 平均性能 | 最差性能 | 算子生成时长 |
|---------|---------|---------|---------|-------------|
| rrelu_with_noise | GLM-5.3 | 0.6 倍库上已有算子 | 0.3 倍库上已有算子 | 约 22 小时（不含性能优化阶段） |

总体情况：纯生成类算子（uniform / normal 家族）可复用公共底座，生成耗时约 20 小时；复合类算子（随机 + 逐元素）需自管编排，耗时约 22 小时。SIMD 路线批量流推进吞吐高，SIMT 路线逐元素变换灵活，生成算子平均性能约为库上已有算子的 0.6 倍；拒绝采样类（截断正态）因消耗数不定，最差场景性能下降明显。

## 支持范围

- **开发语言**：AscendC（SIMD / SIMT / 混合三路线，可复用 `random_common` 公共底座）；CANNBotDSL（SIMD / SIMT 双路线，PRNG 需自研并与流基准对拍）
- **芯片**：Ascend 950PR / 950DT（arch35）及以上；TF 对标接口走 SIMD 路线，不依赖 SIMT 能力；torch 对标接口与随机 + 逐元素复合依赖芯片 SIMT 能力
- **版本**：DSL 0.8.0；SIMT 路线需 CANN ≥ 9.2.0（cannbot-dsl 仓配套 9.2.1）
- **覆盖分布**：uniform / normal / exponential / truncated normal / bernoulli / multinomial / permutation
- **能力边界**：芯片不支持 SIMT 时，torch 对标接口族与复合类算子不适用本范式；随机量参与规约等非逐元素主计算的算子属多范式并集，由主计算范式主导

## 未来支持计划

暂无
