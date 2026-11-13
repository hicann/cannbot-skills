<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# ArgReduce 范式算子档案

ArgReduce（选择归约）范式覆盖沿单一选择轴选出胜者并返回其位置（可选同时返回值）的算子：argmax/argmin、max/min(dim) 双输出、谓词首达、top-k 位置输出等。核心是选择代数（主规则/平局/NaN）与位置流；块内、跨块、跨核三处共享同一套选择规则。

## 典型算子

- argmax / argmin：位置单输出（AscendC arg_max_custom 已交付；argmin 以盲测验证）
- arg_max_with_value / arg_min_with_value：值+位置双输出（CANNBotDSL 两者已交付；AscendC 侧 arg_max_with_value 已交付）
- arg_first / arg_last / find_index：谓词首达变体
- arg_topk / topk 位置输出：top-k 变体（位置输出必含 ArgReduce，排序语义可叠加 SortSelect）

## 典型生成耗时与性能

| 算子名称 | 模型类型 | 平均性能 | 最差性能 | 算子生成时长 |
|---------|---------|---------|---------|-------------|
| arg_max_with_value（AscendC） | CANNBot | 9.79 us（isolated p50，1.32× aclnnArgMax） | —（未采集） | 约 27 小时（含验证与性能摸底） |
| arg_max_with_value（CANNBotDSL） | CANNBot | 14.206 us（kernel 10 采样最小值） | 15.916 us（10 采样最大值） | 约 9.9 小时（5 轮） |
| arg_min_with_value（CANNBotDSL） | CANNBot | 4.987 us（kernel 10 采样最小值） | 6.693 us（10 采样最大值） | 约 21 小时（2 轮） |
| arg_min_with_value（AscendC） | CANNBot | 约 0.52× aclnnArgMin | —（未采集） | 约20小时 |

总体情况：AscendC 侧 arg_max_custom 在 [2,100,4000] fp16 dim=1 的 isolated p50 为 9.79 us，优于官方 aclnnArgMax 12.93 us（1.32×），语义门 8000/8000 bitwise；CANNBotDSL 侧 arg_max_with_value / arg_min_with_value 分别以 14.206 us / 4.987 us（kernel Task Duration 最小值）完成性能摸底，均无性能硬门槛。DSL 首轮设计曾因跨 tile UB 状态槽机制在 cannbot-dsl 0.7.1 不可实现而失败，修订为 GM workspace + launch 边界合并后完成；生成耗时含设计修订与多轮验收，不含后续性能优化。模型类型按生成时使用的 CANNBot 会话记录，底层模型版本未单独归档。

## 支持范围

- **开发语言**：AscendC（regbase，8 模式体系）；CANNBotDSL 0.8.0（SIMD VF，B0/B1 路径 + EMPTY/COPY/AR/ARA 场景三件套）
- **芯片**：Ascend 950PR / 950DT（arch35）及以上；本范式不依赖 SIMT/原子，核心是向量比较选择与位置流
- **版本**：CANNBotDSL 0.8.0（按 0.8.x 文档核对）；CANN 9.2.1（cannbot-dsl 仓配套）；AscendC 侧按 CANN 版本
- **覆盖算子族**：位置单输出（argmax/argmin）、值+位置双输出（arg_max/arg_min_with_value）、谓词首达（arg_first/arg_last/find_index）、top-k 位置（arg_topk/topk(k)）、选择前后接 elementwise 的复合
- **能力边界**：只输出值的归约（sum/max/min）属 Reduction；纯排序/argsort 的排序语义属 SortSelect（位置输出仍叠加 ArgReduce）；输出 shape 依赖数据值的 nonzero/unique 属 VariableOutput；稀疏、量化、矩阵乘不属本范式

## 未来支持计划

暂无
