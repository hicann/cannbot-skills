<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式算子档案

Sparse（稀疏算子）范式覆盖数据面为稀疏表示的算子：数据位置由索引张量的值决定、工作量由 nnz 主导的计数累积、散射累积、稀疏行状态更新与索引重映射。

## 典型算子

- sparse_bincount：稀疏计数累积（Count 族），范式最小样例
- sparse_segment_sum_grad：散射累积（ScatterAdd 族），分段求和反向
- sparse_apply_adadelta / sparse_apply_ftrl：稀疏行状态更新（RowApply 族）
- sparse_reshape：索引坐标重映射（IndexRemap 族），值依赖输出 shape

## 典型生成耗时与性能

| 算子名称 | 模型类型 | 平均性能 | 最差性能 | 算子生成时长 |
|---------|---------|---------|---------|-------------|
| sparse_bincount | GLM-5.3 | 0.7 倍库上已有算子 | 0.3 倍库上已有算子 | 约 18 小时（不含性能优化阶段） |

总体情况：Count 族（bincount）与 ScatterAdd 族（segment_sum_grad）复用零初始化 + 散射累积两段 launch 骨架，RowApply 族（apply_adadelta/ftrl）行级状态回写与 IndexRemap 族（reshape）值依赖输出 shape 的论证较重；算子生成耗时约 18 小时。SIMT 索引驱动离散访存吞吐受 nnz 与原子冲突域影响，生成算子性能约为库上已有算子的 0.3～0.7 倍，重复索引密集的冲突场景最差性能下降明显。

## 支持范围

- **开发语言**：AscendC（`__simt_vf__` 索引驱动离散访存 + 标量原子）；CANNBotDSL（SIMT 数据面 + 原子 + AI CPU metadata 调度）
- **芯片**：Ascend 950PR / 950DT（arch35）及以上（标量原子依赖）；目标芯片不支持 SIMT 时不进入本范式，改走通用方法论论证非 SIMT 路线
- **版本**：DSL 0.8.0；SIMT（含混合）需 CANN ≥ 9.2.0（cannbot-dsl 仓配套 9.2.1）
- **覆盖算子族**：Count / ScatterAdd / RowApply / IndexRemap
- **能力边界**：dense 数据面的 gather / scatter 属 IndexGather / ScatterUpdate 范式；稀疏注意力块跳过属 FA 组合范式；稀疏×稠密矩阵乘属 Contraction 范式

## 未来支持计划

暂无
