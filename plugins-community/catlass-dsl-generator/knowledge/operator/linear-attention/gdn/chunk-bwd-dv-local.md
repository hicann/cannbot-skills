---
type: CATLASS DSL Operator Example
title: Chunk Bwd Dv Local
description: 分块门控 Delta Rule 反向的块内 Value 梯度核。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/README.md
    title: 接口、算法与支持范围
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/op_host/op_api/aclnn_chunk_bwd_dv_local.h
    title: aclnn 接口原型
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/op_host/chunk_bwd_dv_local_tiling.cpp
    title: host tiling、模板选择与分核
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/op_kernel/chunk_bwd_dv_local.cpp
    title: kernel 入口与任务类型分派
  - id: impl
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/op_kernel/arch35/chunk_bwd_dv_local_vector.h
    title: 目标 kernel 的流水与数据路径
  - id: cube
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dv_local/op_kernel/arch35/chunk_bwd_dv_local_cube.h
    title: Cube 基本块、矩阵乘与写回流水
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

先由 Q/K 形成每个 chunk 的分数矩阵，再对每个 Value head 施加指数 gate 与上三角含对角 mask，最后与 dO 相乘得到块内 dV；`H_do/H_qk` 决定 grouped-head 映射。[^guide]

Q/K 为 `[B,H_qk,T,K]`，dO/dV 为 `[B,H_do,T,V]`，g 为 `[B,H_do,T]`；`H_do % H_qk == 0`，K=128，V 为 128 或 256，chunk 为 64 或 128。累计长度与 chunk 索引必须同时出现，变长模式要求外层 B=1；两个预留输入必须为空。[^guide][^api]

# 用法

## 分核策略与基本块切分

入口模板同时分派定长/变长策略、Q 与 gate dtype、以及 V 基本块；混合任务按 1 个 AIC 配 2 个 AIV。Cube 以 Q/K head 为任务生成可共享分数，Vector 和后续 Cube 再按 Value head 消费，尾 chunk 由策略返回实际长度。[^tiling][^entry][^impl][^cube]

优化关键是复用 `K·Q^T`：该分数按 Q/K head 生成一次，再由同组多个 Value head 的 AIV gate 阶段消费；若直接按 Value head 重算，会把 GVA 场景的 Cube 工作放大 `H_do/H_qk` 倍。[^guide][^impl][^cube]

# 代码模式

## 数据路径与存储层级

Q/K 从 GM 经 L1、L0 进入 MMAD，分数写入 workspace；AIV 在 UB 读 gate 和分数，生成 gated score 回 workspace；AIC 再读 gated score 与 dO，L0C 经 Fixpipe 写 dV。[^entry][^impl][^cube]

## 流水排布、同步关系与数值精度

目标实现用可反向复用的 cross-core flag 串接 score、gated score 与最终 MMAD；本核 hard event 保护 MTE2/Vector/MTE3，workspace 是两类核的显式交接面。[^impl][^cube]

Q/K/dO 支持 FP16 或 BF16，g 允许同类低精度或 FP32；指数、mask 和逐元素乘在向量路径提升到 FP32，Cube 累加后按 dV dtype 写回。[^guide][^impl][^cube]

# 优化决策

先按 `chunkSize`、`V`、Q/K 与 gate dtype 确认 tiling key。定长任务数为
`B * ceil(T/chunkSize)`，变长任务数来自 chunk 索引；host 只启用
`min(AIC核数, 任务数)` 个核。每核 workspace 为
`headBufNum * chunkSize^2 * sizeof(q/k dtype)`，因此小任务首先受并行度限制，
大 `chunkSize`/head buffer 则增加 GM 中转和同步成本。[^tiling]

profiler 中 Cube 高且利用率正常对应两次矩阵乘；Vector/Exp 高对应 gate 与上三角 mask；
workspace 写读或 AIC/AIV 等待高对应 score 交接；活跃核少对应 chunk 任务不足。

按单轴顺序验证：

1. 先增加独立 chunk 数或重排任务，不改变 chunk 内上三角语义；若尾块比例或调度开销上升则回退。
2. 再评估同一 Q/K score 是否可连续服务其映射的多个 dO head，目标是减少重复 `K @ Q^T`；代价是延长 score 生命周期并增加 head buffer 压力。
3. workspace 成为主因时，尝试缩短 score 的 GM 往返或流水化 Phase 1/1.5/2；必须保持 cross-core 信用闭环，出现等待长尾或非确定误差即回退。
4. `V=256`、变长和尾 chunk 单独验证，不能把 `V=128` 整块结论外推。[^impl][^cube]

# 约束

- K=128，V 为 128/256，chunk 为 64/128，且 `H_do % H_qk == 0`。[^guide][^tiling]
- 累计长度和 chunk 索引必须同时出现；变长模式只支持外层 B=1。[^guide][^tiling]
- gamma 与 A 为预留输入，当前必须为空。[^guide][^tiling]

# 失败表现

未成对提供变长元数据、V/K 或 chunk 超出支持集合会被 host 拒绝；head 映射错误使同组 Value head 使用错误的 Q/K score；尾块 mask 或 flag 次序错误会产生三角区泄漏或旧 workspace 数据。[^guide][^tiling][^impl][^cube]

# 验证方法

覆盖 V=128/256、chunk=64/128、`H_do=H_qk` 与 grouped-head、定长和不整除尾块的变长序列，并用显式 masked 矩阵乘 reference 比较 dV。[^guide][^entry][^impl][^cube]

[^guide]: 固定提交中的接口语义、算法说明、shape、dtype、layout 与支持边界。
[^api]: 固定提交中的 aclnn 参数顺序、必选/可选输入、属性与输出原型。
[^tiling]: 固定提交中的 host 参数检查、tiling 数据、模板选择、block dim、workspace 和尾块规划。
[^entry]: 固定提交中的 kernel 参数顺序、目标实现选择、tiling 读取和任务类型。
[^impl]: 固定提交中的基本块、存储层级、AIC/AIV 分工、流水同步、类型转换和特殊分支。
[^cube]: 固定提交中的 Cube 基本块、矩阵乘、L1/L0/Fixpipe 路径及同步。
