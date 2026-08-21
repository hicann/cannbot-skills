---
type: CATLASS DSL Operator Example
title: Chunk Scaled Dot Kkt
description: 按 chunk 计算缩放 Key-Key 转置乘积的矩阵核。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/README.md
    title: 接口、算法与支持范围
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/docs/aclnnChunkScaledDotKkt.md
    title: aclnn 接口原型
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/op_host/chunk_scaled_dot_kkt_tiling.cpp
    title: host tiling、模板选择与分核
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/op_kernel/chunk_scaled_dot_kkt.cpp
    title: kernel 入口与任务类型分派
  - id: impl
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_scaled_dot_kkt/op_kernel/chunk_scaled_dot_kkt.h
    title: 目标 kernel 的流水与数据路径
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

在每个 chunk 内计算缩放后的 K 与 K 转置乘积，并按算法要求处理三角区域；结果作为后续 WY/Delta Rule 阶段的块内矩阵。[^guide]

K 与输出分别使用序列特征布局和带 chunk 最后一维的矩阵布局；支持固定/packed 序列，scale、chunk 大小、layout 以及累计长度/chunk 索引共同确定 shape 和访问。[^guide][^api]

# 用法

## 分核策略与基本块切分

tiling key header 定义模板字段，host 依据 dtype、layout 和块形状选择实例；block scheduler 按 batch、head、chunk 切分，尾 chunk 以有效 M/N 限制写回。[^tiling][^entry][^impl]

任务数为 `B * Hk * NT`，其中 fixed `NT=ceil(T/BT)`、varlen `NT` 来自 chunk 索引。
AIC 数为 `min(taskNum,AIC核数)`，AIV 最多按 AIC 1:2 配对；K tile 同时作为 A/B
转置视图，尾块只缩小有效 M/N。[^tiling][^impl]

# 代码模式

## 数据路径与存储层级

K×Kᵀ 先以 FP32 写入 `taskNum*BT^2*4B` GM workspace；AIV 再将 score、g、beta
搬入 UB，按 8 行一组计算 gate、严格下三角 mask 和写回。该实现以 GM workspace 交接
Cube 与 Vector，不是直接 Fixpipe 到最终输出。[^tiling][^entry][^impl]

## 流水排布、同步关系与数值精度

L1/L0 ping-pong 由 MTE1/MTE2/M/FIX event 保护；每个输出 chunk 独占写入，不需要 atomic，跨核也不共享部分和。[^impl]

低精度 K 输入由 Cube 做 FP32 累加，scale 在输出转换前应用；三角无效区必须显式清零，不能依赖 padding 的偶然值。[^guide][^impl]

# 优化决策

先确认 dtype 与 `BT=16/32/64/128` 对应 tiling key。成本为 `BT^2*K` MMAD、
`BT^2` FP32 workspace 写读、g/beta 向量读和严格下三角后处理；BT 增大时计算与中转均按平方增长。
Cube 高对应 KKT，MTE2/MTE3 高对应 score 往返，Vector 高对应 gate/mask，AIC/AIV wait
高对应 1:2 配对失衡，任务数不足则表现为活跃核少。

候选按顺序单独验证：

1. 先比较 BT，兼顾 chunk 数、尾块比例和 `BT^2` workspace；不改变算法 chunkSize。
2. Cube 主导时保留 K 的 A/B 复用并调 M/N/K tile；L1/L0 超限或尾块效率下降即回退。
3. GM 中转主导时流水化 score 生产消费或缩短 workspace 生命周期；必须保持 AIC/AIV 所有权同步。
4. Vector 主导时增大 8-row gate tile或融合 mask/scale；严格下三角、FP32 输出和 varlen 边界不变。[^impl]

# 约束

- 输入低精度 dtype、layout 与 chunk 输出矩阵格式必须匹配 host 分派。[^guide][^tiling]
- 尾 chunk 的无效行列及算法要求的三角无效区必须清零。[^guide][^tiling]
- scale 只应用一次，K 的转置通过正确 stride/view 表达。[^guide][^tiling]

# 失败表现

K 转置 stride 错会得到非对称矩阵；scale 应用两次或遗漏产生固定倍率偏差；尾 chunk 无效区和三角 mask 错误会污染后续求解。[^guide][^tiling][^impl]

# 验证方法

比较 FP32 `K @ K^T * scale` reference，覆盖不同 chunk、尾块、定长/变长和多 head；额外断言对称/三角约束与 padding 区处理。[^guide][^entry][^impl]

[^guide]: 固定提交中的接口语义、算法说明、shape、dtype、layout 与支持边界。
[^api]: 固定提交中的 aclnn 参数顺序、必选/可选输入、属性与输出原型。
[^tiling]: 固定提交中的 host 参数检查、tiling 数据、模板选择、block dim、workspace 和尾块规划。
[^entry]: 固定提交中的 kernel 参数顺序、目标实现选择、tiling 读取和任务类型。
[^impl]: 固定提交中的基本块、存储层级、AIC/AIV 分工、流水同步、类型转换和特殊分支。
