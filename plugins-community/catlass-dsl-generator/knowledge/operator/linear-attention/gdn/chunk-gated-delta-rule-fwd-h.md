---
type: CATLASS DSL Operator Example
title: Chunk Gated Delta Rule Fwd H
description: 分块门控 Delta Rule 的跨 chunk 隐藏状态传播核。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/README.md
    title: 接口、算法与支持范围
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/op_api/aclnn_chunk_gated_delta_rule_fwd_h.h
    title: aclnn 接口原型
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_host/chunk_gated_delta_rule_fwd_h_tiling.cpp
    title: host tiling、模板选择与分核
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/chunk_gated_delta_rule_fwd_h.cpp
    title: kernel 入口与任务类型分派
  - id: impl
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_gated_delta_rule_fwd_h/op_kernel/arch35/gemm/kernel/gdn_fwd_h_kernel.hpp
    title: 目标 kernel 的流水与数据路径
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

按时间顺序消费每个 chunk 的 K/W/U 与 gate 表示，传播隐藏状态并输出供最终阶段使用的 chunk 状态；可选初始状态决定首 chunk 起点。[^guide]

接口按 batch、Value head、chunk、K/V 维组织状态和 WY 中间量，支持定长/变长 chunk 索引；状态布局与输出布局必须和 host 推导的 stride 一致。[^guide][^api]

# 用法

## 分核策略与基本块切分

GEMM scheduler 在可并行的 batch/head 上分核，同一状态链内按 chunk 顺序执行；目标入口选用专门的 scheduler 与 epilogue，尾 chunk 由实际长度控制。[^tiling][^entry][^impl]

跨 chunk 状态依赖限制了序列方向并行度，性能主要来自 batch/head 链并行、状态 tile 常驻以及 MMAD 与 epilogue 的交叠；同一链不能为了增加 block 数拆给多个核。[^tiling][^impl]

# 代码模式

## 数据路径与存储层级

状态和 WY 矩阵由 GM 经 L1/L0 进入 MMAD，L0C 累加后由 epilogue 在 UB 融合门控并写回下一 chunk 状态与 workspace。[^entry][^impl]

## 流水排布、同步关系与数值精度

同一状态链具有严格先后依赖；AIC/AIV tile 通过 cross-core flag 和 hard event 交接，阶段外依赖以 GM workspace/状态写回可见性维持。[^impl]

低精度矩阵通过 Cube 计算，状态更新与门控在 FP32 中间量上完成，再按 state/output dtype 转换；重复 chunk 的累计误差应以最终状态重点检查。[^guide][^impl]

# 优化决策

先确认 state dtype、是否有 `g/gk`、K/V、chunkSize 与 fixed/varlen 路径。host 使用全部 AIC 核，
并为每核双 slot 分配 `V`、`VUpdate`、可选 `KDecay` 及 `H(K*V)` 的 FP32 workspace；
短任务仍支付全核容量和 AIC/AIV 交接成本。[^tiling]

成本分为块内 `K/W/U/V` 变换、跨 chunk 状态更新、gate 向量处理和 workspace 往返。
Cube 高对应状态矩阵乘；Vector/Exp 高对应 gate；MTE 与 cross-core wait 高对应双 slot 生产消费；
尾核时间高常对应 varlen chunk 数或 Value-head 分配不均。

候选按顺序单独验证：

1. 先调整 chunk/head 到 core 的映射，目标是缩小最长任务；必须保持每条序列的状态前向顺序。
2. workspace 主导时减少 `V/VUpdate/KDecay` 的重复写读或缩短 slot 生命周期；代价是 UB/L1 占用和更紧的 flag 编排。
3. GVA 下复用同一 key head 的 K/W，并连续处理其 Value heads；若引入尾核不均或缓存挤压则回退。
4. state 为 FP32、存在 `gk`、尾 chunk 三条路径分别验证；低精度无 gate 快路径不能代表它们。[^impl][^cube]

# 约束

- 同一 batch/head 的 chunk 必须按时间顺序更新状态。[^guide][^tiling]
- 可选初始状态为空时走零状态分支；状态 K/V 末维顺序必须与接口一致。[^guide][^tiling]
- 尾 chunk 的有效长度只影响当前块，不能改变下一条序列的状态起点。[^guide][^tiling]

# 失败表现

chunk 顺序或初始状态选择错误通常首块之后快速发散；状态 V/K 转置或 stride 错误形成规则转置偏差；同步缺失会产生非确定的旧状态。[^guide][^tiling][^impl]

# 验证方法

覆盖有无初始状态、多 chunk、单个尾 chunk、变长 batch 与 grouped heads；同时比较每个 chunk 状态和最终状态，不只比较 attention 输出。[^guide][^entry][^impl]

[^guide]: 固定提交中的接口语义、算法说明、shape、dtype、layout 与支持边界。
[^api]: 固定提交中的 aclnn 参数顺序、必选/可选输入、属性与输出原型。
[^tiling]: 固定提交中的 host 参数检查、tiling 数据、模板选择、block dim、workspace 和尾块规划。
[^entry]: 固定提交中的 kernel 参数顺序、目标实现选择、tiling 读取和任务类型。
[^impl]: 固定提交中的基本块、存储层级、AIC/AIV 分工、流水同步、类型转换和特殊分支。
