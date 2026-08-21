---
type: CATLASS DSL Operator Example
title: Prepare Wy Repr Bwd Da
description: WY 表示反向中计算块内 A 矩阵梯度的混合核。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/README.md
    title: 接口、算法与支持范围
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/op_host/op_api/aclnn_prepare_wy_repr_bwd_da.h
    title: aclnn 接口原型
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/op_host/op_tiling/arch35/prepare_wy_repr_bwd_da_tiling_a5.cpp
    title: host tiling、模板选择与分核
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/op_kernel/prepare_wy_repr_bwd_da.cpp
    title: kernel 入口与任务类型分派
  - id: impl
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/op_kernel/arch35/prepare_wy_repr_bwd_da_vector.h
    title: 目标 kernel 的流水与数据路径
  - id: cube
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/op_kernel/arch35/prepare_wy_repr_bwd_da_cube.h
    title: Cube 基本块、矩阵乘与写回流水
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

根据 K、V、beta、前向 A、dW、dU 与 gate 形成 dA；Key head 可广播到多个 Value head，A 的最后一维与 chunk 基本块对应。[^guide]

K 为 `[B,HK,T,K]`，V/dU 为 `[B,HV,T,V]`，beta/g 为 `[B,HV,T]`，A/dA 为 `[B,HV,T,BT]`，dW 为 `[B,HV,T,K]`；要求 `HV` 是 `HK` 的整数倍，K=128，V 为 128 或 256，变长模式要求累计长度与扁平 chunk 索引。[^guide][^api]

# 用法

## 分核策略与基本块切分

目标 host tiling 生成独立数据结构并由公共入口引用；按 chunk 与 Value head 分核，grouped-head 映射选择 K head，V=256 通过 V 向基本块循环覆盖，尾 chunk 按实际长度处理。[^tiling][^entry][^impl][^cube]

K head 在一个 group 内广播，V=256 则沿 V 维分成两个 128 基本块；最有价值的复用是让同一 K tile 留在 L1，连续服务该 group 的 Value head/V tile，避免为每个 dA tile 重搬 K。[^guide][^impl][^cube]

# 代码模式

## 数据路径与存储层级

AIC 将 K/V 与梯度矩阵从 GM 搬到 L1/L0 做 MMAD，AIV 在 UB 处理 beta、gate、三角边界和逐元素项；workspace 保存 Cube 与 Vector 之间的 FP32/低精度阶段量，Fixpipe 完成 L0C 写回。[^entry][^impl][^cube]

## 流水排布、同步关系与数值精度

AIC/AIV 通过 cross-core flag 交接 workspace tile，核内 hard event 保护 MTE 与计算流水；双缓冲只允许在前一 tile 消费完成后复用。[^impl][^cube]

输入主体可为 FP16/BF16，beta/g 还可为 FP32；向量表达式和 MMAD 累加保留 FP32，再将 dA 转换为声明的低精度输出。[^guide][^impl][^cube]

# 优化决策

Ascend 950 只有一个 tiling key，host 启用全部 AIC 核；fixed chunk 数为
`B*ceil(T/chunkSize)`，varlen 来自 chunk 索引。user workspace 为
`2*B*HV*T*(chunkSize+max(K,V))` 字节，并以 schedule mode 1 配合全核同步。[^tiling]

成本来自 K/dW 与 V/dU 两支 MMAD、beta/g 向量项、`chunkSize*max(K,V)` 中间量的 GM
往返和全核 barrier。Cube 高且 V=256 更长对应 V 维两 tile；Vector 高对应 beta/g；
MTE 或 SyncAll 高对应 workspace 交接；chunk/head 少时全核启动会产生尾核等待。

按单轴顺序验证：

1. 先改善 `(chunk,hv,V-tile)` 任务分配，缩小最长核；保持 `hv -> hk` 映射和同一 dA 的归并顺序。
2. GVA 下让共享 key head 的 K/dW tile连续服务多个 Value heads；代价是 L1 生命周期和尾核不均。
3. workspace 主导时复用 K/V 两支不重叠的中间区或缩短 GM 往返；flag/barrier 不闭环即回退。
4. V=128/256、FP32 gate/beta、varlen 尾块分别门禁；不能把整块低精度结果外推。[^impl][^cube]

# 约束

- K=128，V 为 128/256，`HV` 必须是 `HK` 的整数倍。[^guide][^tiling]
- A/dA 的最后一维与 chunk 基本块一致；V/dU、beta/g 的 batch、head、序列维必须对齐。[^guide][^tiling]
- 变长模式要求累计长度与扁平 chunk 索引，且外层 B=1。[^guide][^tiling]

# 失败表现

A/dA 的 BT 与 chunk 不一致、HV/HK 非整数比或变长元数据缺失会导致 host 校验失败；分组广播或尾块三角边界错误表现为特定 head/chunk 的 dA 错位。[^guide][^tiling][^impl][^cube]

# 验证方法

覆盖两种 V、多个 head group、FP32 gate/beta、定长/变长和不足一个 chunk 的序列；与 WY 反向 reference 比较完整 dA，额外验证非法 BT 和元数据组合。[^guide][^entry][^impl][^cube]

[^guide]: 固定提交中的接口语义、算法说明、shape、dtype、layout 与支持边界。
[^api]: 固定提交中的 aclnn 参数顺序、必选/可选输入、属性与输出原型。
[^tiling]: 固定提交中的 host 参数检查、tiling 数据、模板选择、block dim、workspace 和尾块规划。
[^entry]: 固定提交中的 kernel 参数顺序、目标实现选择、tiling 读取和任务类型。
[^impl]: 固定提交中的基本块、存储层级、AIC/AIV 分工、流水同步、类型转换和特殊分支。
[^cube]: 固定提交中的 Cube 基本块、矩阵乘、L1/L0/Fixpipe 路径及同步。
