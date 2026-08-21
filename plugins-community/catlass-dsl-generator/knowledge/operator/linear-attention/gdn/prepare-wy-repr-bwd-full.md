---
type: CATLASS DSL Operator Example
title: Prepare Wy Repr Bwd Full
description: WY 表示反向中联合生成 K、V、beta 与中间分支梯度的混合核。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_full/README.md
    title: 接口、算法与支持范围
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_full/op_host/op_api/aclnn_prepare_wy_repr_bwd_full.h
    title: aclnn 接口原型
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_full/op_host/op_tiling/arch35/prepare_wy_repr_bwd_full_tiling_a5.cpp
    title: host tiling、模板选择与分核
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_full/op_kernel/prepare_wy_repr_bwd_full.cpp
    title: kernel 入口与任务类型分派
  - id: impl
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_full/op_kernel/arch35/prepare_wy_repr_bwd_full_vector.h
    title: 目标 kernel 的流水与数据路径
  - id: cube
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_full/op_kernel/arch35/prepare_wy_repr_bwd_full_cube.h
    title: Cube 基本块、矩阵乘与写回流水
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

消费 K、V、beta、A、W、U 及上游梯度，按 chunk 回传 WY 表示的多个梯度分量；块内三角依赖限制计算只使用当前 chunk 的有效 token。[^guide]

接口采用 B/H/T/D 张量和可选变长累计长度/chunk 索引，主体低精度支持 FP16/BF16，标量控制量可用 FP32；K/Value head 按整数分组映射，K、V 与 chunk 大小受 README 中的离散范围限制。[^guide][^api]

# 用法

## 分核策略与基本块切分

目标 host tiling 与公共入口组合，按 batch、Value head、chunk 分派，并用模板覆盖 dtype、基本块和定长/变长策略；尾 chunk 不补入语义数据，只在对齐存储上 padding。[^tiling][^entry][^impl][^cube]

多个输出共享 K/V/beta/gate 与块内 A；workspace 的阶段量应按最后消费者释放后复用，而不是为每个梯度永久分配。尾块只对存储做对齐 padding，mask 必须阻止 padding 进入三角计算和归约。[^tiling][^impl][^cube]

# 代码模式

## 数据路径与存储层级

多次 Cube MMAD 在 GM、L1、L0A/L0B、L0C 与 Fixpipe 间传递矩阵项；AIV 在 UB 计算 gate、beta、mask、归约和类型转换；workspace 按阶段复用承载中间梯度。[^entry][^impl][^cube]

## 流水排布、同步关系与数值精度

cross-core flag 将 AIC 产生的矩阵 tile 交给 AIV，并把已消费信用反向送回；MTE2/V/MTE3 与 MTE1/M/FIX hard event 维护各存储层 buffer 生命周期。[^impl][^cube]

矩阵输入使用接口低精度，Cube 和向量归约使用 FP32 累加；输出前显式转换，gate/beta 为 FP32 时避免先降精度。[^guide][^impl][^cube]

# 优化决策

先按 `V=128/256` 选择 tiling key 1/2；fixed chunk 数为 `B*ceil(T/chunkSize)`，
varlen 来自 chunk 索引。host 启用全部 AIC 核，user workspace 为
`2*B*HV*T*(V+K)` 字节，并依赖 schedule mode 1 的全核同步。[^tiling]

成本由多支 dK/dV/dBeta/dG 矩阵项、行归约/门控、workspace GM 往返和 SyncAll 组成。
Cube 高对应 dK/dV 矩阵乘，Vector/Reduce 高对应 dBeta/dG，MTE 或 barrier 高对应阶段交接；
V=256 会增加 V 向 tile 与 workspace 流量，少 chunk 时尾核等待占比更高。

候选按顺序单独验证：

1. 先调整 `(chunk,hv,feature-tile)` 分配并测最长核；所有输出的同一 token/head 映射不变。
2. 复用 K/V/beta/g/A/dA 的公共输入，减少多输出分支重复搬运；代价是 UB/L1 压力。
3. workspace 主导时按最后消费者复用两段中间区或融合相邻生产消费；出现 cross-core 等待或任一梯度漂移即回退。
4. Vector 归约主导时只改变 dBeta/dG 行归约；保持 FP32 累加和尾块 mask。
5. 四个输出逐一设门禁，V=128/256、GVA、FP32 控制量与 varlen 分开比较。[^impl][^cube]

# 约束

- 主体张量为 FP16/BF16，beta/gate 可为 FP32；K、V、chunk 必须落在 host 声明的离散支持集合。[^guide][^tiling]
- K head 到 Value head 的 group 映射必须为整数比。[^guide][^tiling]
- 尾 chunk 的 padding 不能参与三角运算、归约或输出。[^guide][^tiling]

# 失败表现

workspace 段偏移错会同时污染多个梯度；遗漏反向信用会死锁，过早复用会产生非确定误差；尾 chunk mask、head group 或 dtype 分派错误会形成局部梯度异常。[^guide][^tiling][^impl][^cube]

# 验证方法

用自动微分 reference 覆盖所有输出、两种主体 dtype、控制量 FP32、V 上界、grouped heads、定长/变长和尾块；逐输出设独立容差并检查非法 shape。[^guide][^entry][^impl][^cube]

[^guide]: 固定提交中的接口语义、算法说明、shape、dtype、layout 与支持边界。
[^api]: 固定提交中的 aclnn 参数顺序、必选/可选输入、属性与输出原型。
[^tiling]: 固定提交中的 host 参数检查、tiling 数据、模板选择、block dim、workspace 和尾块规划。
[^entry]: 固定提交中的 kernel 参数顺序、目标实现选择、tiling 读取和任务类型。
[^impl]: 固定提交中的基本块、存储层级、AIC/AIV 分工、流水同步、类型转换和特殊分支。
[^cube]: 固定提交中的 Cube 基本块、矩阵乘、L1/L0/Fixpipe 路径及同步。
