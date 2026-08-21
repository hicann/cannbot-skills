---
type: CATLASS DSL Operator Example
title: Recompute W U Fwd
description: 门控 Delta Rule 前向重计算 W/U 表示的混合核。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_w_u_fwd/README.md
    title: 接口、算法与支持范围
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_w_u_fwd/op_host/op_api/aclnn_recompute_w_u_fwd.h
    title: aclnn 接口原型
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_w_u_fwd/op_host/op_tiling/recompute_w_u_fwd_tiling.cpp
    title: host tiling、模板选择与分核
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd.cpp
    title: kernel 入口与任务类型分派
  - id: impl
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_vector.h
    title: 目标 kernel 的流水与数据路径
  - id: cube
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/recompute_w_u_fwd/op_kernel/recompute_w_u_fwd_cube.h
    title: Cube 基本块、矩阵乘与写回流水
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

从 K、V、beta、块内 A 与 gate 重新构造 W 和 U，供前向输出或反向使用；重计算避免长期保存对应中间量。[^guide]

输入按 B/H/T/K、B/H/T/V 与 B/H/T/chunk 组织，支持 grouped heads、定长和变长；主体支持 FP16/BF16，门控/系数可采用 FP32，chunk 与 K/V 范围由 host 校验。[^guide][^api]

# 用法

## 分核策略与基本块切分

按 batch、Value head 和 chunk 分核，模板分派定长/变长、dtype 与 V 基本块；K head 通过 group 比例复用，尾 chunk 只处理有效 token。[^tiling][^entry][^impl][^cube]

重计算的收益来自省去长期保存 W/U，而 kernel 内应复用同一 K/V/A tile 同时生成两个输出；若两条输出路径各自重搬输入，会抵消 recompute 的带宽优势。[^guide][^impl][^cube]

# 代码模式

## 数据路径与存储层级

AIC 用 L1/L0 矩阵乘形成 W/U 主项，AIV 在 UB 完成 beta、gate、mask 与逐元素校正；中间 tile 经 workspace 往返，L0C 由 Fixpipe 写出低精度矩阵。[^entry][^impl][^cube]

## 流水排布、同步关系与数值精度

AIC/AIV 以 cross-core flag 建立生产者/消费者关系，信用反向通知 workspace slot 可复用；hard event 保护各自的搬运和计算流水。[^impl][^cube]

Cube 使用 FP32 累加，向量门控也在 FP32 中计算，写 W/U 时转换为接口 dtype；重计算结果需与保存前向中间量使用相同舍入顺序。[^guide][^impl][^cube]

# 优化决策

先按 `V=128/256` 选择 tiling key 1/2，并区分 fixed/varlen。chunk 数分别为
`B*ceil(T/chunkSize)` 和 chunk 索引对数；user workspace 为 `2*B*HV*T*V` 字节，
但 `vb` 与 `k*beta*exp(g)` 在实现中复用同一逻辑阶段区。[^guide][^tiling]

成本由两次逐元素预处理、`A @ vb`、`A @ kbg_exp`、workspace GM 往返和 AIC/AIV
交接组成。Cube 高对应两个 A 矩阵乘，Vector/Exp 高对应 g 分支，MTE/wait 高对应复用区交接；
V=256 增加 u 分支与 workspace，K 分支则受固定 K 影响。

按单轴顺序验证：

1. 先平衡 `(chunk,hv,V-tile)`，保持 `hv -> hk` 的 GVA 映射和 varlen chunk 解码。
2. 若 Cube 主导，分别调 U 或 W 的 matmul tile，一次只改一支并用两输出门禁。
3. 若 GM 主导，缩短复用 workspace 的生产消费距离，或让 A tile连续服务 U/W；代价是 A/中间量驻留压力。
4. 若 Exp 主导，只优化 `k*beta*exp(g)` 融合；FP32 g 与低精度 g 分开验证。
5. V=128/256、chunkSize=64/128、短尾 chunk 独立比较，任何一支退化即回退。[^impl][^cube]

# 约束

- K/Value head 必须满足整数 group 映射，V 支持实现声明的基本块。[^guide][^tiling]
- 定长/变长策略及尾 chunk mask 必须与生成 A 时一致。[^guide][^tiling]
- 重计算 W/U 的 dtype 转换和舍入顺序必须与消费方约定一致。[^guide][^tiling]

# 失败表现

A 的三角语义或 gate 指数方向错误会同时影响 W/U；workspace slot 过早复用造成随机块错误；grouped-head 或尾块偏移错误只污染特定 head/chunk。[^guide][^tiling][^impl][^cube]

# 验证方法

将重计算 W/U 与独立前向 reference 对比，覆盖 dtype、FP32 gate、V=128/256、grouped heads、两种 chunk、定长/变长和尾块。[^guide][^entry][^impl][^cube]

[^guide]: 固定提交中的接口语义、算法说明、shape、dtype、layout 与支持边界。
[^api]: 固定提交中的 aclnn 参数顺序、必选/可选输入、属性与输出原型。
[^tiling]: 固定提交中的 host 参数检查、tiling 数据、模板选择、block dim、workspace 和尾块规划。
[^entry]: 固定提交中的 kernel 参数顺序、目标实现选择、tiling 读取和任务类型。
[^impl]: 固定提交中的基本块、存储层级、AIC/AIV 分工、流水同步、类型转换和特殊分支。
[^cube]: 固定提交中的 Cube 基本块、矩阵乘、L1/L0/Fixpipe 路径及同步。
