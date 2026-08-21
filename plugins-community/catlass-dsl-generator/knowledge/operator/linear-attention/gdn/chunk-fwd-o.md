---
type: CATLASS DSL Operator Example
title: Chunk Fwd O
description: 分块门控 Delta Rule 前向输出阶段的矩阵乘与融合后处理核。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/README.md
    title: 接口、算法与支持范围
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_host/op_api/aclnn_chunk_fwd_o.h
    title: aclnn 接口原型
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_host/chunk_fwd_o_tiling.cpp
    title: host tiling、模板选择与分核
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/chunk_fwd_o.cpp
    title: kernel 入口与任务类型分派
  - id: impl
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp
    title: 主 GEMM、scheduler 与融合 epilogue 实现
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

将块内 Q/K 相关项、Value 表示和 chunk 状态组合为 attention 输出；epilogue 对块内因果区域及门控项做融合，避免另起逐元素 kernel。[^guide]

输入输出采用 B/H/T/D 及 chunk 中间矩阵布局，支持定长与通过累计长度/chunk 索引描述的变长序列；head group、K/V 维和 chunk 参数由 host 统一校验。[^guide][^api]

# 用法

## 分核策略与基本块切分

GEMM scheduler 将 batch、head 和 chunk 的 M/N tile 映射到物理核，epilogue 接收实际尾块尺寸；入口 tiling key 选择编译期 dtype/shape 模板。[^tiling][^entry][^impl]

主 GEMM 与 qk-mask/output epilogue 已融合；调优应联合选择 scheduler 的 M/N tile 和 epilogue UB 占用，避免只放大 Cube tile 后使 mask、门控与输出转换成为串行尾巴。[^tiling][^impl]

# 代码模式

## 数据路径与存储层级

主路径从 GM 经 L1/L0 读矩阵，L0C 保存 FP32 累加；epilogue 在 UB 融合 qk mask/门控与输出项，再经 Fixpipe 或向量搬出写回 GM。[^entry][^impl]

## 流水排布、同步关系与数值精度

GEMM mainloop 用 hard event 保护多级搬运，epilogue 与主循环按 tile 所有权衔接；调度器保证同一输出 tile 只由一个任务写入，无需原子累加。[^impl]

低精度矩阵输入在 L0C 以 FP32 累加，epilogue 保持 FP32 运算后转换为输出 dtype；尾块 padding 必须在 mask 后不参与有效输出。[^guide][^impl]

# 优化决策

先确认 `chunkSize=64/128`、K/V、GVA 比例和定长/变长路径。任务由 chunk 与 Value head
共同决定，但 host 固定按 AIC 核数预留双缓冲 workspace：每核分别有两段
`chunkSize*V` FP32、两段 `chunkSize^2` FP32，并另有共享 mask；短序列仍承担该固定容量与交接成本。[^tiling]

成本由状态项 `Q @ H`、块内 score、mask/gate 向量处理、再乘 V，以及多段 workspace
GM 往返组成。Cube 忙而 Vector 等待低说明矩阵乘主导；Vector/Exp 或 mask 高说明门控主导；
MTE3/MTE2 与 cross-core wait 高说明 workspace 交接主导；活跃任务少说明 chunk/head 并行不足。

候选按以下顺序单独验证：

1. 优先调整 `(chunk,hv)` 任务分配，减少尾核空转；不得改变 `hv -> hk` 的 GVA 映射。
2. workspace 流量主导时，压缩 `attn/aftermask` 的生命周期或复用 ping-pong 段；UB/flag 冲突或等待上升即回退。
3. mask/Exp 主导时，复用固定 chunk mask并合并 gate 后处理；必须保留尾块无效区和 FP32 gate 精度。
4. 分别评估 `chunkSize=64/128`：较大块提高 MMAD 工作量，也按平方扩大 score/mask 流量，不能预设更优。[^impl]

# 约束

- 输入、中间矩阵和输出的 head/chunk stride 必须与 host tiling 一致。[^guide][^tiling]
- 变长尾 chunk 只写实际 token；因果无效区不得通过 epilogue 进入输出。[^guide][^tiling]
- 每个输出 tile 由单一任务写入，不能引入未配套的跨核 partial sum。[^guide][^tiling]

# 失败表现

因果 mask 方向错误会在 chunk 对角线出现泄漏；状态项 stride 或 head 映射错误导致整头偏差；尾块有效 M/N 误用会写越界或污染 padding。[^guide][^tiling][^impl]

# 验证方法

与分块公式 reference 比较定长/变长、grouped heads、完整与尾 chunk，并分别隔离状态项和块内项；检查输出 padding 不被当作有效 token。[^guide][^entry][^impl]

[^guide]: 固定提交中的接口语义、算法说明、shape、dtype、layout 与支持边界。
[^api]: 固定提交中的 aclnn 参数顺序、必选/可选输入、属性与输出原型。
[^tiling]: 固定提交中的 host 参数检查、tiling 数据、模板选择、block dim、workspace 和尾块规划。
[^entry]: 固定提交中的 kernel 参数顺序、目标实现选择、tiling 读取和任务类型。
[^impl]: 固定提交中的基本块、存储层级、AIC/AIV 分工、流水同步、类型转换和特殊分支。
