---
type: CATLASS DSL Operator Example
title: Solve Tri
description: 分块线性注意力中下三角矩阵求逆核。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/README.md
    title: 接口、算法与支持范围
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/op_host/op_api/aclnn_solve_tri.h
    title: aclnn 接口原型
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/op_host/solve_tri_tiling.cpp
    title: host tiling、模板选择与分核
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/op_kernel/solve_tri.cpp
    title: kernel 入口与任务类型分派
  - id: impl
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/solve_tri/op_kernel/arch35/solve_tri_ascend950.h
    title: 目标 kernel 的流水与数据路径
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

使用矩阵链减半与矩阵块减半方法递归求下三角矩阵逆，AIC 执行子块矩阵乘，AIV 构造辅助块；输出 shape 与输入一致。[^guide]

输入支持 FP16/BF16；固定布局 BSND/BNSD 使用四维张量，packed 布局 TND/NTD 使用三维张量并必须提供累计长度与 chunk 索引；最后一维只支持 64 或 128。[^guide][^api]

# 用法

## 分核策略与基本块切分

入口位于公共目录并在目标编译条件下包含专门实现；host 根据 layout、矩阵阶数和序列模式设置 tiling，按 batch/head/chunk 分配独立三角矩阵，递归层级由矩阵阶数决定。[^tiling][^entry][^impl]

阶数仅为 64 或 128，递归减半后的子问题可映射到固定 Cube tile；优化重点是复用对角块逆和 Schur 更新中间量，并让 AIV 辅助矩阵生成与 AIC 子块乘交叠。[^guide][^impl]

# 代码模式

## 数据路径与存储层级

矩阵从 GM 经 L1 进入 L0，子块 MMAD 在 L0C 形成 Schur 更新，AIV 在 UB 生成符号、单位阵和辅助数据，Fixpipe 把结果写回 GM/workspace。[^entry][^impl]

## 流水排布、同步关系与数值精度

递归层之间存在严格依赖；AIC/AIV 通过 cross-core flag 交换辅助块，Cube 内用 MTE/M/FIX event 控制 L1/L0 复用，独立矩阵之间无需 atomic。[^impl]

低精度输入以 FP32 矩阵累加和向量中间量求解，最后转换回输入 dtype；接近奇异或对角尺度极端时误差会被递归乘法放大。[^guide][^impl]

# 优化决策

先确认 `chunkSize=64/128` 与四种 layout。任务数 fixed 为
`B*ceil(T/chunkSize)*H`，varlen 为 `chunkPairCount*H`；每核领取
`ceil(totalTiles/AIC核数)` 个 tile，只启用实际需要的核。Ascend 950 全程片上，仅保留系统
workspace；因此不应把 GM workspace 当成优化对象。[^tiling]

成本由 MCH/MBH 各级辅助矩阵生成、块矩阵乘和 layout 搬运组成。Cube 高对应递归块乘；
Vector 高对应辅助矩阵；MTE2 高且 BNSD/NTD 正常而 BSND/TND 偏高，对应跨 head 的
分段 DataCopy；活跃核少对应 tile 数不足，尾核长对应 varlen 长度不均。

候选按顺序单独验证：

1. 先按总 tile 调度并减少尾核空转；一个矩阵 tile 仍由同一核完成，避免跨核依赖。
2. BSND/TND 主导时合并跨 head 分段搬运或转为连续布局；转换成本必须计入端到端结果。
3. Cube 主导时分别调整 MBH 级数或 matmul tile；保持单位对角与严格下三角结构。
4. UB 允许时复用 MCH/MBH 辅助矩阵并双缓冲相邻 tile；片上容量、event 数或数值误差超限即回退。
5. 64/128、四种 layout、尾块与 varlen 分开比较，不从连续布局外推到分段布局。[^impl]

# 约束

- 输入仅 FP16/BF16，矩阵阶数仅 64/128。[^guide][^tiling]
- BSND/BNSD 为四维，TND/NTD 为三维；packed layout 必须提供累计长度和 chunk 索引。[^guide][^tiling]
- 输入必须符合求解算法假设的下三角结构；近奇异矩阵不属于稳定性能样例。[^guide][^tiling]

# 失败表现

layout 或 chunk 索引错会选择错误矩阵起点；递归块偏移错常表现为对角块正确而非对角块错误；同步缺失产生间歇性 workspace 旧值。[^guide][^tiling][^impl]

# 验证方法

构造单位下三角、随机良态下三角和两种阶数，检查 `X @ inverse(X)` 接近单位阵；覆盖四种 layout、变长尾段和多 head。[^guide][^entry][^impl]

[^guide]: 固定提交中的接口语义、算法说明、shape、dtype、layout 与支持边界。
[^api]: 固定提交中的 aclnn 参数顺序、必选/可选输入、属性与输出原型。
[^tiling]: 固定提交中的 host 参数检查、tiling 数据、模板选择、block dim、workspace 和尾块规划。
[^entry]: 固定提交中的 kernel 参数顺序、目标实现选择、tiling 读取和任务类型。
[^impl]: 固定提交中的基本块、存储层级、AIC/AIV 分工、流水同步、类型转换和特殊分支。
