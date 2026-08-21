---
type: CATLASS DSL Operator Example
title: Chunk Local Cumsum
description: 按 chunk 重置的局部门控前缀和向量核。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/README.md
    title: 接口、算法与支持范围
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/op_host/op_api/aclnn_chunk_local_cumsum.h
    title: aclnn 接口原型
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/op_host/chunk_local_cumsum_tiling.cpp
    title: host tiling、模板选择与分核
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/op_kernel/chunk_local_cumsum.cpp
    title: kernel 入口与任务类型分派
  - id: impl
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_fwd/chunk_local_cumsum/op_kernel/chunk_local_cumsum_tiling_data.h
    title: 目标 kernel 的流水与数据路径
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

沿序列维对每个 head 的 gate 做 chunk 内累计和，到新 chunk 或新序列边界时重置；变长模式用累计长度定位每条序列。[^guide]

输入与输出 shape 相同，支持固定 B/H/T 与 packed token 布局；layout、chunk 大小和可选累计长度由 host 决定，输出 dtype 跟随接口定义。[^guide][^api]

# 用法

## 分核策略与基本块切分

纯向量任务把 `outer * chunk * ceil(H/512)` 切成独立工作项；定长 chunk 数为
`ceil(T/chunkSize)`，变长 chunk 数来自索引，block 数为 `min(AIV核数,taskNum)`。
每个工作项内部按 token 顺序扫描，尾 chunk 使用剩余长度。[^tiling][^entry][^impl]

这是纯 AIV scan，调优变量是每核 chunk 数、单次 GM↔UB token tile 和尾块比例；增加 workspace 或 Cube 路径不会减少 chunk 内的顺序依赖。[^tiling][^entry][^impl]

# 代码模式

## 数据路径与存储层级

gate 从 GM 分片搬入 UB，向量单元完成累计并将结果经 MTE3 写回 GM；无 Cube、L1、L0 或 Fixpipe 数据路径，也不需要 user workspace 交换矩阵。[^entry][^impl]

## 流水排布、同步关系与数值精度

单工作项内的前缀依赖串行保持，工作项之间互不共享输出；队列和 MTE2/V/MTE3 event 防止 UB buffer 被提前复用，不需要跨核 flag 或 atomic。[^impl]

累计按 kernel 声明的计算类型执行；低精度输入的长 chunk 更易积累舍入误差，验证时应与 FP32 reference 比较并关注 chunk 末值。[^guide][^impl]

# 优化决策

成本近似为每元素一次 GM 读写、FP32 累加和可选 scale/cast；没有 Cube 或 user workspace，
所以优化重点是 AIV 占用、搬运连续性和 chunk 内顺序链。profiler 中活跃核少对应
`outer*chunk*headTile` 不足，MTE2/MTE3 高对应小 tile/非连续访问，Vector 长且随 chunkSize
增长对应 scan 依赖。

按单轴顺序验证：

1. 先联合调整 host/kernel 的 512-head tile 或工作项映射提高活跃核数；若 head 尾 tile 增多或搬运变碎则回退。
2. 搬运主导时扩大连续 GM↔UB tile或双缓冲；必须保留每个 `(sequence,head,chunk)` 的独立累加器。
3. scan 主导时只评估分段前缀和加段间修正；代价是额外 UB/同步，结果必须逐 token 等价。
4. fixed 与 varlen 分开判断；变长索引解析和大量短尾块可能抵消大 tile 收益。[^tiling][^impl]

# 约束

- 累计在每个 chunk 和每条 packed 序列边界重置。[^guide][^tiling]
- 输出 shape/layout 与输入一致，尾 chunk 只处理剩余 token。[^guide][^tiling]
- 空段是否支持以 host 校验和 kernel 分支为准，不能读取不存在的首 token。[^guide][^tiling]

# 失败表现

忘记在 chunk/序列边界清零会使首 token 带入前块和；尾块长度错导致越界；layout stride 错会表现为 head 间交叉累计。[^guide][^tiling][^impl]

# 验证方法

覆盖长度小于、等于和大于 chunk，非整除尾块，多 batch/head 以及含空段的变长边界；逐 token 比较而非只验 chunk 末值。[^guide][^entry][^impl]

[^guide]: 固定提交中的接口语义、算法说明、shape、dtype、layout 与支持边界。
[^api]: 固定提交中的 aclnn 参数顺序、必选/可选输入、属性与输出原型。
[^tiling]: 固定提交中的 host 参数检查、tiling 数据、模板选择、block dim、workspace 和尾块规划。
[^entry]: 固定提交中的 kernel 参数顺序、目标实现选择、tiling 读取和任务类型。
[^impl]: 固定提交中的基本块、存储层级、AIC/AIV 分工、流水同步、类型转换和特殊分支。
