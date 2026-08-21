---
type: CATLASS DSL Operator Example
title: Recurrent Gated Delta Rule
description: Ascend 950 Recurrent Gated Delta Rule 的逐 token 状态更新、变长分核、UB 数据路径和 VF 数值策略。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, recurrent, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T12:10:09Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T12:10:09Z'}
sources:
  - id: guide
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/recurrent_gated_delta_rule/docs/aclnnRecurrentGatedDeltaRule.md
    title: Recurrent Gated Delta Rule API contract
  - id: entry
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/recurrent_gated_delta_rule/op_kernel/recurrent_gated_delta_rule_apt.cpp
    title: kernel entry and state dtype dispatch
  - id: kernel
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/recurrent_gated_delta_rule/op_kernel/arch35/recurrent_gated_delta_rule.h
    title: recurrent vector kernel
  - id: vf
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/recurrent_gated_delta_rule/op_kernel/arch35/vf_vec_mul_mat.h
    title: vector-times-matrix VF primitive
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

该核按时间顺序直接实现 Gated Delta Rule：先用 `exp(g + gk)` 衰减状态，再计算
`delta = beta * (v - S k)`，执行 `S = S + delta k^T`，最后输出 `o = S(q*scale)`。
`g` 是每 token/head 标量，`gk` 是 Dk 向量，二者均可选；状态按 `ssmStateIndices`
寻址，MTP 时从 `seq0 + numAcceptedTokens - 1` 选择初始状态槽。[^guide][^kernel]

# 用法

## 分核策略与基本块切分

kernel 是 AIV-only。Host 给出每 batch 的序列长度，核内先计算 `realT * Nv` 的平均负载，
然后以“完整 batch/head 序列”为不可拆单元分给连续 block；同一 head 的 token 递推绝不跨核。
Dv 再按 `vStep = floor(min(255,8192/alignDk)/16)*16` 切片，使一片状态行、Q/K/V 和临时
FP32 数据容纳于 UB。状态 dtype 由 tiling key 选择 BF16 或 FP32。[^entry][^kernel]

主要 ABI 为 Q/K `(T,Nk,Dk)`、V/O `(T,Nv,Dv)`、beta/g `(T,Nv)`、gk
`(T,Nv,Dk)`、state `(BlockNum,Nv,Dv,Dk)`，以及 sequence length、state index 和
accepted-token 元数据。`Nv % Nk == 0`，Q/K head 用 `head_i / (Nv/Nk)` 映射。[^guide][^kernel]

# 代码模式

## 数据路径与存储层级

```text
state BF16/FP32 GM -> queue -> FP32 state UB (Dv tile x alignDk)
Q/K/V BF16 GM     -> queue -> FP32 Q/K/V UB
g/gk FP32 + beta BF16 GM -> scalar/vector UB
FP32 VF recurrence -> BF16/FP32 state GM + BF16 output GM
```

Q/K/V 对完整 head sequence 各搬入一次并 cast 到 FP32；状态按 Dv tile 搬入。每个 token
依次调用 VF 的 row-vector×matrix、ReduceSum 和 outer-add，避免依赖 Torch 或外部融合算子。
Dk/Dv 尾部按 32B 对齐并补零，reduce 使用 `alignDk`，因此 padding 必须保持为零。[^kernel][^vf]

## 流水排布、同步关系与数值精度

单 AIV 上以 `TQue` 管理输入、状态输出和 attention 输出，以 `TBuf` 常驻 FP32 状态、Q/K/V
及 reduce 临时区。阶段间用 `PipeBarrier<PIPE_V>()` 保证衰减、`S k`、delta、outer-add 和
`S q` 的依赖；copy queue 负责 MTE2/MTE3 所有权。无跨核 flag，因为状态序列不跨核。
[^kernel]

输入输出是 BF16；g/gk、衰减、状态计算、点积、delta 和输出累加为 FP32。仅回写 state/O
时分别按 tiling key cast 为 BF16 或保留 FP32、以及 cast 为 BF16。[^entry][^kernel]

# 优化决策

先按 BF16/FP32 state 选择两个 tiling key，并记录序列长度、batch、Nv、Dk/Dv 与 varlen 分布。
入口是 AIV-only；分核沿 `(sequence,value-head,Dv block)`，每个任务内部逐 token 串行更新状态，
所以增加 T 不增加可并行任务。[^entry][^kernel]

成本每 token 约为读取 Q/K/V/g/beta、两次状态矩阵-向量作用、一次 K×V 外积更新和 O 写回；
state 常驻 UB 可避免逐 token GM 往返，但 Dk×Dv 占用限制并行 tile。profiler 中活跃核少对应
`sequence*Nv*DvBlock` 不足，Vector 高对应 matvec/outer，Exp 高对应 gate，MTE 高对应 Q/K/V
小批搬运；不同长度序列造成尾核长尾。

按单轴顺序验证：

1. 先调整 Dv block 与任务映射，平衡 UB 状态容量和活跃核；状态分片边界及 O 拼接不变。
2. 变长场景按估算 token 工作量而非任务数分配；不能拆开同一 `(sequence,head,Dv block)` 的时间链。
3. 搬运主导时扩大 token micro-batch或双缓冲 Q/K/V/g/beta；代价是 UB，且不能提前越过序列边界。
4. Vector 主导时分别优化 matvec 或 outer-add VF，一次只改一个；FP32 state 更新顺序保持。
5. recurrent 适合 decode/短序列；长 prefill 若顺序链主导，应比较 chunk 算子而非宣称本核可消除依赖。[^kernel]

# 约束

- 仅提取 Ascend 950 的 AIV-only 路径，不应套用 arch22 实现。
- 每条序列长度 `0 < Li <= 8`，`Nk <= Nv` 且 `Nv % Nk == 0`；Dk/Dv 不超过 512。
- `ssmStateIndices[i] < BlockNum`，MTP accepted token 必须位于当前序列范围内。
- state 只允许 0/1 轴非连续；stride 必须通过 tiling 传入并用于读写。
- token 循环必须串行，不能把同一 head 的不同 token 独立分核。

# 失败表现

- 仅首 token 正确：outer-add 后状态未在 UB 中原地保留，或 token 循环被并行化。
- 不带 g/gk 时输出非有限：缺省衰减 UB 未置零后取 exp。
- Dk 非 16 倍数错误：padding 未清零却按 `alignDk` reduce。
- MTP 状态错位：初始 `stateBlockIdx` 未使用 accepted-token 偏移。
- BF16 state 正常、FP32 state 错：queue 大小、cast 分支或 state stride 按 BF16 假定。

# 验证方法

用逐 token FP32 oracle 比较 O 和每个 `ssmStateIndices` 对应的 state；覆盖 g/gk 的四种组合、
BF16/FP32 state、GQA、Dk/Dv 非对齐尾块、batch 变长、MTP 与非连续 state stride。静态检查
Ascend 950 入口为 `KERNEL_TYPE_AIV_ONLY` 且仅有两个 state tiling key。性能需另在空闲 Ascend
950 上测量。[^guide][^entry]

[^guide]: 固定提交中的完整 ABI、shape、限制和调用验证入口。
[^entry]: 固定提交中的 Ascend 950 APT 入口与 BF16/FP32 state 分派。
[^kernel]: 固定提交中的变长分核、UB 切片、递推顺序、stride 和精度实现。
[^vf]: 固定提交中的向量乘矩阵 VF 基本块。
