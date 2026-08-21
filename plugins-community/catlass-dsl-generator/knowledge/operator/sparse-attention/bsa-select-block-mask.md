---
type: CATLASS DSL Operator Example
title: BSA Select Block Mask
description: Ascend 950 BSA 稀疏掩码选择的池化、QK、两遍 Online Softmax、Radix TopK 与 AIC/AIV 同步。
tags: [catlass-dsl, operator, sparse-attention, block-sparse-attention, topk, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T12:10:09Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T12:10:09Z'}
sources:
  - id: guide
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/bsa_select_block_mask/README.md
    title: BSA Select Block Mask algorithm and interface guide
  - id: entry
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/bsa_select_block_mask/op_kernel/bsa_select_block_mask.cpp
    title: kernel entry and implementation selection
  - id: kernel
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/bsa_select_block_mask/op_kernel/arch35/bsa_select_block_mask_base.h
    title: pipeline orchestration
  - id: matmul
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/bsa_select_block_mask/op_kernel/arch35/bsa_matmul_service.h
    title: Cube matmul service
  - id: softmax
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/bsa_select_block_mask/op_kernel/arch35/bsa_vec_sm_service.h
    title: two-pass online softmax service
  - id: topk
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/bsa_select_block_mask/op_kernel/arch35/bsa_radix_topk_service.h
    title: distributed radix TopK and mask writer
operator_families: [sparse-attention, block-sparse-attention]
---

# 接口与概念

## 算子算法

该算子是 Block Sparse Attention 前处理：先对每个 Q/K block 做 mean pooling，再计算
`score = scale * Qcmp @ Kcmp^T`，逐 Q block 对 K blocks 做 softmax，最后全局选择
`round(sparsity * Xblocks * Yblocks)` 个最大值并输出 INT8 二值 mask。部分压缩时每块均值
分母来自 actual block length。[^guide]

# 用法

## 分核策略与基本块切分

入口使用 mixed AIC 1:AIV 2，按模板参数选择 Q/K 的 TND 或 BNSD。每 batch 根据 actual
sequence length 重新计算有效 X/Y blocks；K pooling 在有效 AIV 上均分，Q pooling 在同一
AIC 对应的两个 AIV 间二分。AIC 按 Q chunk 和 K chunk 做 QK，AIV 紧随其后累积 softmax；
一个 batch/head 的 score 完成后，所有 AIV 协作 Radix TopK。[^entry][^kernel]

固定 head dim 为 128，blockShapeX/Y 是 64 的倍数；只支持 MHA。TND 用 token 和 block
前缀和定位每 batch，BNSD 用固定 stride。Q/K 为 FP16 或 BF16，mask 为 INT8。[^guide][^kernel]

# 代码模式

## 数据路径与存储层级

```text
Q/K GM -> AIV UB mean pooling -> qCmp/kCmp GM
qCmp/kCmp GM -> AIC L1 -> L0A/L0B -> FP32 L0C -> score FP32 GM
score GM -> AIV UB two-pass online softmax -> attention score FP16 GM
score FP16 GM -> distributed radix histograms/workspace -> INT8 mask GM
```

pool、softmax 和 radix TopK 共用各自 UB 服务，完整 score 及 TopK 的 tile histogram/count
保存在 workspace。Matmul 服务显式使用 L1 与 L0 双缓冲；TopK 每轮处理 2 个 radix bit，
FP16 共 8 轮，最后对阈值大于与等于项分别写 mask，从而在重复值下精确满足 K。[^matmul][^softmax][^topk]

## 流水排布、同步关系与数值精度

Q pooling 后 AIV 通过 ping/pong cross flag 通知 AIC；每个 K chunk 的 FIX 完成后 AIC 再
通知 AIV 做 Online Softmax 第一遍。第二遍在全部 K chunks 的全局 max/sum 已知后归一化并
cast FP16。batch/head 边界使用 `PipeBarrier<PIPE_ALL>` 与 `SyncAll`，TopK 各 radix 轮也有
全核 histogram 归并屏障。[^kernel][^softmax][^topk]

Q/K 输入保留 FP16/BF16；pool 输出供 Cube 使用，QK 与 softmax max/sum 为 FP32；归一化
score cast FP16 后参与 radix 排序，最终只回写 uint8/int8 mask。[^matmul][^softmax]

# 约束

- 仅 `__CCE_AICORE__ == 310` 时选用这里记录的 Ascend 950 headers。
- Q/K layout 仅 TND、BNSD，head 数必须相同，head dim 固定 128。
- blockShapeX/Y 必须为 64 的倍数，且 `Xblocks * Yblocks > 1`。
- `post_block_shape` 必须为空；actual block length 必须位于对应 block shape 范围。
- workspace 的 qCmp/kCmp、FP32 score、FP16 score、TopK histogram 偏移必须与 tiling 一致。

# 失败表现

- TND 第二个 batch 后错位：token 或 block 前缀和未累计。
- softmax 每个 K chunk 单独归一化：第二遍未使用跨 chunk 的 global max/sum。
- mask 中 1 的数量不等于 K：阈值等值项的配额或 radix 最后一轮累计错误。
- AIC 偶发读到旧 pooling：V→C ping/pong flag 与 qChunk parity 不一致。
- 尾 batch 多出 1：未按 actual sequence length 截断有效 X/Y blocks。

# 验证方法

reference 按 pool→QK→softmax→TopK 逐步比对，并覆盖 TND/BNSD 四种组合、FP16/BF16、
完整/部分 block、变长 batch、Q/K block 尾块、重复阈值和 sparsity 边界。除 mask 值外还要
检查每 batch/head 的 1 数量。静态检查编译日志确实选择 Ascend 950 路径。性能须在空闲
Ascend 950 上单独 profile。[^guide][^entry]

[^guide]: 固定提交中的数学定义、布局、dtype、shape 与接口限制。
[^entry]: 固定提交中的 Ascend 950 条件选择、模板布局和 mixed-core 入口。
[^kernel]: 固定提交中的动态有效块切分、阶段次序、前缀和与跨核同步。
[^matmul]: 固定提交中的 L1/L0 双缓冲 QK Cube 数据路径。
[^softmax]: 固定提交中的 FP32 两遍 Online Softmax 和同步实现。
[^topk]: 固定提交中的分布式 Radix TopK、阈值等值处理和 mask 回写。
