---
type: CATLASS DSL Operator Example
title: Lightning Indexer
description: Ascend 950 Lightning Indexer 的 QK 分块、ReLU 加权归约、流式 TopK、PagedAttention 寻址和 C/V 流水。
tags: [catlass-dsl, operator, sparse-attention, lightning-indexer, topk, paged-attention, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T12:10:09Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T12:10:09Z'}
sources:
  - id: guide
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/lightning_indexer/README.md
    title: Lightning Indexer API and algorithm guide
  - id: entry
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/lightning_indexer/op_kernel/lightning_indexer.cpp
    title: kernel entry, implementation selection and template dispatch
  - id: kernel
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/lightning_indexer/op_kernel/arch35/lightning_indexer_kernel.h
    title: split-core and pipeline orchestration
  - id: cube
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/lightning_indexer/op_kernel/arch35/lightning_indexer_service_cube.h
    title: QK Cube service
  - id: vector
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/lightning_indexer/op_kernel/arch35/lightning_indexer_service_vector.h
    title: weighted reduction and TopK service
  - id: vf
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/lightning_indexer/op_kernel/arch35/vf/lightning_indexer_vector1.h
    title: ReLU weighted-reduction VF
operator_families: [sparse-attention, lightning-indexer]
---

# 接口与概念

## 算子算法

对每个 query token，先计算 group 内 Q 与全部 K 的点积，对点积执行 ReLU，再用 group
权重 W 加权求和，最后返回最大 `sparse_count` 个 K 位置及可选 value：
`TopK(sum_g(W_g * ReLU(Q_g K^T)))`。该核支持普通 BSND/TND K，以及通过 block table
寻址的 PA_BSND K，并支持 right-down causal 裁剪。[^guide]

# 用法

## 分核策略与基本块切分

任务按 `(batch, K-head, grouped-Q block, K block)` 展平后连续分核。固定 D=128、K head=1；
Q 的基础 M 块通常为 `4*g`，`sparse_count>2048` 时改为 `2*g`，K 基础块为 128。
一个 AIC 对应两个 AIV：AIC 计算 `QK^T`，两个 AIV 沿 query token 二分，执行 ReLU、权重
归约和最终 TopK。sparse mode 3 会按 query 位置缩短 K block 循环。[^kernel][^entry]

模板覆盖 FP16/BF16 Q/K、BSND/TND Q、BSND/TND/PA_BSND K、paged 开关及 weight dtype
标志。输出 index 是 INT32；optional value 与 K dtype 一致。[^guide][^entry]

# 代码模式

## 数据路径与存储层级

```text
Q/K GM (PA 时 K 经 block table) -> L1 -> L0A/L0B -> FP32 L0C
  -> dual-destination FIX -> 两个 AIV 的 ping/pong UB
weights GM -> AIV UB
  -> ReLU(QK) * W -> group reduce -> per-core score FP32 workspace
score workspace -> streaming/merge TopK VF -> index/value GM
```

每个 AI core 的 workspace 保存 `s1BaseSize * align(S2,128)` 个 FP32 score。Cube 侧 Q L1
双缓冲、K L1 三缓冲、L0 与 L0C 双缓冲；FIX 的 dual destination 将 M 维一分为二送到对应
AIV。Vector 侧在所有 S2 blocks 写完后才对整行做 TopK。[^kernel][^cube][^vector]

## 流水排布、同步关系与数值精度

mode-4 cross flags 构成双向 ping/pong：AIV 初始释放两个 UB 槽；AIC 等待 V→C 后写 FIX，
再发 C→V；AIV 消费 QK、加载 W、写 score workspace 后反向释放。Cube 内 MTE2→MTE1→M→FIX
使用独立 event 管理多级缓冲，TopK 内也使用 MTE2/V/MTE3 events 做历史候选 ping/pong。
[^kernel][^cube][^vector]

Q/K/W 可为低精度输入，QK L0C、ReLU、加权归约和 score workspace 使用 FP32；TopK
比较保留相应 score/value 语义，最终 index 为 INT32，value cast/保留为 K dtype。[^vf][^vector]

# 约束

- Ascend 950 实现由 `__CCE_AICORE__ == 310` 或 DAV_310R6 路径选择。
- D 固定 128、K_N 固定 1；Q_N 仅支持 8/16/24/32/64。
- PA block size 是 16 的倍数且不超过 1024；block table 必须覆盖实际 K 长度。
- `sparse_count` 支持 1..2048 及 3072 到 8192 的指定步长值。
- sparse mode 仅 0 或 3；return_values 只用于非 PA 训练路径。

# 失败表现

- TopK 只来自最后 128 个 K：未把所有 S2 block 的 score 写入/合并 workspace。
- group size>1 数值错误：ReLU、W 广播或 group reduce 次序错误。
- PA 仅跨物理 block 时错误：block-table 索引或 K L1 gather stride 错误。
- 两个相邻 query 行交换：dual-destination FIX 与 AIV sub-block 映射不一致。
- causal 边缘包含未来 token：right-down mask 的实际 Q/K 长度和 block 上界错误。

# 验证方法

用 FP32 reference 计算完整 `QK→ReLU→W reduce→TopK`，覆盖 FP16/BF16、BSND/TND、
PA_BSND、变长、causal/default、group sizes、TopK 小/大两条 M-base 路径及 optional values。
对 TopK ties 采用实现允许的索引集合判定，同时严格核对 value。静态确认 Ascend 950 header 和
mixed AIC:AIV 1:2。性能须在空闲 Ascend 950 NPU 上另测。[^guide][^entry]

[^guide]: 固定提交中的公式、布局、dtype、TopK 范围和产品约束。
[^entry]: 固定提交中的 Ascend 950 选择和模板特化集合。
[^kernel]: 固定提交中的基本块、分核、workspace 与 C/V 主流水。
[^cube]: 固定提交中的多级 Cube 缓冲、MMAD、dual-destination FIX 和事件同步。
[^vector]: 固定提交中的权重归约、score 落盘、TopK 和输出路径。
[^vf]: 固定提交中的 ReLU、权重乘和 group reduce VF 实现。
