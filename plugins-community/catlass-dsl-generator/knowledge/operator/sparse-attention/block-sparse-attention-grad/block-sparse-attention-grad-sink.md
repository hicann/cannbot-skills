---
type: CATLASS DSL Operator Example
title: Block Sparse Attention Grad Sink
description: >-
  bsag-sink 扩展层：在块稀疏反向之上追加 attention sink 梯度。块稀疏侧沿用 vendor Arc35
  语义（blockSparseMask 块参与、NonZero index 或 latin-square 任务生成、TND 累计 seqlen、
  dQ 每 task atomic、dK/dV 同 KV 块 L0C 累加）；sink 侧对齐有 sink 的反向算子语义
  （flash_attn_grad 的可选 sinks 输入、FlashAttentionScoreGradV3 的 dsinkOut 公式），
  dsinks[h] = -sum exp(sinks[h]-LSE)*D 在 post 阶段按 query head 做 FP32 原子归约。
  强调 LSE 口径必须与产生 O 的同一次前向一致，并给出全屏蔽行 LSE=+inf 的处理约定。
tags: [catlass-dsl, operator, sparse-attention, block-sparse-attention, bsag-sink, attention-sink, softmax-lse, atomic-add, block-mask, tnd, varlen, cumulative-length, gqa, mqa, fp32, extension]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-09-23T00:00:00Z'}
verified:
  - {by: human:maintainer, at: '2026-09-23T00:00:00-07:00'}
sources:
  - id: sink-fwd
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/flash_attn/README.md
    title: FlashAttn sink definition, its role in the softmax denominator and max
  - id: sink-fasgv3
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/flash_attention_score_grad/docs/aclnnFlashAttentionScoreGradV3.md
    title: FlashAttentionScoreGradV3 sinkInOptional, dsinkOut and dSink formula
  - id: sink-fag
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/flash_attn_grad/README.md
    title: FlashAttnGrad optional sinks input and TND cumulative lengths
  - id: sink-fagapi
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/flash_attn_grad/docs/torchapi_flash_attn_grad.md
    title: FlashAttnGrad torch API sinks shape and dtype
  - id: sink-bsag
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/README.md
    title: BSAG block-sparse mask shape, semantics and input list
  - id: sink-maskread
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_kernel/arc35/bsa_common_header.h
    title: Arc35 KV-major mask lookup and cumulative sequence lengths
  - id: sink-addr
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_kernel/arc35/bsa_addr_compute.h
    title: Arc35 NonZero index traversal and latin-square task schedule
  - id: sink-cube
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_kernel/arc35/cube_op_bsag_arch35.h
    title: Arc35 five-MMAD path and FP32 atomic gradient workspace
  - id: sink-vec
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_kernel/arc35/vec_op_bsag_arch35.h
    title: Arc35 SoftmaxGradFront, SimpleSoftmax and post scale path
  - id: sink-aclnnapi
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_host/op_api/aclnn_block_sparse_attention_grad.cpp
    title: aclnn adapter mask transpose for DAV_3510
  - id: sink-api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: CATLASS core API for broadcast load, range and tile_view binding
  - id: sink-tensor
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/tla/tensor.py
    title: CATLASS Tensor scalar load semantics
  - id: sink-atomic
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul_atomic_add.py
    title: CATLASS AtomicMode.ADD accumulation pattern
operator_families: [sparse-attention, block-sparse-attention, attention]
arch: [c310, ascend-950]
---
# 接口与概念

本条目描述 **bsag-sink 扩展层**：在块稀疏反向之上追加 attention sink 梯度。默认主体（不
带 sink 的块稀疏反向）见 [Block Sparse Attention Grad](block-sparse-attention-grad.md)。

先分清两件事：

- **块稀疏侧**沿用 vendor Arc35 语义：`blockSparseMask` 决定块是否参与，任务生成走 NonZero
  index 列表或 latin-square 顺序，`dQ` 每个 task 原子累加、`dK/dV` 在同一 KV 块内跨 task
  L0C 累加后一次原子写出，TND 用累计 seqlen。
- **sink 侧不属于 Arc35**：`block_sparse_attention_grad` 的输入表里没有 `sinks`，也没有
  `dsinks` 输出。[^sink-bsag] 有 sink 的 vendor 反向算子是 `flash_attn_grad`（可选 `sinks`
  输入，shape `(N1,)`、FP32）与 FlashAttentionScoreGradV3（`sinkInOptional` / `dsinkOut`）。
  [^sink-fag][^sink-fagapi][^sink-fasgv3]

## 算子算法

sink 是一个 value 恒为 0 的虚拟 key，logit 为 `sinks[h]`：[^sink-fwd]

```text
m   = max(sinks[h], max_j s_j)
LSE = log( sum_j exp(s_j - m) + exp(sinks[h] - m) ) + m
P   = exp(scale * S - LSE)
O   = sum_j P_j * V_j
```

**LSE 口径必须先固定。** vendor 文档里存在两种写法：

- FlashAttn 前向 README：sink 进入 softmax 分母**并参与 max**，因此随前向一起产出的 LSE
  含 sink 项。[^sink-fwd]
- FlashAttentionScoreGradV3：`m = max(sink, max(S))`，但 `dSink` 写成
  `reduce(-P*dP*SimpleSoftmax(sink, x_max, x_sum))`，归约用的是**只覆盖真实 key** 的
  `x_max`/`x_sum`。[^sink-fasgv3]

两种写法在同一份 `LSE` 输入下给出的 `dsinks` 是一致的，因为

```text
D          = rowsum(dO * O) = sum_j P_j * (dO · V_j)
dsinks[h]  = -sum_{b,i} exp(sinks[h] - LSE[b,h,i]) * D[b,h,i]
```

`sum_j P_j*(dO·V_j)` 正是 `reduce(-P*dP*...)` 的加权和，`exp(sinks-LSE)` 就是
`SimpleSoftmax(sink, x_max, x_sum)`。**关键是 `LSE` 必须与产生 `O` 的同一次前向一致**：前向
把 sink 算进分母就用含 sink 的 LSE，前向只覆盖真实 key 就用不含 sink 的那份。混用会让
`dsinks` 系统性偏移。[^sink-fwd][^sink-fasgv3]

`dsinks` 不是块稀疏反向的文档化输出，而是本扩展层推导：`flash_attn_grad` 只把 `sinks` 列为
可选输入，输出仍只有 `dq/dk/dv`。[^sink-fag][^sink-fagapi] `sinks = null` 时必须跳过整段
归约，`dsinks` 保持精确全零。

sink 不进入 `O`，`dO` 对 sink 因此没有直接路径；sink 也不改变 `P`、`D`、`dS` 的算式，所以
在 LSE 口径一致的前提下，`dQ/dK/dV` 的数值与无 sink 时完全相同。[^sink-bsag]

**块级稀疏与任务生成**沿用 Arc35：`blockSparseMask` 的每个元素控制一个
`blockShapeX × blockShapeY` 块是否参与，公开 shape 是
`[batch, headNum, ceilDiv(Sq,blockShapeX), ceilDiv(Skv,blockShapeY)]`（Q 块为行）；
`DAV_3510` 的 aclnn 适配层用 `perm = {0,1,3,2}` 把它转成 KV 块为行的连续布局后下发，kernel
内 `IsValidBlock` 的线性 offset 是 `kvBlock*QBlocks + qBlock`。
[^sink-bsag][^sink-maskread][^sink-aclnnapi] 非确定性路径由 host NonZero 预 kernel 生成有效
块下标列表、kernel 按列表取块，不再逐核扫 mask；确定性路径用 latin-square 顺序直读 mask。
两条路径给出的有效块集合必须与 sink 归约看到的集合一致。[^sink-addr]

**TND 变长。** Q 与 KV 各自提供数组，kernel 按**累计结束位置**解释：`len(0)=x[0]`、
`len(i)=x[i]-x[i-1]`，`x[i-1]` 作段 GM 基址。FlashAttnGrad 接口表写明「累积序列长度」，块稀疏
反向的 README 只写「准确切分界定各序列的真实边界」，措辞不同、含义须按累计边界统一。
[^sink-fag][^sink-maskread]

# 用法

## 分核策略与基本块切分

sink 不引入新的分核维度：`(batch, q_head, query tile, KV tile)` 的归属完全沿用块稀疏路径，
sink 只在 post 阶段追加一次逐 query head 的标量归约。[^sink-addr]

```text
每个 AIV 只处理自己拥有的 query 行：
  acc_h += exp(sinks[h] - LSE_row) * D_row
post 阶段：dsinks[h] += -acc_h      # FP32 atomic ADD
```

`D` 与 `LSE` 行和复用软最大值前处理/主循环里已有的量，sink 归约不额外读 `dO`/`O`。`dsinks`
是 `[Nq]` FP32 向量，跨 batch 与全部 query 行汇聚、没有唯一 writer，必须走 atomic。[^sink-vec]

`sinks` 的索引维度是 **query head**，不是 KV head：GQA/MQA 下多个 query head 共享 KV head，
但每个 query head 有独立 sink。[^sink-fagapi]

# 代码模式

## 数据路径与存储层级

```text
GM sinks[Nq] -> UB broadcast（铺满 lane）
UB: D_row, LSE_row（同一次前处理/主循环已算出）
for each owned query row: acc += exp(sink_lane - LSE_lane) * D_lane
dsinks[Nq] <- -acc        # AtomicMode.ADD
```

清零阶段必须把 `dsinks` 一并写 0，并 publish 到全部参与核之后才能开始 atomic 累加。[^sink-atomic]

块稀疏侧的数据路径与主卡一致：mask 只在任务生成时消费（NonZero 列表或 latin-square 直读），
`dQ/dK/dV` 落在三路 FP32 原子 workspace，post 统一乘 `softmax_scale`（`dV` 除外）并 cast
回输入 dtype。[^sink-cube][^sink-vec]

## 流水排布、同步关系与数值精度

sink 归约不改变 Cube/Vector 流水结构，也不改变 flag 次数：它挂在 post 阶段，与 `dQ/dK/dV`
的 scale/cast 共用同一阶段屏障。因此在 mask 判定或空任务跳过时顺手跳过 sink 归约，会破坏
flag 对称并挂死；sink 必须无条件走完整的阶段路径。[^sink-addr]

数值上全程 FP32：`exp(sinks[h]-LSE)`、`D` 的行和与跨核累加都是 FP32。`LSE = +inf` 的行按
`exp(-inf) = 0` 处理，所以全屏蔽行对 `dsinks` 的贡献精确为 0，不需要特判分支。[^sink-fwd]

DSL 侧实现要点：`sinks`/`LSE` 用 broadcast load 铺满 lane 后做向量乘累加，避免逐 lane 标量
循环；`dsinks` 用 `AtomicMode.ADD` 写回；`sinks`/`cu_seqlens` 这类 1-D 张量可用 dynamic
layout 绑定后按标量 load。[^sink-api][^sink-tensor][^sink-atomic]

# 约束

- `block_sparse_attention_grad`（Arc35）不接受 `sinks`、也不输出 `dsinks`；带 sink 是扩展层
  能力，不要把 sink 写进主卡的输入表。[^sink-bsag]
- sink 只通过 `LSE` 影响 `P_j`，不进入 `O`、不改变 `D/P/dS` 的算式，因此不改变
  `dQ/dK/dV`（前提是 LSE 口径与产生 `O` 的前向一致）。
- `LSE` 必须来自产生 `O` 的同一次前向，且与 `dsinks` 公式口径一致：前向把 sink 算进分母
  就用含 sink 的 LSE，前向只覆盖真实 key 就用不含 sink 的 LSE。[^sink-fwd][^sink-fasgv3]
- `sinks` 索引维度是 query head；`sinks = null` 时 `dsinks` 必须精确为 0。
- `dsinks` 必须在清零阶段写 0 并 publish，之后才能 atomic 累加。
- sink 归约必须挂在 post 阶段并复用既有阶段屏障，不得改变主循环的 flag/credit 配对。
- `LSE = +inf`（全屏蔽行）按 `exp(-inf)=0` 处理，不要给这类行单开分支。[^sink-fwd]
- `blockSparseMask` 公开 shape 是 Q 块为行，kernel 内按 KV 块为行线性读取（适配层转置）；
  自研实现必须用非对称、逐 head 不同的 mask 双向确认。[^sink-bsag][^sink-maskread][^sink-aclnnapi]
- `ceilDiv(Sq, blockShapeX)` / `ceilDiv(Skv, blockShapeY)` 取自 mask 张量实际维度，不由当前
  序列长度反推。[^sink-bsag]
- 非确定性路径不得在 kernel 内再读 mask 判块；确定性路径必须保持 latin-square 顺序。
  [^sink-addr]
- TND 的 seqlen 按累计结束位置解释：`cu[0]=0`、`cu[B]=T`，段长 `cu[b+1]-cu[b]`、段基址
  `cu[b]`；Q/KV 各用各的数组，不得互相推导，不得假设等长。[^sink-fag][^sink-maskread]
- 零长度段不产生 task，也不得让后续段基址错位。
- 标量 load 坐标必须都是标量索引，不支持 `None`/`_` 占位。[^sink-tensor]

# 失败表现

- `sinks` 非空但 `dsinks` 全零：sink 分支未置位，或归约被跳过。
- `dsinks` 整体偏大或偏小：`LSE` 口径与产生 `O` 的前向不一致。[^sink-fwd][^sink-fasgv3]
- `dsinks` 等于「两份 LSE 之差」：调用方把两次前向的 `O` 与 `LSE` 混搭。
- 只有部分 head 的 `dsinks` 错：清零未 publish，或索引用了 KV head 而不是 query head。
- 同一行被多核重复计入：`dsinks` 直接写而不是 atomic 累加。[^sink-atomic]
- `dQ/dK/dV` 随 sink 变化：错误地把 sink 当成参与计算的 KV token，或改了 `P`/`D`。
- `dsinks` 偶发非有限值：`LSE` 与 `sinks` 量级差过大让 `exp` 溢出，按 FP32 域检查。
- 空 query 行的 `dsinks` 非零：该行 `LSE` 不是 `+inf`，或没有按 `exp(-inf)=0` 处理。[^sink-fwd]
- 非对称 mask 下结果像沿块对角线转置：轴语义被当成 KV-major。[^sink-maskread]
- 偶发挂起或读到上一 task：mask 判定与 flag 次数不一致，或空任务被跳过同步路径。[^sink-addr]
- 稀疏区域出现旧 `dQ`：空任务未写零。
- 单 batch 正确、多 batch 错：线性 mask 索引漏乘 `QBlocks`/`KVBlocks`。[^sink-maskread]
- 第一个 batch 正确、第二个起整体错位：`cu_seqlens` 被当成逐 batch 长度，或基址用了段内相对
  下标。[^sink-fag]
- 某 batch 之后整段为空：零长度段让累计起点错位。
- 仅 TND 变长失败、等长通过：段基址/段长仍按等长写死。

# 验证方法

- oracle 用**同一份** `O`/`LSE` 计算 `D` 与 `dsinks`，并对 `sinks=null` 断言 `dsinks` 精确为
  0；逐 head 报告相对误差，不只报全局均值。矩阵至少覆盖有/无 sink、`q_heads != kv_heads`
  （GQA/MQA）、含零长度段与空 query 行的 TND。[^sink-fwd][^sink-fasgv3]
- LSE 口径专项：分别用「含 sink」与「不含 sink」两份 LSE 跑同一组输入，确认 `dQ/dK/dV` 只随
  LSE 口径变化、`dsinks` 的变化方向与公式一致，避免把口径差异当成实现缺陷。[^sink-fwd]
- 掩码至少构造全 1、单块、随机非对称（`QBlocks != KVBlocks`）、逐 query head 不同；对称
  掩码只能作回归，不能作轴语义证据；参考侧必须接收与 kernel 同一份按公开语义构造的 mask。
  [^sink-bsag][^sink-aclnnapi]
- `cu_seqlens` 至少覆盖各段等长、各段不同长、含零长度段；Q 与 KV 段长应显式不同。断言逐段
  task 数之和与总 task 数一致，并单独检查零长度段前后两段的输出。[^sink-fag]
- 与 vendor 对照：无 sink 时用块稀疏反向算子；带 sink 且 mask 全 1 时用支持 sink 的反向
  算子对（`dSink` 是其一个输出）；带 sink 且 mask 非全 1 时没有对等 vendor 算子，只能做
  candidate 自身测量。[^sink-bsag][^sink-fasgv3]

---

[^sink-fwd]: 固定提交 flash-attention 前向文档中 sink 进入 softmax 分母并参与 max 的定义。
    
[^sink-fasgv3]: 固定提交 FlashAttentionScoreGradV3 接口文档中 `sinkInOptional`、`dsinkOut`、
    以及 `dSink = reduce(-P*dP*SimpleSoftmax(sink, x_max, x_sum))` 的定义。
    
[^sink-fag]: 固定提交 FlashAttnGrad 文档中的可选 `sinks` 输入、`dq/dk/dv` 输出，以及
    `cu_seqlens_q`/`cu_seqlens_kv` 作为 TND 累积序列长度的定义。
    
[^sink-fagapi]: 固定提交 FlashAttnGrad torch 接口文档中 `sinks` 的形状 `(N1,)`、dtype
    float32 与 layout ND。
    
[^sink-bsag]: 固定提交块稀疏反向文档中的输入/输出清单、blockSparseMask shape 与块参与语义。
    
[^sink-maskread]: 固定提交 Arc35 `bsa_common_header.h` 中 `IsValidBlock` 的 KV-major 线性偏移
    与累计序列长度读取。
    
[^sink-addr]: 固定提交 Arc35 `bsa_addr_compute.h` 中的 NonZero index 取块、latin-square 确定性
    调度与任务到核的分配。
    
[^sink-cube]: 固定提交 Arc35 five-MMAD Cube 路径与三路 FP32 原子 workspace。
    
[^sink-vec]: 固定提交 Arc35 的 SoftmaxGradFront、SimpleSoftmax 与 post scale/cast 路径。
    
[^sink-aclnnapi]: 固定提交 aclnn 适配层对 `DAV_3510` 的 mask `perm = {0,1,3,2}` 转置。
    
[^sink-api]: 固定提交 `core_api.py` 中的 broadcast load、`range`、`tile_view` 与 dynamic
    layout 绑定接口。
    
[^sink-tensor]: 固定提交 `tla/tensor.py` 中标量 load 的地址空间分发与 layout 限制。
    
[^sink-atomic]: 固定提交 `basic_matmul_atomic_add` 中的 `AtomicMode.ADD` 累加模式。
    
