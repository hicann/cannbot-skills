---
type: CATLASS DSL Operator Example
title: Chunk Kda Fwd Intra Token Parallel
description: KDA 块内 token-parallel 前向核，将 gate 预处理、两个 16x16 因果矩阵乘和 mask/scale 融合到单个 CATLASS mixed kernel。
tags: [catlass-dsl, operator, linear-attention, kda, chunk, intra, token-parallel, mixed, ascend]
status: verified
generated: {by: process:catlass-dsl-operator-development, at: '2026-08-14T00:00:00Z'}
verified:
  - {by: process:full-workload-correctness, at: '2026-08-14T00:00:00Z'}
  - {by: process:msprof-device-profile, at: '2026-08-14T00:00:00Z'}
sources:
  - id: definition
    resource: attention-bench/KDA/chunk_kda_fwd_intra_token_parallel/definition.json
    title: 算子输入、输出和数学契约
  - id: workload
    resource: attention-bench/KDA/chunk_kda_fwd_intra_token_parallel/workload.jsonl
    title: TND/BSND、FP16/BF16 测试 workload
  - id: kernel
    resource: python/tla_dsl/examples/end_to_end/chunk_kda_fwd_intra_token_parallel/chunk_kda_fwd_intra_token_parallel_kernels.py
    title: CATLASS DSL fused mixed-kernel 实现
  - id: host
    resource: python/tla_dsl/examples/end_to_end/chunk_kda_fwd_intra_token_parallel/chunk_kda_fwd_intra_token_parallel.py
    title: 输入 packing、运行、精度验证和性能入口
  - id: correctness
    resource: output/chunk_kda_fwd_intra_token_parallel/correctness.log
    title: 完整 workload 精度证据
  - id: performance
    resource: output/chunk_kda_fwd_intra_token_parallel/msprof
    title: 完整 workload 的 msprof device kernel 证据
operator_families: [linear-attention, kda]
---

# 接口与概念

## 算子算法

该算子只计算 KDA 的 chunk 内 token-parallel 矩阵项。一个 task 对应
`(sequence/batch, sub_chunk, value_head)`，固定子块大小为 `BC=16`、特征维
`K_DIM=64`。令 `mid` 为该子块末 token 的 gate 值，则：

```text
qg[t]  = q[t]  * exp2(g[t] - mid)
kg[j]  = k[j]  * exp2(midt[j] - gt[j])
kbg[t] = k[t]  * exp2(g[t] - mid) * beta[t]

Aqk = tril(qg @ kg.T) * scale
Akk = strict_tril(kbg @ kg.T)
```

`Aqk` 输出按调用方格式恢复为 token 维度上的 `BT=64/128` 紧凑表示，`Akk`
保留每个 16-token sub-chunk 的严格下三角项。输入支持 TND 和 BSND，Q/K/gate/beta
支持 FP16 或 BF16；MMAD 累加、mask 和 `Akk` 输出使用 FP32。[^definition][^host]

# 用法

## 分核与阶段划分

任务由 `task_id = block_idx + n * block_dim` 静态分配。每个 AIC 配对两个 AIV
sub-block；AIV0/AIV1 各处理 16x64 输入的一半，分别负责 8 行 `qg/kbg` 和 32 行
`kg`。当前基线使用 `block_dim=28`。[^kernel]

单 task 的流水分为四个阶段：

```text
Stage 1  AIV: GM -> UB，计算 qg 与 kg
Stage 2  AIV: UB -> L1 发布 qg/kg；AIC: L1 -> L0，MMAD 得到 Aqk
Stage 3  AIV: 计算 kbg 并 UB -> L1；AIC: 同时进行 Aqk 的 Fix/L0C -> UB
Stage 4  AIC: L1 -> L0，MMAD 得到 Akk；AIV: 对 Aqk/Akk 做三角 mask、scale、store
```

因此 `Aqk` 的 Cube 路径与独立的 `kbg` Vector 路径可以并行，不能把 `Aqk` 和
`kbg` 误认为存在数据依赖。[^kernel]

## 已保留的优化

1. **单 launch 融合。** preprocess、`Aqk` MMAD、`Akk` MMAD 和 Vector mask/scale
   在 `chunk_kda_fwd_intra_token_parallel_fused_kernel` 内完成，避免公开多 kernel
   边界和 raw result 的 GM 中转。
2. **AIV 半块预处理。** 两个 AIV 各自搬运并计算半个 task，降低单 AIV 的 UB 占用和
   向量工作量。
3. **AIV UB -> 共享 L1。** `qg/kg/kbg` 从 UB 直接写共享 L1；保留的 qg/kg/kbg GM
   参数仅为 ABI 兼容，不属于 fused 数据路径。
4. **KG 的 L0 复用。** `kg` 在 L0B 中保留，`Akk` 只需要装载 `kbg`，不重复搬运 KG。
5. **L0C -> 共享 UB。** `CopyL0C2DstParams(SPLIT_M)` 将 Aqk/Akk FP32 结果直接交给
   两个 AIV 的 Vector mask 阶段，不经 GM raw workspace。
6. **ready/free 反压。** `qgkg_ready`、`kbg_ready`、`aqk_ready`、`akk_ready` 与
   `aqk_ub_free`、`akk_ub_free` 保证 AIC/AIV 的 producer-consumer 顺序，避免共享
   L1/UB 槽位覆盖。[^kernel]

## 成本模型与调优判据

当前 `msprof` 显示 AIV scalar、MTE2、MTE3 与跨 pipeline 同步占比较高；Cube MMAD
本身不是首要瓶颈。因而不应先靠增大 MMAD tile 或 block_dim 来解决延迟。

- AIV MTE2/标量主导：优先减少独立 GM->UB descriptor 和输入搬运指令，或实现有明确
  生命周期的 task 级 UB 预取。
- AIC wait 主导：检查 qg/kg/kbg 到 L1 的发布时机及 cross-core flag，而不是扩大 Cube。
- AIV wait 主导：检查 L0C->UB result 槽位回收和 mask/store 的延迟。
- 低 task 数：block_dim 过大可使核空转；大 task 数：block_dim 过小会降低并行度。必须
  用相同 workload 的 device profile 选择，不能只看 host `fused_ms`。[^performance]

## 后续候选

1. 将连续的 `q/k/g/mid` 合并为一次 MTE2，连续的 `kt/gt/midt` 合并为一次 MTE2，
   再以 UB `tile_view` 切分；目标是把每半 task 的 8 次输入搬运降为 3 次，不改变 host
   ABI。
2. 实现真正的 task 级 ping-pong：输入 UB、三份 L1 及 ready/free flag 均使用双槽，
   使 AIV 在 AIC 消费 task N 时预取 task N+1。需要 prologue/steady-state/epilogue，
   不能只复制 L1 指针。
3. 按 `num_tasks` 分桶测试 block_dim；这是 host 调度策略，需完整精度和 msprof 门禁。

两项未保留的尝试：将固定 mask 常驻 UB 的两种实现均产生 NaN；删除若干 layout
descriptor 的候选未越过 profile 噪声门槛，均已回退。[^correctness][^performance]

# 代码模式

## 数据路径与存储层级

```text
packed Q/K/G/Mid, KT/GT/MidT, beta GM
  -> AIV UB: qg/kg/kbg
  -> shared L1: qg/kg/kbg
  -> AIC L0A/L0B: Aqk MMAD, then Akk MMAD (reuse kg)
  -> L0C -> shared UB with SPLIT_M
  -> AIV UB: triangular mask, Aqk scale, output GM
```

Cube 使用 MTE1、MMAD 和 FIX；Vector 使用 MTE2、SIMD 与 MTE3。AIV halves 对
L1 与 shared UB 的 tensor view 必须非重叠，并与 `SPLIT_M` 的行分片一致。[^kernel]

## 同步与数值精度

`qgkg_ready` 在 qg/kg 的 UB->L1 结束后发布，Cube 才可启动 Aqk；`kbg_ready` 在
kbg UB->L1 后发布，Cube 才可启动 Akk。AIC 在 L0C->UB 后发布 Aqk/Akk ready，AIV
消费并写回输出后再发布对应 UB free。删除或合并 flag 前必须证明 producer-consumer
关系仍被同一 queue event 覆盖。[^kernel]

指数多项式、mask 和 accumulations 用 FP32 语义；FP16/BF16 仅用于输入与 `Aqk`
公开输出转换。Aqk 是含对角线下三角，Akk 是严格下三角，二者不能共用同一个 mask。[^definition][^host]

# 约束

- `chunk_size` 为 64 或 128，且应能按 16-token sub-chunk 切分。[^definition][^workload]
- TND varlen 的 `cu_seqlens` 与 BSND 的 batch/sequence 展开必须映射为相同 canonical
  task 顺序；AIC 与 AIV 的 `task_range` 不得改变该对应关系。[^host]
- fused 路径不得依赖 Torch reference；Torch 仅用于 host 输入构造和精度 oracle。[^host]

# 失败表现

- Aqk/Akk 均为 NaN：检查 AIV mask UB 生命周期、MTE2->Vector flag 和 shared UB
  分片是否在首次消费前可见。
- 只有 Akk 错：通常是 `kbg_ready` 时机、KG L0 reuse 或严格下三角 mask 错。
- 只有 Aqk 错：检查 qg/kg L1 layout、scale 或含对角线下三角 mask。
- 偶发旧 tile/死锁：ready/free 成对关系、跨核 flag 的 pipeline 归属或共享 L1/UB 回收
  顺序错误。
- 小 case 反而变慢：同步和 scalar descriptor 的固定成本超过 Cube 计算，需分桶评估。

# 验证方法

使用 `workload.jsonl` 覆盖 TND/BSND、FP16/BF16、`BT=64/128` 与不同
batch/head/task 数。精度验证比较 Aqk 与 Akk，打印大写阈值和小写实际值。[^workload][^correctness]

性能仅以 `msprof` 的 `op_summary.csv` 中
`chunk_kda_fwd_intra_token_parallel_fused_kernel` 的 `Task Duration(us)` 为准；host
侧 `fused_ms` 不作为 kernel 性能结论。[^performance]

[^definition]: 算子数学、输入输出与约束定义。
[^workload]: 回归 workload 的 shape、dtype、layout 和任务规模。
[^kernel]: 当前 CATLASS DSL fused kernel 的 AIC/AIV 数据路径、局部内存和同步实现。
[^host]: packing、TND/BSND task 映射、Torch oracle 与 CATLASS launch 实现。
[^correctness]: 完整 workload 的精度日志。
[^performance]: 完整 workload 的 msprof 输出及 `op_summary.csv`。
