---
type: CATLASS DSL Operator Example
title: Chunk Kda Bwd Intra
description: KDA 块内 Q/K/gate/beta 梯度修正 mixed kernel，包含三角打包、四槽交接、head-window 流水和调优判据。
tags: [catlass-dsl, operator, linear-attention, kda, backward, mixed, triangular, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_host/chunk_kda_bwd_intra_def.cpp
    title: 接口 dtype、属性和平台范围
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_host/op_api/aclnn_chunk_kda_bwd_intra.cpp
    title: aclnn shape、layout 与 varlen metadata 校验
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_host/chunk_kda_bwd_intra_tiling_processor.h
    title: 模板选择、任务数和四槽 workspace 规划
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/chunk_kda_bwd_intra.cpp
    title: mixed AIC:AIV 入口与模板分派
  - id: common
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/chunk_kda_bwd_intra_common.h
    title: row block、head window、slot 和地址映射
  - id: vector
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/chunk_kda_bwd_intra_vector.h
    title: Vector-Pre 打包、Vector-Post 梯度合并与流水
  - id: cube
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/arch35/chunk_kda_bwd_intra_cube.h
    title: A5 lower/upper MMAD 与 cross-core 交接
  - id: regbase
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/chunk_kda_bwd_intra/op_kernel/arch35/chunk_kda_bwd_intra_regbase.h
    title: A5 mask、gate scale、reduce 和梯度融合向量原语
operator_families: [linear-attention, kda]
---

# 接口与概念

## 算子算法

本核消费前序反向阶段得到的 `dAqk/dAkk` 以及已有 `dq/dk/db/dg`，在每个 64-token chunk 内补上因果矩阵和 key-wise gate 引起的梯度项。它不是完整 KDA backward，也不计算跨 chunk 状态梯度。[^api][^vector]

对每个 16 行 block，Vector-Pre 构造两组带三角 mask 和 gate 比例的矩阵：lower 路径只遍历当前行之前的 prefix，upper 路径只遍历当前行之后的 future。AIC 执行：

```text
[raw_dq; raw_dk_lower] = [lower(dAqk); lower(dAkk)] @ gated_K_prefix
raw_dk_upper            = [upper(dAqk), upper(dAkk)] @
                          [gated_Q_future; beta*gated_K_future]
```

Vector-Post 用 anchor token 的 `gk` 恢复正/反方向 `exp2` 比例，并合并已有梯度：

```text
dq_out = dq + raw_dq
dk_out = dk + beta*raw_dk_lower + raw_dk_upper
db_out = db + reduce_sum(raw_dk_lower * k, K)
dg_out = dg + q*raw_dq + k*(raw_dk_lower - raw_dk_upper)
```

这里 `raw_dk_lower` 在写入 dk 前乘 beta，而 db 的 reduce 使用乘 beta 前的值。把顺序交换会同时破坏 dk 与 db。gate 比例直接使用已经按 base-2 累计的 `gk`。[^vector][^regbase][^cube]

输入 Q/K 为 BF16，gk、矩阵梯度、已有梯度和四个输出为 FP32，beta 为 BF16/FP32。当前只支持 safe gate 与 chunk=64；dense BNSD 支持 K=64/128/256，varlen TND 只支持 K=128。[^guide][^tiling]

# 用法

## 分核策略与基本块切分

外层任务组是 `(chunk, head-window)`，一个 window 最多包含 2 个 head；任务数为 `chunkNum*ceil(H/2)`，block 数取任务数与 AIC 数的较小值。每个 chunk 再按 16 行切成最多 4 个 row block，两个 AIV sub-block 在 Vector-Pre 中分别承担一部分矩阵/列打包，在 Vector-Post 中各拥有最多 8 个输出行，因此输出不需要 atomic。[^tiling][^common][^vector][^cube]

一个 row block 的顺序是：

```text
Vector-Pre(head0) -> ready
Vector-Pre(head1) -> ready     # 与 Cube(head0) 可重叠
Cube(head0)       -> done
Cube(head1)       -> done
Vector-Post(head0/head1)
```

workspace slot 为 `((window_index & 1) << 1) + head_in_window`，即相邻两个 row-window × 两个 head 共四槽。每槽包含 lower/upper A/B 输入区和 dq/dk 结果区；所有偏移 512B 对齐。四槽既承载 AIV↔AIC GM 交接，也允许下一 head/window 的打包与当前 MMAD/后处理重叠。[^tiling][^common][^vector][^cube]

Lower MMAD 的形状为 `32 x K x prefix`：前 16 行对应 dAqk，后 16 行对应 dAkk。Upper MMAD 为 `16 x K x (2*future)`。K=64/128 时 L0 reduction tile 为 64，K=256 时降为 32；尾 chunk 的 prefix/future 使用真实长度，不做无效 64 行 reduction。[^cube]

## 成本模型与瓶颈判断

每个 16 行 block 都经历 FP32 三角打包写 workspace、AIC 从 workspace 读两次 MMAD、Fixpipe 写 FP32 result、AIV 再读结果并与原梯度合并。因此小 K/短尾块常由 GM↔UB 打包和同步主导，大 K/完整块更可能由 Cube 或 Vector gate/reduce 主导。[^tiling][^vector][^cube]

- AIC 时间高且 Cube 利用率低：prefix/future 很短、尾块多或 K=256 reduction tile 不合适；先按 row block 分桶分析。
- AIV 时间高、MTE2/MTE3 stall 高：三角矩阵逐行 copy、FP32 workspace 往返或非对齐尾行主导。
- AIC 等 ready 的空洞长：Vector-Pre 打包慢；不要继续扩大 Cube tile。
- AIV 等 done 的空洞长：MMAD/Fixpipe 慢，或 head-window 无法形成重叠。
- H 为奇数且小：最后一个 head-window 只有一个 head，双 head overlap 失效。
- varlen 明显退化：检查 packed scalar stride、chunk metadata 热读和短尾块比例，而不是只对比总 token 数。

## 优化候选与门禁

1. 先按 `(dense/varlen,K,validLen,rowStart,H parity,beta dtype)` 建基线，并记录 AIC/AIV 两侧等待。只在正确归因后修改一个轴。
2. Vector-Pre 主导时，优先保留已存在的多行 `DataCopyPad`、双 matrix-input buffer 和 A5 regbase mask/gate-scale；可扩大批量的前提是 UB 仍在 192 KiB 预算内，且非对齐 tail 有回退。
3. AIC 主导时，分别测 lower 的 prefix 与 upper 的 `2*future` reduction。结构零只可通过缩短这两个真实 reduction 裁剪，不能跳过数据相关非零块。
4. GM workspace 往返主导时，可评估 L0C→UB 的片上交接，但 copy mode 必须匹配两个 AIV sub-block 的行所有权，ready/free flag 覆盖完整 buffer 生命周期；若新增 flag 或 L1/UB 占用抵消收益则回退。
5. 同一 head 的四个 row block 具有不同 prefix/future 成本，静态 round-robin 可能失衡；可按估算 MMAD 和打包字节做 task 排序，但必须让 AIC/AIV 使用完全相同的顺序，否则 flag 会错配。
6. odd H 或低任务数导致低并发时，可重新组合跨 chunk/head window，但不能让两个核写同一梯度行，也不能破坏 canonical varlen metadata 顺序。
7. K=256 独立调整 reduction tile时需同时检查 L1/L0 容量、MMAD 次数和 Fixpipe；K=128 fast 结论不能外推。
8. 删除 barrier/event 只允许在已有 queue 或 cross-core flag 已精确覆盖同一 producer-consumer 的情况下进行；随机压力测试是必要门禁。

# 代码模式

## 数据路径与存储层级

```text
dAqk/dAkk, Q/K, gk, beta GM
  -> AIV UB (mask, exp2 scale, beta broadcast)
  -> per-core 4-slot FP32 GM workspace [A_lower/B_lower/A_upper/B_upper]
  -> AIC L1/L0A/L0B -> FP32 L0C -> Fixpipe
  -> workspace [raw_dq/raw_dk_lower/raw_dk_upper]
  -> AIV UB (scale, reduce db, merge dq/dk/dg)
  -> dq_out/dk_out/db_out/dg_out FP32 GM
```

A5 Vector 使用 8 KiB 双 input/output queue、两个 matrix staging buffer、96 KiB arena 和 32 KiB reduce scratch，总量受 192 KiB 静态断言约束。arena 以 4 KiB plane 划分；K=128 的 lower 每个 sub-block 恰好打包 `16x64 FP32`，upper 一次可处理 `16x128 FP32`。[^vector][^regbase]

## 流水排布、同步关系与数值精度

每个 head 的 Vector-Pre 写完槽后以 `PIPE_MTE3` 发布 vec-to-cube ready；AIC 等待后执行 lower/upper MMAD，并在 Fixpipe 完成后发布 cube-to-vector ready。AIV 等待结果，再执行 Vector-Post。slot 复用依赖四槽循环和严格一致的 window/head 顺序，不能只在一侧改变遍历。[^common][^vector][^cube]

Q/K BF16 在 AIV 打包时转为 FP32；gate 比例、beta、dA、已有梯度、MMAD A/B/C、reduce 和输出均为 FP32。AIC 显式关闭 HF32。A5 regbase 与普通 Vector 路径必须在三角边界、Exp2、beta-before/after-reduce 顺序上等价。[^cube][^vector][^regbase]

# 约束

- 只支持 `chunk_size=64` 和 `safe_gate=true`；unsafe 模板保留但不实例化。[^tiling][^entry]
- dense 仅 BNSD，K=64/128/256；varlen 为 TND 兼容 rank 3/4，rank 4 时 B=1 且 K=128。[^api][^tiling]
- varlen 必须同时提供 cu_seqlens 和每 chunk 四个 INT64 的 metadata；kernel 热路径读取 metadata 的 global begin/end。[^api][^tiling][^common]
- `q/k/gk/dq/dk/dg` shape 一致，`dAqk/dAkk` 末维固定 64，beta/db 去掉 K 维。[^tiling]
- 每个 AIV sub-block 只写其拥有的最多 8 行；workspace slot 和 flag 次序在 AIV/AIC 两侧必须一致。[^vector][^cube]

# 失败表现

- db 与 dk 同时偏差：raw_dk_lower 在 db reduce 前误乘 beta，或 beta broadcast stride 错。
- dg 符号错误但 dq/dk 正常：`raw_dk_lower - raw_dk_upper` 的方向被交换。
- 只在 rowStart=16/32/48 错：anchor、prefix/future 或 lower/upper 三角 mask 错。
- 偶发旧结果/死锁：四槽复用、head-window 顺序或 vec/cube flag 不匹配。
- dense 正常、varlen 错：packed tensor/scalar stride 或 chunk metadata begin/end 错。
- K=256 独有错误：reduction tile、K 分段或 result stride仍按 128 假定。

# 验证方法

正确性用显式 FP32 masked-matmul reference 覆盖 dense K=64/128/256、varlen K=128、beta BF16/FP32、H=1/2/奇数/偶数、完整 chunk 与长度 1/8/15/16/17/63 的尾块，以及 rowStart 四个位置。分别比较 dq/dk/db/dg，并构造只激活 dAqk、只激活 dAkk、只激活已有梯度的用例定位分支。[^api][^tiling][^entry][^vector][^cube]

性能测试在空闲 NPU 上按 row block 成本分桶，记录 `aicore_time`、AIC/AIV 时间、Cube 利用率、MTE stall、cross-core wait 和 workspace 字节。修改流水后增加多轮随机压力测试捕获 slot/flag 问题；只有全量正确性与设备 profile 同时通过的候选才能作为优化结论。

[^guide]: 固定提交中的输入输出 dtype、属性、平台和动态 shape 契约。
[^api]: 固定提交中的 aclnn shape、layout、可选 metadata 和参数组合校验。
[^tiling]: 固定提交中的支持矩阵、tiling key、任务数、block dim 和 512B 对齐四槽 workspace。
[^entry]: 固定提交中的 mixed AIC:AIV=1:2 入口、K/layout/beta 模板分派和 safe-only 实例化。
[^common]: 固定提交中的 16 行基本块、双 head window、四槽公式和 dense/packed 地址映射。
[^vector]: 固定提交中的两阶段 AIV 顺序、lower/upper 打包、UB 预算、梯度公式、GM 读写与同步。
[^cube]: 固定提交中的 lower/upper MMAD shape、K=256 reduction tile、L1/L0/Fixpipe 和 cross-core flag。
[^regbase]: 固定提交中的 A5 mask、Exp2 gate scale、beta、reduce 与 dq/dk/dg 融合向量实现。
