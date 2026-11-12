---
type: CATLASS DSL Operator Example
title: Block Sparse Attention Grad (BSAG)
description: >-
  Ascend 950 Arc35 Block Sparse Attention 反向主路径：以 blockSparseMask 为唯一稀疏元数据，
  重算被选稀疏块的 score 与概率，五路 MMAD 产出 dQ/dK/dV。dQ 每个 task 原子加到 FP32
  workspace；dK/dV 在同一 KV 块内跨 task 在 L0C 累加、换块时一次原子写出。统合 BSND、BNSD、
  TND 三种 layout 的 stride 寻址与 TND 累计长度，覆盖 GQA/MQA 的 Q-head 到 KV-head 映射和
  共享 dK/dV 所有权。非确定性路径由 host NonZero 预 kernel 生成有效块下标列表、kernel 按
  列表取块并把 Q/K/V/dY 做三槽轮转预取；确定性路径用 latin-square 顺序直接读 mask。同一
  kernel 的 constexpr 分支覆盖 FP16/BF16 与 1000..1005/1008..1013 两族 tiling key。文末
  「扩展层」记录细 token mask 与 head_dim 32/64 两个实现侧变体的出处与边界。
tags: [catlass-dsl, operator, sparse-attention, block-sparse-attention, backward, softmax-grad, ascend-950, bsag, block-mask, non-zero-index, deterministic, latin-square, triple-buffer, bnsd, bsnd, tnd, stride, variable-length, cumulative-length, gqa, mqa, fp32, fp16, bf16, head-dim, constexpr, five-mmad, d-precompute, atomic-add, workspace, task-granularity, tiling-key, tail, empty-row, benchmark, vendor-baseline, extension]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-09-15T13:44:19Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-09-15T13:44:19Z'}
  - {by: human:maintainer, at: '2026-09-18T00:00:00-07:00'}
  - {by: human:maintainer, at: '2026-09-20T00:00:00-07:00'}
  - {by: human:maintainer, at: '2026-09-23T00:00:00-07:00'}
sources:
  - id: bsag-readme
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/README.md
    title: BSAG public interface, block-sparse mask contract and formulas
  - id: bsag-aclnn
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/docs/aclnnBlockSparseAttentionGrad.md
    title: aclnnBlockSparseAttentionGrad layout, dtype and block-shape contract
  - id: bsag-entry
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_kernel/block_sparse_attention_grad.cpp
    title: kernel entry, mixed-core attribute and Arc35 tiling-key dispatch
  - id: bsag-arch35
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_kernel/arc35/block_sparse_attention_grad_arch35.h
    title: Arc35 AIC/AIV pipeline, triple buffering and cross-core flags
  - id: bsag-addr
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_kernel/arc35/bsa_addr_compute.h
    title: Arc35 task generation on NonZero index and latin-square schedule
  - id: bsag-common
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_kernel/arc35/bsa_common_header.h
    title: Arc35 layout offsets, sequence lengths and mask lookup
  - id: bsag-cube
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_kernel/arc35/cube_op_bsag_arch35.h
    title: Arc35 five-MMAD Cube path, dK/dV accumulation and atomic copy-out
  - id: bsag-vec
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_kernel/arc35/vec_op_bsag_arch35.h
    title: Arc35 vector zero, softmax front, softmax grad and post scale
  - id: bsag-vfcast
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_kernel/arc35/vector_api/bsa_vf_cast_nd2nz.h
    title: Arc35 ND to NZ cast helper for P and dS
  - id: bsag-nonzero
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_kernel/arc35/vector_api/non_zero_big_mask_dim1.h
    title: NonZero pre-kernel producing the valid block index list
  - id: bsag-tiling
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_host/arch35/block_sparse_attention_grad_tiling_arch35.cpp
    title: Arc35 capability checks, base tiles, core count and workspace layout
  - id: bsag-nztiling
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_host/arch35/non_zero_tiling_arch35.h
    title: NonZero pre-kernel host tiling for the mask index list
  - id: bsag-tilingh
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_host/block_sparse_attention_grad_tiling.h
    title: Arc35 tiling data fields including index workspace offsets
  - id: bsag-aclnnapi
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/op_host/op_api/aclnn_block_sparse_attention_grad.cpp
    title: aclnn adapter mask transpose for DAV_3510
  - id: bsag-test
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/block_sparse_attention_grad/examples/test_aclnn_block_sparse_attention_grad.cpp
    title: BSAG launch smoke test and its verification boundary
  - id: fag-fwd
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/flash_attn/README.md
    title: FlashAttn invalid-row convention for attn_out and softmax_lse
  - id: fagrad-readme
    resource: https://gitcode.com/cann/ops-transformer/blob/2b67400cf8284717cb38f1d4ecdc69df9191943f/attention/flash_attn_grad/README.md
    title: FlashAttnGrad cu_seqlens_q and cu_seqlens_kv as cumulative lengths
  - id: tla-api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: CATLASS core API for mmad layout, cross_flag, L0C2UBMode and extern
  - id: tla-atomic
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul_atomic_add.py
    title: CATLASS AtomicMode.ADD copy pattern
  - id: tla-mixed
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py
    title: CATLASS mixed Cube and Vector kernel pattern
  - id: tla-crossflag
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/dsl_tutorials/cross_flag_two_way/cross_flag_two_way.py
    title: CATLASS two-way cross flag pairing
  - id: tla-cast
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/cast_multi.py
    title: CATLASS vector cast and register slot usage
  - id: tla-fa
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/attention/flash_attention_infer/flash_attention_infer.py
    title: CATLASS attention tail view and DINTLV store patterns
operator_families: [sparse-attention, block-sparse-attention, attention]
arch: [c310, ascend-950]
---
# 接口与概念

本条目描述 Ascend 950 Arc35 **Block Sparse Attention Grad**（`aclnnBlockSparseAttentionGrad`）
主路径：以 `blockSparseMask` 为唯一稀疏元数据，重算被选稀疏块的 score 与概率，五路 MMAD
产出 `dQ/dK/dV`，三路梯度经 FP32 workspace 归并后再 scale/cast 落盘。覆盖 FP16/BF16 ×
BSND/BNSD/TND、GQA/MQA 与 TND 变长。带 attention sink 时读
[Block Sparse Attention Grad Sink](block-sparse-attention-grad-sink.md)。

文末「扩展层」记录两个**不属于 vendor Arc35 基线**的实现侧变体（细 token mask、`head_dim`
32/64）；本卡默认主体只承诺 Arc35 行为。

## 算子算法

Arc35 反向接收 `dO/Q/K/V/O/LSE` 与 UINT8/BOOL 二值 block mask，重新计算被选稀疏块的 score
和概率，不保存完整前向概率。公开公式链为：[^bsag-readme]

```text
softmax_grad = softmaxGrad(dOut, attentionOut)
D  = rowsum(dO * O)
S  = Q @ K^T
P  = exp(scale * S - LSE)
dP = dO @ V^T
dS = P * (dP - D)
dV = P^T @ dO
dQ = scale * (dS @ K)
dK = scale * (dS^T @ Q)
```

`D` 先对全部有效 query 行计算一次并写入 workspace；主循环随后对每个有效稀疏 tile 发起
`QK`、`dO·V^T`、`dV`、`dK`、`dQ` 五次 MMAD，源码中分别落在 `ComputeMM12` 的两次 MMAD 与
`ComputeMMDKV<DV>`、`ComputeMMDKV<DK>`、`ComputeMMDQ`。[^bsag-cube]

Arc35 只在 `__CCE_AICORE__ == 310` 分支进入，入口设置 mixed-core
`KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2)`（1 AIC : 2 AIV）。tiling key 由 dtype、
Q/KV layout 与 `deterministic` 开关共同决定，低位是 dtype/layout、`+0b1000` 表示确定性路径：
[^bsag-entry][^bsag-tiling]

| tiling key | dtype | layout | 任务生成路径                |
| ---------: | ----- | ------ | --------------------------- |
|     `1000` | BF16  | BSND   | 非确定性：NonZero index 列表 |
|     `1001` | FP16  | BSND   | 同上                        |
|     `1002` | BF16  | BNSD   | 同上                        |
|     `1003` | FP16  | BNSD   | 同上                        |
|     `1004` | BF16  | TND    | 同上                        |
|     `1005` | FP16  | TND    | 同上                        |
|     `1008` | BF16  | BSND   | 确定性：latin-square 顺序    |
|     `1009` | FP16  | BSND   | 同上                        |
|     `1010` | BF16  | BNSD   | 同上                        |
|     `1011` | FP16  | BNSD   | 同上                        |
|     `1012` | BF16  | TND    | 同上                        |
|     `1013` | FP16  | TND    | 同上                        |

`1006`/`1007` 未被使用。非确定性路径不再逐核读 mask 判块，而是消费 host 生成的 index 列表，
这条约束被源码写成编译期断言 `static_assert(DETERMINISTIC_ENABLE || INDEX_ENABLE)`。[^bsag-addr]

**块级稀疏。** `blockSparseMask` 是唯一稀疏元数据：bool/UINT8 张量，每个元素控制一个
`blockShapeX × blockShapeY` 连续块是否参与计算（1 参与、0 不参与）。公开 ABI 的 mask shape
是 `[batch, headNum, ceilDiv(maxQSeqLength, blockShapeX), ceilDiv(maxKvSeqLength, blockShapeY)]`，
逻辑上以 **Q 块为行**。[^bsag-readme]

`DAV_3510` 下 aclnn 适配层先用 `perm = {0,1,3,2}` 把 mask 转置成连续的
`[batch, headNum, KVBlocks, QBlocks]` 再下发，因此 kernel 内 `IsValidBlock` 的线性 offset 是
`n1*(QBlocks*KVBlocks) + kvBlock*QBlocks + qBlock`，在 kernel 里是 **KV 块为行**。
[^bsag-aclnnapi][^bsag-common] 这是 ABI 分层而不是漂移：调用侧按 README 的 Q-major shape
构造、kernel 侧按转置后的 KV-major 读取。全 1 或对称 mask 无法区分两者，必须用非对称
pattern 双向核对。

README 声明 mask 传 `nullptr` 时退化为「全计算」，但 Arc35 的 NonZero（非确定性）路径在
host tiling 阶段明确要求 mask 非空，不要把该退化语义当作 Arc35 的可用配置。
[^bsag-readme][^bsag-tiling]

**两条任务生成路径。**

- **非确定性（`1000..1005`）**：host 侧先由 NonZero 预 kernel 把 mask 的真值元素展平成一维
  下标列表写入 `indexWorkspace`、有效块个数写入 `indexShapeWorkspace`；主 kernel 按该列表
  grid-stride 取块，不再读 mask 判块。[^bsag-nonzero][^bsag-nztiling][^bsag-tilingh]
- **确定性（`1008..1013`）**：kernel 自己在 `s1_outer × s2_outer` 稠密 tile 网格上按
  latin-square 顺序推进并读 `blockSparseMask` 判块，使加法顺序可复现；GQA 与非 GQA 分别用
  `MapGqaIndex` / `MapDenseIndex` 反解 `(batch, q_head, q_tile, kv_tile)`。[^bsag-addr]

**布局与寻址。** 三个 layout 只改变 GM stride 与 LSE/sftg 寻址。源码给出统一 offset 公式
（TND 用上一段累计起点作为基址）：[^bsag-common]

```text
BSND : b*(S*N*D) + s*(N*D) + n*D
BNSD : b*(N*S*D) + n*(S*D) + s*D
TND  : last_sum*(N*D) + s*(N*D) + n*D
LSE  : TND -> last_sum*N + s*N + n ; BNSD/BSND -> b*(N*S) + n*S + s
sftg : BNS8 布局，TND 按 last_sum*N*8 + n*S*8 + s*8
```

Q/dO/O 使用 Q stride 族，K/V 使用 KV stride 族，LSE 与 sftg 各有独立 stride 族；GQA 下不能
从 Q stride 推导 K/V stride。地址计算在 tile 外完成，不进入逐行 SIMD 热循环。[^bsag-common]

**TND 变长。** TND 把 batch 压进 token 维，Q 与 KV **分别**提供数组；kernel 把它解释为
**累计结束位置**：`len(0) = x[0]`、`len(i) = x[i] - x[i-1]`，`x[i-1]` 又直接作为段 GM 基址。
BSAG README 只写「准确切分界定各序列的真实边界」而未点明累计语义；明确写「累积序列长度」
的是 FlashAttnGrad 接口表。调用与 reference 都必须按累计边界理解。[^bsag-common][^fagrad-readme]

**GQA/MQA。** 令 `group_size = q_heads / kv_heads`，Q head `h` 使用
`kv_head = h // group_size`。mask 与任务仍按 query head 独立遍历；dQ 由单一 Q head 所有，
dK/dV 汇聚同组全部 Q head，多个 task 会写同一 KV 地址，必须靠 FP32 atomic 归并。
[^bsag-cube][^bsag-addr]

**dtype 与 head dim。** Arc35 只接受 FP16/BF16，且 `headDim` **必须为 128**，其余取值在
`IsCapable` 阶段直接判不可用。Q/K/V/dO/O 的 dtype 必须一致；LSE、sftg 与三路梯度 workspace
为 FP32。[^bsag-tiling]

**尾块、空行与零长度段。** 稀疏块固定为 `blockShapeX × blockShapeY`（默认 `128×128`），末
query/KV 块常为尾块：`q_size = Sq - qb*blockShapeX <= 128`、
`k_size = Skv - kb*blockShapeY <= 128`。物理 buffer 与 layout 不随有效长度变化，尾块只收缩
搬运与写出 extent。无效行 `LSE` 按 `+inf`（3e+99）、`attn_out` 清零，这是 FlashAttn 前向的
行无效约定，反向侧沿用同一约定判断空行。[^fag-fwd] TND 允许段长为 0；零长度段不应产生
task，但状态机切换必须复位 prev-valid 与预取状态。

# 用法

## 分核策略与基本块切分

**基本块。** host 侧固定 `baseM = min(blockShapeX, 128)`、`baseN = min(blockShapeY, 128)`，
再取 `singleM = AlignTo(2048, baseM)`；核数取全部可用 AIC（`GetCoreNumAic()`）并
`SetBlockDim(cubeCoreNum)`。[^bsag-tiling] AIV 数为 2 × AIC，地址模块用 `GetBlockIdx()/2`
复现与 AIC 相同的任务序列。[^bsag-addr]

**task 粒度。** 非确定性路径的每一个 index 条目就是一个 `baseM × baseN` tile：
`unravel_index` 把展平下标还原成 `(batch, q_head, q_block, kv_block)`，
`s1Idx = q_block*baseM`、`s2Idx = kv_block*baseN`，再按真实序列长度裁剪
`s1Len/s2Len`。核内以 `index_block_num = singleM / baseM` 个连续条目为一组取任务、组间以
`cubeCoreNum` 为步长（`block_local_idx` 与 `block_global_idx`）。[^bsag-addr] 确定性路径则把
`s1_outer × s2_outer` 网格按 latin-square 次序分给各核并跳过无效 tile；非 GQA 时 KV 方向
变化最慢，同一 KV 块的多个 query tile 因此落在连续轮次上。[^bsag-addr]

**dQ/dK/dV 的所有权与 atomic 边界。** 这里最容易写反：

```text
dQ   : 每个 task 都 MMAD + Fixpipe atomic 到 FP32 dQ workspace
dK/dV: 同一 KV 块内跨 task 在 L0C 累加（is_kv_change 决定 init），
       换到下一 KV 块时（is_kv_end）一次 atomic 写出
```

源码中 `ComputeMMDQ` 的 `cmatrixInitVal = true` 且每次调用都走 `MM345CopyOut<true>`
（即 `SetAtomicType<float>()`）；`ComputeMMDKV<DK|DV>` 的 `cmatrixInitVal = is_kv_change`，
只有 `is_kv_end` 为真才 copy-out，`unitFlag = is_kv_end ? 3 : 2`。因此 dQ 的累加边界是
**单个 task**，dK/dV 的累加边界是**同一 KV 块的一段连续轮次**。[^bsag-cube][^bsag-arch35]

**固定成本。** 除稀疏 MMAD 外，每次都要：全量清零 `dQ/dK/dV` FP32 workspace、全量读
`dO/O` 算 `D`、收尾再把三路 workspace 全量读回做 scale/cast/写出。确定性路径还要每个核
各自从 latin-square 起点扫 mask；非确定性路径把 mask 扫描成本提前交给 NonZero 预 kernel，
代价是额外的 index workspace 与一次 AIV 预 kernel。高稀疏小 workload 若耗在 Vector 前后
处理或调度上，应先看这些固定项，而不是直接归因 MMAD。[^bsag-vec][^bsag-nonzero]

静态主计算量近似为：

```text
MMAD FLOPs ~= 10 * D * sum_{t in V}(M_t * N_t)
```

`V` 是 mask 有效且经 `baseM/baseN` 展开后的实际 tile 集。任务按有效 tile 轮转分配、不按
tile 内工作量加权，mask 分布偏斜时尾核可能不均衡。[^bsag-addr]

# 代码模式

## 数据路径与存储层级

```text
dO/O GM -> AIV UB SoftmaxGradFront（每行 8 槽）-> D FP32 GM workspace
blockSparseMask GM -> AIV NonZero 预 kernel -> index / indexShape workspace
Q/K/V/dY GM -> AIC L1（三槽）-> L0A/L0B -> QK、dO·V^T FP32 L0C -> S/dP FP32 UB ping/pong
S/dP/LSE/D UB -> AIV P、dS FP32 -> ND2NZ cast FP16/BF16 -> P/dS L1 ping/pong
P/dY L1 -> AIC dV FP32 L0C -> 同 KV 块累加 -> atomic FP32 dv workspace
dS/Q L1 -> AIC dK FP32 L0C -> 同 KV 块累加 -> atomic FP32 dk workspace
dS/K L1 -> AIC dQ FP32 L0C -> 每 task atomic -> atomic FP32 dq workspace
dq/dk/dv workspace -> AIV post（dq/dk 乘 scale）-> RINT cast -> dq/dk/dv GM
```

P 与 dS 由同一次 `SimpleSoftmax` 从 `S`、`dP`、`LSE`、`D` 一起算出，再经 `CastND2NZ` 转成
输入 dtype 的 NZ 分形写入 L1，供后三路 MMAD 使用。[^bsag-vec][^bsag-vfcast]

**workspace 布局。** 用户 workspace 依次为 `sftg → dq → dk → dv`，非确定性路径在前面额外
放 `index → indexShape → nonZero 计数`，全部 FP32。元素数为 `sftg: B*Nq*Sq*8`、
`dq: B*Nq*Sq*D`、`dk/dv: B*Nkv*Skv*D`；TND 用 total token 数替换 `B*S`。这里的 8 是
`SoftmaxGradFront` 每 query 行的输出槽数，不是 head dim。总字节数为这些元素数之和乘 4，
外加平台 LibAPI 系统 workspace。[^bsag-tilingh][^bsag-vec]

**片上容量。** 顶层固定申请 247 KiB UB 与 512 KiB L1；L1 内的 P/dS ping-pong 与 Q/K/V/dY
三槽共享这 512 KiB。Cube 侧 L0A 64 KiB、L0B 64 KiB、L0C 256 KiB，其中 L0C 划成 S/dP
ping-pong 与 dK、dV 专用累加区四个 64 KiB 区。[^bsag-arch35][^bsag-cube]

**Q/K/V/dY 三槽与预取。** `taskId % 3` 选本轮的 Q/dY L1 槽、`(taskId-1) % 3` 选上一轮的槽；
K/V 另有一组三槽，但只在 `is_kv_change` 时轮转，同一 KV 块的连续轮次复用同一份 K/V。
`ComputeMM12` 在算当前轮的同时把**下一轮**的 Q/dY 搬进下一个槽，只有
`nextRunTimeInfo.is_kv_change` 为真时才顺带搬 K/V，这就是三缓冲的重叠点。[^bsag-arch35]
[^bsag-cube]

**尾块搬运视图。**

```text
尾块 KV 搬运: 按真实 k_size 收缩 blockCount/blockLen（物理 layout 不变）
score/dP    : L0C 按 align 后的 mSize/nSize Fixpipe 到 UB
P/dS 落盘   : 按 (half_row, k_size) 写出；尾部 lane 用 cast mask 屏蔽
```

UB 行 stride 取物理行宽而不是有效列数，否则尾块会串行。[^bsag-vec]

**GQA atomic 写出。**

```text
Q head h -> KV head h // group_size
dQ  : 每 task atomic 到 dq workspace（同一 Q 行会被多个 KV 块 task 命中）
dK/dV: 同一 KV 块跨 task 在 L0C 累加，换块时 atomic 到共享 dk/dv workspace
```

dK/dV 地址使用 KV stride，不能使用 Q head stride；FP32 workspace 避免每个 Q head 先 cast
再累加造成重复低精度舍入。[^bsag-cube]

## 流水排布、同步关系与数值精度

**阶段划分。** AIV 先并行清零 `dq/dk/dv` workspace，再做 sftg 前处理（算出 `D`），
`PipeBarrier` + `SyncAll` 后用 `FLAG_CUBE_POST` 放行 AIC。主循环按有效 tile 一轮延迟：
Cube 第 r 轮算当前轮的 `QK` 与 `dO·V^T` 并 Fixpipe 到 UB，同时消费第 r-1 轮的 P/dS 做
`dV/dK/dQ`；AIV 第 r 轮消费本轮的 S/dP 生成 P/dS。收尾 drain 后 AIV 全核同步执行 post。
[^bsag-arch35]

```text
round r Cube : Q/K -> S, dO/V -> dP -> shared UB
round r Vec  : S,dP,LSE,D -> P,dS -> L1
round r Cube : P/dS from r-1 -> dV/dK/dQ
```

**flag 表。** 源码用 6 个固定 cross-core flag 编号：`0` cube(QK)→vec(softmax)、
`1` cube(dO·V^T)→vec(softmaxGrad)、`2` vec(softmax)→cube(dV)、`3` vec(softmaxGrad)→cube(dQ/dK)、
`4` 阶段屏障（前处理完成/结束），`5` 仅确定性路径用于给 mm345 的波次排序。
[^bsag-common][^bsag-arch35] 确定性路径在相邻 task 间插一对 `mode=0` 的 `FLAG_DETER_FIX`
把 mm345 发射顺序固定下来，非确定性路径没有这一对。跳 tile 时两侧必须对称跳过：一侧 break
而另一侧仍等 flag 会直接挂死。

**数值精度。** Q/K/V/dO/O 是 FP16/BF16；QK 与 dP 的 MMAD 累加和 Fixpipe 输出为 FP32。P 由
FP32 score、FP32 scale 与 FP32 LSE 重算，dS 也在 FP32 形成，但进入后三路 MMAD 前会 cast 成
输入 dtype。dQ/dK/dV 以 FP32 累加并原子加到 FP32 workspace；post 对 dQ/dK 乘 softmax scale、
对三者按 `CAST_RINT` 写回输入 dtype，dV 不乘 scale。[^bsag-vec][^bsag-cube] oracle 对比必须
允许「P/dS 先低精度化再 MMAD」的舍入边界。

dQ/dK/dV 的 workspace 写回显式启用 FP32 atomic add；`deterministic=true` 走独立
latin-square 分支而不是拒绝 atomic。不同运行的低位差异属于实现允许的现象，但偏差仍必须
落在已定义容差内。[^bsag-cube][^bsag-addr]

**CATLASS DSL 复现要点。** 一次调用只 launch 一个 `@tla.kernel`，host 侧只做元数据检查、
输出/workspace 分配与参数透传；三路 atomic 落盘用 `AtomicMode.ADD`；L0C→UB 用
`L0C2UBMode.SPLIT_M` 让两个 AIV 各取半行；阶段切换用一对 `mode=0` 的全核 `cross_flag`，
两侧都配 `local_mem_bar` + `pipe_barrier(ALL)`，必要时用 `@tla.extern` 内联
`AscendC::SyncAll()` 作全核屏障逃生舱；Cube/Vector 混排骨架照 `basic_mixed`，成对握手照
`cross_flag_two_way`。[^tla-api][^tla-atomic][^tla-mixed][^tla-crossflag] P/dS 的
f32→b16 全宽 cast 需要 even/odd 两个 RegSlot，只用一个会丢一半 lane。[^tla-cast]

# 约束

- 本条目只覆盖 `__CCE_AICORE__ == 310`、`DAV_3510` 注册的 Ascend 950 Arc35 路径，不能
  外推到 `arc22`。[^bsag-entry]
- Arc35 host tiling 只接受 FP16/BF16、`headDim == 128`、`blockShapeX/Y` 为 16 的正整数倍，
  且 Q/K/V 的 batch、序列长度、head dim 必须相同（比 README 允许 Q/KV 长度分别描述更严格）；
  Q heads 必须等于 `qGroup * kvHeads`。[^bsag-tiling][^bsag-aclnn]
- Q/KV layout 必须相同且只允许 BSND、BNSD、TND；TND 必须同时提供两组 seqlen，非 TND 忽略
  它们。attention mask、非零 mask type、滑窗 pre/next token 均不在当前路径内。[^bsag-tiling]
- mask 的公开 shape 是 Q-major，但 kernel 内按 KV-major 线性读取（适配层负责转置）；构造
  reference 必须用非对称 pattern 双向核对，全 1/对称 mask 证明不了轴语义。
  [^bsag-readme][^bsag-aclnnapi]
- 非确定性路径必须由 host 先跑 NonZero 预 kernel 生成 index 列表，kernel 内不得再读 mask
  判块；该约束由 `static_assert(DETERMINISTIC_ENABLE || INDEX_ENABLE)` 在编译期强制。
  [^bsag-addr]
- TND 的 seqlen 数组必须按累计结束位置理解：单调不减、首项即首段长度、末项等于对应总 token
  数，Q/KV 两个数组长度相同。README 未成文，reference 与调用都必须用累计边界。
  [^bsag-common][^fagrad-readme]
- dQ 的累加边界是单个 task（每 task atomic）；dK/dV 的累加边界是同一 KV 块的一段连续轮次
  （`is_kv_change` 起、`is_kv_end` 止）。不要为了负载均衡拆开 dK/dV 的 L0C 累加，也不要给
  dQ 加跨 task 的 L0C 累加。[^bsag-cube]
- dK/dV 地址必须用 KV stride，不能复用 Q head stride。[^bsag-common]
- 物理 buffer 与 layout 不随有效长度变化。尾部列若会被整块消费，必须显式填充归零哨兵；空
  任务仍必须写出零 `dQ`；无效行 `LSE` 按 `+inf`（3e+99），不要改成 `-inf` 或 0。[^fag-fwd]
- 填充、稳态、排空三阶段必须显式闭合；禁止只加 wait 而不加对应 set，或把 set/wait 放在不同
  运行时分支。跳 tile 时两侧必须同时跳过。[^bsag-arch35][^tla-crossflag]
- Q/K/V/dY 三槽与 K/V 三槽的轮转规则不同：Q/dY 按 `taskId % 3` 无条件轮转，K/V 只在
  `is_kv_change` 时轮转；把两者绑成同一个计数器会在共享 KV 的连续轮次上写坏数据。
  [^bsag-arch35]
- 非确定性路径需要 index、indexShape 与 NonZero 计数三块额外 workspace，且 `indexWorkspace`
  上界按 mask 元素个数分配；改 mask shape 必须同步重算该上界。[^bsag-tiling][^bsag-tilingh]
- workspace 元素数用 64-bit 计算，但 AIV 的均分结构与循环入口用 32-bit
  `data_size/start_idx/len`；接近或超过 `2^32` 元素的梯度区必须做 host 上界检查与专项验证。
  [^bsag-vec]
- 用 vendor 做性能分母时，直接调用 vendor backward API；不用 Torch 数学 oracle 的执行时间
  当分母，不把 compile、输入生成、mask 转换或 correctness 比较计入 kernel 时间。[^bsag-aclnn]

# 失败表现

- TND 第二个 batch 起整体错位：seqlen 传成逐 batch 长度而不是累计结束位置，或段基址与段内
  下标重复相加。[^bsag-common]
- 空段后结果串段：跳段后未复位 prev-valid 与预取状态。
- Q 长 KV 短时越界：KV 尾块误用 Q 段长裁剪。
- 只有多 batch GQA 错：段基址与 KV head stride 组合错误。[^bsag-common]
- 非对称 pattern 像沿块对角线转置：mask 轴序不一致（漏了适配层转置，或 reference 用了
  转置后的 mask）。[^bsag-aclnnapi]
- 稀疏 tile 偶发挂起：Cube 与 Vector 对有效 tile 判定不一致，或一侧在跳过的 task 上仍等
  cross flag。[^bsag-arch35]
- 只有 BSND 错：head/token stride 对调。GQA 只有 K/V 侧错：错误复用了 Q token/head stride。
  dQ 正确而 LSE 相关结果错：LSE stride 族与 Q stride 混用。[^bsag-common]
- MHA 正确、GQA 错：KV head 映射或共享地址错误。dK/dV 随 group_size 增大偏小：部分 Q head
  未 atomic 归并。重复运行叠加历史结果：workspace 未清零。
- dQ 偏小或后写覆盖：把 dQ 误当成跨 KV 块 L0C 累加（少写）或漏掉 atomic（后写覆盖前写）。
- dK/dV 只在部分块上少贡献：`is_kv_change`/`is_kv_end` 与 K/V 三槽轮转不一致，或 index 列表
  里同一 KV 块的连续轮次被拆给不同核。[^bsag-cube][^bsag-addr]
- 确定性路径重复运行不一致：latin-square 的 `(round, core, m, n)` 反解与 mask 判块次序
  不一致。[^bsag-addr]
- 完全不计算直接出零：index 列表为空，或 `indexShapeWorkspace` 的有效块个数读到旧值。
  [^bsag-nonzero]
- sftg 约为正确值一半：缺少 odd RegSlot 或只 cast even lane。[^tla-cast]
- 尾块串行数值：UB 行 stride 用了有效列数。只在尾块出现大负值或未定义值：哨兵未填，或
  mask 后下游仍读满整块。空行输出 `nan`：`LSE` 用了 0 或 `-inf`。空行 `dQ/dK/dV` 非零：
  `D` 或 `P` 未按无效行归零。[^fag-fwd]
- 长序列更慢且尾核明显：有效 tile 在核间分布失衡（mask 偏斜而调度只按轮转）。[^bsag-addr]
- 单核正确、多核缺梯度：atomic Fixpipe 未排空就放行 Vector。
- 平均核很快但个别核空转：latin-square 组内的无效轮次未跳过。[^bsag-addr]
- 高稀疏 workload 加速不随有效块数下降：先看 workspace 清零/post、`D` 前处理与 NonZero
  预 kernel 的固定占比。[^bsag-vec][^bsag-nonzero]
- 偶发读到上一 task：三槽轮转与 flag 次数不同步，或 K/V 槽在非 `is_kv_change` 轮次被覆写。
  [^bsag-arch35]
- 超大 tensor 仅尾段未初始化或回写缺失：64-bit workspace size 到 32-bit AIV 均分字段的
  截断。[^bsag-vec]
- vendor 参数报 mask dtype 错：mask 未转 BOOL/UINT8。全 1 mask 正常、随机 mask 错：mask 轴
  顺序不一致。聚合达标但长序列退化：逐 case 被均值掩盖。[^bsag-aclnn]

# 验证方法

先记录 tiling key，确认命中 `1000..1005`（非确定性）或 `1008..1013`（确定性）且与预期
dtype/layout 一致，再用 Torch FP32 oracle 按有效稀疏 tile 计算上述公式。oracle 必须接收
同一次前向产生的 O 与 LSE；除最终 dQ/dK/dV 外，至少抽查 `D`、重算 P、dS，才能区分前向
中间量、SoftmaxGrad 与后三次 MMAD 的误差。[^bsag-readme][^bsag-cube]

- FP16/BF16 × BSND/BNSD/TND × MHA/GQA/MQA，并覆盖 `deterministic=true|false` 两条任务
  生成路径的对照（同一输入应给出同样的数学结果，允许低位差异）。[^bsag-addr]
- `64x64`、`128x128`，以及能暴露 tile 边界的 `64x128`、`128x64` block shape（均为 16 的
  倍数）。[^bsag-tiling]
- 对齐/非对齐尾长（分桶 `1/15/16/17/63/64/65/127`，并验证
  `blockShapeX != blockShapeY`），空行与零长度段用独立用例断言输出精确为 0、无 NaN/Inf。
  [^fag-fwd]
- mask 至少构造全 1、单块、随机非对称、QBlocks 与 KVBlocks 不相等、逐 query head 不同；
  对称 mask 只能作回归，不能作为轴语义证据；参考侧必须接收同一份按 README 语义构造的 mask，
  而不是转置后的 mask。[^bsag-readme][^bsag-aclnnapi]
- TND 多 batch 累计边界、中间零长度段、仅 Q 空、仅 KV 空、非 128 对齐尾段，以及 Q/KV 每
  batch 长度不同但总 token 数满足 host 约束的场景；累计数组异常必须在 host 侧确定性拒绝。
  [^bsag-common][^fagrad-readme]
- group size 1/2/4/最大值，多 Q head 写同一 KV head，以及不同 task 完成顺序的重复运行；
  分别对比每个 Q head 的 dQ 与按 KV head 聚合的 dK/dV。[^bsag-cube]
- 同一逻辑 tensor 构造 BNSD、BSND、TND 三种物理布局，覆盖 Q/KV 长度不等、MHA/GQA、多
  batch、尾块；地址测试必须包含非对称数值。[^bsag-common]
- index 路径专项：断言 `indexShapeWorkspace` 里的有效块个数等于手工统计值；把同一 mask
  分别喂给两条路径并逐 tile 比对 `(batch, q_head, q_block, kv_block)` 集合。
  [^bsag-nonzero][^bsag-addr]
- 确定性路径专项：固定输入重复运行，要求逐元素 bit 级一致；覆盖组内无效轮次（mask 空洞）
  与行尾不足一组的场景。[^bsag-addr]
- sftg：单独对拍每行 `rowsum(dO.float()*O.float())`，覆盖 FP16/BF16、三种 layout、TND
  多段、尾行与多核；主循环测试同时检查 workspace 在首个 sparse tile 读取前已完成写入。
  [^bsag-vec]
- 固定提交的硬件测试源码只有一个 FP16/BNSD/MHA、`128x128` 序列、`64x64` block、全 1 mask
  的 launch smoke；O/LSE 被置零，测试只检查调用成功并打印 dQ，不断言 tiling key/目标架构，
  也不与 golden 比较，不能作为 Arc35 数值正确性证据。[^bsag-test]
- 与 vendor NPU 算子比较时，独立构造 BOOL/UINT8 block mask、同步计时并逐 workload 报告
  比值；正确性未通过的 case 不进性能汇总。[^bsag-aclnn]

优化时一次只验证一个轴：若 profile 显示高稀疏场景由 index 生成与每核取块开销主导，可验证
放大 `index_block_num` 分组粒度；若尾核显著且与 mask 偏斜相关，可在 host 侧按每核有效块
数做 cost-aware 分配（代价是调度元数据）；若小 workload 主要耗在 workspace 清零、`D`
前处理与 post，可分别验证减少冗余扫描。dQ 必须保持每 task atomic、dK/dV 必须保持同 KV 块
L0C 累加，不要为负载均衡破坏这两个边界。以上均为固定源码导出的可证伪候选，只有同配置完整
correctness 与 benchmark/profile 才能进入 learned knowledge。[^bsag-cube][^bsag-tiling]

若目标含 attention sink，读
[Block Sparse Attention Grad Sink](block-sparse-attention-grad-sink.md)。

# 扩展层（非 vendor Arc35 基线）

以下内容**不属于** `ops-transformer` 的 Arc35 实现，出处是本仓库的 `code/flex_fag_kernel.py`
（一份 CATLASS DSL 变体）。放在这里是为了不丢信息并明确边界：**默认主体仍是前面的 vendor
Arc35 行为**，不要把本节机制当作 Arc35 的既有能力。

## 细 token mask

该变体在块级稀疏之外再做 token 级可见性，压缩成三层元数据：

- `sparse_compute`：packed int32 位图，1 表示该 tile 可见/需要计算；
- `sparse_mask`：同款位图，1 表示该 tile 需要逐行细 mask，0 表示整 tile 无需细 mask；
- `maskr`：逐 query 行的右边界；`holel`/`holes`：逐行 hole 起点与长度，hole 数有编译期上界
  `holeMaxNum`（`_MAX_HOLES = 3`），零 hole 仍需合法占位地址。

位图按 **KV-major** 存放为 `[B*Tk, Tqw]`，word 内每一 bit 对应一个 Q tile；tile 粒度为
`_TILE_M = _TILE_N = 128`。`sparse_compute = 0` 的 tile 不进入五路 MMAD；`sparse_compute = 1`
时再由 `sparse_mask` 决定是否加载逐行边界与 hole；hole 区间必须裁剪到当前 KV tile 与实际
KV 长度。稠密语义须先由独立 mask compiler 转成这套压缩元数据，Cube 与 Vector 必须使用
同一份判定。

```text
K-major sparse_compute/sparse_mask GM -> 标量位测试（跳块）
maskr / hole 起点 / hole 长度 GM -> UB -> Vector 逐行谓词
clean tile（sparse_mask=0）与 fine tile（sparse_mask=1）在行循环外分成两个静态循环体；
fine 在进入 SIMD 前准备逐行谓词，两者共用同一 cast zN buffer 写 L1
```

clean 的有效 lane 只受 Q/KV tail 控制；fine 额外应用逐行 token mask，被 mask lane 的 P/dS
精确置零。fine 谓词预计算与当前 Fixpipe 重叠；两条路径的 flag 数量、输出槽与尾行处理必须
一致，不能采用不同 scale 或 cast 协议。该变体另外把 Q/dY 的 L1 做成**三槽私有轮转**
（`qcyc` 0/1/2，预取下一有效 sub-tile），并在 task 内做 first/last valid 预扫：Cube 侧用
last-valid 决定 dQ/dK/dV 的落盘时机，跳过死前缀与空任务，但空任务仍写零 `dQ`。

验证要点：覆盖全可见/全屏蔽、单/三 hole、跨 tile hole，比较有效 tile 集、P/dS 置零位置与
最终梯度；不得用全 1 mask 证明 fine 路径正确。同一 tile 数据分别走 clean 与「全可见 fine」
并要求结果一致；profile 分别统计 clean/fine tile 数与 Vector 时间。

实现侧口径（同源）：P/dS 经 `BlockStoreParams(block_stride=65)` 以 zN 分形写 UB（64 数据行
外带 1 行 bank 错开），UB 分配按 65 行足迹而不是逻辑 64 行；f32→b16 全宽 cast 需要
even/odd 两个 RegSlot；`L0C2UBMode.SPLIT_M` 让两路 AIV 各取半行。
[^tla-api][^tla-cast][^tla-fa]

## head_dim 32/64

vendor Arc35 只允许 `headDim == 128`（见上文约束）。`code/flex_fag_kernel.py` 的
`get_flex_fag_tiling_data` 里 `assert head_dim in (32, 64, 128)`，即 D=32/64 是该变体额外
支持的特化：D=32/64 的 UB/L1 view 与归约 mask 必须收缩，不能继续按 128 lane 读；全宽
b16→f32 仍必须组合 even/odd RegSlot，只用一个会丢一半 lane。[^tla-cast]

---

[^bsag-readme]: 固定提交中 BSAG 的功能说明、稀疏块契约、公开公式与 mask shape。
    
[^bsag-aclnn]: 固定提交的 aclnnBlockSparseAttentionGrad 接口文档，含 layout、dtype、block
    shape 与约束条文。
    
[^bsag-entry]: 固定提交中的 AICORE 310 条件、mixed-core 属性与 `1000..1005`/`1008..1013`
    模板分发。
    
[^bsag-arch35]: 固定提交中的 AIC/AIV 流水、三槽轮转、跨核 flag、阶段屏障与 post。
    
[^bsag-addr]: 固定提交中的 NonZero index 取块、latin-square 确定性调度、mask 判块与任务
    到核的分配。
    
[^bsag-common]: 固定提交中的 layout offset、三套 stride 族、TND 累计长度与 mask 线性读取。
    
[^bsag-cube]: 固定提交中的五次 MMAD、dQ 每 task atomic、dK/dV 跨 task L0C 累加与
    `MM345CopyOut<true>`。
    
[^bsag-vec]: 固定提交中的 workspace 清零、SoftmaxGradFront、SimpleSoftmax、AttentionOut 后
    处理与 AIV 均分。
    
[^bsag-vfcast]: 固定提交中的 P/dS ND→NZ cast 辅助实现。
    
[^bsag-nonzero]: 固定提交中的 NonZero 预 kernel，把 mask 真值元素展开成一维下标列表。
    
[^bsag-tiling]: 固定提交中的 Arc35 capability 检查、base tile、`singleM`、核数与
    workspace 布局。
    
[^bsag-nztiling]: 固定提交中的 NonZero 预 kernel host tiling。
    
[^bsag-tilingh]: 固定提交中的 Arc35 tiling data 字段，含 `indexWorkspaceOffset` 与
    `indexShapeWorkspaceOffset`。
    
[^bsag-aclnnapi]: 固定提交中 aclnn 适配层对 `DAV_3510` 的 mask `perm = {0,1,3,2}` 转置。
    
[^bsag-test]: 固定提交中的单一 BNSD FP16 hardware launch smoke 及其验证边界。
    
[^fag-fwd]: 固定提交 flash-attention 前向文档中无效行（softmax max 保持 -inf）的
    attn_out 清零与 softmax_lse=+inf 约定。
    
[^fagrad-readme]: 固定提交 FlashAttnGrad 接口表中 `cu_seqlens_q`/`cu_seqlens_kv` 作为 TND
    累积序列长度的定义。
    
[^tla-api]: 固定提交 `core_api.py` 中的 `mmad` layout 契约、`cross_flag` mode 与
    `L0C2UBMode.SPLIT_M`、`tla.extern` 接口。
    
[^tla-atomic]: 固定提交 `basic_matmul_atomic_add` 中的 `AtomicMode.ADD` copy 模式。
    
[^tla-mixed]: 固定提交 `basic_mixed` 中的 Cube/Vector 混合与跨核 flag 骨架。
    
[^tla-crossflag]: 固定提交 `cross_flag_two_way` 中的双向 cross flag 配对模式。
    
[^tla-cast]: 固定提交 `cast_multi` 中的多 RegSlot cast 与寄存器布局用法。
    
[^tla-fa]: 固定提交 flash-attention 示例中的尾块 `tile_view`、DINTLV lane 与 zN 落盘用法。
    
