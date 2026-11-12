---
type: CATLASS DSL Operator Example
title: Flash Attention Backward (FAG / flash_attn_grad)
description: Ascend950 Flash Attention 反向（FAG）在 CATLASS DSL 下的算法、分核、数据路径、流水同步与优化地图，以及重做这个算子时的方向与判据（预算先行、先判绑定管道、flag 台账先行、单轴候选与回退信号；不含逐 case 观测值）；来源只引用 catlass 仓库 catlass-dsl 自身的 kernel/example 与 API，以及固定提交的 ops-transformer 上游实现与设计文档。
tags: [catlass-dsl, operator, flash-attention, attention, backward, fag, flash-attn-grad, five-mmad, l0c, fixpipe, atomic, tnd, gqa, budget, credit-ledger, ablation, vendor-baseline]
status: stable
generated: {by: process:fasg-knowledge-consolidation, at: '2026-09-29T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-09-28T11:35:00Z'}
  - {by: process:fasg-evidence-gate, at: '2026-09-29T00:00:00Z'}
  - {by: process:fasg-knowledge-max-sum-contract, at: '2026-10-08T08:35:00Z'}
sources:
  - id: upstream-readme
    resource: https://gitcode.com/cann/ops-transformer/blob/adc5fd30003da4c73bc478cf42c6f031b491b973/attention/flash_attention_score_grad/README.md
    title: ops-transformer FlashAttentionScoreGrad README（固定提交 adc5fd30）：反向公式、输入输出、layout 与 shape 约束
  - id: upstream-design
    resource: https://gitcode.com/cann/ops-transformer/blob/adc5fd30003da4c73bc478cf42c6f031b491b973/attention/flash_attention_score_grad/docs/FAG算子设计介绍.md
    title: FAG 算子设计介绍（固定提交 adc5fd30）：六步计算流程、950 tiling 与 CV 流水、多模板与确定性计算设计
  - id: upstream-kernel
    resource: https://gitcode.com/cann/ops-transformer/blob/adc5fd30003da4c73bc478cf42c6f031b491b973/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad.cpp
    title: FAG 算子入口（固定提交 adc5fd30）：按 tiling key 分派 BN2/BN2GS1S2/S1S2/BasicDet/MLA 等模板，并在同一 kernel 内区分 AIC/AIV 执行段
  - id: upstream-bn2
    resource: https://gitcode.com/cann/ops-transformer/blob/adc5fd30003da4c73bc478cf42c6f031b491b973/attention/flash_attention_score_grad/op_kernel/arch22/flash_attention_score_grad_s1s2_bn2.h
    title: BN2 模板实现（固定提交 adc5fd30）：workspace 清零 AtomicClean/EODClean、kvAtomic/qAtomic 首写不累加、CrossCore 同步 flag 编号、ping-pong 事件编号分配
  - id: fa-infer-kernel
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/attention/flash_attention_infer/flash_attention_infer.py
    title: catlass-dsl Ascend950 FlashAttention 推理 example 的 kernel：grid-stride 任务循环、L1/L0/L0C 缓冲、L0C2UBMode.SPLIT_M、zN 视图、cross flag 编号与 PRE_LAUNCH
  - id: fa-infer-tiling
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/attention/flash_attention_infer/fa_tiling.py
    title: 同 example 的 tiling：Q/KV 基块与 L1 tile 形状、各 buffer 份数、GQA 的 group_size
  - id: fa-infer-readme
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/attention/flash_attention_infer/README.md
    title: 同 example 的文档：分块 + online softmax、中间计算恒 FP32 的 dtype 组合、编译期常量与 CLI 覆盖
  - id: dsl-core-api
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/core_api.py
    title: CATLASS DSL core API：l0c fixpipe 可窄化的目标 dtype，以及 atomic 的目标位置与累加 dtype 白名单
  - id: dsl-atomic-add
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul_atomic_add.py
    title: CATLASS DSL atomic add 示例：MMAD 结果直接累加到 GM（GM C 必须为 f32）
  - id: dsl-cross-flag
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/dsl_tutorials/cross_flag_two_way/cross_flag_two_way.py
    title: CATLASS DSL 跨核同步示例：mode=4 双向 cross flag 与同核 pipe-to-pipe flag
operator_families: [flash-attention, attention]
arch: [c310, ascend-950]
---
# 接口与概念

> **证据分级**：本文 `sources` 分两类——(a) `cann/catlass` 固定提交（`7b574fb3`）里 catlass-dsl 自身的
> kernel/example 与 API；(b) 固定提交的 ops-transformer 上游 FAG 实现与文档。**不引用 catlass 仓库
> `examples/` 下的 C++ example，也不引用本项目自己产出的设计/benchmark artifact。**

## 算子算法

本仓 FAG 一次 launch 同时产出三路梯度，输入为 Q、K、V、O（前向输出）、dO 和前向的**真实**逐行统计量
`softmax_max`/`softmax_sum`（`[B, N1, S1, 8]` fp32，列 0 为有效值，其余 8 lane 复制）：

```text
S  = scale * Q @ K^T      Sm = where(atten_mask, -inf, S)
P  = exp(Sm - softmax_max) / softmax_sum
D  = rowsum(dO * O)
dP = dO @ V^T             dS = P * (dP - D)
dQ = scale * dS @ K       dK = scale * dS^T @ Q      dV = P^T @ dO
```

因为 `softmax_max`/`softmax_sum` 是前向给的真实逐行统计量（等价地 `LSE ≜ max + ln(sum)`；本文其余处
一律用这两个统计量，不写单张量 `LSE`），`P` 的行和恒为 1，反向不需要 online-softmax 重缩放，也不需要
自己维护 max/sum；
`D` 可以预计算成 per-row workspace（本实现可选），`scale` 折叠进 `dS` 从而只应用一次，`dV` 不乘
`scale`。**这是本实现的算子契约**。

同一组公式在上游公开实现里有直接出处：`dV = P^T dY`、`dQ = ((dS)*K)/sqrt(d)`、
`dK = ((dS)^T*Q)/sqrt(d)`，其中 `S = Mask(QK^T/sqrt(d) + pse)`、`P = Dropout(Softmax(S))`。[^upstream-readme]
上游实现按六步组织，并且**不显式输入逐行统计量**——`FlashSoftmaxGrad` 的两个入参是 `dy` 和正向输出
`attention_in`，`D` 由它算出；本实现把前向的 `softmax_max`/`softmax_sum` 作为显式输入，
`D = rowsum(dO*O)` 是本实现对同一量的取法。[^upstream-readme]

上游算子的 shape 约束（BSND/BNSD 视图）：`B ∈ [1, 2M]`、`N ∈ [1, 256]`、`S ∈ [1, 1M]`、
`D ∈ [1, 768]`（Ascend 950PR/950DT；A2/A3 为 `[1, 512]`），`keepProb ∈ (0, 1]`，且
query/key/value 的 dtype 与 `input_layout` 必须一致。[^upstream-readme]

轴命名沿用本领域惯例：`N1 = N2 * G`（`G` 为共享同一组 K/V 的 query head 数）。catlass-dsl 的写法是
在 tiling 里算 `group_size = num_heads // kv_heads`，kernel 里取 `kv_head_idx = head_idx // group_size`，
所以 dK/dV 必须把 `group_size` 个 query head 的贡献归约到一起。[^fa-infer-kernel]

变长（TND）场景下，实际序列长度由 `actual_q_seqlen` / `actual_kv_seqlen` 以张量形式传入 kernel，而不是
在 kernel 内重新推导布局；同一 example 里 `mask` 只做占位（全 0）。[^fa-infer-readme]

上游实现把 layout 也作为编译期模板参数：BN2 模板里 `TND` 是 `LAYOUT == 3` 的常量，TND 分支用
`tndSplitCoreParams.bN2idxStarts/Ends` 按核划分 `(B, N2)` 段。[^upstream-bn2]

**本实现**支持三种 layout（`0 = BNSD`、`1 = BSND`、`2 = TND`）：TND 用累积前缀和形式的
`actual_seq_qlen`/`actual_seq_kvlen`（以 0 开头、末值为总长），任务空间按 segment 展开，一个任务只走自己
segment 内的 KV 区间。单次入口调用只 launch 一个 `@tla.kernel`。

**方向与判据**：

- 出口路径由 `(G, layout, KV 尾块是否整除)` 一起决定，不是一个可以后补的开关：单写者张量走非 atomic
  出口，多写者张量才需要在 atomic 与 fp32 workspace 归约之间选；判据是"一个元素被多少个 writer 累加"
  加目标精度门禁，而不是 dtype 本身。
- `G > 1`、`head_dim = 192`、tail（`s2 < bn`、`s1 % bm != 0`）是这类算子的常态而非例外，必须做成
  编译期分叉点；把它们留成运行期分支会让每个 shape 都退化到最慢路径。
- 先定路径再算成本：算法层取舍（前向统计量是否显式传入、是否预计算 `D`、`scale` 折叠在哪一步）会整段改变
  后面的数据路径与出口，先定下来再做 tiling。

# 用法

## 分核策略与基本块切分

catlass-dsl 的 attention example 给出了这套 DSL 里可用的分核骨架：任务号由
`tla.range(tla.arch.block_idx(), total_task_num, tla.arch.block_num())` 做 grid-stride，任务号再分解成
`(q_block_idx, head_idx)` 并加 batch 偏移；内层对 KV 轴循环，循环上界比实际块数多 `PRE_LAUNCH = 2`
（CV 提前发射）；AIV 侧再用 `tla.arch.sub_block_idx()` 把自己该处理的一半拿走。基本块是
`Q_BLOCK = KV_BLOCK = HEAD_DIM = 128`，`Q_BLOCK_SUB = Q_BLOCK // 2 = 64`（AIV 每个 sub-block 处理 64
行）。[^fa-infer-kernel]

L1 侧的分块和缓冲份数由 tiling 统一给出：`Q_BASE_TILE = KV_BASE_TILE = 128`，
`QK_L1_TILE = (M 128, N 256, K 192/192)`，`PV_L1_TILE = (M 128, N 128, K 256/128)`，buffer 份数
`Q 1 / K 2 / V 2 / P 3`。[^fa-infer-tiling]

上游 950 设计文档给出的基本块与核数配比：Cube 侧 `(S1_i, D) × (D, S2_i) → (S1_i, S2_i)` 取
`128 × 128`，Vector 侧因 UB 单块上限约 32 KiB 且按 float32 计算取 `(S1_i/2, S2_i) = 64 × 128`，
两者按 **Cube:Vector = 1:2 的核数比**配对；CV 之间一轮的数据量约 512 KiB，核内基本块一般 32 KiB。[^upstream-design]

上游把 tiling 结论压成模板，由 tiling key 在算子入口分派，同一份 kernel 源码覆盖多 layout、dtype 与
mask/dropout 组合，模板名与切分轴一一对应：

| 架构  | 模板                                | 切分轴                         | 走入条件（摘）                                                                                                        |
| ----- | ----------------------------------- | ------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| A2/A3 | B / N2 / SameAB / S1S2 / TND        | 多核切 B、切 N2、UB 切 S1S2 … | 按`S1*S2`、是否确定性计算、是否 TND 选择 [^upstream-design]                                                         |
| 950   | BN2 / BN2S2 / 确定性计算 / BN2GS1S2 | 多核切 BN2、BN2S2、BN2GS1S2    | `S1<=128 and S2<=128 and G=1` 走 BN2；TND 且 `B*N2*S2 > cubeCoreNum` 走 BN2S2；其余走 BN2GS1S2 [^upstream-design] |

**本实现**的任务是 `(batch, query head, query chunk)`，任务内部扫完整条 KV 轴（Q-outer，Q/dO 在任务内
常驻 L1），基本块 `bm = 128`（`head_dim <= 128`）或 `bm = 64`（`head_dim = 192`），`bn = 128`。输出
所有权决定归并方式，这是本算子最重要的一条分核约束：

- **dQ 单写者**：一个任务独占一个 query chunk，`l0c_q` 跨整条 KV 扫累加，循环结束后**每 task 回写
  一次**，不需要 atomic。
- **dK/dV 多写者**：`s1 / bm` 个 query tile（GQA 再乘 `G`）都贡献同一个 dK/dV 元素且分属不同核，因此在
  L0C 里累加完一个 tile 就立刻把它 `fixpipe atomic ADD` 出去，归约顺序不影响结果。

**方向与判据**：

- 先算尾轮量化 `(ceil(tasks / cores) * cores - tasks) / (waves * cores)`，它是一阶固定开销；只有它
  显著时才考虑尾块再切（方向见 `optimization/multicore-load-balancing.md`），否则优先改任务粒度本身。
- 输出所有权（单写者 / 多写者）决定归并方式与是否需要 atomic / workspace，是分核的第一约束；先定所有权，
  再定 Q/KV 哪个轴做外层。
- 任务轴方向（Q-outer vs KV-outer）的取舍标准是"每个输出元素被多少个不同核写"：Q-outer 下 dQ 单写者、
  dK/dV 多写者；KV-outer 能把 dK/dV 收敛成单写者，代价是同时活着的 accumulator 更多（见"数据路径与
  存储层级"的预算约束）。

## 数据路径与存储层级

catlass-dsl 推理 example 的数据路径组织（可作为 FAG 同类布局的参照）：

- L0A/L0B 各开 ping-pong 两份；L0C 也开 ping-pong，且 S 与 PV 各占一对 fp32 缓冲。[^fa-infer-kernel]
- MMAD 用 `tla.mmad(l0c, l0a, l0b, init_c=True)`；出 L0C 时用
  `tla.params.CopyL0C2DstParams(l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M)` 把结果按 M 方向切给 AIV。[^fa-infer-kernel]
- 需要回喂 MMAD 的中间量（P）在 UB 上直接按 `tla.arch.zNUnAlign` / `tla.arch.zN` 排布，然后整块
  copy 进 L1 的 `zN` 视图，避免在拷贝路径上做 layout 转换。[^fa-infer-kernel]
- README 明确：输入 f16/bf16 时中间计算恒为 FP32，最后再转回 fp16/bf16 写回 GM。[^fa-infer-readme]

上游 950 设计文档对这条路径的表述是：950 新增 AIC↔AIV 的 `L0C -> UB` 与 `UB -> L1` 高速通路，Cube
侧 `Bmm` 结果放 L0C 后经 `fixp_l0c_to_ub` 按 `S1_i/2` 均分给两个 V 核，Vector 算完再 `copy_ub_to_l1`
聚合回 L1，之后才做 dq / dkv 两个 Bmm。[^upstream-design]

DSL 层面对出口路径的约束（决定 dQ/dK/dV 能不能省掉 fp32 workspace）：

- `l0c` 经 fixpipe 出到 `f32`/`f16`/`bf16` 是允许的，其它目标 dtype 直接报错——"L0C 直接窄化写
  fp16 输出"在 DSL 层面成立。[^dsl-core-api]
- atomic 的目标位置只能是 **GM**；累加 dtype 白名单是 `f32/f16/i16/i32/i8/bf16`。atomic add 的示例里
  GM 侧累加张量要求是 f32。[^dsl-atomic-add]

**本实现**的数据路径（先结构、后预算；重做时要复现的是层与所有权，字节数由 tile/dtype/buffer 数决定）：

```text
Q / dO input GM -> L1 l1_q / l1_y（任务内常驻，单写者）
K / V input GM -> L1 l1_k,l1_kb / l1_v,l1_vb（双槽）-> L0A/L0B（双槽，仅 HD<=128）
  -> S MMAD（Q zN × K^T nZ）-> L0C cwide = S fp32
  -> FIX SPLIT_M -> UB ub_s/ub_sb（fp32 主体）+ ub_s16（fp16 zNUnAlign 出口）
  -> Vector exp(Sm - softmax_max) / softmax_sum -> 整块 store -> L1 l1_p/l1_pb（zN）= P
dO（L1 zN）× V^T（L1 nZ）-> dP MMAD -> L0C cw_dp = dP fp32
  -> FIX SPLIT_M -> UB ub_p/ub_pb（fp32 主体）+ ub_p16（fp16 zNUnAlign 出口）
  -> Vector P*(dP-D) -> 整块 store -> L1 l1_s/l1_sb（zN）= dS
dS（L1 zN）× K（L1 zN）-> dQ MMAD -> L0C cq（跨整条 KV 扫常驻，单写者）-> 循环末窄化 -> GM dQ
dS^T（L1 nZ）× Q（L1 zN）-> dK MMAD -> L0C cwide（S 出片后相位轮转）-> FIX 窄化 atomic -> GM dK
P^T（L1 nZ）× dO（L1 zN）-> dV MMAD -> L0C cw_dp（dP 出片后相位轮转）-> FIX 窄化 atomic -> GM dV
G>1 出口分叉：dK/dV 的 FIX 改写 fp32 ws_k/ws_v，主循环后按块 cast 回输出
```

逐行因子：`softmax_max`/`softmax_sum` 经 MTE2 进 `ub_m`/`ub_l`；`O`/`dO` 分块进 `ub_o`/`ub_y`，向量算出
`D = rowsum(dO * O)` 存 `ub_d`；`ub_m`/`ub_l` 与 `ub_d` 一起参与 `exp(S - softmax_max)`、`/softmax_sum`
和 `P*(dP-D)`。

所有权与预算：L1 上 Q/dO 单写者常驻，K/V/S/P 双槽（`l1_kb`/`l1_vb`/`l1_sb`/`l1_pb` 是槽 1）；UB 上
S/dP 的 fp32 主体与 fp16 zNUnAlign 出口分开占位，zN 排布让 L1←UB 是一次整块 copy，转置交给 L0 操作数
的 layout，不做转置 copy。L0C 是本节最硬的预算：三个 fp32 accumulator（S/dK 轮转的 `cwide`、dP/dV
轮转的 `cw_dp`、dQ 常驻的 `cq`）按 `(bm, bn)`/`(bn, head_dim)` 计最多放三块，第 4 块放得下但吃满余量；
`D=192` 时 `L0C3` 谓词为假，`cw_dp`/`cq` 别名回 `cwide`，dK/dV 的相位轮转消失、退化成单 accumulator
串行。`DIRECT_OUT` 谓词是"直写 fp16 还是走 fp32 workspace"的编译期开关（形如见"约束"一节）。
**实测**：UB 上的 zN 排布 + 整块 store 相比 row-major 分块 copy，把 s4096 的 `aiv_mte3` 压下一个量级
（量级：大——它直接决定 AIV 侧是不是关键路径）。

**方向与判据**：

- 先列逐级预算表再动手：L0C 能同时放几个 fp32 accumulator（每个是 `(bm, bn)` 或 `(bn, head_dim)` 块，
  全库上限 256 KiB）、UB 能放几份 fp32 副本与低精度副本、L1 各输入开几份 buffer。预算决定"哪些槽位
  可用"，不是先写好代码再砍。
- 预算不足时的正确退路是**共享 buffer + 相位轮转**（同一块物理 buffer 在不同相位承载不同张量，用彼此
  独立的 flag 链表达所有权），不是硬加槽位；相位轮转的前提是这些 flag 链不落在同一个物理 flag 上。
- 出口优先级：能用 fixpipe 直接窄化写出就不要落 fp32 workspace；只有精度门禁不允许时才付 workspace 的
  清零、回读与后处理代价。
- 给低精度 atomic 出口留一个编译期谓词位置：能否直写取决于尾块是否整除与 writer 个数，凡是"先跑通再补
  尾块判断"的实现都会在边界 shape 上挂死，而不是算错。

# 代码模式

## 流水排布、同步关系与数值精度

**DSL 侧的同步原语**：跨 AIC/AIV 用 `tla.cross_flag(name[, mode])`，同核 pipe 之间用
`tla.flag(name, src_pipe, dst_pipe)`。推理 example 里跨核 flag 是**按方向编号**的：QK 方向用 0/1，
softmax 方向用 2/3/4，PV 方向用 5/6；每个方向的编号都只用了 2–3 个。[^fa-infer-kernel]
`cross_flag_two_way` 示例给出最简形态：两个 `mode=4` 的 cross flag 做双向握手，加一对同核 flag 串起
MTE2→VECTOR→MTE3。[^dsl-cross-flag]

**CV 流水**：推理 example 用 `PRE_LAUNCH = 2` 表达"CV 提前发射"——KV 循环的上界多算两块，
`MAX_CROSS_CORE_BUF_STAGES = PRE_LAUNCH + 1` 决定跨核缓冲的份数。[^fa-infer-kernel]

上游设计文档把同一思想写成 **"Cube 双发"（CV 间 preload）**：提前发射两块 Cube 计算，使 V 侧算完上一轮
就能立刻启动当前轮；C 侧总耗时大于 V 侧时目标是 MTE2/Cube bound，反之目标是 Vector bound。V 侧自身用
double buffer 把搬运与计算 ping-pong 开。[^upstream-design]

上游 BN2 模板把同步资源显式编号，可作为"编号预算"的参照：跨核 flag 只定义了 5 个
（`SYNC_C1_V1_FLAG=0`、`SYNC_V1_C2_FLAG=1`、`SYNC_DQ_FLAG=2`、`SYNC_DK_FLAG=3`、`SYNC_DV_FLAG=4`）；
核内 ping-pong 用 `AllocEventID<HardEvent::V_MTE2>` / `<MTE2_V>` / `<MTE3_MTE2>` / `<MTE3_V>` /
`<V_MTE3>` 成对申请、`ReleaseEventID` 归还。[^upstream-bn2]

**本实现的同步**：主循环对 KV 轴做一轮滞后——A 相产 S/dP，B 相消费上一轮的 P/dS；同轮内
MM1/MM2 在 A 相、MM3/MM4/MM5 在 B 相。每个 pipe 方向只有 4 个 event id（正好用满），同核 buffer 所有权
用普通 flag，跨 AIC/AIV 用 mode-4/mode-2 cross flag。两处踩出来的要求：UB→L1 的 zN 整块 store 之前必须
有一条 `pipe_barrier(ALL)`（否则结果是确定的但错的）；跟 fixpipe copy 串起来的 MMAD 用 `unit_flag=0b11`，
跨轮累加的 MM3 不需要。

**本实现的数值精度**：Q/K/V/dO/O 按输入 dtype 读，乘积与累加全部 fp32（L0C 累加器、UB 向量状态），
P/dS 只在进入 MMAD 操作数时按输入 dtype 舍入一次，dQ/dK/dV 只在出口再舍入一次。**实测**：低精度
atomic 的可用性取决于一个元素被多少个 writer 累加——`G == 1` 时 dK/dV 是 `q_blocks` 次，可用 fp16
atomic；`G > 1` 时是 `G * q_blocks` 次，fp16 舍入的抵消在量级上会顶穿"逐元素必须完全命中"的门禁。
所以出口路径必须按 `G` 分叉：`G == 1` 走 fixpipe 直接窄化写输出，`G > 1` 保留 fp32 精确归约。

上游对同一取舍的做法是**让 workspace 承担 fp32 累加、出口再 cast**：BN2 模板里
`sizeof(T1) == sizeof(float)`（fp32 输入）时直接清 `dvGm`，否则清 `dvWorkspaceGm`（`dq`/`dk` 同理）；
workspace 元素数按 `B32_BLOCK_NUM = BLOCK/sizeof(int32_t)` 向上取整，另有 `EODClean` 处理真实 T
之外的尾部。[^upstream-bn2]

**方向与判据**：

- 先建 flag 台账再写第一条 flag：每条 flag 记 producer pipe、consumer pipe、它保护的物理 buffer、启动
  credit、归还距离、drain。台账是"删一个阶段 / 删一条 wait"之前必须过的自查——删掉生产阶段会把该阶段的
  资源所有权一起删掉，结果是挂死而不是回退；自查清单见 `debug/cube-vector-deadlock-and-credit-audit.md`。
- "一轮滞后 + 双槽"本质是把循环重构成 prologue / steady / drain 三段，不是把某个 wait 换个位置；跨 task
  时必须重置 prev-valid，否则第一个 task 会消费上一个 task 遗留的中间量。
- barrier 只在结果可见性处保留（UB→L1 的 zN 整块 store 之前），其余按依赖收窄成定向 flag；写法见
  `optimization/pipeline-and-synchronization.md`。
- 数值取舍写成可判定的门禁：低精度累加的可用性取决于"一个元素被多少个 writer 累加"，所以出口路径按
  `G` 分叉（`G == 1` 直接窄化，`G > 1` 走 fp32 精确归约），并且这条门禁在 `G` 和 tail 上都要成立。

# 本仓实现：方向与已否证清单

本节只保留方向结论与已否证项；逐版本观测数值留在项目 evidence，不随本文件发布。

## 关键路径迁移史："为什么之前慢"

这条链路的关键路径换过多次。**唯一可复用的结论是：绑定管道会随版本迁移，每次优化前必须先量当前 gate
再动手**——照抄别处的优化点、或继承上一轮的瓶颈判断，大概率打在非 gate 上。

历史里出现过完整的翻转：先是 AIV 侧搬运受粒度限制；补上"UB 按目标 zN 排布 + 整块 store"后转为
L1→L0 载入与 MMAD 不重叠；再往后发现 fixpipe 掉一大截而 Duration 完全不动（当时 fixpipe 不在关键
路径上，真正的 gate 是 fp32 workspace 的 zero-fill 与后处理 pass）；等出口改成 fixpipe 直接窄化直写
之后，fixpipe 反而"成为"新的 gate。同一份 pipe 计数在不同阶段会给出相反的结论。

因此：**不要因为"某个 pipe 很满"就直接优化它**——要被删掉的工作量真的让 Duration 动，才说明它在
关键路径上。逐版本的观测数值不写进本文。

## 每 workload 的主导成本

| workload 家族                | 先看哪一项                                                                                   |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| 长稠密（大`s2`、多 batch） | 出口（fixpipe / atomic RMW）与 MAC 效率：先比 per-pipe ratio，再判是"出口受限"还是"MAC 空闲" |
| `head_dim = 192`           | 每 pair 固定开销与 L0C/UB 预算：`bm` 被迫减半后 L0 槽位与 accumulator 数量都会变紧         |
| GQA/MQA（`G > 1`）         | 精确路加`G * q_blocks` 次累加：出口必须走 fp32 归约，成本高于同尺寸 MHA                    |
| TND 变长                     | AIV 向量段扫描与段边界判断，而不是 cube                                                      |
| 小尺寸 / tail                | launch 与固定开销：该量级上比值不可靠，不要用小 shape 的比值下结论                           |

## profiler 症状 → 先查哪里

| 症状                                                            | 先查                                                                                                                                                  |
| --------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `fixpipe` ≈ `Duration`，`cube_utilization` 高但 MAC 不满 | 先确认它是不是**当前** gate：与 MAC/MTE 同段比较，并确认删掉这部分工作量后 Duration 真的动；出口的 atomic 读改写（L0C→GM RMW）只是候选解释之一 |
| `aic_mte1 + aic_mac ≈ Duration`                              | L1→L0 载入与 MMAD 没有重叠：查 L0 操作数槽位数（是否真的开了第二个槽）                                                                               |
| `aiv_mte3` 占 40% 以上                                        | UB→L1 的 P/dS 搬运粒度：row-major UB + 分块 copy 会退化成 32 B burst                                                                                 |
| `aiv_mte2` 高且 fixpipe 只到 MAC 的一半                       | 存在 fp32 workspace 回读的后处理 pass                                                                                                                 |
| `aiv_vec` 在 varlen case 里顶到 60%                           | TND 的逐行段扫描/tail mask，而不是 cube                                                                                                               |
| kernel 编译成功但结果确定性地错在少数 tile/head                 | 缺`pipe_barrier(ALL)` 之类的 UB/MTE3 可见性，先按"前几个 tile 精确、后面损坏"这种 signature 找                                                      |
| 加进第 5/6 组同步 flag 就 hang（永不返回）                      | 同步编号预算：IR 里会出现`EVENT_ID4`/`EVENT_ID5`，工具链不报错但硬件不接受                                                                        |

## 优化轴：已落地 / 已否证

已落地（历史观测，用于估计同类改动的量级）：

| 轴                                | 修改对象                                       | 预期变化（量级 + 受影响 pipe）                                                      |
| --------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------------------------- |
| L0A/L0B 双操作数槽                | 给需要第二组操作数的 MMAD 单独的第二组 L0 slot | 大；把 L1→L0 载入与 MMAD 的重叠打开，`aic_mte1` 不再与 `aic_mac` 相加          |
| UB 按目标 zN 排布 + 整块 store    | P/dS 的 UB 排布与 UB→L1 copy                  | 大；UB→L1 搬运退化成小 burst 时先做这条，`aiv_mte3` 直接下降                     |
| fp16 直接出口（fixpipe 窄化）     | 删掉 fp32 workspace、zeros 与整个后处理 pass   | 大；`aiv_mte2` 与后处理整条消失，之后 `fixpipe` 成为新的候选 gate               |
| L0C 相位轮转（多个 accumulator）  | 让两个出口不再共用一块 L0C accumulator         | 中；缩短 MAC 与 FIX 的交接串行链                                                    |
| AIV 热路径 barrier → 定向 credit | 把`pipe_barrier(ALL)` 换成显式归还 credit    | 小；只在`tla.flag(MTE3, VECTOR)` 被前端丢弃、必须用显式 credit 表达归还时才值得做 |

已否证（不要再试）：

- K/V 预取：中性，不改变 Duration。
- 删掉 L0C→UB 的 S/dP copy 却不改出口：Duration 不动——"某个 pipe 很满"不等于它在关键路径上。
- 删掉任何一条 wait、或整体删掉 dK/dV 的 fixpipe 出口：会把 `unit_flag=0b11` 的消费者或该阶段的资源
  所有权一起抽掉，结果是挂死而不是变快；消融前先过 flag 台账。
- **dK/dV 首写非 atomic、后续 ADD**：把全部 dK/dV 出口改成非 atomic 后 Duration 不变，说明 atomic 本身
  不构成开销，这条轴不再投入。
- 只做部分 zeroing / 后处理：workspace 链路的收益是整段的，不能按比例拆着做。
- 为 `head_dim = 192` 强上 `bm = 128`（在 L0C/UB/操作数维度分别排除后仍 `507015`）。

# 约束

- 单次入口调用只能 launch 一个 kernel；不得用 GM 临时张量串联多个 executor，也不得用 Torch 计算 API
  实现。
- 编译期形状/开关集中在文件顶部常量，由 CLI 覆盖；**改值触发重新编译**，因此不要把它当作运行期参数
  通道。[^fa-infer-readme]
- GQA 要求 `HEAD_NUM` 能被 `KV_HEAD_NUM` 整除（`group_size = num_heads // kv_heads` 必须是整数）。[^fa-infer-tiling]
- 输入 f16/bf16 时中间计算恒为 FP32；出口按目标 dtype 只舍入一次。[^fa-infer-readme]
- atomic 只能打到 GM，且累加 dtype 必须在 `f32/f16/i16/i32/i8/bf16` 白名单内。[^dsl-core-api]
- `keepProb` 只在 `(0, 1]` 内合法；query/key/value 的 dtype 与 `input_layout` 必须一致；950 上 `D`
  最大 768（A2/A3 为 512）。[^upstream-readme]
- 同一个算子入口用编译期模板/tiling key 覆盖多 layout、dtype 与 mask/dropout 组合，而不是为每个组合
  写独立算子。[^upstream-kernel]
- L0C 可用 256 KiB；本实现占用三块 64 KiB accumulator（S/dK、dP/dV、dQ），第 4 块可放但会吃满余量，
  第 5 块立即 `device error type 3, error code 507015`（**实测**）。
- 同步编号是稀缺的固定资源：跨核 flag 按方向编号，同一方向只分配少数几个。[^fa-infer-kernel]
- TND 必须按 segment 展开任务并只扫自己的 KV 窗口（上游对应 `tndSplitCoreParams` 的按核分段）。[^upstream-bn2]

# 失败表现

- dK 的逐元素命中率差一点点、达不到要求的 1.0：低精度 atomic 累加次数过多，检查是否漏掉了 `G>1`
  的精确路分叉（上游对 fp16 输入一律走 fp32 workspace）。[^upstream-bn2]
- 前几个 kv tile 精确、后面的 tile 或 head 损坏，且结果完全确定：UB→L1 zN 整块 store 前缺
  `pipe_barrier(ALL)`。
- kernel 永不返回：同步编号用超了预算（IR 里会出现 `EVENT_ID4`/`EVENT_ID5`）。
- 首个 synchronize 就报 `device error type 3, error code 507015`：L0C 超预算。
- `G == 1` 走 fp16 直写出口时，在**非整除的 KV 尾块**上首个 synchronize 就挂住，报
  `device error type 3, error code 507014`（是挂死不是数值错）：出口必须有尾块谓词门禁
  （形如 `(G == 1) && (S2 % KB == 0)`），不满足就退回 workspace 出口；边界 shape 的处理见
  `dsl/dynamic-tail-layout-and-origin-shape.md`。
- 消融实验里删掉一条 wait 或整段生产阶段后挂住：flag 的 set/wait 计数失衡，或者该阶段还承担着所有权
  归还；先过台账再动手（见 `debug/cube-vector-deadlock-and-credit-audit.md`）。
- O 或 P 整行/整段错：前向统计量偏移（按 head / batch / token 三种 layout 的 stride 不同）或 TND 段边界
  扫描错误。
- GQA 的 dK/dV 串 head：`kv_head = q_head // group` 或 K/V offset 错误。[^fa-infer-kernel]
- workspace 元素数不是 1024 的整数倍时读到未初始化内存、多次调用结果不一致：清零与后处理的分块表达式
  不一致（上游 `AtomicClean` 用 `B32_BLOCK_NUM` 向上取整，并有 `EODClean` 处理真实 T 之外的
  尾部）。[^upstream-bn2]
- benchmark 报 `profiler 未生成可用的 kernel 性能数据` 且 `anti_hack=not_run`：先看宿主负载、原样
  重跑，不要改 kernel。

# 验证方法

1. **静态**：确认只有一个 `@tla.kernel` 入口和一次 launch；确认 wrapper 只做元数据检查、空输出分配和
   launch；确认没有为编译期变体声明独立 kernel。
2. **正确性**：用冻结的 fp32 Torch oracle 和**前向真实统计量 `softmax_max`/`softmax_sum`** 输入跑全部
   required cases，三路输出逐元素
   要求 `matched_ratio == 1.0`；覆盖 BNSD/BSND/TND、fp16/bf16、`head_dim ∈ {64,128,192}`、
   GQA/MQA（`G>1`）、tail（`s2 < bn`、`s1 % bm != 0`）和非 1024 整数倍的元素数；多跑几次并在
   reference **之后**再调用候选，以暴露 workspace 未清零类问题。
3. **变长**：TND 用 `actual_q_seqlen`/`actual_kv_seqlen` 形态的实际长度输入，确认每个任务只扫自己的
   segment。[^fa-infer-kernel]
4. **anti-hack**：每次 launch 只允许出现声明的 kernel 名字、每 workload 一次 launch；profiler 的
   `kernel_details.csv` 必须保留。
5. **性能（交付口径）**：只在 NPU 上计时，按**冻结的 benchmark 口径**测量（配置与 warmup/trials 固定，
   不随探索变化），逐 workload 报告而不是只报聚合；宿主负载高时先看 load 再解释异常值。
6. **profiling**：按 pipe 分解（`aic_mac`/`aic_fixpipe`/`aic_mte1`/`aic_mte2`/`aiv_mte2`/`aiv_mte3`/
   `cube_utilization`）而不是只看 Duration，并用 `fixpipe ≈ Duration` 这类关系判断关键路径换了没有。
7. **DSL 约束回查**：出口走 fixpipe 时确认目标 dtype 在 `f32/f16/bf16` 内；走 atomic 时确认目标是 GM
   且 dtype 在白名单内。[^dsl-atomic-add]
8. **测试契约（探索期）**：每个 shape 用全新进程验证（同一进程连续跑多个 shape 会把后几个小 shape 误判为失败）；
   bench 输出目录每次新建、不复用（复用会把上一次的 profiling 结果与 anti-hack 报告带进来）；候选与
   baseline 用交错 A/B 取中位数比较，不用单次快值。
9. **设备纪律**：单条命令限时、长跑放后台再轮询，避免一条挂住的 kernel 阻塞整台设备；判定性能前先按
   pipe 分解并与 vendor baseline 用同一采集口径对比（见 `profiler/metrics-and-bottlenecks.md` 与
   `profiler/collection-workflows.md`），必要时用 `Block Num`/`Mix Block Num` 确认 launch 覆盖。

[^fa-infer-kernel]: catlass-dsl `examples/python/tla_dsl/tla/kernel/attention/flash_attention_infer/flash_attention_infer.py`：`tla.range(block_idx(), total_task_num, block_num())` 任务循环与 `q_block_idx`/`head_idx` 分解、`sub_block_idx()` 与 `Q_BLOCK_SUB`、`PRE_LAUNCH`/`MAX_CROSS_CORE_BUF_STAGES`、L0A/L0B/L0C ping-pong 缓冲、`tla.mmad(..., init_c=True)`、`CopyL0C2DstParams(l0c2ub_mode=L0C2UBMode.SPLIT_M)`、P 的 `zNUnAlign`/`zN` 视图、按方向编号的 cross flag（0/1、2/3/4、5/6）、`actual_q_seqlen`/`actual_kv_seqlen`。
    
[^upstream-readme]: ops-transformer FlashAttentionScoreGrad 算子 README（固定提交 adc5fd30）：反向公式 `dV=P^T dY`/`dQ`/`dK`、输入输出表、layout 一致性与 B/N/S/D（950 上 D ≤ 768）取值范围、`keepProb ∈ (0,1]`。
    
[^upstream-design]: ops-transformer FAG 算子设计介绍（固定提交 adc5fd30）：六步计算流程与 `FlashSoftmaxGrad` 入参、950 的 `(128,128)`/`(64,128)` 基本块与 1:2 核数比、CV 基本块 512 KiB/核内 32 KiB、`fixp_l0c_to_ub`/`copy_ub_to_l1` 高速通路、CV preload 流水、多模板与确定性计算模板表。
    
[^upstream-kernel]: FAG 算子入口 `op_kernel/flash_attention_score_grad.cpp`（固定提交 adc5fd30）：按 tiling key 分派 BN2/BN2GS1S2/S1S2/BasicDet/MLA 等模板，同一 kernel 内用 `ASCEND_IS_AIV` 区分 AIV 执行段。
    
[^upstream-bn2]: BN2 模板 `op_kernel/arch22/flash_attention_score_grad_s1s2_bn2.h`（固定提交 adc5fd30）：`LAYOUT == TND` 常量与 `tndSplitCoreParams` 分段、`AtomicClean`/`EODClean` 的 `B32_BLOCK_NUM` 向上取整清零、fp32 输入走 `dvGm` 而 fp16 走 `dvWorkspaceGm`、`kvAtomic`/`qAtomic` 首写不累加、`SYNC_*_FLAG` 5 个跨核 flag 编号、`AllocEventID/ReleaseEventID` ping-pong。
    
[^fa-infer-tiling]: catlass-dsl `examples/python/tla_dsl/tla/kernel/attention/flash_attention_infer/fa_tiling.py`：`Q_BASE_TILE`/`KV_BASE_TILE = 128`、`QK_L1_TILE(M,N,K_LEFT,K_RIGHT) = (128,256,192,192)`、`PV_L1_TILE = (128,128,256,128)`、`Q/K/V/P_L1_BUF_NUM = 1/2/2/3`、`group_size = num_heads // kv_heads`。
    
[^fa-infer-readme]: catlass-dsl `examples/python/tla_dsl/tla/kernel/attention/flash_attention_infer/README.md`：分块 + online softmax 公式、输入 f16/bf16 时中间计算恒 FP32、编译期常量被 CLI 覆盖且改值触发重新编译、`HEAD_NUM` 必须被 `KV_HEAD_NUM` 整除、mask 仅占位。
    
[^dsl-core-api]: catlass-dsl `python/tla_dsl/catlass/core_api.py`：`l0c` fixpipe 的 `f32 → f32/f16/bf16` 窄化白名单；atomic 目标必须为 GM，累加 dtype 白名单 `f32/f16/i16/i32/i8/bf16`。
    
[^dsl-atomic-add]: catlass-dsl `examples/python/tla_dsl/tla/kernel/matmul/basic_mmad/basic_matmul_atomic_add.py`：MMAD 结果直接累加到 GM 的写法，GM 侧 C 必须为 f32。
    
[^dsl-cross-flag]: catlass-dsl `examples/python/tla_dsl/dsl_tutorials/cross_flag_two_way/cross_flag_two_way.py`：`tla.cross_flag(name, mode=4)` 双向握手与 `tla.flag(name, src_pipe, dst_pipe)` 同核同步。
