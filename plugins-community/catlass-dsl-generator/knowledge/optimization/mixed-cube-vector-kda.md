---
type: CATLASS DSL Optimization Guide
title: Linear Attention 与 Mixed Cube/Vector 局部块候选
description: 面向包含独立局部块的 Linear Attention 等 mixed 算子的任务聚合、双 AIV、共享操作数与条件 solve 候选。
tags: [catlass-dsl, optimization, mixed, cube, vector, kda, attention, causal, sparse-tile, load-balance, task-coarsening, dual-aiv, shared-rhs, grouped-dma, packed-solve]
status: stable
generated: {by: human:caijianlong, at: '2026-08-06T00:00:00+08:00'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-06T00:00:00+08:00'}
  - {by: process:catlass-dsl-msprof, at: '2026-08-11T16:52:15+08:00'}
  - {by: process:catlass-dsl-kda-architecture-review, at: '2026-10-11T00:00:00+08:00'}
sources:
  - id: mixed
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py
    title: CATLASS DSL mixed Cube and Vector example
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/6ccf88e89723b65461e9921047c7970a71b67b42/python/tla_dsl/catlass/core_api.py
    title: CATLASS DSL copy and synchronization APIs
  - id: sparse-balance-kernel
    resource: project-evidence:python/tla_dsl/examples/end_to_end/chunk_gla_fwd_kernel_o/chunk_gla_fwd_kernel_o_kernels.py?kernel-sha256=9678ff7347025489d5153413075a577123b4e7f527803def05aadd0d2f31d0b0
    title: BT128 triangular tile balancing implementation
    kind: implementation
  - id: sparse-balance-result
    resource: project-evidence:.catlass-dsl/optimize-runs/chunk-gla-triton-align-20260811/manual/iter-005-bt128-triangular-balance/result.json?kernel-sha256=9678ff7347025489d5153413075a577123b4e7f527803def05aadd0d2f31d0b0
    title: Final correctness and msprof result
    kind: profiling
operator_families: [mixed, attention, kda, linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

适用于 Linear Attention（包括 GDN/KDA）等含独立局部块的 mixed Cube/Vector 计算、
block-local solve、L0C 到 UB handoff 和双 AIV 分工。只修改 Python DSL 算子；
编译器、lowering、runtime wrapper 和 backend
限制应保留原始错误并回退候选，不能作为本优化轴的修改目标。[^api]

# 用法

先确认 launch 覆盖全部任务，再按 `cube_utilization`、`aicore_time`、`aiv_time`、任务数
扩展性和同步尾部提出单一假设。可用 axis 包括 `mixed-fused-diagonal-subtile`、
`mixed-aiv-subblock-postprocess-split`、`mixed-l0c-to-ub-copy-mode-alignment`、
`mixed-solve-brc-beta-load`、`mixed-packed-output-tail-reuse`、
`mixed-raw-l0c-footprint-reduction` 和 `mixed-redundant-barrier-cleanup`。

## SIMD 与 Cube/Vector 并行门禁

不要使用 SIMT 风格，用 SIMD 风格，充分利用 Cube 以及 Cube 和 Vector 之间的并行。
规则化的 row、column、feature、gate、mask 和 solve 使用 `tla.vec.func(mode="simd")`；
Cube 承担可表达的矩阵运算，Vector 承担准备与后处理。不能用增加 SIMT thread 数代替向量化
或双 AIV owner 分工，也不能把串行 Cube/Vector 阶段称为重叠。

每个候选写清 SIMD lane 到有效元素的映射、尾部与结构零 mask、Cube/Vector 生产消费关系，
以及相邻任务的重叠区间。双缓冲或任务聚合必须同时证明 owner 地址、私有 slot、ready/free
代际、最后消费者和 drain。对照固定 mixed 示例与目标 API 核实真实执行能力。[^mixed][^api]
完整正确性通过后，用总时延及 Cube/Vector/MTE、同步间隙判断收益；未测量前只记录候选。

生成新算子时不能只把“一个最小数学块对应一个 task、只使用 AIV0”作为唯一方案。若多个局部块
互相独立、在 GM 上连续，并且共享相同的 shape、dtype 和计算图，必须先完成下面的性能架构门禁；
单任务/AIV0-only 方案可以作为正确性回退，但不能在没有比较表的情况下直接冻结为唯一性能 baseline。

# 生成前性能架构门禁

对重复且独立的 Linear Attention 等局部块，先枚举 `group_size in {1,2,4}`，再结合目标平台实际可用的
AIV 数量比较 owner 映射。候选必须记录逻辑 task 数、每个 AIV 的 task 数、GM 搬运次数、执行的
MMAD 元素数、有效输出元素数、UB/L1/L0 峰值、ready/free 次数和尾组处理。不能只以实现简单或
单 task 容量更小为由拒绝聚合方案。

| 决策轴 | 正确性回退 | 必须评估的吞吐候选 |
| --- | --- | --- |
| task 粒度 | 每个 AIV 一次处理一个局部块 | 一个 AIV 处理 2/4 个 GM 连续且独立的局部块 |
| AIV owner | 仅 AIV0 | AIV0/AIV1 各自拥有不重叠 group、UB 和 L1 slot |
| 输入搬运 | 每个局部块分别搬 q/k/g/beta | 地址连续且不跨语义边界时做 group DMA（整组 DMA），尾组回退逐块搬运 |
| 相关点积 | 两个独立 MMAD | 共享 RHS 时把多个左操作数沿 M 轴拼接，一次 MMAD 产出多个结果 |
| 多块 CUBE | 扩大 M/N 后计算全部交叉块 | 比较逐对角 subtile MMAD，避免不消费的跨块乘加 |
| solve（仅契约包含局部 solve 时） | 每块单独递推 | packed solve：多块按独立 lane/row 区间打包，保持块间零依赖 |
| 结构零 | 每 task 全量清零 | 地址同余和最后消费者可证明时跨 task 保留结构零，仅覆盖变化区 |

共享 RHS 融合的代数候选示例：若同一局部块需要 `Q @ K.T` 和 `Kp @ K.T`，可把 `Q` 与 `Kp`
沿 M 轴拼成 `A=[Q;Kp]`，以一次 `A @ K.T` 同时产生两个结果。该候选只有在累加 dtype、计算顺序、
layout、有效区和输出拆分与 reference 相容时才成立；否则保留两个 MMAD。多个局部块聚合时还要比较：

```text
大 M/N MMAD 的执行乘加数
vs.
逐局部块紧凑 MMAD 的执行乘加数 + 额外 FIX/flag/copy 成本
```

不消费的跨局部块乘积属于明确的计算浪费，必须计数，不能只因大 tile 看起来更规则就采用。反之，
紧凑 MMAD 增加的提交和同步成本也必须由同配置 profile 裁决。上述变换都是待验证候选，不继承其他
算子的性能结论。CUBE/AIV 片上交接和双 AIV 的真实 API 能力必须由固定 mixed 示例与当前目标源码
核对。[^mixed][^api]

设计文档记录适用候选及前提。“聚合 task + 双 AIV”仅在存在独立块和对应执行能力时比较；
“共享 RHS/对角 subtile”分别要求共享操作数或只消费对角输出，solve 候选要求局部 solve 契约。
缺少这些计算结构时记录不适用，不要求将每种候选纳入所有 Linear Attention 设计；因公式、
地址连续性、tail、容量或 DSL API 排除具体路径时记录可复核证据。只要公开 ABI、reference、
输出所有权和单 kernel 契约不变，这些预先批准的轴可在 optimize 阶段逐轴实验，
不需要把 AIV0-only owner 或单 task 粒度当成不可变条件。

## 耦合架构包与实现可行性门禁

上表中的轴并不总能独立产生收益。若 task 聚合只有在双 AIV owner、私有 slot、group DMA 和 packed
postprocess 同时成立时才改变关键路径，应把这些依赖声明为一个“最小可行架构包”，而不是依次测试
彼此不完整的单轴版本。一个架构包必须记录 `package_id`、组成轴、前置依赖、允许的降级边界和整体
判伪条件。单独放大 `group_size` 后没有收益，不能用于否定依赖双 AIV 或 group DMA 的完整包。

首次编码前完成下面的实现可行性表；表中任何结论都要引用目标版本 Python DSL 源码、固定示例或
最小编译探针，不能由相似 C++ API 推断：

1. **能力表**：逐条列出 copy/MMAD/vector/flag API 的源与目标存储空间、layout、dtype、静动态
   shape 和合法参数组合。一个 layout 转换不支持时，只能判该数据路径失败；检查合法替代路径后，
   才能判整个架构包不可行。
2. **owner/address 表**：对每个 `(logical_task, group_member, AIV)` 写出输入和输出半开区间、物理
   slot 及 layout。有效输出必须恰好被一个 owner 覆盖，不能重叠，也不能把不同 group member 写入
   同一 row/column tile。
3. **私有资源表**：双 AIV 默认使用不重叠的 UB/L1 slot；若共享物理 buffer，必须证明所有写者和
   读者的先后关系，不能只依赖逻辑 owner 名称。
4. **同步代际表**：按 `(task, slot, generation, producer, consumer)` 统计 ready/free 的发送和等待
   次数，覆盖首轮、稳态、tail、空任务、slot 回绕和退出排空。
5. **数值 DAG 表**：列出 solve/postprocess 的依赖、累加 dtype、cast/round 次序和结构零条件。
   focused case 通过但完整 BF16 workload 失败时，应判为数值顺序或边界问题，不能据此否定 task
   grouping、owner 或 DMA 架构。

实现失败必须区分“本次写法失败”和“架构包不可行”。编译/lowering 错误只证明具体 API/layout
组合不可用；越界、NaN、跨次不稳定或固定区域未写通常先指向 owner/address/slot/flag 表错误。只有
能力、地址、容量、同步或公式证据排除了包内所有合法实现路径，才允许把主吞吐包降级为串行回退。

# 局部块与分工

若最终只消费 32x32 task 中两个 16x16 diagonal block，优先验证 CUBE 直接产出所需
sub-tile，避免完整 raw matrix 及随后 VECTOR mask/gather。拆分可能减少无效计算和后处理，
但额外 MMAD、flag 和 copy 也可能更慢，必须 profile 判定。[^mixed]

`L0C2UBMode` 必须匹配消费者：AIV0/AIV1 各消费完整 tile 时用对应 `NO_SPLIT_VEC_*`；
按行或列分工时使用 `SPLIT_M` 或 `SPLIT_N`。每个 `raw_ready` 必须覆盖相应 UB 写完成，
`raw_free` 只能在对应消费者完成后释放。双 AIV 分割后处理时，两个 AIV 不得写同一输出
tile，beta、mask 和 row index 必须按 local sub-block 修正。[^mixed]

# 代码模式

```text
task rows 0..31
  sub-block 0: rows 0..15, cols 0..15 -> AIV0
  sub-block 1: rows 16..31, cols 16..31 -> AIV1
```

先确认 CUBE 的 copy mode、`raw_ready` 和 `raw_free` 与消费者一一对应，再让两个 AIV
分别写自己的输出 tile。[^mixed]

# 结构零 Tile 的联合裁剪与 Sub-block 均衡

对 causal、triangular、banded 或 block-diagonal 矩阵，先画 tile 级非零图。只做元素级
`where(mask, value, 0)` 时，结构零 tile 仍可能被 AIV 读取、转换和写回，再被 AIC MMAD
消费。有效候选必须同时考虑 producer AIV 搬运/转换、consumer AIC MMAD，以及裁剪后两个
AIV sub-block 的负载均衡。[^sparse-balance-kernel]

以 128x128 下三角矩阵和 64x64 tile 为例：

```text
diag-0 | structural-zero
-------+----------------
full   | diag-1
```

有效工作只有三个 tile。简单按上下 64 行分工会形成 1 tile 对 2 tiles；kernel 仍由较慢的
sub-block 决定。可把 `full` tile 按 32 行拆半，使每个 sub-block 处理一个 diagonal tile
加半个 full tile，即各 6144 个 FP32 元素。

补丁顺序是：

1. 用 compile-time shape 构造全零、全有效、对角 mask 的 tile 非零图。
2. AIC 仅遍历结构非零 K tile，例如 `tla.range_constexpr(0, row_tile + 1, 1)`。
3. 分别统计两个 AIV 的有效 load、cast、mask 和 store 元素，以最大值而非总和为均衡目标。
4. 必要时拆分一个全有效 tile，与两个 diagonal tile 重新配对。
5. 用 `tla.range_constexpr` 生成专用路径；不要用会在 AST lowering 中引入未初始化 Tensor
   变量的动态 Python `if`。
6. 同步更新 UB 工作区、GM/UB stride、cache key 和尾部同步。

仅跳过 AIC MMAD 可能因 AIV 仍是瓶颈而没有收益；仅裁剪 tile 而不重平衡，也会受最慢
sub-block 限制。在已验证 workload 上，这个联合候选将 focused case 从 784.619 us 降到
692.437 us，最终稳定 profile 为 656.550 us。[^sparse-balance-result]

# 约束

局部下三角 solve 先缩短 live range：beta 可用单元素 broadcast load，连续 `1x16` row
slice 比从 32x32 raw 的逐 lane gather 更稳。将多个 vector region 合并或使用动态 rank1 UB
tensor 可能触发 VF stack 或 lowering 失败，记录并回退，不能宣称为有效经验。[^api]

复用 raw f32 workspace 放置 packed f16/bf16 输出时，压缩输出不能从 raw 起点覆盖仍要读取
的行；在 raw buffer 尾部保留不重叠空间，并用 recast pointer 构造 packed 输出视图。缩小
raw UB、L0C 或 postprocess workspace 时按 raw UB、L0C、large-task profile 的顺序逐项验证。

结构裁剪只适用于恒零 tile；数据相关 mask 不能静态跳过。拆分后的 GM/UB layout stride
必须分别反映原矩阵行宽和紧凑工作区行宽，两个 sub-block 不得重叠写入或遗漏 full tile。
正确性至少覆盖三角边界、BT64/BT128、FP16/BF16、TND/BSND 和多 task。

# 失败表现

- copy mode 与 AIV 消费者不匹配，导致 stale、NaN 或稳定局部 mismatch。
- 聚合后所有 group member 落到同一 row/column tile，或只有第一个成员有输出：owner/address 表
  没有形成一一覆盖，不能把现象归因于 group 本身无效。
- 双 AIV focused case 通过而完整 workload 出现跨次 NaN/幅值漂移：优先检查私有 slot、flag 代际、
  尾组和输出重叠，再判断数值公式。
- 将 packed 输出从 raw f32 起点紧凑写入，覆盖尚未读取的 raw 行。
- 动态 rank1 UB tensor 或合并 vector region 触发 VF stack 或 lowering 失败。

# 验证方法

只可删除已由精确 producer/consumer flag 覆盖的冗余 barrier。buffer 生命周期、最后一轮
MTE/FIX/CUBE/VECTOR 完成和重用前的等待必须保留，除非有独立正确性和 profile 证据。[^api]

[^mixed]: 固定提交 mixed 示例中的片上 Cube 到 Vector 数据路径。
[^api]: 固定提交 Core API 的 copy 与同步原语。
[^sparse-balance-kernel]: `chunk_gla_fwd_kernel_o` 的三 tile AIV 紧凑布局、sub-block 分工和 AIC K 范围。
[^sparse-balance-result]: focused 与全量正确性、msprof latency 和最终接受结果。
