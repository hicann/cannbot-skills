---
type: CATLASS DSL Operator Example
title: Recompute W U Fwd
description: KDA recompute_w_u_fwd 的 C310 mixed kernel：每个 64-token chunk 同时重算 w、u、qg 与 kg，并以 AIV UB→AIC L1 交接合并两路 MMAD。
tags: [catlass-dsl, operator, linear-attention, kda, recompute, mixed, aic, aiv, c310]
status: stable
generated: {by: human:codex, at: '2026-08-14T00:00:00+08:00'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-14T00:00:00+08:00'}
sources:
  - id: kernel
    resource: project-evidence:python/tla_dsl/examples/end_to_end/recompute_w_u_fwd/recompute_w_u_fwd_kernels.py?kernel-sha256=3dcf084d983548bcd7d9b490d56be620961f7d01eea30504fc4b11425d261139
    title: 当前 mixed kernel 的 task、AIC/AIV、L1/UB 与同步实现
    kind: implementation
  - id: host
    resource: project-evidence:python/tla_dsl/examples/end_to_end/recompute_w_u_fwd/recompute_w_u_fwd.py?kernel-sha256=17f553a0f7c8c67981e1194ed4686b89107bdd0b3d4e9653bbd8ba68900230d9
    title: 当前 host 的 workload 校验、task layout、硬件 block 数与验证入口
    kind: implementation
  - id: guide
    resource: project-evidence:python/tla_dsl/examples/end_to_end/recompute_w_u_fwd/README.md?kernel-sha256=447b50d03fea5131e498cd95bace6087cd804e19d2d86f1bd91d77f94098d0cb
    title: 算子公式、支持矩阵、历史验证与性能采集说明
    kind: documentation
  - id: workload
    resource: project-evidence:/home/npu_user7/panhangzhen/attention-bench/KDA/recompute_w_u_fwd/workload.jsonl
    title: 当前 12 个 TND/BSND workload 定义
    kind: test
operator_families: [linear-attention, kda]
arch: [c310]
---

# 接口与概念

## 算子算法

该算子对每个 `(batch/sequence chunk, value_head)` 的 64-token block 重算 KDA 前向所需的
`w`、`u`、`kg`，以及可选的 `qg`。令 `g_last` 是该 chunk 最后一行的 `gk`，其不可改变的
逐元素/矩阵语义为：[^guide]

```text
gate     = exp2(gk)
v_beta   = cast(v * beta)
u        = cast(A @ v_beta)
kbg      = cast(cast(k * beta) * gate)
w        = cast(A @ kbg)
kg       = cast(k * exp2(g_last - gk))
qg       = cast(q * gate)                 # q 存在时
```

其中 `A`、`kbg` 与 `v_beta` 均按 task 看作 `64x64`。kernel 将两次左矩阵相同的乘法合并为：

```text
[w | u] = A[64,64] @ [kbg | v_beta][64,128]
```

因此宽 MMAD 是实现机制；它不改变 `w/u` 的数学定义。[^kernel]

当前 host 支持 FP16/BF16 的 TND 与 BSND 输入。TND 的 `beta` 可为 `[T,HV]` 或 `[1,T,HV]`，
并要求 `cu_seqlens` 的边界和每条 sequence 长度都按 64 对齐；BSND 不接受 varlen metadata。
`HV % H == 0` 时，host 将 K/Q 按 group 扩展到 value-head 视图。当前 tile 固定
`BT=K=V=64`，不是动态 tile 路径。[^host][^guide]

# 用法

## 分核策略与基本块切分

逻辑 task 数为：

```text
TASKS = batch * (tokens / 64) * value_heads
```

host 将每个 TND/BSND 输入规范化为连续的 `[task,64,feature]` task layout；每个 task 独占
`w/u/qg/kg` 的 64 行输出，因而不需要 atomic。[^host]

kernel 两个区域都以相同的步长 task loop 工作：

```python
for task_id in tla.range(
    tla.arch.block_idx(), TASKS, tla.arch.block_dim()
):
    ...
```

logical block `b` 处理 `b, b+block_dim, b+2*block_dim, ...`。不同 block 的 task 可并行，
同一 block 的 task 循环串行；尾部 task 使不同 block 的迭代次数可能不同。因此 ping-pong
索引由 block 内局部迭代状态切换，而不是由 `task_id % 2` 推导。[^kernel]

默认 launch block 数读取当前 device 的 AICore 数；`--block-dim` 可显式覆盖。host 同时打印
AICore、VectorCore 与最终 block 数，便于将实际并行度与 profile 的 `Block Num/Mix Block Num`
对照。mixed block 的 AIV 区域通过 `sub_block_idx()` 分为两路，每路处理 task 的 32 行：

```text
AIV0: rows [0, 32)
AIV1: rows [32, 64)
```

这两路各自产生其半块 `kbg/v_beta/qg/kg`，并只写其拥有的输出半块。[^kernel][^host]

## 成本模型与 profiler 归因

单 task 的主要工作是：AIV 读取 5 路或 6 路输入、做 gate/乘法/exp，AIC 做一个
`64x64x128` MMAD，随后由 FIX 写回 `w/u`。`q=None` 时输入段数从 6 降为 5，且不计算/写回
`qg`。[^kernel]

- `aiv_mte2_time` 高：首先检查 task-layout 输入的 GM→UB load、q 存在与否、输入段数以及
  task 数/连续性。
- `aiv_vec_time` 或 `aiv_scalar_time` 高：检查 gate 的 `exp`、FP32 widen/narrow、beta 乘法、
  `g_last-gk` 和地址/循环开销；不要优先调整 MMAD。
- `aic_mte2_time` 或 `aic_mte1_time` 高：检查 A 的 GM→L1→L0A 以及 RHS L1→L0B 路径；
  它们不等价于 Vector 的 GM→UB 瓶颈。
- Cube/FIX 高：检查宽 MMAD、L0C 分左右 tile 写回和 A/RHS slot 的交接；不能将一次宽 MMAD
  误报为两次独立计算。
- AIC/AIV 周期性空洞或偶发 hang：先检查两个 slot 的 cross flag、AIV0/AIV1 是否都发布 ready，
  以及 block 内迭代次数是否一致。[^kernel]

优化前后保持 workload、q presence、dtype、block 数和 cache 策略一致；README 中的历史 profile
仅是此前实现/环境下的测量证据，当前修改必须重新采集，不能直接外推。[^guide]

## 优化候选与优先级

1. 先确认实际 task 覆盖与 block 数。若 `TASKS < block_dim`，应先做小 block 数对照；若大 task
   profile 的 AIC/AIV 空闲，则检查 task 分配与尾部，而不是盲目增大 tile。
2. 若 RHS 交接或 AIV MTE3/MTE2 主导，分别测 UB→L1 slot、输入 UB 双缓冲和 A/RHS L1 双缓冲；
   任何修改只能改变一个 slot/同步轴，并保留 `w/u/qg/kg` 所有权。
3. 若 Cube 主导，宽 MMAD 已消除重复的 A 读取和 MMAD launch；后续候选应从 A tile、L0A/L0B
   搬运或有效矩阵结构入手，不能退回为两个相同的 `64x64x64` MMAD。
4. 若输入 staging 主导，才评估减少 task-layout 转换或读取原 layout；必须同时重新证明
   TND/BSND、GQA、q 有无、尾 task 和输出 layout 的正确性。当前实现仍选择连续 task layout。[^host]
5. q 为空是已存在的 specialization；不要在 q-present workload 上复用 5 段 UB layout 或跳过
   `qg` buffer。

每个候选必须以完整精度检查为门禁，再用 DSL-only 与全量 profile 判断；仅源码分析不能证明
UB→L1、双缓冲或 overlap 在当前环境提升性能。[^guide]

# 代码模式

## 数据路径与存储层级

```text
TND/BSND GM
  -> host task layout [task,64,64]
  -> AIV MTE2: k,beta,gk,v,(q),gk_last -> one UB slot
  -> AIV VECTOR: kbg, v_beta, (qg), kg
  -> AIV MTE3: kbg/v_beta -> shared L1 RHS slot
  -> AIC MTE1: L1 RHS -> L0B
  -> AIC: A GM -> L1 -> L0A; wide MMAD -> FP32 L0C
  -> FIX: L0C left/right tiles -> w/u GM

AIV MTE3: qg/kg -> output GM
```

`kbg` 和 `v_beta` 没有经 GM workspace 落地：两个 AIV 各把自己的 `32x64` half tile 写入同一个
`64x128` RHS L1 slot 的不同坐标。Cube 读取 RHS 前等待两个 AIV 的 ready，再搬到 L0B。该
UB→L1 AIV→AIC 交接是 A5 专用能力；本条目架构标记为 C310，但不能据此假定所有 C310 target
均支持此路径，跨架构移植应先验证 lowering、共享 L1 可见性和 cross-core 同步。[^kernel]

以 2-byte 输入类型计，单个 RHS L1 slot 是 `64*128*2 = 16 KiB`；A 的单个 L1 slot 是
`64*64*2 = 8 KiB`。每个 AIV 的一个输入 UB slot 为 `segments*32*64*2`，q-present 为
24 KiB，q-absent 为 20 KiB；这些值未包含 allocator 对齐和其它 live buffer。L0C 使用 FP32，
其宽输出占 `64*128*4 = 32 KiB`。容量改动应按同一时刻 live range 汇总，而不能只检查某一
单独 tensor。[^kernel]

## 流水排布、同步关系与数值精度

Cube 侧为 A 的 L1 slot 使用 `l1a_free[0/1]` 与 `l1a_loaded[0/1]`，并用单个 L0A/L0B/L0C
路径的 `l0_free/l0_loaded/mm_done/fix_free` 管理 GM→L1→L0→MMAD→FIX。RHS L1 的所有权跨
AIV/AIC：

```text
AIC MTE1:  set rhs_slot_ready[slot] -> wait rhs_ready[slot] from AIV0/AIV1
AIV MTE3:  wait rhs_slot_ready[slot] -> UB->L1 half-tile -> set rhs_ready[slot]
```

Vector 输入 UB 也有两个 slot：`ub_free -> MTE2 load -> ub_loaded -> VECTOR -> vec_done -> MTE3
store -> ub_free`。`vector_buf_idx` 与 `cube_buf_idx` 都在各自 block-local task iteration 尾部
翻转，以便在 task N 的 Cube MMAD/FIX 与 task N+1 的 AIV 输入/Vector/UB→L1 工作之间形成
重叠。[^kernel]

gate 通过 `exp(gk * ln(2))` 实现 `exp2(gk)`。寄存器处理 128 个低精度元素；
`RegSlot.ZERO/ONE` 分别覆盖偶/奇半寄存器的 FP32 widen/narrow。`w/u/qg/kg` 最终以输入
FP16 或 BF16 写回；改变 cast slot、将 `exp2` 替换为自然 `exp`、或改变 beta/gate 的运算顺序
都会改变数值语义。[^kernel][^guide]

# 约束

- TND varlen 的 sequence boundary 必须按 64 对齐；可选 `chunk_indices` 必须是 canonical
  `(sequence_id, local_chunk_id)` 顺序。[^host]
- BSND 不接受 `cu_seqlens/chunk_indices`；TND/BSND 的输出在 host 端从 task layout 恢复为原
  layout。[^host]
- `q=None` 与 q-present 具有不同 `INPUT_SEGMENTS` 和 UB 片段位置，二者都必须独立验证。[^kernel]
- 任何 AIV/AIC slot 修改必须保证 AIV0、AIV1 都发布当前 slot 的 ready，Cube 才能读取 RHS；
  不得让两个 sub-block 重叠写同一半块。[^kernel]

# 失败表现

- `w/u` 都错而 `qg/kg` 正常：优先检查 RHS L1 half-tile 坐标、宽 MMAD 的左右 output tile，或
  Cube 对 `rhs_ready` 的等待。
- `qg` 仅在 q-present case 错：检查 6 段 UB layout、q segment 与 `gk_last` segment 的索引。
- 只有多 task 或大 workload hang：检查 block-local ping-pong 迭代、tail block 的 slot release、
  两个 AIV 的 mode-4 cross flag 和 L1 RHS reuse。
- TND 正常而 BSND 错，或相反：检查 host `_to_tasks/_from_tasks` 的 reshape/permute 及 beta
  规范化，而不是先改 Cube kernel。
- FP16/BF16 只在小值或特定位型失败：检查 `RegSlot.ZERO/ONE` 的 widen/narrow、FP32 中间值
  和参考的 tolerance，而不是将输出直接改为 FP32。
- profile 显示 block 数正确但结果未改善：同步/容量开销可能抵消 overlap；保留正确版本并回退
  候选，不把静态数据路径当成性能结论。

# 验证方法

功能验证以指定 workload 文件的全部 case 为准，至少覆盖：TND/BSND、FP16/BF16、q present/
absent、不同 H/HV、不同 token 数、多个 sequence 边界和 batch。运行时先确认打印的
`tasks`、AICore/VectorCore 和最终 `block_dim`；再分别检查 `w/u/qg/kg` 的 shape、dtype、
finite 与误差门限。[^host][^workload]

```bash
python python/tla_dsl/examples/end_to_end/recompute_w_u_fwd/recompute_w_u_fwd.py \
  --run --run-all-cases \
  --workload /home/npu_user7/panhangzhen/attention-bench/KDA/recompute_w_u_fwd/workload.jsonl \
  --device <device>
```

性能测量先完成同配置 warm-up，再使用 `--dsl-only` 跳过 CPU reference、D2H 与比较；仍保留输入
准备、kernel launch 和必要 synchronize。新 profile 目录中单独读取
`recompute_w_u_fwd_mixed_kernel` 的 Count、Task Duration、Block Num、AIC/AIV MTE、Vector、
Cube/FIX 指标，并与完全相同的 dtype/layout/q/block_dim/cache 条件比较。[^host][^guide]

```bash
msprof --output=output/recompute-w-u-fwd-current \
  python python/tla_dsl/examples/end_to_end/recompute_w_u_fwd/recompute_w_u_fwd.py \
  --run --run-all-cases --dsl-only \
  --workload /home/npu_user7/panhangzhen/attention-bench/KDA/recompute_w_u_fwd/workload.jsonl \
  --device <device>
```

[^kernel]: 当前 kernel 固定 tile、mixed task loop、AIV half-tile、UB/L1/L0 分配、wide MMAD、flag 与 cast 实现。
[^host]: 当前 host 的输入契约、task/layout 变换、编译 specialization、硬件 block 选择、DSL-only 与验证入口。
[^guide]: README 的算子公式、历史实现说明、支持范围和验证/采集命令；历史性能数字不构成当前版本结论。
[^workload]: 当前 workload 文件列出的 12 个 TND/BSND shape、dtype、q 和 sequence 配置。
