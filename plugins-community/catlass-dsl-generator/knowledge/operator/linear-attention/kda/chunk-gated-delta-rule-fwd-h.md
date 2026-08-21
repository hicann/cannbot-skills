---
type: CATLASS DSL Operator Example
title: Chunk Gated Delta Rule Fwd H
description: 64x64 chunk gated delta rule recurrent FwdH 的 CATLASS DSL 单 launch task-loop mixed kernel，涵盖 task-major host packing、AIC/AIV 片上交接、状态链和负载均衡诊断。
tags: [catlass-dsl, operator, linear-attention, kda, gated-delta-rule, fwd-h, recurrent, mixed, ascend-950pr, c310]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-14T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-14T00:00:00Z'}
sources:
  - id: guide
    resource: https://gitcode.com/m0_53222058/catlass/blob/e533d4e2aee145e5e5863c2933f95aaf66bab859/python/tla_dsl/examples/end_to_end/intracard_fwd_h/README.md
    title: 算法、接口、支持范围与实现状态
  - id: host
    resource: https://gitcode.com/m0_53222058/catlass/blob/e533d4e2aee145e5e5863c2933f95aaf66bab859/python/tla_dsl/examples/end_to_end/intracard_fwd_h/intracard_fwd_h.py
    title: host 路径选择、task packing、workspace 和 launch
  - id: kernel
    resource: https://gitcode.com/m0_53222058/catlass/blob/e533d4e2aee145e5e5863c2933f95aaf66bab859/python/tla_dsl/examples/end_to_end/intracard_fwd_h/intracard_fwd_h_kernels.py
    title: kernel 入口、task-loop 任务分派与 AIC/AIV 流水
  - id: workload
    resource: /home/npu_user7/panhangzhen/attention-bench/KDA/intracard_fwd_h/workload.jsonl
    title: 12 个 TND/BSND FP16/BF16 workload
operator_families: [linear-attention, gated-delta-rule, kda]
arch: [c310, ascend-950pr]
---

# 接口与概念

## 算子算法

该实现计算 chunk gated delta rule 的 recurrent `FwdH`。每个 sequence 的 state 沿 chunk 严格递推；task
粒度是 `(sequence, value_head)`，同一 task 内的 chunk 不可重排：[^guide][^host][^kernel]

```text
h_chunk = S
decayed = W_chunk @ S
v_new = U_chunk - decayed
v_update = v_new * exp2(g_last - g_chunk)                  # g 存在时
S = S * exp2(g_last) * exp2(gk_last)[:, None] + K_chunk^T @ v_update
```

无 `g` 时 `v_update = v_new`；无 `gk` 时省略 key-wise state 衰减。内部 state 统一使用
`[K,V]`，`state_v_first=True` 仅改变公开 initial/final state 及 H 的末两维解释。支持
FP16/BF16 K/W/U/g/gk 输入、FP32 recurrent state、`chunk_size=64`、K/V 为 16 的倍数且不大于
128。task-loop 快路径还要求 K=V=64、`H=HV`、`group_size=1` 和所有 sequence 的 chunk 数相同；
其余 shape 进入逐 chunk 的回退路径。[^host]

# 用法

## 分核策略与基本块切分

快路径任务数为：

```text
TASKS = sequence_count * value_heads
task_id = sequence_id * value_heads + value_head
```

每个 block 通过 `range(block_idx, TASKS, block_dim)` 领取 task。一个 task 内遍历
`CHUNKS_PER_SEQ` 个 64-token chunk，确保 state 依赖不跨 block；不同 task 可以并行。[^host][^kernel]

因此并行度上限由 `sequence_count * HV` 决定，不由 token 数决定。12-case workload 的 task 数为
`1/4/8/16/32`；其中 case 005 和 case 012 各有 32 个 task。[^workload]

### block 负载诊断

`TASKS` 不能被 `block_dim` 整除时，前 `TASKS % block_dim` 个 block 会多执行一个完整 task。
这是静态 round-robin 分配的结果，不是首核初始化异常。应比较可用核数、`block_dim` 与 task
数的因子关系，并通过上板 profile 验证调度波次和总时延，不能只凭单 block 的仿真时长判断。[^kernel]

## 成本模型与 profiler 归因

一个 task 的主要工作是每 chunk 两次 `64x64` MMAD、两次 L0C->UB 的半块交接，以及 values/
state AIV 向量阶段。chunk 依赖使单 task 的 chunks 串行；并行 task 数不足或不均时，最长 task
链决定 launch 时延。[^kernel]

出现下列 profile 症状时，优先检查对应对象：

- 少数 block 时长约为其余 block 的整数倍：检查 `TASKS % block_dim` 和 task range，而非先调
  Cube tile。[^kernel]
- AIV MTE2/MTE3 长、Vector 利用率低：检查 GM<->UB、UB->L1、H/V/state 写回以及是否引入额外
  gate/workspace 中转。[^kernel]
- AIC/AIV 同时大面积 wait：检查按 ping-pong buffer 成对的 cross flag 是否 set/wait 对齐，
  以及消费者是否在错误 pipe 上等待。[^kernel]
- Transpose/ViewCopy/Cast 明显：检查 `k_work/w_work/u_work`、packed gate 和 H/V output 的
  `.float().permute(...).contiguous()` 路径。[^host]

## 优化候选与优先级

1. **先匹配 block 与 task 数。** 只调整 `block_dim`；预期减少长尾 block。保持 task 内 chunk
   顺序、task 输出所有权及 flag 名称不变。若硬件波次造成总时延不降，回退。[^kernel]
2. **保持 AIC 输出直达 AIV。** `decayed` 和 `delta` 使用 L0C->UB `SPLIT_M`；优化时不得恢复
   这些中间量的 GM round-trip。预期 AIV MTE2/MTE3 字节下降；若 UB live range 挤掉可用缓冲或
   新 flag 增加等待，应回退。[^kernel]
3. **只预取无数据依赖的输入。** AIV 可在等 decayed 前搬 U/value gate；AIC 可在等 values
   ready 前搬 K；AIV 可在等 delta 前搬 state/state gate 并计算 scaled state。不得跨越 state
   recurrent 依赖预取下一 chunk 的 state。[^kernel]
4. **原生 ABI 是独立且高风险的轴。** 当前 host 将输入转 FP32 并重排为 task-major 连续布局，
   换来连续的 AIC 读取。原生 BSND/TND 多 head 直接读会扩大 GM row stride；应先用 profile
   验证，不可把消除 Transpose 自动等同于 kernel 加速。若目标 MMAD dtype contract 不能直接
   接受低精度 K/W 和 FP32 state/vupdate，则需要在 AIV/UB widen 后再 UB->L1 给 AIC，可能加重
   已敏感的 AIV MTE。此项应单独验证精度、MTE 和端到端耗时。[^host][^kernel]

# 代码模式

## 数据路径与存储层级

当前 task-loop 不是原生输入 ABI。host 先把 BSND/TND 输入重排为 task-major FP32 GM buffer：

```text
K/W/U low precision GM
  -> Torch cast + permute + contiguous
  -> K/W/U task-major FP32 GM

g/gk low precision GM
  -> Torch exp2 + expand
  -> value_gate/state_gate FP32 GM
```

然后单一 mixed kernel 完成每个 task。下图的同一列表示可并行的 CV 工作，而不是把
`A -> B -> C -> D` 误解成 Cube 与 Vector 完全串行：[^host][^kernel]

```text
                         chunk i, ping-pong buffer b

        AIC / Cube lane                                  AIV / Vector lane
  --------------------------------                 ---------------------------------
  A0  W,S: GM -> L1 -> L0 -> MMAD
                    -> L0C[b]                      B0  U,value_gate: GM -> UB
                         |                                |  (不依赖 decayed)
                         | l0c_ready[b]                   |
                         +------------------------------> B1  decayed UB + U/g
                                                           -> v_new, v_update
                                                           -> shared L1[b]
                         <------------------------------+ values_ready[b]
  C0  K: GM -> L1 -> L0      <--- 与 B1 并行预取 K
  C1  K^T @ v_update(L1[b]) -> L0C[b]               D0  state,state_gate: GM -> UB
                    |                                    -> H, scaled_state
                    | l0c_ready[b]                         (不依赖 delta)
                    +----------------------------------> D1  scaled_state UB + delta UB
                                                           -> next_state GM
                         <------------------------------+ state_ready[b]
```

这里的 CV 融合有三个明确边界：`A0` 与 `B0` 重叠、`B1` 与 `C0` 重叠、`C1` 与 `D0` 重叠。
只有收敛点才等待 cross flag：B1 等 `l0c_ready[b]`，C1 等 `values_ready[b]`，D1 等第二次
`l0c_ready[b]`；下一 chunk 的 A0 则等 `state_ready[b]`。因此状态递推保持正确，且无依赖的
GM->片上搬运和 Vector 计算不会被 Cube 结果阻塞。[^kernel]

L0C->UB 使用 `CopyL0C2DstParams(L0C2UBMode.SPLIT_M)`；两个 AIV sub-block 各消费一个
`32x64` 半块。`v_update` 不落 GM，而从 AIV UB 直送共享 L1，随后由 AIC 送入 L0B。state 的
`scaled_state = state * state_gate` 留在 UB，等 delta 到达后完成加法，避免额外 GM 中间结果。[^kernel]

公开 H/V 仍经 FP32 storage 重塑、transpose 和 cast 回输入 dtype；可选 final state 从 task-major
state workspace 复制，并在 V-first 时 transpose。[^host]

## 流水排布、同步关系与数值精度

Cube 侧有 L1/L0A/L0B/L0C 双缓冲和 local `free/loaded/done` flag；AIV 用两个 sub-block
处理半 tile。跨 AIC/AIV 的 task-loop flag 按 ping-pong buffer 拆为：[^kernel]

```text
task_loop_l0c_ready0/1: AIC FIX -> AIV Vector   # decayed / delta
task_loop_values_ready0/1: AIV MTE3 -> AIC MTE1 # v_update 已写共享 L1
task_loop_state_ready0/1: AIV MTE3 -> AIC MTE2  # 当前 chunk state 已更新
```

每个 AIC producer 对 `aiv_id=0` 和 `aiv_id=1` 都 set；两个 AIV sub-block 都 wait 自己的
flag 实例。buffer index 在 chunk 间切换，避免 AIC 覆盖 AIV 仍在读的 L0C->UB 数据，或 AIV
覆盖 AIC 尚未消费的 L1 v_update。[^kernel]

MMAD、L0C、UB vector 中间结果和 recurrent state 为 FP32。输入 FP16/BF16 在 host `.float()`
后进入 kernel；H/V 在 host 侧 cast 回输入 dtype。g/gk gate 采用 `exp2` 的 base-2 语义，
不得改为自然指数或重复乘 `ln(2)`。[^host]

# 约束

- 同一 `(sequence,value_head)` 的 chunk state chain 不可跨 task/block 拆分或乱序。[^kernel]
- `SPLIT_M` 半块 AIV view 的坐标是 tile 坐标：`coord(sub_block_idx, 0)`；把它写为
  `sub_block_idx * 32` 会造成错误地址和精度失败。[^kernel]
- `task_loop_*_ready0/1` 的 producer、consumer、buffer index 和 pipe 必须一一对应；不能用
  单组 flag 替代 ping-pong 版本。[^kernel]
- task-major packing 与 H/V restore 是当前 ABI 的一部分；移除任一重排必须同步修改 GM stride、
  task 映射、output 所有权和完整 12-case 精度验证。[^host][^workload]

# 失败表现

- 首批 block 明显慢、其余 block 空闲：`block_dim` 与 `TASKS` 不整除，少数 block 多拿 task；
  检查 range 映射。[^kernel]
- 偶发旧 tile、hang 或只在多 task 出错：ready/free flag 被跨 buffer 或跨 task 错配，或 AIV
  sub-block 只 set/wait 了一侧。[^kernel]
- values 正确但 next state 错：`scaled_state` 在 final add 前被覆盖，或 delta 的 L0C->UB
  half-tile 坐标/flag 错误。[^kernel]
- H/V layout 正确性失败而核心公式通过：host task-major packing 或 output reshape/permute 的
  sequence/head/chunk 维度顺序错误。[^host]
- 尝试原生多 head 输入后 kernel 变慢：GM strided access 降低 AIC 搬运效率，或 AIV widen/UB->L1
  新增路径使 MTE 成为长尾。[^host][^kernel]

# 验证方法

正确性运行当前 12-case workload，覆盖 TND/BSND、FP16/BF16、H=1/2/4/8/16、1/2/4 sequence、
`g`/`gk` 的有无、initial/final state、`save_new_value=false` 与 `state_v_first=true`。每轮修改
必须同时比较 H、VNew 和 final state。[^workload][^host]

性能测试先固定 device、环境、workload 和 `block`，每 case 单独保存 `msprof` 输出；导出后读取
`op_statistic_*.csv` 的目标 kernel 耗时。对 task 调度候选，还要检查 block-level 统计中的
AIC/AIV 时间和 Cube 指令数，而不是只看总平均。候选只有在完整精度通过且同配置 profile 的关键路径
下降时保留。[^kernel]

[^guide]: 算子目录 README 中的算法、接口、支持范围与实现状态。
[^host]: 当前 host 源码中的快路径判定、task-major packing、gate materialization、state workspace、输出 restore、artifact cache 与 launch。
[^kernel]: 当前 kernel 源码中的 task range、AIC/AIV 四阶段、L0C->UB、UB->L1、双缓冲、本地 flag 与 cross flag。
[^workload]: 当前 workload 的 12 个 case shape、layout、sequence 边界、head 数、gate/state/output 组合。
