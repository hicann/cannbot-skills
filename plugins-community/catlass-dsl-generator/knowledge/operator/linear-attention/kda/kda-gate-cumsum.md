---
type: CATLASS DSL Operator Example
title: Kda Gate Cumsum
description: KDA gate 激活与 chunk 内累计和向量核，包含 dense/varlen 分核、A5 快路径及可执行调优判据。
tags: [catlass-dsl, operator, linear-attention, kda, scan, vector, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/kda_gate_cumsum/op_host/kda_gate_cumsum_def.cpp
    title: 接口、属性与平台范围
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/kda_gate_cumsum/op_host/op_api/aclnn_kda_gate_cumsum.h
    title: aclnn 接口原型
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/kda_gate_cumsum/op_host/kda_gate_cumsum_tiling.cpp
    title: dense/varlen 分核与 dtype 分派
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/kda_gate_cumsum/op_kernel/kda_gate_cumsum.cpp
    title: AIV-only kernel 入口
  - id: impl
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/kda_gate_cumsum/op_kernel/kda_gate_cumsum_kernel.h
    title: 通用流水、A5 regbase 快路径与数据路径
operator_families: [linear-attention, kda]
---

# 接口与概念

## 算子算法

算子先得到逐 token、逐 head、逐 K 元素的自然对数 gate，再在每个 chunk 内沿 token 维做前缀和，输出供下游 `exp2` 使用的 FP32 值：

```text
use_gate_in_kernel = false: gate[t,k] = g[t,k]
safe_gate = false:          gate[t,k] = -exp(A_log[h]) * softplus(g[t,k] + dt_bias[h,k])
safe_gate = true:           gate[t,k] = lower_bound * sigmoid(exp(A_log[h]) * (g[t,k] + dt_bias[h,k]))
gk[t,k] = sum(gate[chunk_start:t,k]) / ln(2)
```

因此 `1/ln(2)` 是算法契约而非可选缩放：后续 `exp2(gk_i-gk_j)` 才与自然指数衰减相同。累计在 dense chunk、packed sequence 边界及每个 head 上独立重置。[^guide][^impl]

输入 `g` 支持 FP16/BF16/FP32，`A_log/dt_bias` 为可选 FP32，输出恒为 FP32；逻辑 rank 为 3 或 4，K 的上限是 256。raw gate 模式与辅助输入的合法组合由调用层保证。[^guide][^api][^tiling]

# 用法

## 分核策略与基本块切分

dense 模式把 `(batch, value-head, chunk)` 作为任务，任务数为 `B*HV*ceil(T/BT)`，天然具备 chunk 级并行；varlen 模式把 `(sequence, value-head)` 作为任务，同一核顺序遍历该序列的真实 chunks，以避免为短序列发射大量空矩形任务。block 数取任务数与 AIV 核数的较小值。[^tiling][^impl]

通用路径一次处理一行 K，K 最大 256，输入和输出各有两个 slot：下一行 MTE2 可与当前行 Vector 计算重叠，MTE3 写回也按 slot 回收。chunk 内 scan 仍有严格的 token 依赖，不能把同一 `(sequence,head,chunk)` 的行拆到多个核后再无代价合并。[^impl]

Ascend 950 有两个重要整块 regbase 路径：

- `FP32 + 已激活 gate + K=128 + BT=64`：整块 `64x128` 搬入 UB，用寄存器累计并整块写回；尾 chunk 也走相同实现但使用真实行数。
- `FP32 + raw safe gate + K=128 + BT=64`：把 bias、sigmoid、`1/ln(2)` 和 scan 合入同一 regbase 循环。

其他 dtype、K、chunk 或 unsafe raw gate 回退到逐行通用路径。优化前必须先确认 workload 是否命中快路径；只优化快路径而基准仍落在通用路径不会产生收益。[^impl]

## 优化诊断与候选顺序

先分别记录 dense/varlen、完整块/尾块、已激活/raw-safe 的时延，并用 profiler 判断 MTE2、Vector 非线性、MTE3 或低并发哪一项主导。源码结构只给出候选，不构成性能结论。[^tiling][^impl]

1. 若 dense 的完整 `64x128 FP32` 明显快、尾块占比高时整体退化，优先增加按真实行数批量搬运/scan 的尾块覆盖；不要改变 scan 边界。
2. 若 varlen 的 `seq_num*HV` 小于 AIV 数且单序列含很多 chunks，瓶颈是任务粒度；可评估按 chunk 建 canonical metadata 后分核，但必须让每个 chunk 的累计从零开始，且 metadata 构造成本计入端到端。
3. 若 `inputSequenceMajor=true`，同一 head 的相邻 token 行在 GM 带 `HV*K` 跨距；当前 `DataCopyExt` 进行多行搬运。若 head-major 输入连续，则整块 copy 更直接。调布局时必须计入调用侧 transpose，不能只比较 kernel。
4. 若 raw safe gate 的 Vector 时间占主导，优先保持 bias 常驻 UB、把 `exp(A_log[h])` 每任务只算一次，并将 gate 激活与 scan 融合；拆成前置 kernel 会增加一遍 FP32 GM 读写。
5. 若通用路径 MTE2/MTE3 等待显著，保留双 slot 所有权协议后再尝试扩大行批量；单纯增加 buffer 深度不会解除 token scan 依赖，还会挤占整块 `64x128` scratch。

## 可观测判据

- `taskCount < AIV core count` 且每任务包含多个 chunks：优先检查 varlen 低并发，而不是向量指令吞吐。
- 完整 `64x128` 与尾块时延/元素差异大：检查是否命中 bulk regbase 分支及尾块比例。
- raw-safe 明显慢于预激活：检查 `Exp/Div` 或 regbase sigmoid 占比，并验证 `A_log` 与 bias 是否被每行重复加载。
- MTE3 尾部明显：检查输出 slot drain 和非 32B 对齐的 `DataCopyPad`，K 小于 8 个 FP32 元素时尤其要单独测量。

# 代码模式

## 数据路径与存储层级

```text
g GM -> (双槽输入 UB | 64x128 bulk UB)
     -> FP32 cast / gate activation / chunk-local scan on AIV
     -> (双槽输出 UB | bulk UB) -> FP32 gk GM
A_log scalar + dt_bias row -> scalar/bias UB, reused by one head task
```

算子不使用 AIC、L1、L0 或 Fixpipe，也不以用户 workspace 交换中间量。通用路径的 `row/acc/tmp/one/bias` 为独立 UB buffer；bulk 路径用两个 `64x128 FP32` 区域保存输入和输出。[^entry][^impl]

## 流水排布、同步关系与数值精度

通用路径对两个 input/output slot 分别维护 `MTE2->V`、`V->MTE2`、`V->MTE3`、`MTE3->V` event；bulk 路径额外用 `MTE3->MTE2` 保证整块写回后才能复用。读取 `A_log` 和 varlen offset 还需要 MTE2 到标量的可见性同步。删除 event 时必须证明对应 buffer 不会被下一行或下一任务提前覆盖。[^impl]

所有 gate 激活和累计都在 FP32 中完成，低精度输入先 cast；输出恒为 FP32。A5 regbase 快路径与通用路径必须对 `1/ln(2)`、safe sigmoid 和尾行给出一致结果。[^guide][^impl]

# 约束

- K 必须在 `1..256`，逻辑 rank 仅 3/4；输入物理布局由 `inputSequenceMajor` 判定。[^tiling]
- raw gate 需要 `A_log`，`dt_bias` 可选；safe/unsafe 公式不能互换。[^guide][^impl]
- varlen 的每个 `(sequence,head)` 任务顺序扫描真实 chunks；packed 边界不得继承前一序列累计值。[^tiling][^impl]
- A5 bulk 快路径只覆盖源码中的精确 dtype/shape/mode 条件，不能把其收益外推到 BF16/FP16 或任意 K/BT。[^impl]

# 失败表现

- 忘记或重复 `1/ln(2)`：下游衰减出现系统性指数偏差。
- dense/varlen 地址或边界错误：跨 batch、sequence 或 chunk 串累计。
- sequence-major 多行 copy 的 stride 错：不同 head 的 gate 周期性串扰。
- 复用 input/output slot 前漏等 event：偶发旧行、局部 NaN 或仅压力 workload 失败。
- bulk 与通用尾块 mask 不一致：只有非 64 整除的末 chunk 失败。

# 验证方法

正确性覆盖三种输入 dtype、已激活/raw-unsafe/raw-safe、有无 bias、K=1/128/256、BT=64 与非 64、dense 两种物理顺序、varlen 空/短/多 chunk 序列和尾块。对每个 chunk 用 FP32 reference 逐 token 比较，并专门交叉比较 A5 bulk 与通用边界实现。[^guide][^tiling][^entry][^impl]

性能测试至少分四组：A5 bulk 命中、同 shape 的 BF16 通用路径、dense 高并发、varlen 低并发。先在空闲 NPU 上采集 `aicore_time`、Vector/MTE stall 和任务扩展性，再一次只改变一个候选；没有设备 profile 的源码推断不得写成已验证收益。

[^guide]: 固定提交中的输入输出 dtype、属性、平台和动态 shape 契约。
[^api]: 固定提交中的 aclnn 参数顺序、必选/可选输入与输出原型。
[^tiling]: 固定提交中的 dense/varlen 任务公式、block dim、dtype 编码、布局判定和 K 上限。
[^entry]: 固定提交中的 AIV-only kernel 入口与 dtype 分派。
[^impl]: 固定提交中的 gate 公式、双槽事件、UB buffer、地址计算、逐行 scan 和 Ascend 950 bulk regbase 分支。
