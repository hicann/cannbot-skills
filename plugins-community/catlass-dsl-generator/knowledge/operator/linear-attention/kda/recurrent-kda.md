---
type: CATLASS DSL Operator Example
title: Recurrent Kda
description: Decode/MTP KDA 的 AIV-only 递推核，包含状态切片、UB 队列选择、低并发诊断和优化门禁。
tags: [catlass-dsl, operator, linear-attention, kda, recurrent, decode, vector, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/recurrent_kda/README.md
    title: 算法、接口、布局与支持范围
  - id: design
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/recurrent_kda/docs/design.md
    title: 任务、内存、精度与性能设计
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/recurrent_kda/op_host/op_api/aclnn_recurrent_kda.h
    title: aclnn 接口原型
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/recurrent_kda/op_host/recurrent_kda_tiling_processor.h
    title: shape 校验、UB 预算、vStep 与队列深度选择
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/recurrent_kda/op_kernel/recurrent_kda.cpp
    title: AIV-only kernel 入口
  - id: impl
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/kda/recurrent_kda/op_kernel/arch35/recurrent_kda.h
    title: 状态切片、递推、输出流水与同步
operator_families: [linear-attention, kda]
---

# 接口与概念

## 算子算法

该算子面向长度不超过 8 的 decode/MTP 序列，在一个 AIV kernel 内按 token 顺序更新状态。可选地融合 Q/K L2 normalize、raw gate 转换和 beta sigmoid：

```text
q = normalize(q) * scale, k = normalize(k)                  # 可选
gate = g                                                    # 已预计算 log gate
gate = -exp(A_log) * softplus(g + dt_bias)                  # raw unsafe
gate = lower_bound * sigmoid(exp(A_log)*(g + dt_bias))      # raw safe
beta = sigmoid(beta) * (2 if allow_neg_eigval else 1)       # 可选
S = exp(gate) * S
delta = beta * (v - S @ k)
S = S + outer(delta, k)
o = S @ q
```

`HV % H == 0` 时，一个 value head 通过 `head_v/(HV/H)` 映射到 Q/K head。`cu_seqlens` 使用 packed 累计 offset；空序列不读取 state metadata。`ssm_state_indices` 可为一维 packed 或二维 speculative 索引，`num_accepted_tokens` 决定 MTP 初始槽。[^guide][^design][^impl]

Q/K/V/O 为 BF16；g/beta 可由公开入口接收 FP32/BF16/FP16，kernel 按实际 dtype 读入并转 FP32；state 为 BF16/FP32，支持 V-first `[capacity,HV,V,K]` 与 K-first `[capacity,HV,K,V]`。当前仅支持 K=128、V=128/256 和每序列 0..8 个有效 token。[^guide][^api][^tiling]

# 用法

## 分核策略与基本块切分

任务是完整的 `(logical sequence, value head)`，总数 `B*HV`，block 以 round-robin 方式领取任务；同一任务的 token 递推绝不跨核。并行度上限因而是非空序列数乘 HV，而不是 token 数或 V tile 数。不同活跃序列若命中同一 state slot 会产生写冲突，调用侧必须保证槽互斥。[^tiling][^impl]

每个任务先把最多 8 个 token 的 Q/K/V/gate 和 beta 搬入 UB，然后沿 V 维以 `vStep` 切片 state。每个 state tile 为 `vStep x alignK`，在 UB 中转成 FP32后完成所有 token 的 decay、矩阵-向量、outer update 和输出；下一 state tile 用 MTE2 预取。Q/K/gate 跨 V tile 复用，不能为了简化循环而在每个 `vStep` 重新从 GM 加载。[^impl]

host 从 UB 容量联合选择 `(stateOutBufferNum, attnOutBufferNum)` 为 `(1,1)/(1,2)/(2,2)`，并计算 8 对齐的 `vStep`。选择顺序是：先最少 V 循环次数，再最大输出队列总深度，再最大 `vStep`。FP32 state 比 BF16 state 每行占用更多 UB，可能降低 `vStep` 或队列深度。[^tiling]

## 成本模型与瓶颈

该实现完全不使用 Cube，因此 `cube_utilization` 不是优化目标。一个 `(sequence,head)` 至少读一个 `V*K` state，若输出 final state 还要写回；每 token 又对 state tile 执行 decay、`S@k`、outer update 和 `S@q`。短序列下固定 state 搬运占比高，长度增加后 Vector 计算和 token 串行链逐渐主导。[^design][^impl]

可用以下现象定位瓶颈：

- `B*HV < AIV core count` 且单核时间高：低并发，优先增加独立 sequence/head 任务或评估安全的 V-split 合并，不能拆 token 链。
- `V=256` 相对 `V=128` 接近两倍且 MTE stall 高：state 带宽或 `vStep` 循环主导。
- FP32 state 明显慢于 BF16 state：检查 state GM 字节量和 UB profile 是否从双缓冲退化，而不是先改数学。
- `output_final_state=false` 明显更快：state 写回/MTE3 是关键；若业务不需要 final state，不应强制保存。
- raw gate、normalize、beta sigmoid 开关造成较大差值：检查非线性和 reduce，而不是 state tile。
- AIV 活跃充分但 Vector pipe 长尾明显：检查逐 token barrier、`S@k/S@q` reduce 和 outer update。

## 优化候选与回退条件

1. 先固定 workload，记录 tiling 日志中的 `vStep/stateOutBufferNum/attnOutBufferNum`。只比较相同语义（state dtype、是否写 final state、融合开关）的版本。
2. state 带宽主导时，优先让一个 state tile跨全部 token 常驻 UB、避免重复 cast/load；调整 `vStep` 时必须同时观察 V 循环数和输出双缓冲是否被挤掉。
3. MTE3 尾部主导时，可保留 pending output 的双缓冲；如果增加队列导致 `repeatTime` 增加，host 当前明确优先更少 V tile，候选通常应回退。
4. 低并发时可研究把一个 `(sequence,head)` 的不同 V tile 分给不同核，因为各 V 行的状态更新彼此独立；但 beta/Q/K/gate 会被复制，输出/state 地址必须互斥，且新增任务调度和 metadata 成本需要 profile。token 维仍不可并行。
5. 非线性主导时，`A_log[h]`、bias、gate、beta 只应按 head/token 计算一次并跨 V tile 复用；不要把 gate 拆成独立 kernel，除非端到端 profile 证明 GM 往返更便宜。
6. K 固定为 128，可比较 add-fold reduce 与普通 reduce；任何替换都必须覆盖 K 尾部零填充、BF16/FP32 state 和全部融合开关，编译成功不等于性能有效。

# 代码模式

## 数据路径与存储层级

```text
Q/K/V BF16, g/beta, metadata GM -> per-head queues -> FP32 token buffers UB
state BF16/FP32 GM -> one V tile queue -> FP32 [vStep,alignK] state UB
FP32 Vector recurrence across token 0..L-1
  -> O BF16 queue -> GM
  -> state BF16/FP32 queue -> selected state slot GM (optional)
```

Q/K/V/gate buffer覆盖最多 8 个 token；state tile、broadcast scratch、delta、attention 和 reduce 临时量占据剩余 UB。算子没有 L1/L0/Fixpipe，也不依赖用户可见 GM workspace 做中间交接。V-first 与 K-first 通过不同二维 copy/stride 访问同一逻辑 state，不应在核内生成完整转置副本。[^design][^tiling][^entry][^impl]

## 流水排布、同步关系与数值精度

state 预取与当前 tile 计算交叠；attention/state 输出按 tiling 选择单/双队列，双队列用 pending slot 延后一次 MTE3。`MTE2->V`、`V->MTE2` 和 `V->S` hard event 保护 copy、向量计算与标量读取；token 内的 decay、reduce、outer update、output 必须保持顺序。算子没有跨核 flag，因为一个 state 链只归一个核。[^impl]

state 在 UB 中统一用 FP32 计算；BF16 state 只在 GM 边界转换。Q/K/V/O 为 BF16，normalize、gate、beta、点积、delta 和 outer update 均使用 FP32 中间量。FP32 state 可减少多 token 累计误差，但增加 state 流量和 UB 占用。[^guide][^design][^impl]

# 约束

- 每个相邻 `cu_seqlens` 差值不超过 8；末项可小于物理 token capacity，padding tail 输出无定义。[^guide][^impl]
- 活跃 sequence 的 state slot 必须有效且互不冲突；空序列不读取索引。[^guide][^impl]
- state 仅允许 slot/head 外层带间隔，最后两维必须为不重叠的稠密行主序矩阵。[^guide][^tiling]
- K/V 只支持 128/128 或 128/256；Q/K/V/O 只支持 BF16。[^guide][^tiling]
- V tile、token 流水和 output queue 的优化不得改变 state 原位更新及 speculative slot 选择语义。[^guide][^impl]

# 失败表现

- token 维被并行或 state tile 未原地保留：首 token 正确，后续 token 逐步偏离。
- state V/K 顺序或 stride 错：V-first 通过而 K-first 失败，或只有非连续 view 失败。
- padding 未清零却按 `alignK` reduce：边界维度出现稳定偏差。
- raw gate/beta 开关与预处理不一致：重复 sigmoid/softplus，衰减或 beta 系统性错误。
- 双队列 slot 提前复用：压力 case 偶发旧输出或旧 state。
- speculative 初始槽偏移错误：普通 decode 正常，MTP `num_accepted_tokens>1` 失败。

# 验证方法

正确性使用逐 token FP32 oracle，覆盖 BSND/TND、V=128/256、state BF16/FP32、V-first/K-first、连续/外层非连续 state、raw safe/unsafe/预激活 gate、normalize、beta sigmoid、空序列、1..8 token、packed/二维 state index、accepted-token 裁剪和未命中 slot guard。比较每个 token 的 O 与所有被更新 state，不只比较 final state。[^guide][^design][^entry][^impl]

性能基线至少分 `L=1/8`、`V=128/256`、state dtype、是否输出 state、任务数低于/高于 AIV 数。先在空闲 NPU 上 profile `aicore_time`、AIV 利用率、MTE2/MTE3 stall 与 Vector 时间，再改变一个候选；必须报告实际 `vStep` 和队列深度，避免把 host tiling 变化误判为 kernel 优化。

[^guide]: 固定提交中的完整语义、layout、dtype、state pool、packed/MTP 元数据和支持边界。
[^design]: 固定提交中的任务分解、内存层级、数值策略、性能瓶颈与测试设计。
[^api]: 固定提交中的 aclnn 参数顺序、属性与输出原型。
[^tiling]: 固定提交中的 shape/stride 校验、UB 字节模型、vStep 和输出队列选择规则。
[^entry]: 固定提交中的 AIV-only 入口、state dtype 实例化和参数分派。
[^impl]: 固定提交中的任务循环、V 切片、QKV/gate 复用、state/output 双缓冲、同步与递推实现。
