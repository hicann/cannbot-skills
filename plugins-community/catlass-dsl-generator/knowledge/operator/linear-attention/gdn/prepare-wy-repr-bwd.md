---
type: CATLASS DSL Operator Example
title: Prepare Wy Repr Bwd
description: Gated Delta Rule 的融合 WY 表示反向核，覆盖 dA 构造、四个输出、head 复用、四槽 workspace 和 mixed 流水。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, backward, fused, mixed, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-13T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-13T00:00:00Z'}
sources:
  - id: host
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/op_host/prepare_wy_repr_bwd_def.cpp
    title: 输入输出 dtype、可选 metadata 与平台范围
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/op_host/op_api/aclnn_prepare_wy_repr_bwd.cpp
    title: aclnn 连续化、输出和 launcher 路径
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/op_host/prepare_wy_repr_bwd_tiling_processor.h
    title: shape 门禁、任务、向量行和 workspace 规划
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/op_kernel/prepare_wy_repr_bwd.cpp
    title: mixed AIC:AIV 入口与模板分派
  - id: common
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/op_kernel/prepare_wy_repr_bwd_common.h
    title: chunk 地址、head 映射和四槽 workspace
  - id: cube
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/op_kernel/arch35/prepare_wy_repr_bwd_cube.h
    title: A5 九组 MMAD、片上驻留与 Fixpipe 流水
  - id: vector
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/op_kernel/arch35/prepare_wy_repr_bwd_vector.h
    title: A5 gate、mask、归约、输出和 cross-core 协议
  - id: reference
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd/test/test_final_golden.py
    title: 融合核对独立 dA/full 链的正确性矩阵
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

该 mixed kernel 把原来的 `prepare_wy_repr_bwd_da` 与 `prepare_wy_repr_bwd_full` 链合并，直接从
`K/V/beta/A/dW/dU/g` 生成 `dK/dV/dBeta/dG`，中间 `dA` 不落为公开输出。对长度为 `L` 的
chunk 和 value head `hv`，`hk=floor(hv/(HV/HK))`，主要块内关系为：

```text
Kbg   = K[hk] * beta * exp(g)
Vb    = V * beta
dKbg  = A^T @ dW
dVb   = A^T @ dU
dA4   = strict_lower(dW @ Kbg^T + dU @ Vb^T)
dA5   = dA4 @ A^T
dA6   = A^T @ dA5
D     = strict_causal(-dA6 * exp(g_i - g_j))
dKb   = D^T @ K[hk]
DK    = D @ (K[hk] * beta)
```

Vector 后处理把 `dKb`、`dKbg`、`DK` 合并成 `dK`，把 `dVb * beta` 写为 `dV`，并结合
`K K^T`、`D` 和上述中间量归约 `dBeta/dG`。同一 key head 对应的多个 value head 对 `dK`
顺序累加，其余三个输出保持 value-head 所有权。[^cube][^vector][^reference]

输入采用 `[B,H,T,D]`；`K/A/dW` 的特征维固定为 `K=128`，`V/dU` 的最后一维为
`V=128/256`，`A` 最后一维等于 `BT=64/128`。`K/V/A/dW/dU` 为 FP16 或 BF16，
`beta/g` 必须同 dtype，可为 FP16/BF16/FP32，四个输出分别跟随对应输入类型。[^host][^tiling]

# 用法

## 分核策略与基本块切分

任务是一个逻辑 chunk；fixed 模式为 `B*ceil(T/BT)`，varlen 模式为
`len(chunk_indices)/2`。`blockDim` 固定为物理 AIC 核数，每个核以 `taskIdx += blockNum`
遍历 chunk，因此实际有用并行度上限是 chunk 数，而不是 `chunk*HV`。每个 chunk 内按
两个 value head 形成 head window；AIV 的两个 sub-block 再按行 tile 分担 Vector 工作。[^tiling][^common][^cube][^vector]

四个 workspace slot 以 `0/1`、`2/3` 两组交替服务相邻 head window。K tile 在 L1
以 ping-pong 驻留，`K K^T` 只在 `hk` 改变时重算；同一 window 内映射到同一 `hk` 的两个
value head 共享这两者。尾 chunk 使用真实 `L`，但编译期矩阵宽度仍是 BT；`L=1` 的 Cube
M 维会补到 16 后按真实范围写回。[^common][^cube]

模板 key 同时编码 K dtype、gate dtype、`V∈{128,256}` 和 `BT∈{64,128}`，不是性能等级。
给定 workload 必须先确认这四个维度及 fixed/varlen 路径，再比较 profile。[^tiling][^entry]

# 代码模式

## 数据路径与存储层级

```text
K/V/beta/A/dW/dU/g GM
  -> AIV UB(FP32): Kbg, Vb, Kbeta
  -> AIC L1/L0/MMAD(FP32 L0C) -> Fixpipe -> low-precision GM workspace
       dKbg, dVb, KKT, dA1, dA2, dA5, dA6, dKb, DK
  -> AIV UB(FP32): mask/gate/dA4/D/reduce/output merge
  -> dK/dV/dBeta/dG GM
```

每个 slot 的低精度容量是 `5*(BT*K) + 2*(BT*V) + 6*(BT^2)` 个 K-dtype 元素，
分别对应 `kbg/kbeta/dkbg/dkb/dk`、`vb/dvb` 和六个 BT×BT 矩阵；每核另有四个
`BT^2` KKT slot。因此 user workspace 为物理 AIC 核数乘每核四 slot，而非仅按活跃
chunk 数分配。[^tiling][^common]

A5 Cube 在 512 KiB L1 中为 K、dW、dU、A 和 scratch 设置双缓冲；L0A/L0B 双缓冲，
L0C 仅在 `BT*max(K,V)*4B` 的两份能容纳时双缓冲。`V=256` 的 dV 路径按 K=64 分段累加，
其余形状按对应 BT/K/V tile 执行。[^cube]

## 流水排布、同步关系与数值精度

每个 head 的阶段链是：Vector 生成 `Kbg/Vb/Kbeta` 后通知 Cube；Cube 产生 `dA1/dA2`
后通知 Vector 做 lower mask；Cube 再做 `dA5/dA6`，Vector 形成门控 D；Cube 做
`dKb/DK`，最后 Vector 合并并写四个输出。两条 cross-core credit flag 分别表示
Vector→Cube ready 与 Cube→Vector ready，必须为每个 head、每个阶段严格收发一次；
MTE2/Vector/MTE3 与 MTE1/MMAD/Fixpipe 另用 ping-pong hard event 回收缓冲。[^cube][^vector]

所有 MMAD 在 FP32 L0C 累加；AIV 将 gate、指数、mask、归约和输出合并放在 FP32 UB，
每个阶段写 workspace 时再转回 K dtype。`exp(g_j-g_i)` 在进入 `exp` 前被裁到非正区间，
三角 mask 和尾块有效范围共同阻止无效位置参与输出。[^vector]

# 优化决策

先记录 `chunkNum`、K/g dtype、V、BT、GVA ratio、tail 比例和 varlen metadata。profile 中：

- 活跃 AIC 少且单核很长，优先对应 `chunkNum < AIC核数` 或每 chunk 的全部 HV 串行；
- Cube/MMAD 高，按 `dKbg+dVb+KKT`、`dA1/2/5/6`、`dKb+DK` 三段定位；
- Vector/Exp 高，对应 Kbg/Vb、三角 gate D 或最终 dBeta/dG 归约；
- MTE/Fixpipe 或 cross-core wait 高，对应低精度 GM workspace 的阶段交接；
- GVA ratio 增大但 KKT 次数未下降，检查 head window 是否跨越 key-head 复用边界。

按单轴顺序验证候选：

1. chunk 数不足时，评估把独立 head window 纳入任务；必须保持同一 `hk` 的 dK 累加单写者，且 workspace 与 flag 也按新所有权重排。
2. GVA 主导时扩大 K/KKT 驻留跨 window 的 live range；代价是 L1 和 KKT slot 占用，`hk` 切换或 L1 event 不闭环即回退。
3. workspace MTE 主导时，只融合一段 producer-consumer（优先 dA mask 或最终 merge）并观察 GM bytes 与 cross-core wait；UB/L1 超限、阶段乱序或低精度边界变化即回退。
4. `V=256` Cube 主导时只调整 dV 的 K/N 分块和 L0C buffer 数；不得把 V128 的最优 tile 外推。
5. tail 比例高时评估减少 BT×BT 无效计算；必须保留 strict-causal mask、`L=1` 补齐和 varlen chunk 寻址。
6. `dBeta/dG` 归约主导时调整 AIV 行所有权或归约树；必须保持两个 sub-block 不重叠写输出，并逐项覆盖 FP32 gate。

# 约束

- `K=128`、`V∈{128,256}`、`BT∈{64,128}`、`HV>0`、`HK>0` 且 `HV % HK == 0`。[^tiling]
- varlen 必须同时提供一维 `cu_seqlens/chunk_indices`，`B=1`，chunk index 长度为正偶数且 host 可读取常量值。[^tiling]
- `beta` 与 `g` dtype 必须相同；源码 aclnn 层只检查非空和连续化，完整 shape/dtype 门禁位于 tiling。[^api][^tiling]
- dK 对 grouped value heads 的累加顺序、三角方向、gate 指数方向和四阶段 cross-core credit 不可改变。[^vector][^cube]

# 失败表现

- 只有 GVA 错：`hv→hk` 映射、KKT cache 生命周期或 dK 分组累加所有权错误。
- 只有 tail/varlen 错：真实 chunk 长度、metadata 顺序、`L=1` 补齐或三角 mask 错误。
- dV 正确而 dK/dBeta/dG 错：D/dKb/DK 阶段或 KKT 归约链错误。
- 偶发旧矩阵、全零或挂起：四 slot、L1/L0 ping-pong、Fixpipe event 或 cross-core credit 未闭环。
- FP32 gate 单独错误：tiling key/gate load-cast 路径与 K dtype 错配。

# 验证方法

以独立 `prepare_wy_repr_bwd_da + prepare_wy_repr_bwd_full` CPU 链为 reference，同时比较
`dK/dV/dBeta/dG`；覆盖 FP16/BF16 K dtype、FP16/BF16/FP32 gate、BT64/128、V128/256、
HK=HV 与多种 GVA ratio、fixed/varlen、完整 chunk、`L=1` 和一般 tail。测试必须检查所有四个
输出的有限性与误差，不能只用全零 smoke 证明流水正确。[^reference]

性能候选在空闲 NPU 上用 profiler 分别记录三段 Cube、四段 Vector、AIC/AIV wait、MTE/Fixpipe、
workspace bytes 和活跃核数；修改后需重跑上述完整正确性矩阵。静态源码中的四槽、resident、fast
命名不构成已提速证据。

[^host]: 固定提交中的输入输出 dtype 组合、可选 metadata、属性与支持平台。
[^api]: 固定提交中的 aclnn 非空检查、连续化、输出和 launcher 行为。
[^tiling]: 固定提交中的 shape/dtype 门禁、fixed/varlen chunk、模板 key、向量行、block dim 与 workspace 偏移。
[^entry]: 固定提交中的 mixed AIC:AIV=1:2 入口和 24 个模板组合。
[^common]: 固定提交中的 fixed/varlen chunk 地址、GVA 映射、四槽和 KKT workspace 地址。
[^cube]: 固定提交中的九组 MMAD、K/KKT 复用、L1/L0 buffer、Fixpipe 和 AIC→AIV credit。
[^vector]: 固定提交中的 beta/g、三角 mask、dA/D、四输出合并、AIV 行所有权和 AIV→AIC credit。
[^reference]: 固定提交中的融合核与独立 dA/full CPU 链、dtype/BT/V/GVA/tail/varlen 用例。
