---
type: CATLASS DSL Operator Example
title: Chunk Gated Delta Rule Bwd Dhu
description: 门控 Delta Rule 的反向状态链 mixed kernel，覆盖四 head window、八槽 workspace、A5 直达 UB 和当前接口漂移。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, backward, recurrent, mixed, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
  - {by: process:catlass-dsl-source-audit, at: '2026-08-13T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/README.md
    title: 公开算法说明、shape 和调用示例
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_host/op_api/aclnn_chunk_gated_delta_rule_bwd_dhu.cpp
    title: aclnn 可选输入、连续化与输出行为
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_host/op_tiling/chunk_gated_delta_rule_bwd_dhu_tiling_processor.h
    title: shape 门禁、任务、workspace 与模板 key
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_kernel/chunk_gated_delta_rule_bwd_dhu.cpp
    title: A2/A3/A5 mixed 入口和当前未消费参数
  - id: common
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_kernel/chunk_gated_delta_rule_bwd_dhu_common.h
    title: sequence/chunk 寻址、canonical metadata 和输出 chunk 映射
  - id: cube
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_kernel/arch35/chunk_gated_delta_rule_bwd_dhu_cube.h
    title: A5 三组 MMAD、L0C 直达 UB 和 AIC/AIV 交接
  - id: vector
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/op_kernel/arch35/chunk_gated_delta_rule_bwd_dhu_vector.h
    title: A5 gate、FP32 状态、dv2/dh0 和同步流水
  - id: reference
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_gated_delta_rule_bwd_dhu/test/test_chunk_gated_delta_rule_bwd_dhu.py
    title: scalar-g fixed/varlen CPU reference 与用例
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

实现为每条 sequence、每个 value head 维护 FP32 状态梯度 `R∈R^(K×V)`，并严格按 chunk
逆序扫描。scalar-g 路径在写 `dh` 后执行：

```text
dh[chunk] = R
dv2_t     = dv_t + (K_t @ R) * exp(g_last - g_t)
R         = R * exp(g_last)
            + scale * (Q_t * exp(g_t))^T @ dO_t
            - W_t^T @ dv2_t
```

`hq=floor(hv/(HV/HK))` 决定 Q/K head，W/dO/dv/dh/dv2 保持 value-head 所有权。
gk 路径只从 chunk 最后 token 读取 K 维 gate，把 base-2 `gk_last` 乘 `ln(2)` 后指数化，
逐 K 行衰减 `R`；Q 和 `K@R` 不走 scalar token gate。[^vector][^reference]

当前实现与 README 存在可执行语义漂移：接口仍接收 `h0/dht`，README 声称 `dht` 初始化
反向状态，但 kernel 入口显式忽略两者并把每个任务的 `R` 清零；`dh0` 仅在 `h0` 非空时申请，
最终写入第一 chunk 对应位置。因此在该固定提交上，非零 `dht`/`h0` 不能按 README 语义使用，
必须先以实现和可执行测试为准修复或补证。[^guide][^api][^entry][^vector]

# 用法

## 分核策略与基本块切分

任务粒度是 `(sequence, head-window)`，每个 window 最多四个连续 value heads：
`taskNum = seqNum * ceil(HV/4)`。host 发射全部物理 AIC；每个 AIC 以 `taskIdx += blockNum`
取任务，两个 AIV sub-block 按 `headOffset % 2` 分担 window 内的 heads。同一任务内部必须从
最后 chunk 扫到第一个 chunk，不能按 chunk 横向扩核。[^tiling][^common][^cube][^vector]

fixed 模式 `seqNum=B`、每条链有 `ceil(T/BT)` 个 chunk；varlen 模式 `seqNum=len(cu_seqlens)-1`
且 `B=1`。kernel 先按 canonical 顺序尝试 `outputChunkBase+localChunkIdx`，metadata 不匹配时
线性搜索 `chunk_indices` 找到 dh 的输出 chunk 位置。乱序 metadata 虽可能得到正确地址，但会
额外引入每 chunk 的 O(totalChunkNum) 搜索。[^tiling][^common]

每个物理 AIC 有八个 workspace slot：偶数 task round 使用 0..3，奇数 round 使用 4..7；
每个 head 独占一个 slot。AIC 侧可在相邻 value heads 映射同一 `hq` 时复用 K resident，
但状态 `R`、dv2 和两个 K×V 更新项保持每 head 独立。[^cube][^vector]

模板 key 编码 Q dtype、gate dtype、`V=128/256` 和 `USE_GK=0/1`；BT 不在 key 中，
但 tiling 仍只接受 64/128。当前实现将 K 固定为 128，而不是旧文档中的 `K≤128`。[^tiling][^entry]

# 代码模式

## 数据路径与存储层级

```text
FP32 R in per-head GM workspace
  -> AIV: store dh; apply scalar-g or gk-last decay; gate Q
  -> AIC: K @ R             -> dvState
  -> AIV: gate/add dv       -> dv2
  -> AIC: Qg^T @ dO         -> termQ
          W^T @ dv2         -> termW
  -> AIV: R += scale*termQ - termW
```

每个 slot 依次分配 `qg(BT*K)`、对齐后的 `state(K*V*FP32)`、`dvState(BT*V)`、
`termQ(K*V)` 和 `termW(K*V)`；`dv2WorkspaceElems` 当前为 0，因为 dv2 直接作为第二个 GEMM
输入。user workspace 是 `AIC核数 * 8 * workspaceElemsPerSubBlock * sizeof(Q dtype)`，
即便实际任务少于核数也按全部物理核预留。[^tiling]

AIC 从 GM 经 L1/L0 执行三组矩阵乘，L0C 用 FP32；A2/A3 经 Fixpipe 写低精度 GM workspace。
A5 对 `K@R` 和 `W^T@dv2` 在满足 tile 条件时可把 L0C 直接拆给两个 AIV 的 UB，减少中间 GM
落地；`Q^T@dO` 仍落 `termQ` workspace。K/W resident、L1 scratch、L0A/L0B/L0C 都用
事件和 ping-pong slot 管理。[^cube]

## 流水排布、同步关系与数值精度

每个 chunk 有三次 producer-consumer 交接：AIV 完成 gate/Q 与 state publish 后通知 AIC；
AIC 完成 `K@R` 后通知 AIV 生成 dv2；AIV 再通知 AIC 消费 dv2，最后 AIC 通知 AIV合并
`termQ/termW`。ready/free credit 必须按每 head 收发，非本 AIV sub-block 所有的 head 也要发
占位 credit，否则另一侧会挂起。[^cube][^vector]

A5 直达 UB 另用 mode-4 flag 区分两个 AIV sub-block：同一逻辑 flag 的第二个 sub-block 偏移
16；AIC 等待两份 free 后才覆写 L0C→UB 目标，并分别发布 ready。`K@R` 和 `W^T@dv2`
各自还有成对的 AIV→AIC 回收 flag，结束时必须 drain 所有 credit。[^cube]

状态 R、Cube 累加、gate 指数、scale 合并均为 FP32；Q/K/W/dO/dv/dh/dv2 在边界转为
FP16/BF16。scalar g 直接用自然指数，gk 被视为 base-2 累计值并乘 `ln2` 后调用 `exp`；
两种路径不能混用缩放。[^vector]

# 优化决策

先记录 `seqNum`、`ceil(HV/4)`、每序列 chunk 数、Q/g dtype、V、BT、scalar-g/gk、
fixed/varlen 和是否请求 dh0。profile 映射如下：

- 活跃核少：`seqNum * ceil(HV/4)` 小于 AIC 核数；增加 chunk 不会增加独立任务；
- Cube 长：三组 GEMM，V256 尤其放大 K×V state 与 term；
- Vector/Exp 长：scalar-g 每 chunk token 指数，或 gk 的 K 维指数与 FP32 state row update；
- MTE/Fixpipe 长：state/qg/dvState/termQ/termW 的 GM workspace 往返；
- AIC/AIV 周期空洞：三次 cross-core credit、A5 mode-4 sub-block flag 或 resident event；
- varlen 短序列异常慢：kernel 线性扫描 cu_seqlens 和非 canonical chunk_indices fallback。

按单轴顺序验证候选：

1. 低并发时先调整 head-window/任务映射；必须让完整 reverse state chain 留在一个任务，并同步修改八槽所有权和两 AIV 分工。
2. GVA 场景扩大 K resident 跨四 head window 的复用；代价是 L1 live range，`hq` 切换或尾 window 即回退。
3. workspace 主导时把一个 A2/A3 中间量改为 L0C→UB 直达；必须复制 A5 的双 sub-block free/ready 协议，任何丢 credit、UB 冲突或尾 tile 错即回退。
4. A5 直达路径 wait 高时只调整一种 credit/slot 距离；不得同时改变 GEMM tile，以便用 ready/free stall 验证。
5. scalar-g Exp 主导时缓存 `exp(g)` 和 `exp(g_last-g_t)` 的共同量；代价是 UB resident，gk 结果不得据此外推。
6. varlen metadata 主导时要求上游提供 canonical `(seq,chunk)` 顺序，或构造 host-side 映射；必须保持 dh output chunk 语义。
7. dh0 清零占比高时只优化实际输出范围；当前输出布局是 `[B,HV,totalChunkNum,K,V]` 且仅 chunk0 被最终状态覆盖，不能按 README 的四维描述缩小写入。

# 约束

- Q/K 完全同形 `[B,HK,T,128]`；W 为 `[B,HV,T,128]`；dO/dv 为 `[B,HV,T,V]`，`V=128/256`，`HV % HK == 0`。[^tiling]
- 必须且只能提供 scalar `g[B,HV,T]` 或 `gk[B,HV,T,K]` 之一；gate dtype 为 FP32 或与 Q 相同。[^api][^tiling]
- varlen 的 cu_seqlens/chunk_indices 必须成对出现、均为一维，`B=1`，chunk_indices 长度为偶数。[^tiling]
- BT 仅 64/128；同一 sequence/head 的 reverse chunk 顺序、dh 写更新前 R、两 AIV head 所有权和 credit 次数不可改变。[^common][^cube][^vector]
- `h0/dht` 在当前 kernel 未消费；不得仅依据 README 声称它们已参与计算。[^guide][^entry]

# 失败表现

- `dht` 改变但输出不变：当前 kernel 入口忽略 dht，不是数值容差问题。
- 仅 gk 错：base-2/自然指数混用、只读 last-token 的 K 向量或 USE_GK key 错误。
- 仅 tail/varlen 错：chunk output 映射、真实长度或 canonical metadata 错误。
- dh 只有末块正确：reverse state workspace 未在 chunk 间持久化，或更新前/后写 dh 的时机颠倒。
- 偶发挂起/旧 tile：未归还另一个 AIV sub-block 的 mode-4 flag、head 占位 credit 或 L1/L0 event。
- dh0 大片非零或越界：清零范围、五维 chunk 布局和只写 chunk0 的所有权不一致。

# 验证方法

scalar-g 路径按上述逐 chunk FP32 reference 比较 dh/dv2，覆盖 FP16/BF16、FP32/同 dtype gate、
V128/256、BT64/128、GVA、fixed/varlen、一般 tail 和单 token tail。另为 gk 构造 base-2 reference，
覆盖每 K 行不同 gate；验证 `g`/`gk` 恰一存在的负例。[^reference][^tiling]

当前源码必须额外加入契约测试：以非零 dht/h0 证明或否定其影响、核对 dh0 五维 shape 和只有每条
sequence 第一 chunk 位置有效的语义、检查非 canonical chunk_indices 的 fallback。修复接口漂移前，
这些组合应标为已知不一致，不能由 README 推断为通过。[^guide][^entry][^vector]

性能实验在空闲 NPU 上分别报告 task 数、活跃核、三组 Cube 时间、AIV Exp/state 时间、GM workspace
bytes、mode-4/cross-core wait 和 fixed/varlen metadata 成本。未完成同配置正确性与 profile 的候选只保留
为静态假设。

[^guide]: 固定提交 README 中的公开算法、shape、可选参数和示例；与实现冲突处已显式记录漂移。
[^api]: 固定提交中的 g/gk 互斥检查、可选输入连续化、dh0 可选输出和 launcher 行为。
[^tiling]: 固定提交中的完整 shape/dtype 门禁、task/window、fixed/varlen、workspace、vecRow 和模板 key。
[^entry]: 固定提交中的 A2/A3/A5 mixed 入口、g/gk 选择以及 h0/dht 未消费事实。
[^common]: 固定提交中的 sequence/chunk 解析、canonical 快路径和线性 fallback 地址映射。
[^cube]: 固定提交 A5 的 K resident、三组 GEMM、L0C→UB、双 sub-block flag 和 AIC/AIV credit。
[^vector]: 固定提交 A5 的 FP32 状态、scalar-g/gk 分支、dh/dh0/dv2 写回和 event 生命周期。
[^reference]: 固定提交中的 scalar-g fixed/varlen CPU reference、GVA 和受控数值用例。
