---
type: CATLASS DSL Operator Example
title: Chunk Kda Fwd
description: KDA 分块前向的五阶段 mixed kernel，包含 A5 融合选择、状态链、workspace 生命周期与分阶段优化方法。
tags: [catlass-dsl, operator, linear-attention, kda, chunk, mixed, state-space, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
  - {by: process:catlass-dsl-source-audit, at: '2026-08-13T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/kda/chunk_kda_fwd/README.md
    title: 顶层算法、接口、输出保留与支持范围
  - id: design
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/kda/chunk_kda_fwd/docs/design.md
    title: 阶段职责、状态布局、tiling key 与性能设计
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/kda/chunk_kda_fwd/op_host/op_api/aclnn_chunk_kda_fwd.h
    title: aclnn 接口原型
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/kda/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp
    title: block、workspace、可选输出与阶段参数
  - id: fast-select
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/kda/chunk_kda_fwd/op_host/arch35/chunk_kda_fwd_tiling_impl.h
    title: Ascend 950 快路径和融合门槛
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp
    title: mixed 入口与两个 tiling key
  - id: orchestration
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/chunk_kda_fwd_common.h
    title: Gate、Prepare、Post-WU、FwdH、Finalize 编排
  - id: fwdh
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_fwd_h.h
    title: A5 dense 状态传播、双 AIV 直达 UB credit 和片上流水
  - id: prepare
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_prepare.h
    title: A5 Prepare mixed 流水与 finalKg 精度边界
  - id: post
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_post_wu.h
    title: A5 Post-WU mixed 流水
  - id: finalize
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/1b2ed3e13a446337d69ab5efbaf64af216adbf06/fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_finalize.h
    title: A5 Finalize 输出流水
operator_families: [linear-attention, kda]
---

# 接口与概念

## 算子算法

该物理 L0 在一个 mixed AIC:AIV=1:2 kernel 中完成五个逻辑阶段，而不是依次启动五个公开算子：

```text
Gate:     gk = cumsum(gate) / ln(2)                         # chunk-local FP32
Prepare:  q,k,v,gk,beta -> Aqk,Akk,qg,qg_scaled,W0,U0
Post-WU:  k,gk,W0,Akk,U0 -> w,u,kg,VNew0
FwdH:     v_new = u - w @ h_prev
          h_next = exp2(gk_last) * h_prev + kg^T @ v_new
Finalize: o = qg_scaled @ h + Aqk @ v_new
```

Prepare 中的 `Aqk/Akk` 含因果三角结构和块内修正，Post-WU 形成递推需要的 W/U/KG；FwdH 严格沿每条 sequence 的 chunk 顺序传播状态，Finalize 再恢复 token 级并行。`state_v_first` 只改变公开 state 的末两维，内部统一按 `[K,V]` 计算。[^guide][^design][^orchestration]

公开 layout 支持 BSND/BNSD/TND/NTD；Q/K/V 为 FP16/BF16，gate/beta 为 FP32/BF16，K/V 是 16 的倍数且位于 16..256，chunk 为 64/128。varlen 使用 `cu_seqlens + chunk_indices` 的 canonical chunk 顺序，最多 1024 个逻辑序列。[^guide][^api][^tiling]

# 用法

## 分核策略与基本块切分

Gate dense 路径按 `(B,HV,chunk)` 分给 AIV，varlen 按 `(sequence,HV)` 顺序遍历真实 chunk。Prepare/Post-WU/Finalize 主要以 `(chunk,value-head)` 为任务，AIV 的两个 sub-block 再按行或列分工；FwdH 以 `(sequence,value-head)` 为不可拆状态链，同一链的 chunks 串行。host 用 AIC block 数设置 `prepareUsedCoreNum`，gate 可使用其两倍 AIV 数。[^tiling][^orchestration][^prepare][^post][^fwdh][^finalize]

两个 tiling key 只表示 shape family：key 2 固定 `BT=64,K=128,V=128`，key 1 覆盖其他支持 shape；它们不是 SoC 编号。Ascend 950 在 key 2 内再选择 fast sub-pipeline，其他架构或不满足条件时运行通用实现。[^design][^entry]

Ascend 950 的关键选择条件如下，优化和 benchmark 必须记录实际命中组合：[^fast-select]

| 开关 | 必要条件 | 被阻断的常见原因 |
| --- | --- | --- |
| `computeGateInPrepare` | A5、BT64/K128/V128、Q=BF16、raw g=FP32、存在 A_log、kernel 内 safe gate | 预激活 gate、BF16 gate、FP16 Q、tail/varlen 本身不是该开关条件 |
| `useDenseFwdH` | A5 shape fast family、dense 且 T 可整除 BT、Q=BF16 | varlen、tail、FP16 Q |
| `fusePostWu` | dense aligned、Q=BF16、safe gate、HV 为偶数 | odd HV、tail/varlen、unsafe gate |
| `fusePostWuIntoFwdH` | 上述条件 + `computeGateInPrepare` + 不导出 qg/v_new/h | autograd/调试要求保存任一中间量 |

## 阶段成本与 profiler 归因

- Gate 是 AIV scan；raw-safe 时非线性重，预激活时更接近搬运加累计。
- Prepare/Post-WU/Finalize 同时含 Cube GEMM 与 AIV mask、gate、solve/epilogue，某一侧变快后另一侧可能成为同步尾巴。
- FwdH 的每条 state chain 串行，batch/head 决定并行度；state KxV 搬运、`w@h`、`kg^T@v_new` 和 exp2 epilogue共同决定时延。
- 可选输出会改变融合选择和 workspace 落地，不只是多一次最终 copy；端到端对比必须保持 `disable_recompute/return_intermediate_states/output_final_state` 一致。[^design][^fast-select][^orchestration][^fwdh]

若 profile 表现为 AIC 时间长且利用率低，检查 K/V tile 与结构零三角 MMAD；AIV 长尾则检查 mask、exp2、solve 和 layout writeback；AIC/AIV 同时有周期性空洞则检查 cross-core flag 与阶段 `SyncAll`；短序列时固定阶段同步占比会高，不能直接外推长序列结论。

## 优化候选与优先级

1. 先固定 shape、dtype、layout、gate 模式和输出保留，记录 tiling key 及四个 fast 开关。先比较阶段路径，避免把“开启融合”误认为某个局部 tile 优化。
2. 若未命中 key 2，先判断业务 shape 能否合法落到 BT64/K128/V128；不能为了命中快路径改变模型 K/V 或 chunk 语义。
3. Gate 主导时使用 gate 条目的 scan 判据；`computeGateInPrepare` 已消除独立 Gate 阶段，候选要检查 Prepare 的 UB/Vector 是否因此变成瓶颈。
4. Prepare/Post-WU 主导时，优先保持 K/K^T 在 L1 复用、AIC L1/L0 ping-pong 和 AIV staging；对 causal/triangular 结构只可裁剪恒零 tile，并同时平衡两个 AIV sub-block。导出 `finalKg` 时必须保留“先完成 fused direct-V 读取，再单独生成 finalKg”的顺序，不能让二者复用同一 UB 输入产生读写竞争。[^prepare]
5. FwdH 主导且链数少时，增加 chunk 并行会破坏状态依赖；应优先减少 state GM 往返、缩短 AIC/AIV producer-consumer 距离，或增加独立 batch/head 链。调整 L0C→UB 直达流水时，必须让 AIC 等待两个 AIV sub-block 都归还 free credit，并保持 state 与 VNew 使用不同的 L1 credit。[^fwdh]
6. Finalize 主导时联合调整 output M/N tile 与 AIV epilogue UB，占用更大的 Cube tile若让 mask/layout 转换串行化，应回退。
7. 关闭中间量导出可触发 `fusePostWuIntoFwdH` 并缩短 workspace 生命周期，但只适用于允许 backward 重计算的调用；不能用改变返回语义的结果宣称同配置提速。
8. tail/varlen 使用 generic backend 时，优化必须单独覆盖 canonical chunk metadata、真实 token 数和 pipe destroy/reset 路径；dense aligned 的收益不能外推。

# 代码模式

## 数据路径与存储层级

```text
Q/K/V/g/beta GM
  -> Gate AIV -> gk FP32 GM/workspace
  -> Prepare AIC(L1/L0/MMAD/Fixpipe) + AIV(UB mask/gate/solve)
       -> Aqk,Akk,qg_scaled,W0,U0
  -> Post-WU mixed -> w,u,kg
  -> FwdH mixed, state chain -> v_new,h,final_state
  -> Finalize mixed -> BSND/TND output GM
```

非公开的 gk/w/u/qg/kg/v_new/h 由 `ResolveAddresses` 映射到 user workspace；公开输出存在时可直接作为对应内部存储。`qgScaled` 和 output scratch 始终有专用偏移。dense A5 FwdH 在 L1 使用四 slot 流水交接 W/Q/KG/Aqk/Akk/U，并在不导出 VNew/H 时复用较小 storage。AIC 的部分 L0C 结果还会直接拆到两个 AIV sub-block 共享的 UB，绕过 GM 中间落地；该 direct UB 的读写所有权由每个 sub-block 独立 credit 管理。[^tiling][^orchestration][^fwdh]

workspace 优化必须基于 live range：Gate 后才能复用其临时区，Prepare 产物至少活到 Post-WU，W/U/KG 活到 FwdH，QG/Aqk/VNew/H 活到 Finalize。不能简单把各阶段峰值相加，也不能让 compact output 覆盖仍被下一阶段读取的 raw FP32 区域。[^tiling][^orchestration][^finalize]

## 流水排布、同步关系与数值精度

非融合路径的全局顺序是 Gate 后 `SyncAll`，Prepare 后 `SyncAll + pipe.Reset`，Post-WU 后再次同步/reset，FwdH 后同步再 Finalize。融合开关删除的是特定阶段边界，不代表所有同步均可删除。dense FwdH 使用 cross-core ready/free flag 管理 L1 slot；AIC 用 MTE1/MMAD/FIX，AIV 用 MTE2/Vector/MTE3。最新 A5 路径对 direct UB 使用 mode-4：AIC 在覆写前分别等待两个 AIV 的 free，发布 ready 时也分别通知，第二个 sub-block 使用 flag 偏移 16；state L1 与 VNew L1 分别使用 flag 9 和 10，避免一种消费者提前释放另一种数据。[^orchestration][^fwdh]

gate/cumsum、solve、state 累加和 Cube C 使用 FP32；公开 Q/K/V/O 和多数中间量按输入低精度存储，在明确边界转换。`gk` 已是 base-2 表示，FwdH/Finalize 的 `exp2` 不能改成 `exp` 或再次乘 `ln2`。A5 fused Prepare 在需要导出 finalKg 时先让 direct-V 融合调用完成，再以 typed helper 单独生成 finalKg；这既避免覆盖仍在读取的 V UB，也保留 BF16 score-safe 指数范围。[^guide][^design][^prepare][^post][^fwdh]

# 约束

- state chain 的 chunk 顺序、canonical varlen `chunk_indices` 和尾块有效长度不可重排。[^guide][^design]
- key 2 与 A5 fast 开关是精确条件集合；只满足其中一部分时必须走对应回退路径。[^entry][^fast-select]
- `Aqk/Akk` 始终公开；其他中间量是否公开会改变 storage 和融合选择。[^guide][^tiling]
- 内部 h 为 head-major，公开 h 为 sequence-major；state 末两维由 `state_v_first` 解释。[^guide][^design]
- mixed kernel 的两个 AIV sub-block 不得重叠写输出或提前释放 AIC 尚在使用的 slot。[^prepare][^post][^fwdh][^finalize]

# 失败表现

- 仅首 chunk 正确：FwdH 未读取上一 chunk state，或阶段全核同步被误删。
- 只有 tail/varlen 错：真实长度、chunk metadata、generic backend 或 pipe 生命周期错误。
- 导出中间量后才错：workspace/public output alias 与 store 标志不一致。
- Q=FP16 或 odd HV 才退化：fast 融合门槛未命中，不一定是局部 kernel 回归。
- gate 衰减系统性错误：自然对数/base-2 转换遗漏或重复。
- 偶发 NaN/旧 tile：cross-core ready/free flag、MTE/FIX event 或四 slot 回收顺序错误；只在第二个 AIV sub-block 出错时首查 mode-4 的 `+16` flag 配对。
- 仅导出 finalKg 时 BF16 错或 VNew 被污染：fused direct-V 尚未完成便复用了 V UB，或 finalKg 未走独立 typed helper。

# 验证方法

正确性覆盖两个 tiling key、FP16/BF16、四种 layout、raw-safe/raw-unsafe/预激活 gate、BT64/128、K/V 边界、dense aligned/tail/varlen、odd/even HV、GQA、两种 state 顺序和全部可选输出组合。逐项比较 O、final state、gk、Aqk/Akk、W/U/QG/KG/VNew/H，并确认不公开中间量时内部 Finalize 结果仍一致。A5 key-2 dense 快路径还要分别覆盖两个 AIV sub-block，并对 `exportFinalKg` 开/关检查 finalKg、VNew 和最终 O，才能捕获 direct UB credit 与 UB alias 回归。[^guide][^design][^entry][^orchestration][^prepare][^fwdh]

性能实验按阶段假设分组，每次只改变一个 fast 开关或一个 tile/流水候选；在空闲 NPU 上用 profiler 记录总 `aicore_time`、AIC/AIV 时间、Cube 利用率、MTE stall、同步尾部、workspace 字节及实际输出保留组合。dense/tail/varlen 分别报告；未经过完整正确性和设备 profile 的候选只能标为静态推断。

[^guide]: 固定提交中的顶层公式、layout、dtype、12 返回值、输出保留和支持范围。
[^design]: 固定提交中的五阶段职责、状态布局、tiling key、重计算策略和性能设计。
[^api]: 固定提交中的 aclnn 参数顺序、必选/可选输入、属性与输出原型。
[^tiling]: 固定提交中的 chunk 计数、block dim、workspace 偏移、storage flag 和阶段 tiling data。
[^fast-select]: 固定提交中的 A5 shape、dtype、dense、gate、head parity 和可选输出融合门槛。
[^entry]: 固定提交中的 mixed AIC:AIV=1:2 入口和 key 1/key 2 分派。
[^orchestration]: 固定提交中的地址解析、阶段调用、SyncAll、pipe reset/destroy 和 generic backend。
[^fwdh]: 固定提交中的 dense state chain、L1 四 slot、mode-4 双 sub-block direct UB credit、state/VNew 独立 flag 与输出复用。
[^prepare]: 固定提交中的 Prepare 基本块、三角结构、Cube/Vector 分工、direct-V/finalKg 顺序和 BF16 精度边界。
[^post]: 固定提交中的 Post-WU 任务映射、A5 pipeline、KG/W/U 数据路径和尾块实现。
[^finalize]: 固定提交中的 output GEMM、AIV epilogue、scratch 布局和尾块写回。
