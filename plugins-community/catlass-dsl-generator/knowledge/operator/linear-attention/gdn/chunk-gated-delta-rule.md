---
type: CATLASS DSL Operator Example
title: Chunk Gated Delta Rule
description: Ascend 950 Chunk Gated Delta Rule 的三阶段分块算法、workspace 数据路径、Cube/Vector 协同和精度边界。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, chunk, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T12:10:09Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T12:10:09Z'}
sources:
  - id: guide
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/chunk_gated_delta_rule/README.md
    title: Chunk Gated Delta Rule interface and algorithm guide
  - id: entry
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/chunk_gated_delta_rule/op_kernel/chunk_gated_delta_rule_apt.cpp
    title: kernel entry and tiling-key dispatch
  - id: kernel
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule.h
    title: three-stage orchestration and workspace layout
  - id: stage1
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_stage1.h
    title: intra-chunk transformation stage
  - id: stage2
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_stage2.h
    title: state propagation stage
  - id: stage3
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/chunk_gated_delta_rule/op_kernel/arch35/chunk_gated_delta_rule_stage3.h
    title: intra-chunk output stage
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

Chunk Gated Delta Rule 将 TND 变长序列按 `chunkSize` 切块，并将每个 batch 再按
`maxGroupLength = p * chunkSize` 分组。它等价实现逐 token 状态递推
`S_t = alpha_t S_(t-1) + beta_t (v_t - alpha_t S_(t-1) k_t) k_t^T` 与
`o_t = S_t (q_t * scale)`，但把块内依赖改写为三阶段矩阵运算，以增加 prefill 的并行度。
输入 Q/K、V、beta 为 BF16，g 为 FP32；状态支持 BF16 或 FP32。[^guide][^entry]

# 用法

## 分核策略与基本块切分

外层严格按 batch 和 chunk group 顺序执行，组内依次运行 Stage1、Stage2、Stage3，每阶段后
`SyncAll<false>()`，因此跨组状态不能重排。Stage1 处理块内变换；Stage2 按 `Nv × chunk`
传播入组状态并写 final state；Stage3 按 `Nv × chunkNum` 均分到核，尾块用实际长度。
AIC:AIV 任务比为 1:2。四个 tiling key 覆盖 BF16/FP32 state 与有/无 gamma。[^kernel][^entry]

接口布局为：Q/K `(T,Nk,Dk)`，V/O `(T,Nv,Dv)`，beta/g `(T,Nv)`，状态
`(B,Nv,Dv,Dk)`，actual sequence lengths `(B,)`；`Nv % Nk == 0`，Dk/Dv 不超过
128。移植时必须保留 TND batch 前缀和、GQA 头映射和 chunk 尾块语义。[^guide]

# 代码模式

## 数据路径与存储层级

```text
Q/K/V/beta/g GM
  -> Stage1 UB + Cube -> workspace gCum, kCumDecay, vInner, qPrime, kg, qkt
initial/final state GM + Stage1 workspace
  -> Stage2 Cube/Vector -> attention state term + updated final state
qkt + gCum + vInner
  -> Stage3 Vector causal/gamma mask -> temporary GM -> Cube -> output GM
```

workspace 连续放置各中间张量；FP32-state 路径额外保存 BF16 `vInner` 和 BF16 当前状态，
供只接受低精度输入的 Cube 路径消费。Stage1 生成下三角块内系统并求逆，形成修正后的
`qPrime/kCumDecay/vInner`；Stage2 计算 `qPrime @ state^T`、`kCumDecay @ state^T`
和 `vInner^T @ kg`；Stage3 计算带因果与 gamma 衰减的 `qkt @ vInner`。[^kernel][^stage1][^stage2][^stage3]

## 流水排布、同步关系与数值精度

Stage1/2/3 都以 GM workspace 作为 AIV 与 AIC 的交接面。Stage1 使用多组 mode-2
cross-core flag 串接 Q/K 预处理、KK/QK MMAD、块内逆、GBK、VBeta 和 QPrime；Stage2
用 flag 交接 gamma、状态 cast、两次 state matmul 与状态更新；Stage3 由 AIV 先生成
masked QKT，再通知 AIC MMAD，AIC 完成后反向释放 workspace。普通 hard event 保护
MTE2/V/MTE3/FIX 的 buffer 所有权。[^stage1][^stage2][^stage3]

向量非线性、衰减累计、状态累加和 MMAD C 使用 FP32；Q/K/V、块间 Cube 输入和 O 使用
BF16。FP32 state 在 GM 保持 FP32，但进入 BF16 Cube 前显式生成低精度镜像。[^kernel][^stage2]

# 优化决策

先记录 `chunkSize`、`maxGroupLength/chunkSize`、序列长度、Nv/Nk、Dk/Dv、state dtype 与 gamma，
由此确定四个 tiling key和每组 chunk 数。外层 batch/group 严格串行，Stage1/2/3 之间均有
`SyncAll`；Stage3 才按 `Nv * chunkNum` 均分，故单看总 token 数会高估并行度。[^entry][^kernel]

成本分别为 Stage1 的块内 KKT/QKT、逆与 WY 变换，Stage2 的状态矩阵乘和跨组传播，
Stage3 的 gate/mask 与输出 MMAD，以及各阶段 workspace 写读和全核同步。profiler 中
Stage1 Vector 长对应逆/Exp，Stage2 长对应状态 K×V，Stage3 Cube 长对应 chunk²×Dv；
`SyncAll` 或尾核长对应阶段负载不均，短 group 则固定 mask/workspace/同步占比上升。

按单轴顺序验证：

1. 先调 group 长度与 chunkSize，比较三阶段最长核、尾块和同步时间；必须保持组间 state 顺序。
2. Stage1 主导时分别优化 K/Q tile 复用或块内逆，不能同时改两者；FP32 中间和下三角语义不变。
3. Stage2 主导时复用当前 state、减少 BF16 镜像往返；代价是 L1/UB 占用，FP32-state 路径必须独立门禁。
4. Stage3 主导时调整 `Nv*chunkNum` 分配或融合 mask/gamma 后处理；尾块有效长度必须保持。
5. 同步/GM 主导时缩短阶段 workspace 生命周期或流水化组内阶段；出现 flag 等待、状态串组或内存增长即回退。[^stage1][^stage2][^stage3]

# 约束

- 仅提取 Ascend 950 实现；不可混用 `arch22` 的入口或同步协议。
- TND 变长序列按每 batch 长度而非累计长度解释；batch/组/阶段次序具有状态依赖。
- `0 < Nk,Nv <= 64`、`Nv % Nk == 0`、`0 < Dk,Dv <= 128`。
- Q/K 建议位于 `[-1,1]`，g 位于 `[-1,0]`，beta 位于 `(0,1)`，否则存在溢出风险。
- mask、workspace 偏移、FP32-state 附加区必须与 host tiling 完全一致。

# 失败表现

- 仅首组正确：后续 group 没有读取上一组 final state，或阶段间全核同步缺失。
- chunk 对角附近错误：下三角 mask、块内逆或尾块有效长度错误。
- GQA 头串扰：`Nv/Nk` 映射或 workspace 的 Nv stride 错误。
- FP32 state 路径错误而 BF16 正常：状态 Cube 镜像或专用 FP32-C tiling 未保留。
- 偶发旧数据：cross-core flag 编号、方向或 MTE/FIX 释放顺序不匹配。

# 验证方法

先检查固定提交的 APT 入口只包含 Ascend 950 实现 header，并对四个 tiling key 分别生成内核；再用
逐 token FP32 reference 覆盖有/无 g、BF16/FP32 state、batch>1、GQA、chunk 尾块和多 group。
检查 final state 及每个 token 输出，而不只比较末 token。性能结论须在空闲 Ascend 950 NPU
上另行 benchmark/profile，本 concept 只记录源码结构。[^entry][^guide]

[^guide]: 固定提交中的算法、接口、shape、dtype 和数值范围说明。
[^entry]: 固定提交中的 Ascend 950 入口、任务类型和四类 tiling-key 实例化。
[^kernel]: 固定提交中的三阶段顺序、workspace 布局、状态类型路径和全核同步。
[^stage1]: 固定提交中的块内变换、矩阵乘、逆与 AIC/AIV 交接实现。
[^stage2]: 固定提交中的跨块状态传播、状态更新和 BF16/FP32 路径。
[^stage3]: 固定提交中的 masked QKT、尾块处理和最终块内输出矩阵乘。
