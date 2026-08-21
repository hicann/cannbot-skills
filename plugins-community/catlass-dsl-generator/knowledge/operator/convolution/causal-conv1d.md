---
type: CATLASS DSL Operator Example
title: Causal Conv1d
description: Ascend 950 因果一维卷积的序列/通道切分、状态缓存、增量更新、APC/MTP 与纯 AIV 数据路径。
tags: [catlass-dsl, operator, causal-convolution, sequence-convolution, state-cache, apc, mtp, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/fused_causal_conv1d/README.md
    title: cache、APC、MTP、残差与模式语义
  - id: entry
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/fused_causal_conv1d/op_kernel/fused_causal_conv1d_apt.cpp
    title: BH/BSH 与 dtype 分派入口
  - id: bh
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/fused_causal_conv1d/op_kernel/arch35/fused_causal_conv1d_cut_bh.h
    title: batch 与通道二维切分实现
  - id: bsh
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/fused_causal_conv1d/op_kernel/arch35/fused_causal_conv1d_cut_bsh.h
    title: 序列与通道二维切分实现
  - id: vf
    resource: https://gitcode.com/cann/ops-transformer/blob/90b41d6d8f2ce716275383a28f5dfb1d7c75ca1e/attention/fused_causal_conv1d/op_kernel/arch35/vf/fused_causal_conv1d_no_state_double_tail.h
    title: 无状态双尾向量基本块
  - id: fla_guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/README.md
    title: 前向、状态更新、接口与支持范围
  - id: fla_api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/docs/aclnnCausalConv1d.md
    title: aclnn 接口、shape、layout 与可选输入
  - id: fla_tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/op_host/causal_conv1d_tiling.cpp
    title: host tiling、模式与核数规划
  - id: fla_entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/op_kernel/causal_conv1d.cpp
    title: kernel 入口与模板分派
  - id: fla_kernel
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d/op_kernel/arch35/causal_conv1d_regbase.h
    title: 目标寄存器向量实现
operator_families: [causal-convolution, sequence-convolution]
---

# 接口与概念

## 算子算法

Causal Conv1d 对每个通道独立计算
`y[t,d] = activation(sum_j weight[j,d] * x[t-j,d] + bias[d])`，序列开头从 state cache
补足历史。整段前向处理完整序列，增量更新读取 cache 后计算当前 token，并把最新历史写回同一
cache line；因此卷积计算与状态更新属于同一个有别名约束的算子语义。[^guide][^fla_guide]

一套来源固定卷积宽 3，并融合残差、APC prefix cache、MTP accepted/computed token、Pangu
开头清零和 inplace y→x；另一套来源支持卷积宽 2/3/4、可选 bias/SiLU，以及由 run mode
选择整段前向或状态更新。两者是同一算子的不同支持域，不能把某一实现独有的属性当成公共接口。
[^guide][^fla_guide][^fla_api]

# 用法

## 分核策略与基本块切分

入口均为 AIV-only。BH 路径二维切分 batch×dim，dim 以 128 元素块分核；BSH 路径切累计
sequence×dim，sequence tile 携带卷积宽减一的 overlap，并把延迟 cache 写回交给拥有完整尾部
的 tile。四个 tiling key 覆盖 BH/BSH 与 FP16/BF16，dim remainder 和 sequence tail 分别处理。
[^entry][^bh][^bsh]

当前源码的寄存器路径按 run mode、dtype、卷积宽和 layout 分派，block 在序列/通道 tile 间分配；
尾 token、通道 remainder、无效 cache slot 和 speculative accepted-token 数走显式分支。短卷积下
性能关键是扩大连续通道搬运、复用 weight/state，并避免更新模式重复加载同一 cache row，而不是
引入 Cube 或跨核 partial sum。[^fla_tiling][^fla_entry][^fla_kernel]

输入既可为累计序列二维布局，也可为固定 batch 三维布局；query start、cache index、initial-state
标记、accepted/computed token 和 APC block metadata 共同确定每条逻辑序列及 cache line。
[^guide][^fla_api]

# 代码模式

## 数据路径与存储层级

```text
x/weight/bias GM -> VECIN/寄存器 -> UB 滑动窗口
cache GM -> UB prefix -> [历史 | 当前 x] 逻辑窗口
UB depthwise conv (+ bias/SiLU/residual/zero reset) -> y GM
窗口尾部 -> cache GM；APC prefix 边界 -> 额外 cache line
```

BH 在 UB 中按 batch 装载 cache 并直接卷积；BSH 使用 sequence overlap、batch metadata 和
deferred write，必要时借 workspace 暂存跨 tile 的 y/cache。寄存器路径针对不同卷积宽和运行
模式直接执行滑窗乘加。所有实现均为 GM↔UB/寄存器↔GM，不经过 L1、L0 或 Fixpipe。
[^bh][^bsh][^vf][^fla_kernel]

## 流水排布、同步关系与数值精度

没有 AIC/AIV 跨核流水；每核依靠 queue 的 EnQue/DeQue 和 MTE2/V/MTE3 hard event 保护输入、
计算与写回 buffer。cache 必须先完整搬到本地才能覆盖同一 GM line；APC 读写冲突走延迟或临时
缓冲，BSH 在 cache prefix 填充结束后才执行 inplace y→x。不同 cache slot 可以并行，同一 slot
内的 token/state 更新必须保持顺序。[^bh][^bsh][^fla_kernel]

FP16/BF16 由入口模板全路径特化，输出和 cache 保持接口 dtype；SiLU、残差和开头清零均在向量
路径融合。该算子没有 Cube MMAD，性能主要受 GM/UB 搬运、通道尾块和 cache 冲突路径影响。
[^entry][^vf][^fla_entry][^fla_kernel]

# 约束

- 两套源码实现的卷积宽支持域分别为固定 3 与 2/3/4；调用和调优时必须以实际入口为准。[^guide][^fla_guide]
- state 长度必须覆盖卷积历史及 speculative offset；更新前不得覆盖尚未消费的 cache 数据。[^guide][^fla_guide]
- packed 序列不能跨 query-start 边界读取历史；cache index、pad slot 和 accepted/computed token 必须在各自合法范围。[^guide][^fla_api]
- APC 的 block table、首尾 block 与 initial-state index 必须一致；无 APC 的实现不能接收或模拟这些元数据。[^guide][^bh][^bsh]
- activation 支持范围由入口限定；bias、residual、开头清零和 inplace 均不是所有实现共有的功能。[^guide][^fla_guide]

# 失败表现

- 序列开头若固定错若干 token：历史窗口方向、零初始化或 cache prefix 拼接错误。
- decode 第二步起错误：accepted-token offset 未用于 cache 读取/写回。
- 仅 APC case 错误：prefix boundary、block-table line 或 cache 读写碰撞处理错误。
- 仅通道/序列尾部错误：最后一个 dim core、sequence overlap 或寄存器尾块 mask 错误。
- inplace 或 update 偶发错误：源 x/cache 在最后一次读取前被覆盖。

# 验证方法

reference 应显式维护每个 cache line，覆盖宽度 2/3/4、有无 bias/SiLU/残差、prefill、逐 token
update、混合 prefill/decode、packed 变长、cache 重映射、pad slot、accepted/computed token、APC
开关、首次 computed=0、开头清零、FP16/BF16、dim/sequence 双尾块及 inplace。每一步同时比较 y
与所有被修改的 cache line；性能需在空闲 Ascend 950 NPU 上分别测 BH、BSH 与寄存器路径。
[^guide][^entry][^fla_guide][^fla_entry]

[^guide]: 固定提交中的固定宽卷积、cache、APC/MTP、残差与模式语义。
[^entry]: 固定提交中的 AIV-only 入口及 BH/BSH、FP16/BF16 分派。
[^bh]: 固定提交中的 batch/dim 分核、UB queue、cache 读写和 APC prefix 实现。
[^bsh]: 固定提交中的 sequence/dim overlap、deferred cache、APC collision 和 inplace 实现。
[^vf]: 固定提交中的无 state 双尾向量卷积基本块。
[^fla_guide]: 固定提交中的前向/状态更新、卷积宽、bias、SiLU 和 cache 语义。
[^fla_api]: 固定提交中的 aclnn 参数、shape、layout 与可选输入约束。
[^fla_tiling]: 固定提交中的运行模式、dtype、卷积宽、layout 与 block 规划。
[^fla_entry]: 固定提交中的 AIV-only 入口和模板分派。
[^fla_kernel]: 固定提交中的寄存器滑窗、尾块、cache 更新与流水实现。
