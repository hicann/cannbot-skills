---
type: CATLASS DSL Operator Example
title: Chunk Bwd Dqkwg
description: Gated Delta Rule 的 DQ/DK/DW/DG 混合反向核，包含四阶段信用流水、workspace 环和可证伪优化路径。
tags: [catlass-dsl, operator, linear-attention, gated-delta-rule, backward, mixed, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/README.md
    title: 算法语义与支持范围
  - id: host
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_host/chunk_bwd_dqkwg_def.cpp
    title: dtype、平台与输出 head 语义
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_host/op_api/aclnn_chunk_bwd_dqkwg.cpp
    title: shape、可选参数和连续化约束
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_host/op_tiling/chunk_bwd_dqkwg_tiling.cpp
    title: 分核、workspace 环深与路径参数
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_kernel/chunk_bwd_dqkwg.cpp
    title: mixed AIC:AIV 入口与 tiling key
  - id: common
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_kernel/chunk_bwd_dqkwg_common.h
    title: chunk 地址、workspace 槽与信用协议
  - id: vector
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_kernel/chunk_bwd_dqkwg_vector.h
    title: A/B/C/D Vector 阶段、UB、GVA 归并与精度
  - id: cube
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/op_kernel/chunk_bwd_dqkwg_cube.h
    title: 七个 GEMM、V 路径与 Cube 流水
  - id: reference
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/tests/chunk_bwd_dqkwg_cpu.py
    title: 四个梯度的 CPU reference
  - id: cases
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/chunk_gdn_bwd/chunk_bwd_dqkwg/tests/cases.py
    title: dense、varlen、GVA、dtype、BT 与 V 用例
operator_families: [linear-attention, gated-delta-rule]
arch: [c310]
---

# 接口与概念

## 算子算法

只保留约束优化合法性的语义。对长度为 `L≤BT` 的 chunk 和 value head `hv`，令
`hk=floor(hv/(HV/HK))`、`gL=g[L-1]`：

```text
E[i,j]    = 1[i>=j] * exp(min(0, g[i]-g[j]))
ds        = scale * (do @ v^T) * E
dq_state  = scale * exp(g)[:,None] * (do @ h^T)
dk_state  = exp(gL-g)[:,None] * (v @ dh^T)
dq_hv     = dq_state + ds @ k
dk_hv     = dk_state + ds^T @ q
dw        = -(dv @ h^T)
```

`dg` 合并 `dq_state*q`、`-dk_state*k` 以及
`ds*(q@k^T)` 的行和减列和；最后一个有效 token 还加入
`exp(gL)*sum(h*dh)+sum(dk_state*k)`。同一 `hk` 对应的 `HV/HK` 份 `dq_hv/dk_hv`
相加得到最终 `dq/dk`。因此因果 mask、gate 差值方向、末 token 注入和 GVA head 归并都不能因
优化而改变。[^host][^reference][^vector]

只影响路径的 workload 轴为：`BT∈{64,128}`、`K=128`、`V∈{128,256}`、QKV dtype、gate
dtype、dense/varlen、`HV/HK`、`B*numChunks` 及 workspace 规模。[^host][^tiling]

# 用法

## 分核策略与基本块切分

任务是完整的 `(batch,chunk)`，`coreLoops=B*numChunks`；dense 的
`numChunks=ceil(T/BT)`，varlen 直接取 canonical `chunk_indices` 的 pair 数。host 使用全部物理
AIC 作为 block dim，各核以 `coreIdx, coreIdx+aicCoreNum,...` 领取 chunk；所以有效并行度上限是
`min(coreLoops,aicCoreNum)`，增加 `HV` 只增加单任务工作量。每个 mixed block 是
`AIC:AIV=1:2`，AIC 逐 head 做 GEMM，两个 AIV 按 head 或半行分工。[^tiling][^entry][^vector][^cube]

每核把自己的 chunks 组成大小 `G=D/4` 的组，在组内执行 `A→B→C→D`；尾组在剩余任务不超过
`2G-1` 时合并。`D` 是 cross-stage ring depth，候选为 16/8/4；short ring depth
`S=max(D/2,2)`。Cube 与 Vector 必须使用相同分组和槽公式。[^tiling][^common]

## 实现路径

所有合法 workload 都使用 tiling key 1 和同一个 mixed kernel，但内部有四个关键分支：

| 条件 | 实际路径 | 优化含义 |
| --- | --- | --- |
| `V=128` | 七个 GEMM 使用单次 `GM→L1→L0→MMAD→Fixpipe` | 无 L1/L0 ping-pong，关注 GEMM 间事件和 workspace 往返 |
| `V=256` | 七个 GEMM 实例化 tiled `BlockMmadTla` | reduction 跨 128 tile，Cube 与 L1/L0 占用上升 |
| `BT=64` / `BT=128` | Vector queue 深度分别为 2 / 1 | BT128 的 `BT²` UB 压力阻断双缓冲 |
| `mainFootprint>512 MiB` | B 阶段现场重算 gate/mask `mul1` | 否则 A 写、B 读 low-precision `BT×BT` workspace |
| `HV>HK` | 每个 value-head 先生成 partial，再对 `dq/dk` 做 GM 读改写 | 额外 MTE3→MTE2 fence，且中间归并按输出 dtype 舍入 |
| varlen | metadata 恢复 `bos/eos`，stage reset 前 `PIPE_ALL` drain | dense 仅 drain MTE3；短尾块另有 padding 路径 |

这里 `mainFootprint = 2*B*HV*T*K + 2*B*HV*T*BT + 4*B*HV*numChunks` 字节。它和
512 MiB 都是固定源码中的选择条件，不是已验证的 Ascend 950 最优阈值。[^tiling][^vector][^cube]

## 成本模型与 profiler 归因

每个 `(chunk,hv)` 的七个 GEMM 约含
`3*L*K*V + L²*V + 3*L²*K` 次乘加；Vector 还承担 `O(L²)` gate/mask 与行列归约、
`O(L*K)` epilogue 和 `O(K*V)` 的 `h*dh` 归约。BT 从 64 增至 128 时 chunk 数约减半，但
`L²` 工作和 `BT²` workspace/UB 项显著增加；V=256 主要放大四个 V-reduction GEMM 和 state
流量。[^vector][^cube]

- `coreLoops<aicCoreNum` 且少数核很长：chunk 任务并行度不足。
- AIC 等 credit 多：Vector 落后；AIV 等 cube-ready 多：Cube 落后。先按 A/B/C/D 阶段定位。
- V256 的 Cube/MTE1 时间突增：检查 tiled reduction，而不是沿用 V128 单 tile 结论。
- MTE2/Fixpipe 与 L2 miss 同时偏高：检查实际 `D/S` 和 ring 字节；不要只扩大 overlap 窗口。
- `mainFootprint` 跨阈值后 Vector Exp 上升、workspace 流量下降：这是 `mul1` 重算分支切换。
- `HV/HK>1` 时 MTE2/MTE3 尾部随 ratio 增长：检查低精度 GM 累加和 fence。
- varlen 短尾多且 Cube 利用率低：检查真实 `L`、metadata 热读、padding 和 `PIPE_ALL` reset。
[^tiling][^common][^vector][^cube]

## 优化候选与回退

1. 先记录 `V/BT/dtype/varlen/HV:HK/coreLoops/D/S/G/mainFootprint` 和实际分支；任何实验只改一个轴。
2. `coreLoops` 限制并行度时，可评估 `(chunk,head-group)` 任务；必须保持同一 `hk` 的 partial
   有唯一归并者。预期活跃核增加；代价是 metadata、workspace 和归约，长尾或 GM 累加上升则回退。
3. AIC/AIV 等待不平衡时，只调整 `D/G`。预期目标侧 wait 缩短；代价是 ring 容量和 L2 驻留。
   Cube/Vector 分组、初始 credit 数 `N=min(G,M)` 和槽公式必须同步，出现死锁、旧槽或 L2 miss 上升即回退。
4. `mul1` GM 往返主导时比较“物化”和“B 阶段重算”。预期 MTE2/MTE3 降低、Vector Exp 增加；
   A/B 必须使用同一判据，且 mask、tail 和 gate dtype 等价。Vector 成为长尾时回退。
5. V256 Cube 主导时，单独调整 reduction tile/ping-pong；保持七个 GEMM 的实际 `L` 与转置布局。
   预期 Cube/MTE1 改善，代价是 L1/L0 和事件数；容量失败或 Fixpipe 尾部增加则回退。
6. `HV/HK>1` 的 GM 读改写主导时，可让一个 `hk` 的 partial 在 FP32 UB 中合并后只写一次。
   预期 fence 和输出流量下降；代价是 `BT*K*4B` 累加区。head 所有权不唯一或 UB 不足则回退。
7. FP16 的 `dw` Vector 长尾明显时，评估用批量向量归约替代当前首 16 列 repair。必须覆盖小值
   舍入门禁；若 `dw` 误差或额外 dv/h 读取增加则回退。[^vector]

# 代码模式

## 数据路径与存储层级

```text
q/k/v/g/h/do/dh/dv GM
  -> AIC L1/L0/MMAD/Fixpipe
  -> low-precision GM ring: dw|mm6, mm5|mm7, ds_temp, optional mul1
  -> AIV UB: FP32 gate/exp/reduce/epilogue
  -> dq/dk/dw low precision, dg gate dtype GM
```

令 `R=min(aicCoreNum,B*numChunks)`、元素宽度 `e=2`、`D` 为 group ring depth、
`S=max(D/2,2)`，user workspace 是下列 32B 对齐区域之和：

```text
R*S*HV*BT*K*e       # dw 与 mm6 复用
R*D*HV*BT*K*e       # mm5 与 mm7 复用
R*D*HV*BT*BT*e      # ds_temp
R*S*HV*BT*BT*e      # 小 footprint 时的 mul1
R*D*HV*4            # dg_last FP32
```

共享成立依赖严格的 A/B/C/D 生命周期，不能仅因两个 region 大小相同就重叠。[^tiling][^common]

## 流水排布、同步关系与数值精度

AIC 每个 stage/chunk 先消费一个 credit，完成全部 heads 并在 Fixpipe 写回后发布 cube-ready；两个
AIV sub-block 都等待 ready，完成 MTE3 后共同返还一个 credit。Vector 启动时预置
`N=min(G,M)` 个 credit，使 Cube 最多领先 N 个任务。阶段间没有 `SyncAll`，flag 跨
`TPipe.Reset` 延续；varlen reset 前使用 `PIPE_ALL`，dense 只等待 MTE3。[^common][^vector][^cube]

Cube 在 FP32 L0C 累加后由 Fixpipe 写为输入低精度；AIV 将中间量转 FP32执行 exp、逐元素运算和
reduce，再把 `dq/dk/dw` 转回 QKV dtype。`dg` 每阶段按 gate dtype 写回，FP16/BF16 gate 会在阶段间
舍入；GVA 的 `dq/dk` 也在每个 value-head partial 后按低精度写回再读取。FP16 `dw` 另有首 16 列
FP32 repair。改变归并位置或消除写回会改变舍入路径，必须以四输出正确性为门禁。[^vector][^cube]

# 约束

- Q/K/V/H/dO/dH/dV 同为 FP16 或 BF16；gate/dg 为相同低精度或 FP32；`K=128`、
  `V∈{128,256}`、`BT∈{64,128}`、`HV%HK=0`。[^host][^tiling]
- varlen 仅支持 `B=1`，`cu_seqlens` 与 flattened `(sequence,chunk)` pairs 必须同时存在；每个
  pair 决定 state 的 chunk 索引和真实 `L`。[^tiling][^common]
- `w/g_gamma` 必须为空，`use_exp2/transpose_state_layout` 必须为 false；user workspace 为空时
  kernel 直接返回。[^api][^entry]
- `g` 预期非正且沿 chunk 单调不增；优化不得去掉 `min(0,g[i]-g[j])`、因果 mask 或尾部 padding。[^guide][^vector]

# 失败表现

- 仅 `HV>HK` 错：`hv→hk` 映射、partial 顺序、GM fence 或输出所有权错误。
- 仅 varlen/短于 8 的尾块错或报 Vector illegal configuration：metadata、有效长度或 gate padding 错。
- 偶发死锁/旧数据：AIC/AIV 的分组、ready/credit 计数或复用槽不一致。
- `dg` 其他位置正常而 chunk 末项错：`dg_last` 生命周期、gate 差值方向或最后位置注入错误。
- FP16 `dw` 首行/首 16 列小值异常：repair 被删除、提前舍入或 dv/h 地址错误。
- 正确但 V256、BT128 或高 GVA ratio 突然变慢：首先确认 tiled、单 buffer 或 GM 累加分支，而非归因为同一瓶颈。

# 验证方法

现有固定提交用例覆盖 FP16/BF16、FP32 gate、BT64/128、V128/256、dense/varlen、尾块及多组
`HV/HK`；CPU reference 分别生成 `dq/dk/dw/dg`。这些用例证明支持面，不证明本文候选有收益。
[^reference][^cases]

修改路径、流水或归并后还应定向覆盖：`coreLoops<aicCoreNum`、每核 `M=1/2/G/2G-1/2G`、
真实 `L=1/7/8/63/64/65/127/128`、`HV/HK=1/2/3/32`、512 MiB 判据两侧，以及 D=4/8/16。
分别比较四个输出，并对 GVA 检查每个 `hk` 的完整 value-head 和；同步修改增加多轮随机压力测试。

性能实验在空闲 Ascend 950 上固定 shape、dtype、BT、输出、Device 和 metric，记录总
`aicore_time`、AIC/AIV 分时、MTE2/MTE3/Fixpipe、Vector Exp、cross-core wait、L2 hit、每核长尾和
实际 `D/S/G`。一次只验证一个候选；没有完整正确性与设备 profile 时只保留为静态假设。

[^guide]: 固定提交中的算法用途、gate 约束、shape 和支持范围。
[^host]: 固定提交中的输入输出 dtype、GVA head 语义和平台注册。
[^api]: 固定提交中的输出 shape、必选结果、参数组合和输入连续化。
[^tiling]: 固定提交中的单 tiling key、任务公式、block dim、ring 选择、workspace 复用和 varlen 条件。
[^entry]: 固定提交中的 mixed AIC:AIV=1:2 入口、user workspace 门禁和 dtype 实例化。
[^common]: 固定提交中的 chunk 地址、group/short ring 槽、尾组和 raw credit 协议。
[^vector]: 固定提交中的四个 Vector 阶段、UB queue、gate 路径、GVA 归并、tail padding 和输出转换。
[^cube]: 固定提交中的七个 GEMM、Ascend 950 架构选择、V128 direct 与 V256 tiled 路径。
[^reference]: 固定提交中的 chunk 级公式、GVA 归并与四输出 reference。
[^cases]: 固定提交中的 dense/varlen、dtype、BT、V、尾块与 head-ratio workload。
