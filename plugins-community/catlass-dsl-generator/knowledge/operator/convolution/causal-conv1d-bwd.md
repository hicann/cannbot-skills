---
type: CATLASS DSL Operator Example
title: Causal Conv1d Bwd
description: Ascend 950 因果一维卷积反向的 BT×BD 分块、RegBase 快路径、FP32 workspace 归约及优化方法。
tags: [catlass-dsl, operator, causal-convolution, sequence-convolution, backward, regbase, reduction, ascend-950]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-11T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-11T00:00:00Z'}
sources:
  - id: guide
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d_bwd/README.md
    title: 算法、layout、dtype 与 state 语义
  - id: api
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d_bwd/docs/aclnnCausalConv1dBwd.md
    title: aclnn 可选项与确定性合同
  - id: definition
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d_bwd/op_host/causal_conv1d_bwd_def.cpp
    title: host 输入输出定义与产品注册
  - id: tiling
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d_bwd/op_host/causal_conv1d_bwd_tiling.cpp
    title: BT/BD、block、UB 与 workspace 规划
  - id: entry
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d_bwd/op_kernel/causal_conv1d_bwd.cpp
    title: AIV 入口、dtype 实例与 user workspace
  - id: kernel
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d_bwd/op_kernel/causal_conv1d_bwd.h
    title: 通用路径、任务映射、state、流水与归约
  - id: regbase
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/fla/ops/ascendc/gdn/gdn_preprocess/causal_conv1d_bwd/op_kernel/arch35/causal_conv1d_bwd_regbase.h
    title: Ascend 950 寄存器基本块
  - id: tests
    resource: https://github.com/flashserve/flash-linear-attention-npu/blob/4c07565db3ab0f4bfc1dd20857154688883427ce/torch_custom/fla_npu/test/test_npu_causal_conv1d_bwd.py
    title: fixed、BNSD、varlen、state 与多输出正确性规格
operator_families: [causal-convolution, sequence-convolution]
arch: [c310]
---

# 接口与概念

## 算子算法

先把上游梯度转为有效梯度，再完成 depthwise causal convolution 的四类反向输出：

```text
activation=0:   g[t,d] = dy[t,d]
activation=1/2: g[t,d] = dy[t,d] * sigmoid(y[t,d])
                          * (1 + y[t,d] * (1 - sigmoid(y[t,d])))

dx[t,d]      = sum_i g[t+i,d] * weight[W-1-i,d]
dw[W-1-i,d] = sum_(b,t) g[b,t+i,d] * x[b,t,d]
db[d]        = sum_(b,t) g[b,t,d]
```

`initial_state[B,W,D]` 补序列开头历史并贡献 `dw/dh0`，`dht[B,W,D]` 加到末 `W-1` 行 dx；
varlen 的每条序列由 `queryStartLoc` 独立裁剪。x/dx 始终是逻辑 `[B,T,D]` 或
`[totalTokens,D]`，只有 y/dy 可按 BSND/BSH/BNSD/TND/NTD 解释物理布局。输入支持
FP32/FP16/BF16，激活、dx、dw/db 和 state 累加均使用 FP32 中间量。[^guide][^api][^kernel]

优化时要保留两个源码边界：activation 1/2 数学等价，但只有 1 命中 Ascend 950 RegBase；API
把 dw/db 标为可选，当前 tiling 却固定 `hasWeight=hasBias=1` 并始终写出二者，因此未修复合同前
不能用“关闭 dw/db”宣称提速。[^api][^definition][^tiling][^kernel]

# 用法

## 分核策略与基本块切分

kernel 为 AIV-only，单一 tiling key 0。通道块 `BD=64` 当 `D%64==0`，否则 `BD=16`；序列块
通用为 `BT=min(T,64)`。Ascend 950 RegBase 的精确条件是 FP16/BF16、activation 1、无
initial_state/dht、BD64、`1<=W<=4`；其中 W4 才由 host 把 BT 从 `min(T,224)` 开始按 32
递减直到 UB 可容纳。FP32、activation 0/2、state、BD16 和其他组合走通用 Vector 路径。
[^tiling][^kernel][^regbase]

```text
fixed:   numChunks = B * ceil(T / BT)
varlen:  numChunks = sum_b ceil(seqLen_b / BT)
blockNum = min(AIVCoreNum, numChunks)
numBlksD = D / BD
```

block 只沿 chunk 维分配，每个 block 再串行遍历全部 dBlock；dx 的 `(chunk,dBlock)` 区域独占，
dw/db 则先写 `[block,dBlock]` FP32 partial，`SyncAll` 后按 dBlock 归约。varlen 的
`ResolveChunk` 在 `B>1` 时从第 0 条序列线性扫描；BNSD/NTD 的 BD tile 跨 head 时会拆成多段
MTE2 copy。[^tiling][^kernel]

## 成本模型与瓶颈判断

对一个 `(chunk,dBlock)`，主要成本是读取 `BT×BD` x、`(BT+W-1)×BD` dy，激活路径再读同尺寸
y；dx 和 dw 各执行约 `W×BT×BD` 组乘加，SiLU 另有 `(BT+W-1)×BD` 个 FP32 exp/div。
每个 `(block,dBlock)` 只加载一次 `W×BD` weight，但写一份 `(W+1)×BD×4` 字节 partial；
最终归约读取 blockNum 份。[^tiling][^kernel][^regbase]

- active block 少且 `numChunks<AIVCoreNum`：当前 chunk-only 分核限制并行度，先看 dBlock 是否足够。
- MTE2 长且 BNSD/NTD 明显慢：检查 BD 跨 head 的分段数，而不是先改 Vector 计算。
- exp/div 或 Vector 长：SiLU 有效梯度主导；activation 0 只能用于机制定位，不能作为等语义对照。
- `SyncAll`、workspace 读或最慢核长：检查 blockNum、partial 字节和单核串行归约。
- tail 比例高：RegBase 虽收到 `validRows`，实现未使用该值，计算仍按 BT/dyRows 满 tile 执行。
- 短序列随 blockNum 增大反而变慢：weight 重复加载、partial 往返和同步超过有效计算。
- B 增大而 totalTokens 不变时变慢：检查 `ResolveChunk` 的逐序列扫描。[^kernel][^regbase]

该算子不使用 Cube/L1/L0/Fixpipe；Cube ratio 与结构零 MMAD 都不是有效优化入口。优先记录
`aicore_time`、AIV/Vector/MTE2/MTE3、每核时间、最慢 block、cross-core wait、实际搬运字节和
workspace 字节。

## 优化候选与门禁

1. **裁剪 tail 满块计算。** RegBase 应用 `validRows` 限制 dx/dw/db 行循环，并只计算必要的
   `validRows+W-1` 个 g；通用路径同步缩短 repeat/reduce。预期 tail Vector 时间下降。保持
   dy/y 越界补零、末 W-1 halo 和 varlen 边界；任一非 tail 回退或 tail 数值错误即撤回。
2. **二维分配 `(chunk,dBlock)`。** 当 `numChunks` 少而 `numBlksD>1` 时，把 dBlock 纳入 block
   映射以利用空闲 AIV。预期 active block 增加、单核 dBlock 循环缩短；代价是新的 dx 所有权和
   partial 布局。若 workspace、同步或 weight 重载抵消收益则回退。
3. **自适应 blockNum。** 当前总是使用 `min(core,numChunks)`；短序列可减少 block，令每核处理
   多个连续 chunk，从而复用 weight、形成双 slot 预取并缩小 partial workspace。扫描候选 Q 时
   同时看 active 核、最慢核和归约时间，不能只取计算阶段最快值。
4. **复用相邻 chunk halo。** Arch35 双 slot 当前重新搬完整 dy/y 窗口；同一序列、同一核的下一
   chunk 可保留末 W-1 行，只搬 BT 个新行。预期 MTE2 字节下降；跨 query boundary、跨 block 或
   非连续 chunk 必须失效，slot live range 增长导致等待时回退。
5. **减少 g 的 UB 往返。** RegBase 先把 SiLU 后 g 写入 FP32 UB，再分别供 dx/dw/db 重读。
   W<=4 时可候选化 W 行寄存器/ring，单次形成 g 并同时推进 dx、dw、db。目标是减少 UB 读写和
   VF 调用；必须保证每个 g 只算一次、FP32 顺序不降级，寄存器压力降低并行时回退。
6. **拆分跨核归约。** 当前一个 dBlock 由一个核串行读取全部 block partial；当
   `blockNum>numBlksD` 时同步后大量核空闲。可按 `(dBlock,wRow/channel)` 分配 reducer，或使用固定
   顺序的分层归约。目标是降低最慢归约核与 barrier 尾巴；必须保持 partial 独占、全局
   happens-before 和确定性，不能直接换成无序 atomic。
7. **统一 RegBase 覆盖。** kernel 已支持 W1..4，host 仅对 W4 使用 direct BT/UB 规划；activation 2
   也与 1 同公式。候选是让 host 与 kernel 条件一致并为 W1/2/3、activation 2 建立独立 direct
   选择。不得改变 activation/state/W 语义；两种 B16、所有 W、tail 和 layout 未全量通过则回退。
8. **优化 head-major 搬运。** BNSD/NTD 中 BD64 跨多个小 Dh 时，比较 head-aligned dBlock、合并
   DMA 描述或一次本地 pack。目标是减少小段 MTE2；必须保持逻辑 D 次序和 dx 逻辑 layout，若
   pack 或更多 block/weight copy 变成新瓶颈则回退。
9. **预计算 varlen chunk map。** 多短序列时由 host 生成 `chunk -> (b,t,bos,len)`，替代每个
   dBlock、每个 chunk 的线性扫描。只有控制时间被 profile 证实时采用；额外 workspace/GM 读取
   超过扫描成本，或零长度/不均匀序列错误时回退。
10. **拆出 state 首尾工作。** initial state 只影响首块，dht 只影响尾 W-1 行，却使全部 chunk
    回退通用路径。可把首尾修正拆成专用任务，让中间 chunk 使用无 state 主路径；必须保持单次
    launch/公开输出语义、dh0 slot 和短于 W 的序列，额外同步超过收益时回退。
11. **修复 Optional 后裁剪输出。** 先让 hasWeight/hasBias 与实际输出指针一致并补齐接口测试，
    再对只需 dx 的调用删除 dw/db buffer、partial workspace、`SyncAll` 和归约。合同未修复前此项
    不得启用；输出集合不同的性能只能分开报告。[^api][^tiling][^kernel][^tests]

优化顺序按 profile 选择：tail 主导先做 1；并行不足做 2/3；MTE 主导做 4/8；Vector/UB 主导做
5/7；同步归约主导做 3/6；多短 varlen 做 9；state 或输出组合分别做 10/11。每轮只改变一个轴。

# 代码模式

## 数据路径与存储层级

```text
x/dy/y/weight GM -> MTE2 -> UB/RegBase
  -> g FP32 -> dx + local dw/db FP32
  -> dx/dh0 MTE3 -> GM
  -> per-core partialDw/partialDb FP32 workspace
  -> SyncAll -> cross-core reduce -> dw/db GM
```

纯 AIV 路径不经过 Cube/L1/L0/Fixpipe。user workspace 为
`blockNum*numBlksD*(W*BD+BD)*4` 字节；不存在 atomic 备选。Arch35 W4 direct 的 UB 估算为
`2*(3A+4G+P)+4*(A+G+P+E)`，其中 `A=BT*BD`、`G=(BT+W-1)*BD`、`P=W*BD`、
`E=BD`（均按 8 元素对齐）。BT、双 slot、g/dx FP32 和 partial 的优化必须联合检查 live range。
[^tiling][^entry][^kernel]

## 流水排布、同步关系与数值精度

RegBase 用两个 x/dy/y slot：等待当前 `MTE2_V` 后预取下一 chunk，再计算当前 chunk；通用路径用
`MTE2_V`、`V_MTE3`、`MTE3_V` 与 `PipeBarrier<PIPE_V>` 串起搬入、Vector 和写回。每核 partial
写完后必须 `SyncAll`，reducer 再按 core id 顺序累加。[^kernel][^regbase]

输入 B16 在 UB/寄存器转 FP32，SiLU exp/div、dx、dw/db、dh0 和 state 修正均为 FP32，最后
`CAST_RINT` 到输入 dtype。不能用近似 SiLU、B16 归约、无序 atomic 或删除同步换取不等价性能。
[^guide][^kernel][^regbase]

# 约束

- activation、state、W、BD、layout 任一变化都可能改变 direct/generic 路径，必须记录实际组合。
- BNSD/NTD 的 BD tile 可能跨 head；varlen halo 不能跨 `queryStartLoc`。[^tiling][^kernel]
- dx 区域、partial slot 和最终 reducer 必须各有唯一所有者；归约前保留全核 happens-before。
- initial state slot 0 不参与历史，dht 从 slot 1 加到尾部；短序列按真实长度裁剪。[^kernel][^tests]
- dw/db Optional、W1/W3 和单条零长度 varlen 尚缺完整测试，不得作为稳定提速结论。
- 本条目只指导 Ascend 950/c310；Arch32 的 BD128/Direct Vector 条件不得外推。

# 失败表现

- 仅 tail 错：validRows、halo、mask 或缩短后的 reduce 行数错误。
- 仅 BNSD/NTD 错：跨 head copy、channel 映射或 pack 后逻辑 D 顺序错误。
- varlen 交界错：halo 复用或 chunk map 越过 query boundary。
- dx 正确而 dw/db 非确定：partial 槽、MTE3 完成、SyncAll 或归约顺序被破坏。
- RegBase 隔块旧数据/死锁：双 slot event 配对或预取 slot 提前复用。
- 性能改善只出现在 activation 0、无 state 或少输出：对照语义已变化，不能合并报告。

# 验证方法

正确性覆盖 FP32 generic、FP16/BF16 generic/RegBase、activation 0/1/2、W1..4、BD16/64、
T<64/64/65/224/225、五种 layout、均匀/非均匀 varlen、四种 state 组合和 tail 高占比；逐项比较
dx/dw/db/dh0，并对 Optional 修复增加缺省输出 case。现有固定测试主要覆盖 W2/W4，其他边界是
候选合入前的新增门禁。[^tests]

性能实验保持设备、频率、shape、dtype、layout、activation、state、输出集合、launch 和 metric
一致；记录多次 `aicore_time`、AIV/Vector/MTE2/MTE3、每核分布、最慢 block、cross-core wait、
workspace 和原始 artifact。每个候选单轴实验，最终 best fresh 复测；本文候选未经设备实测，不能
标为已提速。

[^guide]: 固定提交中的算法、dtype、layout、state、activation 与支持边界。
[^api]: 固定提交中的 aclnn 可选输入输出和确定性合同。
[^definition]: 固定提交中的 OpDef 输入输出与产品注册。
[^tiling]: 固定提交中的 direct 条件、BT/BD、chunk/block、UB 和 workspace 规划。
[^entry]: 固定提交中的 AIV task type、dtype 实例与 user workspace 起点。
[^kernel]: 固定提交中的任务所有权、layout copy、state、event 和确定性归约。
[^regbase]: 固定提交中的 Ascend 950 BD64、W1..4 RegBase 及 validRows 现状。
[^tests]: 固定提交中的 fixed、BNSD、TND/NTD、state 和四输出正确性规格。
