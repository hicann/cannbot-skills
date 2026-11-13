# AscendC 多核 Radix Tiling

## 1 输入与规划结果

输入连续 M 行、每行 N 项。C 是可用核数，U 是 UB 字节数，D 是值字节数，a 是 UB block 字节数。本模板在前述路线未选中时尝试；N≥2^30 不是进入多核路线的前提。

需要联合求出：块长 T、每行协作核数 G、同时处理的行数 P_act、每行块数 K、批次数 R 和实际核数 C_act。计数宽度 S 与输出索引宽度 J 分别选择。

## 2 计数位宽与 UB 容量

状态字高两位承载 lookback 状态，低位承载非负计数/前缀：

```text
N≤2^30−1：CounterT=uint32，S=4
2^30≤N≤2^62−1：CounterT=int64，S=8
```

中间原轴索引实际位宽不超过最终输出索引宽度，但 Host 保守按 S 预算对应 UB/GM 区。局部桶计数仍为 uint16、局部排列仍为 uint32，不随 S 扩大。

按[运行保留预算](../../output-mapping.md#scratch-与-ub-由-host-规划)核对 R_simt（参考 Host 为 32768），扣除后 `U_work=U−R_simt`。单 tile 的每元素区系数为 `F=D+S+6`，固定桶区为 `Efix=256×(3S+4)`。对候选 T：

```text
T 为 256 的倍数
T×F + Efix + sortScratchRequired(T) ≤ U_work
sortScratchRequired(T) = QuerySortTmpSizeRadix(T)采用的查询maxValue
                        # uint8、隐式源索引RADIX_SORT；尚非最终分配值
```

初始容量上界为 `floor((U_work−Efix)/F/256)×256`，不超过 align256(N)。从此上界按 256 递减，查询 scratch 并核对容量。局部 uint16 计数也须能表示真实 tile 内计数。

## 3 行内分核与行间并行的联合搜索

将首个通过容量检查的块长上界记为 T_cap，按以下规则枚举：

```text
Gmax = min(C, max(1,floor(N/1024)))
for g in 1..Gmax:
  T = min(T_cap, align256(ceil(N/g)))
  K = ceil(N/T)
  G = min(g,K)
  P_act = min(M,floor(C/G))
  R = ceil(M/P_act)
  Lt = ceil(K/G)
  score = (R×Lt, T, R)        # 按字典序取最小
```

1024 是当前每核轴内工作量的经验起点；256 是桶数及对齐粒度。评分先减少每核串行 tile 轮次，再减少相同轮次的补齐元素数，再减少行批次，不固定“每行一核”或“单行用满所有核”。

选中 T 后重新查询该 T 的 scratch；若超预算，缩小容量上界并重选。参考 `AdjustRadixTmpUb` 随后将剩余 UB 扩入 tmp：余量足够时保留一个 UB block，不足时不扩容。

```text
remainAfterSort = U_work - (T×F + Efix) - sortScratchRequired(T)
tmpAllocated = sortScratchRequired(T) + floor(max(0,remainAfterSort-a)/a)×a
tmpUbSize = tmpAllocated
```

公式对应参考 Host 的分配策略，物理 buffer 的额外对齐另计。最终容量还须容纳 scatter 复用的 3 个 uint64 槽位（24字节）；参考 Host 不满足时拒绝候选。最终 `C_act=P_act×G`。查询需求、扩容后的实际分配和下节清零容量分别记录；清零分工使用 tmpAllocated。

## 4 Kernel 映射

```text
slot = blockIdx // G
lane = blockIdx % G
第 r 批全局行号 = r×P_act+slot
本核 tileId = lane, lane+G, lane+2G, ... < K
tileStart = tileId×T
valid = min(T,N−tileStart)
```

workspace 按 slot 寻址，最终输入/输出按全局行号寻址。无效尾行可以跳过数据访问，但仍参与所有核间屏障。

## 5 清零分工与下发字段

参考实现为每个值字节保留一组桶区。全局逻辑计数元素数为 `Hg=P_act×K×256×D`，独占桶区为 `He=P_act×256×D`。令 `Z=floor(tmpAllocated/S)`，Z 必须非零；它来自最终 tmpUbSize，而非 Sort 查询值。扩容会改变清零循环次数及其 workspace 余量。

```text
gPerCore = max(ceil(Hg/C_act), clearMinElems)
p0 = ceil(Hg/gPerCore)             # globalHist 清零核数
p3 = ceil(gPerCore/Z)              # 每核清零循环次数
p2 = min(gPerCore,Z)               # 每次清零元素数
p5 = gPerCore
ePerCore = max(ceil(He/C_act), clearMinElems)
p1 = ceil(He/ePerCore)             # exclusiveBins 清零核数
p4 = ePerCore
```

参考代码的 clearMinElems 取 blockUbSize 的数值作为最小清零元素数；它是分配策略，p2/p4/p5 的单位仍是 CounterT 元素，消费时乘 S。两段实际分配必须覆盖该清零分工而不仅是逻辑桶数，见 [Buffer](dag-buffers.md)。

其余字段：`numTileDataSize=T`、`lastDimNeedCore=G`、`unsortedDimParallel=P_act`、`lastDimTileNum=K`、`sortLoopTimes=R`。sch2 的模板键按最终 CounterT 宽度生成；需要跨核同步的路线设置对应调度模式。

## 6 可复算选核例子

假设 N=8192、C=56、T_cap=4096：M=1 时取 G=8、T=1024、K=8、P_act=1，`R×Lt=1`；M=37 时 G=1、T=4096、K=2、P_act=37 的评分为 `(2,4096,1)`，优于 G=8 的多批次方案。这里只演算搜索，不声称某个设备上的周期最优。

来源：[Sort Host](../../example/sort/op_host/arch35/sort_tiling_arch35.cpp) 的 `ComputeRadixTileData/SelectRadixCorePlan/AdjustRadixTmpUb/FillRadixKernelParams`。计算协议见 [Kernel](kernel-template.md)。

本例的任务与输出追踪见 [Kernel §8](kernel-template.md#8-贯穿m37n8192t4096)，工作区与存活阶段见 [Buffer §5](dag-buffers.md#5-贯穿算例的-workspace-账本)。
