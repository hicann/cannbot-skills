# AscendC 两阶段排序 Tiling

## 1 参数与选路

输入为 M 行、每行 N 项，C 是可用核数，U 是 UB 字节数，D 是值字节数，J 是最终索引字节数，A 为按 UB block 对齐。输出每批 B 行，任务数 `Q=ceil(M/B)`，使用核数 `C_act=min(C,Q)`；参考模板键为 sch6。

先应用[小轴经验档位](../../tiling.md#小轴路由)，再规划容量。N≤64 使用逆映射恢复，N>64 使用 uint16 紧凑键恢复；这里的 64 是当前实现策略，不能替代紧凑键表示范围的检查。

## 2 批次容量上界

扣除 SIMT 保留区，`U_work=U−32768`。令 `E=B×N`、q=2（逆映射）或 3（紧凑键）。最终索引别名区宽度 F 默认等于 J；若调用方设置 `finalIdxUbElemBytes`，则使用该显式宽度。参考 Host 的粗略估计在未设置该字段时先取 Frough=4，显式设置时取 Frough=F；精确检查始终使用 F：

```text
P = ceil(M/C)
Bmax = min(P, floor(U_work / (2D+4q+Frough) / N))
若使用紧凑键：Bmax = min(Bmax, floor(sqrt(65536/N)))
```

粗略上界未计各 buffer 对齐和排序 scratch，只用于缩小搜索范围。紧凑键的最大值为 `B²N−1`；整数平方根或修正后的平方根必须保证 `B²N≤65536`。

## 3 精确候选检查

对候选 B，Stage1 查询原 dtype、E 个元素的隐式索引 Radix 排序临时空间；紧凑键分支还查询 E 个 uint16 元素的 Stage2 临时空间，取对齐后的最大值 Ttmp。

```text
required(B) = 2×A(ED) + q×A(4E) + A(FE) + Ttmp
候选可行 ⇔ required(B) ≤ U_work，且 E、任务数及偏移可表示
```

不能把 Sort 的第二个模板参数解释成稳定开关；实际调用合同见 [Kernel](kernel-template.md)。scratch 查询配置须与 Stage1/Stage2 的 dtype、索引模式和算法一致。

## 4 批次搜索与多核分配

```text
best = 无; baseRounds = 未设置
for B 从 Bmax 递减到 1:
  若精确容量或字段检查失败：continue
  Q = ceil(M/B); C_act = min(C,Q)
  rounds = ceil(Q/C_act)
  idle = rounds×C_act−Q
  若这是首个可行候选：保存为 best，记录 baseRounds
  否则若 rounds != baseRounds：停止搜索
  否则若 idle < best.idle：更新 best
返回 best；同分保留先枚举的较大 B
```

本搜索先保持首个可行批次的执行轮数，再减少尾轮空闲槽，允许 B=1；不能替换成“无条件取最大 B”。每核按 `batchId=blockIdx+k×C_act` 处理批次，尾批 `v=min(B,M−batchId×B)`，实际排序元素数 `e=v×N`。

## 5 下发字段

| 逻辑量 | 参考字段 | 单位 |
|---|---|---|
| N、M | `numTileDataSize`、`unsortedDimNum` | 每行元素数、总行数 |
| B、Q | `keyParams0`、`keyParams1` | 行数、批次数 |
| 是否逆映射 | `keyParams2` | 1/0 |
| Ttmp | `tmpUbSize` | 字节 |
| C_act | blockDim | 核数 |

## 6 可复算例子

取 int32、int64 输出索引、M=1088、C=17、N=256，假定 `U_work=131072、a=32、Ttmp=4096`。此处 Ttmp 仅为演算输入，实际值须查询。

P=64；紧凑键限制 B≤16。B=16 时 E=4096，预算为 `2×16384+3×16384+32768+4096=118784`，Q=68、C_act=17、rounds=4、idle=0。下一个 B=15 令 Q=73、rounds=5，搜索停止，结果 B=16；最大紧凑键恰为 65535。

源码核对：[Sort Host](../../example/sort/op_host/arch35/sort_tiling_arch35.cpp) 的 `EstimateSmallAxisTwoStageBatching`、`PrepareTwoStageBatchCandidate`、`SearchTwoStageBatchPlan`。分配和存活见 [Buffer](dag-buffers.md)。

本例的任务与输出追踪见 [Kernel §9](kernel-template.md#9-贯穿b16-的最后一批)，工作区与存活阶段见 [Buffer §5](dag-buffers.md#5-贯穿算例的物理角色变化)。
