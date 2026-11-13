# AscendC 归并排序 Tiling

## 1 路线与符号

输入为连续 M 行、每行 N 项，C 为可用核数，D/J 为值和输出索引字节数。U 为实际 UB，a 为 UB block 字节数，`A(x)=ceil(x/a)×a`。本模板的排序、恢复及区间写回保持 Vector/Queue/MTE3，`R_run=0`、`U_work=U`；按[完整调用链预算](../../output-mapping.md#scratch-与-ub-由-host-规划)计入输出适配新增的 `adaptBytes`。

当前 Host 将 float32 N≤4096、float16 N≤2048、bfloat16 N≤1536 纳入归并候选；这些是选路经验值，不能推导原语的完整支持域。浮点长度筛选后仍要对照[物理分配](dag-buffers.md)核对 UB。Sort32 是初始 run 原语，不单设一条排序路线。

## 2 行批次与核数

参考计划使用 4096 元素作为行批次规划尺度：

```text
H = ceil(N/a)×a                         # 参考 Host 的对齐后轴长，单位是元素
B_capacity = max(1, floor((4096/2)/H))
C_act = min(M,C)
B = min(ceil(M/C_act), B_capacity)
rounds = ceil(M/(C_act×B))
bufferNum = 2 if N≤2048 else 1
```

H 还须满足 Sort32 的 32 项 run 对齐。不要把 `A(N×D)` 的字节长度直接作为 H；前者用于搬运分配，后者用于 proposal 和索引寻址。4096、2048 和双缓冲偏好属于本实现策略，改变后要重新比较资源和性能。

第 c 核第 k 轮的起始行为 `rowStart=(k×C_act+c)×B`；`rowStart≥M` 时跳过，否则 `validRows=min(B,M−rowStart)`。每行完整在同一核内排序，跨行不做归并。

Tiling 字段保持原协议宽度，Kernel 的行号和行步长使用 int64，GM 元素偏移及步长使用 uint64；`C_act×B`、`c×B` 和行号乘 N 均在乘法前提升类型，不能依赖赋值目标的 64 位类型。参考递增骨架见 [行批次及搬入](kernel-template.md#2-行批次及搬入)。

B_capacity 随 H 增大自然降到 1；因此收尾不能只靠行间并行。设 W 为所用 Reg 数据类型的 lane 数，必要的完整轴分类按连续轴位置分配 lane 时约为 `ceil(H/W)` 个向量块；“一行一个 lane、循环 H 次”的实现则有 H 级串行依赖，B=1 时只用一个 lane。最终选 L=1 的常数规模读取与这种完整轴扫描须分开核算。不要为掩盖收尾串行化而强增 B 或越过原选路/容量条件；按 [Kernel 输出恢复](kernel-template.md#区间定位与原值恢复) 修正对应的处理域。

## 3 普通与合批 proposal

普通和合批是本模板的两条条件执行路径。普通模式的索引序列与两份 proposal 各容纳 H 项；合批模式各容纳 E=B×H 项。将候选批次代入以下决策，得到 Host 下发的模式及真实分配规模：

```text
E = B×H
p1 = H×B×D; p2 = H×B×J
castBytes = 4×H×B（BF16）或 0
required(K) = bufferNum×(2p1+p2) + K×(4+2×8) + castBytes + adaptBytes
batchEligible = 输入为float16且输出索引为int64且B≥4且E≤8160
if batchEligible and required(E)≤U_work:
  batchMerge=1; K=E
else if required(H)≤U_work:
  batchMerge=0; K=H
else:
  当前批次候选不可行
```

其中各物理区按 [Buffer](dag-buffers.md) 分别对齐；`adaptBytes` 不重复计算已有区。8160=255×32 对应 Sort32 的 uint8 repeat 次数容量，每项 proposal 8 字节。Host 按此结果分配并传递 `batchMerge`；Kernel 在 `batchMerge=1` 且本批有效行数 v≥4 时执行合批，否则执行普通模式。尾批回退只改变计算域，已分配的 K=E 容量仍保留。

## 4 Host/Kernel 参数

| 字段 | 含义 |
|---|---|
| `p0` | B，每次搬入的行数上限 |
| `p1` | 单个 value 队列 `H×B×D` 字节 |
| `p2` | 单个 index 队列 `H×B×J` 字节 |
| `p3` | H，元素数 |
| `p4` | bufferNum，队列份数 |
| `p5` | batchMerge 标志，仅指定类型组合消费 |

schId=0；`numTileDataSize=N`，`unsortedDimParallel=C_act`，`sortLoopTimes=rounds`。BF16 的 FP32 转换区计入 `castBytes`；输出适配所需的运行保留和额外物理区已参与上面的模式及容量决策。

## 5 演算与检查

取 M=37、C=8、N=65、a=32、float16/int64，H=96，B_capacity=21，B=5，rounds=1。第 7 核处理最后 2 行。整批 E=480，Sort32 repeat=15，原有合批物理区为 `480×[2×(4+8)+20]=21120` 字节；Vector 写回后的总需求为 `21120+adaptBytes`，与查询到的 U 比较，R_run=0。尾核只处理 2 行，使用逐行 proposal，分配仍覆盖 480 项。

检查 H=32、跨 32 的尾 run、单/双缓冲、合批与尾批回退，以及两个输出索引宽度。计划来源：[Sort Host](../../example/sort/op_host/arch35/sort_tiling_arch35.cpp) 的 `ComputeMergeSortPlan`、`FillMergeSortInfo`、`SetMergeSortTiling`；计算流见 [Kernel](kernel-template.md)。
