# AscendC 归并排序 Tiling

## 1 路线与符号

输入为连续 M 行、每行 N 项，C 为可用核数，D/J 为值和输出索引字节数。U 为实际 UB，a 为 UB block 字节数，`A(x)=ceil(x/a)×a`。

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

第 c 核第 k 轮的起始行为 `rowStart=(k×C_act+c)×B`，`validRows=min(B,M−rowStart)`。每行完整在同一核内排序，跨行不做归并。

## 3 普通与合批 proposal

普通模式的两份 proposal 仅容纳 H 项，依次供本批各行复用。当前合批 proposal 只在 float16 输入、int64 输出索引、B≥4 时尝试：

```text
E = B×H
E ≤ 255×32 = 8160
required = E×[bufferNum×(2D+J)+4+2×8]
required ≤ U
```

8160 对应 Sort32 的 uint8 repeat 次数容量；每项 proposal 8 字节。满足时 `batchMerge=1`，否则保留逐行 proposal 模式。尾批实际行数不足 4 时 Kernel 仍走逐行模式，即使 Host 已允许合批。

## 4 Host/Kernel 参数

| 字段 | 含义 |
|---|---|
| `p0` | B，每次搬入的行数上限 |
| `p1` | 单个 value 队列 `H×B×D` 字节 |
| `p2` | 单个 index 队列 `H×B×J` 字节 |
| `p3` | H，元素数 |
| `p4` | bufferNum，队列份数 |
| `p5` | batchMerge 标志，仅指定类型组合消费 |

schId=0；`numTileDataSize=N`，`unsortedDimParallel=C_act`，`sortLoopTimes=rounds`。BF16 另加 FP32 转换区；该路线原有路径不统一扣除 SIMT 保留区，若输出适配新增 SIMT，则须重新预算。

## 5 演算与检查

取 M=37、C=8、N=65、a=32、float16/int64，H=96，B_capacity=21，B=5，rounds=1。第 7 核处理最后 2 行。整批 E=480，Sort32 repeat=15，合批 UB 为 `480×[2×(4+8)+20]=21120` 字节，须与查询到的 U 比较；尾核只处理 2 行，使用逐行 proposal。

检查 H=32、跨 32 的尾 run、单/双缓冲、合批与尾批回退，以及两个输出索引宽度。计划来源：[Sort Host](../../example/sort/op_host/arch35/sort_tiling_arch35.cpp) 的 `ComputeMergeSortPlan`、`FillMergeSortInfo`、`SetMergeSortTiling`；计算流见 [Kernel](kernel-template.md)。
