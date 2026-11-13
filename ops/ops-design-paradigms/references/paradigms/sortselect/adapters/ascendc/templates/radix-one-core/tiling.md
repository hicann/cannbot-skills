# AscendC 单核 Radix Tiling

## 1 输入与适用域

排序轴已连续化，每行 N 项、共 M 行，每个核一次处理一行。C 是可用核数，D/J 是值/输出索引字节数，U 是 UB，a 是 UB block 字节数，`A(x)=ceil(x/a)×a`。

本候选在小轴和归并未选中后尝试，要求当前 `isInt32` 标志非零，原生 Sort 的隐式源索引模式可用，且完整 N 项驻留 UB。这个标志不等于最终输出索引类型；输出仍可为 int64。

## 2 容量推导与双缓冲

本模板的排序、恢复及最终区间选择保持 Vector/Reg，经输出 Queue 由 MTE3 写回，`R_run=0`、`U_work=U`。先按[完整调用链预算](../../output-mapping.md#scratch-与-ub-由-host-规划)计入新增压紧区等 `adaptBytes`，再选择单双缓冲。

```text
X = A(ND)
I32 = A(4N)
Y = I32×(J/4)                  # int64 输出将索引 Queue 扩为两倍
halfIndex = I32/4              # 单位 uint32 元素，不是字节
Z = I32（浮点 sourceOrder）或 0（整数）
tmp = 查询原 dtype、N 项、隐式源索引 Radix Sort 的 scratch
Qbytes = 2X+Y                  # 输入值、输出值、输出索引
```

先检查 `Qbytes+Z+tmp+adaptBytes≤U_work`，不满足则尝试多核路线。满足后检查 `2Qbytes+Z+tmp+adaptBytes≤U_work`，可行时 bufferNum=2，否则为 1。剩余 UB 从 U_work 中扣除已计物理区后按 block 向下对齐，再与 tmp 比较；Z 和 tmp 不随队列份数倍增。区间压紧新增物理区时，可驻留轴长与 bufferNum 由同一组不等式重新求得。

## 3 多核分配

```text
C_act = min(M,C)
rounds = ceil(M/C_act)
核 c 在第 k 轮处理 row = k×C_act+c
若 row≥M：该轮无数据
inputOffset = row×N
outputOffset = row×L
```

本路线不在一行内部拆核，也不把多行展平后进行一次全局排序。并行来自多行；UB 不能容纳完整 N 时，不能擅自只排若干 chunk 再拼接结果。

## 4 下发参数

| 参考字段 | 内容 |
|---|---|
| schId | 1 |
| p0 / p1 | X / Y，字节 |
| p2 | halfIndex，uint32 元素偏移 |
| p3 | bufferNum=1/2 |
| `tmpUbSize` | tmp，字节 |
| `numTileDataSize` | N，元素 |
| `unsortedDimParallel` / `sortLoopTimes` | C_act / rounds |

## 5 容量演算

取 N=513、float32/int64、a=32，X=2080、I32=2080、Y=4160、Z=2080，Qbytes=8320。单缓冲物理区需求 `10400+tmp`，双缓冲需求 `18720+tmp`。若假设查询结果 tmp=4096、adaptBytes=0，Vector 路线为 14496 / 22816 字节，R_run=0；新增区间压紧区按 adaptBytes 另加。此 tmp 仅为演算输入，不替代 SDK 查询。

来源：[Sort Host](../../example/sort/op_host/arch35/sort_tiling_arch35.cpp) 的 `IsRadixSortOneCore/SetSortTmpSize/SetRadixOneCoreTiling`。索引半区及恢复见 [Kernel](kernel-template.md)，完整峰值见 [Buffer](dag-buffers.md)。
