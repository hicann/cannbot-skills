# AscendC 插入排序 Tiling

## 1 输入、适用条件与输出

输入已规范化为连续 `[M,N]`。M 是独立行数，N 是完整排序轴长，C 是可用向量核数，U 是查询到的 UB 字节数，D/J 分别是值和输出索引的字节数。`A(x)=ceil(x/a)×a`，a 为查询到的 UB block 字节数。

插入路线适合极短轴。先在[小轴路由表](../../tiling.md#小轴路由)按 dtype 找到首个覆盖 N 的档位，再检查 `P=ceil(M/C)`。档位是经验选路参数；容量、字段范围和搬运限制另行检查。

输出参数是每批行数 B、批次任务数 Q、使用核数 C_act；参考协议采用 sch5，`p0=B`、`p1=Q`、`numTileDataSize=N`。输出长度 L 只影响最后写回，中间排序仍按 N 项预算。

## 2 UB 切分

先扣除 SIMT 保留量 `R=32768` 字节，得到 `U_work=U−R`；U≤R 时该候选不可用。

普通类型每行预算：

```text
valueBytes = A(N×D)
indexBytes = A(N×J)
rowBytes = valueBytes + indexBytes
```

BF16 比较使用 FP32 转换区。先对齐 BF16 搬运区，再按同一行跨度计算 FP32 区，不能把浮点区简单写成未对齐的 `4N`：

```text
castBytes = A(2N)
castRowElems = castBytes / 2
rowBytes = castBytes + 4×castRowElems + A(N×J)
```

于是 `B_ub=floor(U_work/rowBytes)`。按参考实现取 `B=min(P,B_ub,4095)`；4095 是本路线采用的 DataCopy blockCount 上限。B 为 0 时继续尝试其他路线，不能启动零批次 Kernel。

## 3 多核切分与尾批

```text
Q = ceil(M/B)
C_act = min(C,Q)
w = ceil(Q/C_act)
核 c 处理 batchId ∈ [c×w, min(Q,(c+1)×w))
rowStart = batchId×B
validRows = min(B, M−rowStart)
```

任务按连续批次区间分配，最后一个批次只读写 validRows 行。部分核的起始批次可已达到 Q，此时不处理数据；这与两阶段排序的轮转映射不同。每行 value/index 的 UB pitch 分别按各自类型对齐，GM 行距仍为 N；生成区间输出时，输出 GM 行距为 L。

## 4 整体切分伪码

```text
检查 M/N/C、dtype、输出索引类型和整数乘积
应用 dtype 档位，未命中则返回“不选此路线”
计算 U_work 和各物理 buffer 的行字节数
B = min(ceil(M/C), floor(U_work/rowBytes), 4095)
若 B<1：返回“不选此路线”
Q = ceil(M/B); C_act = min(C,Q)
提交 sch5、B、Q、N、C_act 和扣除保留区后的局部内存大小
```

## 5 算术演算与检查

取 int32 输入、M=103、C=16、N=8、J=8、a=32、U=65536。P=7，未达到该 dtype 两阶段首档的行数门槛 8，但满足插入门槛 2。每行 `32+64=96` 字节，`B_ub=341`，因此 B=7、Q=15、C_act=15；最后一批 5 行。仅对比容量公式时，BF16 同轴长的每行需求为 `32+64+64=160` 字节，其选路仍要另查浮点档位。

连续批次映射可用容量受限的 B=3 情况检查：此时 Q=35、C_act=16、w=3，核 1 处理批次 3/4/5，核 11 处理 33/34，核 12..15 无有效批次。

检查 N 为档位边界前后、M 不能被 B 整除、索引 int32/int64、BF16 转换和不足一行 UB 的情况。算法与地址见 [Kernel](kernel-template.md)，物理分配见 [Buffer](dag-buffers.md)。

源码核对：`ComputeInsertionBytesPerSeg`、`EstimateSmallAxisInsertionBatching`、`FillSmallAxisBatched`，位于 [Sort Host](../../example/sort/op_host/arch35/sort_tiling_arch35.cpp)。
