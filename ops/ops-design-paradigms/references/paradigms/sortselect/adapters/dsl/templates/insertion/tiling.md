# DSL 插入排序 Tiling

## 1 参数与选路

输入已规范化为连续 M 行、每行 N 项，C 为当前流有效向量核数。D 为输入值字节数，S 为保序键存储字节数：int64/uint64 取 8，其他类型取 4。先从总 UB 扣除 SIMT 32768 字节，得到 U。

按 [dtype 路由表](../../tiling.md#dtype-路由表)查首个覆盖 N 的 `(n,p)`，要求 `P=ceil(M/C)≥p`；两阶段先尝试，未选中才尝试插入。经验门槛和容量约束分别检查。

## 2 行宽与 UB 批次

插入 UB 的容量行宽 W 取该 dtype 最后一档的轴长上限 Ncap，不是 N，也不是 align256(N)。连续执行路径在这份容量内按实际 N 紧凑存放；低层跨步路径按 W 存放。每行同时保留原值、保序键和 uint32 索引：

```text
W = Ncap
B = min(1024, floor(U/[W×(D+S+4)]))
Q = ceil(M/B)
blocks = min(C,Q)
```

1024 同时对应本模板的行负责人线程数和批次上限，保证一个线程最多负责一行；B=0 时在编译前报告容量不足。B 由本模板的 UB 预算与线程数共同约束，P 仅作为前面的选路指标。

## 3 编译与运行参数

| 类别 | 参数 |
|---|---|
| 编译期 | dtype、route=插入、width=W、groups=B、descending |
| 运行时 | rows=M、length=N、inner=1、blocks |
| 任务映射 | 核 c 处理 batchId=c+k×blocks，尾批 `v=min(B,M−batchId×B)` |

公开入口只把布局规范化后的 inner=1 交给排序主体。低层模板保留原布局的 inner 参数，实际地址公式见 [Kernel](kernel-template.md)。

## 4 可复算例子

取 float32、N=5、M=1000、C=32、U=65536。P=32 未达到浮点两阶段的门槛 64，可继续尝试插入。Ncap=8、D=4、S=4，每行 96 字节，B=min(1024,682)=682，Q=2、blocks=2，最后一批 318 行。

检查宽度 N 与 Ncap 区别、尾批、相等键和两方向，以及容量刚好容纳一行与不足一行。来源：[entry.py](../../example/sort/entry.py) 的 `PreparedSort` 插入分支；内存表见 [Buffer](dag-buffers.md)。
