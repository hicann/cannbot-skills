# DSL 多核 Radix Kernel

## 1 单次 launch 的阶段顺序

```text
for byte in 0..D−1:
  Histogram：为全部行/tile 生成计数、块内秩以及首轮键/原轴索引
  global_sync_all
  Prefix：为全部行/tile/bucket 生成全局独占起点
  global_sync_all
  Scatter：稳定散射键/索引；末轮直接生成最终字段
  若非末轮：global_sync_all
```

全部已启动核以相同次数参加 barrier，屏障放在任务分支之外。偶数字节读 ping、奇数字节读 pong；首轮从 x 生成键，末轮按原轴索引回读 x。D 为输入 dtype 的有效字节数，与键载体 S、int64 索引宽度无关。

## 2 Histogram 与 ranks

任务 t 对应 row=t//K、tile=t%K、start=tile×T，`valid=min(T,N−start)`。合作填入当前 digit，补齐区设为最大 digit；[局部 permutation](../../local-radix.md#2-一字节的稳定-permutation)只对 valid 项计数和排列。

令排序位置为 p，`j=permutation[p]`，桶为 d，tile 内桶起点为 localStart[d]：

```text
within = p−localStart[d]
ranks[row×N+start+p] = j×T+within
hist[(row×K+tile)×256+d] = 此 tile 的桶 d 频数
```

注意 ranks 的下标 p 是排序位置，编码中的 j 是原 tile 位置；二者不能互换。首轮同时生成 `key[rowN+start+j]` 和原轴索引 `start+j`。

## 3 Prefix 的 256 桶计算

一行由 256 个线程各负责一个桶。线程 d 先累加该行所有 tile 的 hist，得到 total[d]。每 32 lane 以 shuffle-up 距离 1、2、4、8、16 求 inclusive scan；8 个 warp 总数写入 UB 并 barrier 后可读。

```text
bucketBase[d] = sum(total[x], x<d)
before = bucketBase[d]
for tile in 0..K−1:
  offsets[(row×K+tile)×256+d] = before
  before += hist[(row×K+tile)×256+d]
```

warp 内扫描时全部参与 lane 执行 shuffle，再按 lane 选择是否累加；不能把 shuffle 放进只由部分 lane 执行的分支。32 lane 是当前 Prefix 实现的线程组宽度，不是其他排序模板的通用线程数。

## 4 Scatter 与最终名次

```text
从 ranks[rowN+start+p] 解码 j = encoded//T，within=encoded%T
从当前键[rowN+start+j] 取得桶 d
rank = offsets[(rowK+tile)×256+d] + within
原轴坐标 axis = 当前 ids[rowN+start+j]
非末轮：写另一侧 key[rowN+rank]、ids[rowN+rank]
末轮：values[row,rank]=x[row,axis]，indices[row,rank]=axis
```

生成区间输出时，只在末轮判断 `first_rank≤rank<first_rank+L`，输出地址 rowL+rank−first_rank；此前的键、indices、ranks、hist/offsets 都不裁剪。

## 5 可见性与复用

跨阶段 GM 数据使用 SIMT L2 load/store，生产 VF 结束前 `threadfence()` 发布，再由 `global_sync_all()` 建立全 launch 依赖。只加全核 barrier 不能省略实际的发布条件。

最后一轮没有下一 digit 消费中间数据，不再执行第三个全核屏障；D 字节共 `2D+(D−1)=3D−1` 次屏障。它们位于任务分支之外，无 Prefix 行任务的核也须参与。

当前全部 M 行的状态必须在各阶段间保留。若改为有界行批次，需要一起修改任务域、每核循环、workspace 行基址和全核 barrier 次数，并验证精度及性能；不得以 Host 逐批 launch 直接替换当前执行方式。

来源：[RadixMoreCore](../../example/sort/sort_radix_more_core.py) 的 `histogram/prefix/scatter`。计数和内存规模见 [Buffer](dag-buffers.md)。

## 6 贯穿：两个长行与单元素尾 tile

沿用 [Tiling 例子](tiling.md#5-演算)：M=2、N=16385、T=2048、K=9、tasks=18、blocks=8。

| 阶段 | 核 0 | 核 1 | 核 2..7 |
|---|---|---|---|
| Histogram / Scatter | task 0/8/16 | task 1/9/17 | 各处理 c、c+8 |
| Prefix | row 0 | row 1 | 无有效行，仍参与阶段屏障 |

task8 和 task17 分别是两行的最后一个 tile，valid=1；其余 tile 为 2048 项。Histogram 仅写有效 ranks，但仍覆盖该 task 的全部 256 桶 hist，空桶写 0。

task17 的唯一有效位置为 rowN+start=32769；局部 permutation、rank 编码均只涉及第 0 项，全局名次仍由该桶的 offsets 决定。该例同时检查任务域、单元素尾块及空闲 Prefix 核的同步参与。
