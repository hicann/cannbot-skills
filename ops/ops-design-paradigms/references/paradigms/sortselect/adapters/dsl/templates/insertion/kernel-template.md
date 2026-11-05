# DSL 插入排序 Kernel

## 1 布局与任务

编译参数 W 为 UB 行跨度，B 为批次行数。核 c 按 `batchId=c+k×blocks` 处理批次；行起点 `r0=batchId×B`，有效行数 v。

对行 r 和原轴 j，低层原布局的 GM 元素地址为 `(r//I×N+j)×I+r%I`，I 是原排序轴后的元素数。公开调用已规范化 I=1，因此输入地址为 rN+j。UB 地址为 `(r−r0)×W+j`。

## 2 加载、行内排序、写回

```text
SIMT 1024 个线程合作搬入 v×N 项
  values[bW+j] = 原值
  keys[bW+j] = 按 dtype/方向构造的保序键
  ids[bW+j] = j
syncthreads
线程 tid<v 时负责第 tid 行：
  对 j=1..N−1 执行稳定插入
  只移动 keys[pos−1] > savedKey 的前项
  value / key / id 一起移动
syncthreads
全体线程合作写出 v×N 项
syncthreads，之后才覆盖下一批 UB
```

键已包含 descending 转换，插入比较始终按无符号键升序执行。严格大于才移动，因此同值项、同键正负零及 NaN 保持输入次序；最终写的是保留的原值，不由键反算。

## 3 区间输出与线程参与

生成 L<N 时，合作写回工作域改为 v×L，每个 t 读取 `first_rank+t` 处的值/原轴索引，输出 GM 为 `r×L+t`。比较循环仍排序 N 项。L=1 仍在行间并行，不新增逐行 VF 调用。

无效尾行只跳过数据访问，不跳过本 VF 的三个 barrier。当前模板在一次 VF 内加载、排序、写回多批次；保留“读者结束后才覆盖”的批末 barrier。

键规则见[局部 Radix 与保序键](../../local-radix.md#1-保序键和有效字节)；这里复用键变换，不调用 Radix。源码：[InsertionSort](../../example/sort/sort_small_axis_insertion.py)。
