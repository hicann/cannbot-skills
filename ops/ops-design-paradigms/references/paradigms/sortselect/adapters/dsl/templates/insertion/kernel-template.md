# DSL 插入排序 Kernel

## 1 布局与任务

编译参数 W 是 UB 容量行宽，B 是批次行数，实际轴长 N≤W。核 c 按 `batchId=c+k×blocks` 处理批次；有效行数 `v=min(B,M−batchId×B)`。公开包装已规范化 inner=1。

## 2 连续路径

Scalar 管理批次和 64 位 GM 基址，DMA 将本批 vN 个原值连续搬入 UB。64 位存储使用字节视图，不发生数值转换。UB 容量仍为 BW，实际使用紧凑 vN 布局，行 b 的基址是 bN。

```text
for batchId=core; batchId*B<M; batchId+=blocks:
  rowStart=batchId*B; v=min(B,M−rowStart)
  DMA x[rowStart*N : (rowStart+v)*N] → values[0:v*N]
  1024 线程 SIMT VF:
    每个有效线程 b 独立生成本行 keys 和 uint32 ids
    for j=1..N−1:
      暂存 value/key/id，逐项移动严格更大的键及配对 value/id，插回暂存项
    syncthreads
    全体线程按 flat=t,t+1024,...<v*N 连续写回 values 和 int64 ids
```

行内坐标、循环计数使用 int32；地址使用 int64。每行只有一个线程生成和消费 keys/ids，不需要加载后的协作 barrier；排序与合作写回之间保留 barrier，所有线程都参加。下一批 DMA 覆盖前的流水依赖由 DSL Buffer 同步处理，不使用忽略同步的裸拷贝。

键包含 descending 转换，比较始终按无符号键升序执行。严格大于才移动，因此同值项、正负零及 NaN 保持输入次序；写回保存的原值，保留位模式。

## 3 低层跨步路径

inner≠1 的低层直调保留协作加载、逐行插入、协作写回的 SIMT 路径，UB 行距仍为 W。地址为 `(r//I×N+j)×I+r%I`。三个 barrier 分别保护完整行加载、排序结果可读及下一批覆盖；无有效数据的线程仍参加。公开调用先转置后排序，不进入该路径。

## 4 区间输出

生成 L<N 时，排序、键和索引初始化仍覆盖 N 项。写回线程按 vL 个输出映射至 `bN+first_rank+q`，GM 目标为 `rL+q`，不能把排序容量或比较循环上限改成 L。L=1 仍须完成输入 N 项的排序。

来源：[InsertionSort](../../example/sort/sort_small_axis_insertion.py)，键规则见[局部 Radix](../../local-radix.md#1-保序键和有效字节)。
