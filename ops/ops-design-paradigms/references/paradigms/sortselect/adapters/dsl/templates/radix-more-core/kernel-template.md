# DSL 多核 Radix Kernel

## 1 执行层次

单次 launch 内逐批完成稳定 LSD，D 为输入元素字节数，K=ceil(N/T)，v 为本批有效行数。workspace 使用批内行号，输入和最终输出使用全局行号。

```text
for row_start in [0,M), step=P_act:
  clear 本批全部 D 轮的 lookback 状态与全局桶起点
  global_sync_all
  for byte in [0,D):
    Histogram：构造 digit → SIMD histogram → 发布频数、原子累加桶起点
    global_sync_all
    Scatter：重载 digit → 稳定局部 permutation → lookback → 成对散射原值/id
    若有下一 digit：global_sync_all
  若有下一批：global_sync_all
```

每批 digit 内为 `2D−1` 次全核屏障，另有一次批首初始化屏障；批间复用另计。屏障在任务分支外，所有启动核以相同次数参加。最后一轮直接恢复最终值和 int64 原轴索引，不增加恢复 kernel。

## 2 统计及构键

Histogram 与 Scatter 均按 `task=core+k×blocks` 遍历 `v×K` 个任务，row=task//K、tile=task%K、start=tile×T，valid=min(T,N−start)。

每轮连续输入均用 `cb.mem_copy` 搬入 raw_tile：首轮来自 x，后续轮来自当前原值 ping/pong。GM 中间值与输入同 dtype，不保存归一化键；原值位模式与原轴索引始终一起重排。

`prepare_digits` 为 Histogram 和 Scatter 共用入口。连续路径调用 `extract_value_digit`，每个 SIMD 循环处理 256 项，并同时生成 digit 与桶独占起点。`load_digit_word` 在寄存器内按原 dtype 生成当前字节：浮点归一 ±0/NaN 比较键，有符号整数翻转符号位，降序反射；这些转换每轮都从原值重新执行，不改写 raw_tile。64 位原值先从 uint32 视图选高/低字；1/2/4 字节分别以对应无符号寄存器表示，随后移位及 DeInterleave 压紧到 256 个字节。

低层跨步路径首轮按原布局读取 x，后续轮读取紧凑的中间原值，使用 `ordered_key/radix_digit` 生成 digit，再用共享 `get_exclusive_sum` 统计。所有路径尾部 digit 填 255，只统计真实 valid 项；统计阶段不生成局部排列。256 桶分别得到：

```text
SIMD 生成 localStart[0..255] = 本 tile 各桶的独占起点
256 线程 SIMT，每线程 d 负责一个桶：
  end = d<255 ? localStart[d+1] : valid
  frequency[d] = end−localStart[d]
  L2Store(hist[(row×K+tile)×256+d], CounterT(frequency[d]))
  atomic_add(globalBucketBase[row,byte,d], CounterT(localStart[d]))
  threadfence
```

全行的 localStart 之和正好是全局桶起点。当前按每 tile 的 256 桶分别累加，没有先对本核多个 tile 聚合；两种 CounterT 均用 SIMT 原子，不宣称已移植 AscendC 的 32 位 DMA 原子分支。频数发布、原子累加结束后全部核汇合。

## 3 无等待 lookback

排列阶段调用 `prepare_digits`，随后复用 `eight_bits_sort/local_sort`。多核调用 `local_sort(..., pack_digits=True)`，在原 uint32 排列项中同时携带原位置与桶号：`packed[p]=j+(d<<16)`。局部宽度不超过 65280，j 占低 16 位，d 占高 16 位；没有截断全局原轴索引。其他模板默认关闭此选项，仍返回纯局部排列。复用已有寄存器 Interleave 生成该编码，不增加 UB buffer，Scatter 一次读取即可解码，省去 `digits[j]` 的随机读取。
随后在一个 256 线程 SIMT VF 中，每线程负责一个桶，独立读取前驱 tile 的状态。0 表示未发布，`−inclusive−1` 表示该 tile 及此前全部 tile 的同桶包含式前缀。负编码避免宽计数在 N 为 int64 最大值时需要表示正数 N+1。

```text
inclusive = hist[currentTile,d]
previous = currentTile−1
while previous≥0:
  cached = L2Load(state[row,byte,previous,d])
  if cached≠0:
    inclusive += −(cached+1)
    结束回溯
  else:
    inclusive += hist[previous,d]
    previous -= 1
L2Store(state[row,byte,currentTile,d], −inclusive−1)
threadfence
```

未发布时读取统计屏障前已完成的频数，**不等待前驱核**。因此不依赖 tile 的实际调度先后，也不轮询尚未调度的工作；读取范围有限，最坏回退扫描此前全部 tile。已发布状态仅缓存相同的整数前缀，命中与否不改变数值和稳定顺序。它与 AscendC 的 aggregate/prefix-ready 等待协议不同；共同点是复用前驱完整前缀，避免独立逐行 Prefix 阶段。

频数区每轮复用；状态和全局桶起点为各 digit 保留独立切片。先清零全部切片再启动生产者，不能将上一轮/上一批的状态误当本轮前缀。状态的 L2 读写、生产 VF 的 threadfence 和跨轮/批次屏障均保留。

## 4 稳定散射与输出

lookback 的同一 256 线程 VF 同时生成 UB scatter bases 并以 threadfence 发布；随后另一个 1024 线程 SIMT VF 处理有效排列项，两个 VF 的 UB 生产消费依赖保留：

```text
base[d] = globalBucketBase[d] + inclusive[d] − frequency[d] − localStart[d]
j = packed[p] % 65536
d = packed[p] // 65536
rank = base[d] + p
originalAxis = 首轮 ? start+j : currentIds[row×N+start+j]
```

`p−localStart[d]` 是当前 tile 的桶内秩，因此 rank 等于“全行较小桶数 + 前驱 tile 同桶数 + 当前桶内秩”。连续路径从 raw_tile[j] 取得原值，后续轮另将 ids DMA 到 index_tile 后按 j 读取；首轮 originalAxis=start+j。低层跨步路径首轮按原布局读取 x，后续轮从紧凑原值/id 工作区读取。

非末轮将原值与 originalAxis 成对散射到另一侧；末轮直接将当前原值写到最终名次，并写 int64 原轴索引。原值逐轮保存，不需要按 originalAxis 再次回读 x，NaN payload 和零符号保持原位模式。派生输出只在末轮按 `first_rank≤rank<first_rank+L` 过滤，目标地址为 `globalRow×L+rank−first_rank`；此前仍排序完整 N 项，不能提前裁剪 digit、频数、排列或中间索引。

## 5 生命周期与验证

本核 permutation、digits 和 localStart 保留至本 tile Scatter 完成，才能处理下一 tile。UB scatter bases 在准备 VF 完成后由散射 VF 消费。不同核可写同一缓存行的不同目标，跨核读写继续使用已核对的 L2 接口与 threadfence，不能仅凭 destination 唯一删除可见性要求。

每个 digit 的全部 Scatter 完成后，下一轮才能覆盖 hist 或读新 ping/pong；本批末轮全部消费者结束后才复用状态。尾批无任务的核仍参加屏障。

验证覆盖 K>blocks、单核和多核、v<P_act、单元素尾 tile、32/64 位计数、非尾轴、同地址重跑与更新输入。`test_more_core_digit_preparation_boundaries` 另覆盖尾 tile 的 255/256/257 项边界、连续原值写回、桶号打包及低层跨步原值读取。宽计数小输入实例化不能替代 N≥2^30 的真实规模证据。性能包含完整排序、恢复和实际转置，不能仅以减少阶段数推断收益。

来源：[RadixMoreCore](../../example/sort/sort_radix_more_core.py)；资源与地址布局见 [Buffer](dag-buffers.md)、[Tiling](tiling.md)。

## 6 地址、轮次与尾 tile 核对

令 r=row_start+row、I=inner，原输入和完整 Sort 输出的逻辑地址是 `g(r,j)=(r//I)×N×I+r%I+j×I`。公开调用 I=1；低层 strided 的原值回读、输出写入用 g，GM 中间原值/id 始终按批内 row×N+j 紧凑寻址，与 I 无关。

| 字节轮 | digit 的输入 | Scatter 原值/索引来源 | 非末轮目的 |
|---|---|---|---|
| 0 | 原输入 x；不读未初始化 ping | 连续路径按 j 读 raw_tile，跨步路径读 x；索引直接初始化 | pong / pong_ids |
| 奇数轮 | pong 原值 | pong / pong_ids | ping / ping_ids |
| 后续偶数轮 | ping 原值 | ping / ping_ids | pong / pong_ids |

末轮一律直接写 values/indices，不再写另一侧中间原值/id，不能把“D 偶数最终读 ping_ids”的单核规则套到本模板。D=1 时 GM 两侧原值/id 没有有效读写，但当前 Host 仍完整分配。

沿用 M=v=2、N=16385、T=2048、K=9、blocks=8 的算例，task17 是第二行的尾 tile，属于核 1，start=16384、valid=1。假定本轮该元素 digit=7、全局较小桶数为 4，前驱 tile 的同桶频数合计为 2：

```text
当前 localStart[7]=0，frequency[7]=1
inclusive[7]=3；发布 state=-4
base[7]=4+3−1−0=6；p=0 得到本行 rank=6
```

以 byte=0 为例，hist 元素下标为 `(1×9+8)×256+7=4359`，states 的当前 tile 下标为 `((0×2+1)×10+8)×256+7=4615`，全局桶起点在 `((0×2+1)×10+9)×256+7=4871`。这些都是对应 CounterT Tensor 的元素下标，不是字节偏移。若前驱已有 prefix=2，则缓存编码为 −3；未发布时回退频数也得到同一 inclusive=3。非末轮将原值/id 写到下一侧 `1×16385+6=16391`，末轮才按输出名次写最终字段。
