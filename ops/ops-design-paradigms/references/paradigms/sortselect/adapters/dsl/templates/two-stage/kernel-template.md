# DSL 两阶段排序 Kernel

## 1 任务、原位置和第一阶段

核 c 遍历 `batchId=c+k×blocks`，`rowStart=batchId×B`，v 为有效行数，e=vN。连续 float32/int32/uint32 用 DMA+SIMD 构键；其余构键用 1024 线程的 SIMT VF，线程 t 按 f=t+k×1024<e 遍历。令 b=f//N、j=f%N，以原输入生成方向化键，紧凑写入 ping，`ping_ids[f]=uint32(f)`，再调用[局部 LSD](../../local-radix.md)对完整 e 项排序。

连续 32 位输入由 `load_keys32` DMA 搬入 ping 的原 dtype 视图，再原位 SIMD 构键并生成批内原位置；其他 dtype 与低层跨步输入仍通过 SIMT 从 GM 读取原值来构键。没有独立 Stage1 有序 values Buffer；Stage1 中间轮重排键和批内原位置，末轮只重排原位置。`sort_local_keys` 的每字节步骤是“SIMD 提取 digit → SIMD 字节 permutation → SIMD UB Gather 键和索引”，不调用原生 Sort。初始排序循环次数由原 dtype 的字节数确定。D 次字节轮后，D 偶数的输出在 ping，D 奇数在 pong。Stage1 输出索引含义始终是批内原位置 f，不能提前改为原轴 j。

## 2 逆映射恢复

```text
并行：f = D 偶数 ? ping_ids[rank] : pong_ids[rank]
      inverse[f] = rank    # 直接读取最终 ids
syncthreads
并行处理 f=bN+j：
  rank = inverse[f]
  pos = count(inverse[bN+k] < rank，k∈[0,N))
  output[rowStart+b,pos] = 原输入[rowStart+b,j]
  indices[rowStart+b,pos] = j
```

该分支用一个 1024 线程 SIMT VF 完成建表、全块 barrier 和恢复，线程按 rank 或 f 各遍历 e 个有效项。permutation 缓冲在 Stage1 字节 gather 的最后读取结束后转作 inverse 表。构表完成后才能被所有线程扫描。通过原输入回读值保留零符号和 NaN payload。恢复复杂度为 O(vN²)，不用第二次排序。行内比较以 8 个独立 int32 累加器处理完整的 8 项组，再单独处理余下 0..7 项；先合并累加器再加尾项，比较域仍为完整 N。

## 3 稳定行号分组

```text
1024 线程 SIMT 遍历 r<e：
  f = D 偶数 ? ping_ids[r] : pong_ids[r]
  ping[r] = f//N            # 仅行号，不再打包 rank
  ping_ids[r] = f           # 直接携带原展平索引
调用 sort_local_keys(passes=1, ids_only=True)
对最终位置 t：
  f = pong_ids[t]
  原轴索引 j = f%N；行号 b = t//N
  从原输入[rowStart+b,j]回读 values 并输出 j
```

Stage1 已按方向化数值键稳定排序。第二阶段仅按行号升序做稳定分组，等行号元素保持 Stage1 顺序，因此等价于按 `(行号, Stage1排名)` 排序，也保留相等值的原轴顺序。降序输入只改变 Stage1，行号分组始终升序。

公开入口限制 B≤256，行号最大 255，只需一个 8-bit digit；载体仍是 uint32/uint64。低层直接调用 B>256 时处理两个字节，两轮后的结果从 ping_ids 读取。第二阶段不依赖原 dtype 的字节数。

两阶段均设置 `ids_only=True`：最后一个 digit 仅重排原位置，不写后续不再读取的键。中间 digit 仍同步重排键与索引；该选项不能用于还要消费最终有序键的 caller。

## 4 布局与区间写回

公开入口传 inner=1，原值 GM 地址为 `(rowStart+b)N+j`；低层 inner=I 时统一为 `g(r,j)=(r//I×N+j)×I+r%I`，r=rowStart+b。构键和最终原值回读都按此映射，UB 始终紧凑使用 f=bN+j，不把 GM 跨度 I 带入局部排序。完整 Sort 输出值/索引按同一 g(r,rank) 写回；索引的数值仍为原轴 j。

生成区间输出时，在规范化布局中，逆映射的 pos 或 Stage2 的 `t%N` 是本行排序名次。仅最终写回判断 `first_rank≤pos<first_rank+L`，输出偏移为 `(rowStart+b)L+pos−first_rank`；Stage1 和第二阶段恢复仍覆盖完整 e。

## 5 并行与同步

除连续 32 位输入的 DMA+SIMD 构键外，构键、建立 inverse/行号键以及恢复均以 1024 线程按本批有效元素并行展开；这与局部 Radix 内部的 SIMD 排列独立，不改变局部排序原语。两个分支都通过 SIMT 直接恢复 GM 输出，不添加合批 DMA 输出区；逆映射的 O(vN²) 扫描与 行号分组的第二次稳定排序各按选定分支完整执行。局部 Radix 的各 VF 与 vstore/vload 依赖按[共用原语](../../local-radix.md#4-发布与存储依赖)处理；inverse 建表到扫描之间额外保留全块 barrier。不能把最终恢复改成外层逐行启动 VF。

验证 `[3,1,5]`、`[4,2,6]` 两行时，两分支应恢复 `[1,3,5,2,4,6]` 和 `[1,0,2,1,0,2]`。源码：[SmallAxisTwoStage](../../example/sort/sort_small_axis_two_stage.py)。

## 6 完整调用顺序

构造阶段按 W=align256(BN) 分配局部 Buffer，每批以 e=vN 作为有效长度：

```text
for batch=blockIdx; batch<Q; batch+=blocks:
  连续 32 位输入 DMA+SIMD 构键，其余 SIMT 构键；ping_ids[f]=f，原值继续保留在 GM
  Stage1：sort_local_keys(length=e,width=W,passes=D,ids_only=True)
  选 D 奇偶对应的最终 ids 侧
  inverse：SIMT 建 permutation[f]=rank → syncthreads
           → 行内计秩 → 从原 GM 读 values 并写回
  row-group：构 key2=f//N，ping_ids 携带 f
             → sort_local_keys(passes=1, ids_only=True)
             → SIMT 按 pong_ids[t] 恢复原轴索引并回读 GM 值
```

第二阶段覆盖 ping 键时，Stage1 的键已无消费者；原位置通过 ids 直接传递。两次 LSD 各自按轮数选择输出侧，不沿用第一阶段 selector。稳定行号分组对应 `ROUTE_TWO_STAGE_GROUPED`（5）。

## 7 贯穿：最后一批的原位置与名次

沿用 [Tiling 算例](tiling.md#4-演算)。核 16 处理 batch=16/33/50/67；末批 rowStart=1072、e=W=4096。Stage1 四轮后在 ping，Stage2 一轮后索引在 pong。

恢复位置 t=258 对应批内 b=1、行内 rank=2。若 `pong_ids[258]=263`，则该项原轴索引为 `263%256=7`，值从原输入 `(1072+1)×256+7` 回读。

完整 Sort 写入 `(1072+1)×256+2`；L=5、first_rank=0 时写入 `(1072+1)×5+2=5367`。最后恢复在本批有效元素中判断名次，或直接按 v×L gather；两种组织均保持原批次粒度。只输出 indices 时可省原值回读，索引恢复链仍保留。
