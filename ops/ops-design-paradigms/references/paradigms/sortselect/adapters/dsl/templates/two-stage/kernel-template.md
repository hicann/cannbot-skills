# DSL 两阶段排序 Kernel

## 1 任务、原位置和第一阶段

核 c 遍历 `batchId=c+k×blocks`，`rowStart=batchId×B`，v 为有效行数，e=vN。将本批保序键按 `f=bN+j` 紧凑写入 ping，`ping_ids[f]=f`，用[局部 LSD](../../local-radix.md)对完整 e 项排序。

初始排序循环次数由原 dtype 的字节数确定。D 次字节轮后，D 偶数的输出在 ping，D 奇数在 pong。Stage1 输出索引含义始终是批内原位置 f，不能提前改为原轴 j。

## 2 逆映射恢复

```text
并行：inverse[stage1_ids[rank]] = rank
syncthreads
并行处理 f=bN+j：
  rank = inverse[f]
  pos = count(inverse[bN+k] < rank，k∈[0,N))
  output[rowStart+b,pos] = 原输入[rowStart+b,j]
  indices[rowStart+b,pos] = j
```

该分支将 permutation 缓冲转作 inverse 表，构表完成后才能被所有线程扫描。通过原输入回读值保留零符号和 NaN payload。恢复复杂度为 O(vN²)，不用第二次排序。

## 3 紧凑键恢复

```text
stage1_ids[r] = 第一阶段的批内原位置
key2[r] = (stage1_ids[r]//N)×e+r
新的 ping_ids[r] = r
对 key2 的低两个字节做稳定 LSD，得到第二阶段排列 perm2
对最终位置 t：
  f = stage1_ids[perm2[t]]
  原轴索引 j = f%N
  行号 b = t//N
  从原输入[rowStart+b,j]回读 values 并输出 j
```

key2 载体仍为本模板的 uint32/uint64，但有效键只占 uint16 范围；第二阶段固定两轮、固定升序。stage1_ids 独立保存，不能在生成新键/新索引时丢失原位置。两轮后排列位于 ping_ids。

## 4 布局与区间写回

公开入口传 inner=1，原值 GM 地址为 `(rowStart+b)N+j`；低层 inner=I 时为 `((rowStart+b)//I×N+j)×I+(rowStart+b)%I`。

生成区间输出时，逆映射的 pos 或 Stage2 的 `t%N` 是本行排序名次。仅最终写回判断 `first_rank≤pos<first_rank+L`，输出偏移为 `(rowStart+b)L+pos−first_rank`；Stage1 和第二阶段恢复仍覆盖完整 e。

## 5 并行与同步

构键、建立 inverse/紧凑键以及恢复均按本批有效元素并行展开。局部 Radix 的各 VF 与 vstore/vload 依赖按[共用原语](../../local-radix.md#4-发布与存储依赖)处理；inverse 建表到扫描之间额外保留全块 barrier。不能把最终恢复改成外层逐行启动 VF。

验证 `[3,1,5]`、`[4,2,6]` 两行时，两分支应恢复 `[1,3,5,2,4,6]` 和 `[1,0,2,1,0,2]`。源码：[SmallAxisTwoStage](../../example/sort/sort_small_axis_two_stage.py)。

## 6 完整调用顺序

构造阶段按 W=align256(BN) 分配局部 Buffer，每批以 e=vN 作为有效长度：

```text
for batch=blockIdx; batch<Q; batch+=blocks:
  构键到 ping，ping_ids[f]=f
  Stage1：sort_local_keys(length=e,width=W,passes=D)
  选 D 奇偶对应的最终 ids 侧
  inverse：建 permutation[f]=rank → syncthreads → 行内计秩并写回
  packed：保存 stage1_ids → 构 key2 / 重置 ping_ids
          → sort_local_keys(passes=2) → 按 stage1_ids[ping_ids[t]] 恢复
```

packed 分支重新使用 ping 前，第一阶段原位置已保存到独立 stage1_ids，第一阶段键不再有消费者。两次 LSD 各自按轮数选择输出侧，第二阶段固定从 ping 开始、两轮后回到 ping，不能沿用第一阶段的 selector。

## 7 贯穿：最后一批的原位置与名次

沿用 [Tiling 算例](tiling.md#4-演算)。核 16 处理 batch=16/33/50/67；末批 rowStart=1072、e=W=4096。Stage1 四轮后在 ping，Stage2 两轮后也在 ping。

恢复位置 t=258 对应批内 b=1、行内 rank=2。若 `stage1_ids[ping_ids[258]]=263`，则该项原轴索引为 `263%256=7`，值从原输入 `(1072+1)×256+7` 回读。

完整 Sort 写入 `(1072+1)×256+2`；L=5、first_rank=0 时写入 `(1072+1)×5+2=5367`。最后恢复在本批有效元素中判断名次，或直接按 v×L gather；两种组织均保持原批次粒度。只输出 indices 时可省原值回读，索引恢复链仍保留。
