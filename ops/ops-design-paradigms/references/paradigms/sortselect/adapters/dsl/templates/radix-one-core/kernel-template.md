# DSL 单核 Radix Kernel

## 1 32/64 位一行的处理骨架

```text
for row = blockIdx; row<M; row+=blocks:
  连续 32 位输入：DMA 到 ping 的原 dtype 视图，SIMD 构键并生成 ids
  64 位或跨步输入：SIMT 合作加载 N 项，keys[j]=ordered_key(x[row,j])，ids[j]=j
  调用局部稳定 LSD，轮数为输入 dtype 的字节数 D
  D 偶数读 ping_ids，D 奇数读 pong_ids
  SIMT 合作：values[row,rank]=x[row,ids[rank]]
             indices[row,rank]=ids[rank]
```

本节对应 `RadixOneCore` 的 D=4/8 路径，连续 float32/int32/uint32 构键使用 `load_keys32` 的 DMA+SIMD 路径，64 位与跨步构键仍用 256 线程 SIMT；恢复保持 256 线程 SIMT，字节排序调用 `sort_local_keys`，不调用原生 Sort。D=1/2 走同文件的 `NarrowRadixOneCore`，见第 5 节。UB 按 W=align256(N) 分配，所有有效排序、计数和写回只使用 N。保序键、permutation 与每轮 ping/pong 关系见[共用原语](../../local-radix.md)。

## 2 GM 寻址与输出

公开入口内布局为连续 `[M,N]`，源地址 rN+j。低层接口 inner=I 时，行基址为 `(r//I)×N×I+r%I`，沿轴跨度为 I；保留这一映射可以检查直接调用，公开包装仍统一传 I=1。

最终索引始终是原轴 j。本节 32/64 位路径的 values 从原输入按 j 回读，连整数也不从最终 key 反算；原值读取结束前不能释放或覆盖布局规范化后的输入。1/2 字节路径的恢复来源见第 5 节，不能把 GM 回读套用到全部单核变体。

生成输出区间时，在规范化布局中令 rank=`first_rank+t`、目标地址 `rL+t`；本节 GM 原值源地址仍用 N 和原轴索引。只输出 indices 时可以省最终原值回读，完整 N 项的键/索引或窄类型排列仍不能缩到 L 项。

## 3 同步与核间独立性

digits 的 N..W−1 槽填最大字节，计数、permutation 和最终写回仍只使用 N。SIMD 内 byte_order 原位写后读保留 `vmem_bar("vst_vld")`，不能用 SIMT barrier 替代。各行之间没有数据依赖，不需要全核屏障。局部 VF、字节排列及其缓冲交接保留原来的存储依赖；下一行才能重用同一 UB。groups=1 是物理容量合同，不能只修改 groups 元数据来声称支持 UB 合批。

验证 N 非 256 整数倍、奇偶字节轮数、相等键、原索引回读以及同地址更新输入后重跑。源码：[RadixOneCore](../../example/sort/sort_radix_one_core.py)。

## 4 多字节手算检查

以升序 uint32 `[257,1,256,1]` 为例：最低字节为 `[1,1,0,1]`，一轮后值序列为 `[256,257,1,1]`、原索引 `[2,0,1,3]`；下一字节后为 `[1,1,256,257]`、索引 `[1,3,2,0]`。高两字节为零，后续稳定轮次保持顺序。该例检查低位先排、键和索引同移及重复值稳定性。

## 5 一/二字节稳定 LSD 特化

此分支实现 [generic 排列映射](../../../../generic/compute-flow.md#局部排列与原位置映射)，仍为同一单核 Radix 模板：

```text
D=1：构造 digit → 稳定排列 A → 原轴坐标 A[p] → 恢复值/索引
D=2：保存原始值、生成低字节 → 稳定排列 A
     按 A 读取原值，生成高字节 → 稳定排列 B
     原轴坐标 A[B[p]] → 从原始 UB 值恢复 values，写 int64 indices
```

两条窄路径的调用与保存内容如下，构造 digit 和最终恢复均用 256 线程的 SIMT VF：

```text
for row=core; row<M; row+=blocks:
  D=2 且 I=1：DMA 载入 N 项原值到 original_values
  SIMT 遍历 j<W：
    j<N 时从原输入或 original_values 构造最低字节；否则 digit[j]=255
    D=2 且 I≠1 时，同时将跨步读取的原值保存到 original_values[j]
  radix_permutation(digits, first_order, byte_order, offsets, N)
  if D=2:
    SIMT 遍历 p<W：有效项按 first_order[p] 读原始 UB 值，构造高字节
    radix_permutation(digits, second_order, byte_order, offsets, N)
  SIMT 遍历 p<N：
    j = D=1 ? first_order[p] : first_order[second_order[p]]
    D=1：raw=digit[j]；降序时 raw=255−raw；int8 时 raw XOR=128
         按原 dtype bitcast 写 values
    D=2：values=original_values[j]
    indices=int64(j)
```

完整 N 项参加每次排列；最终写回才选择输出区间。D=1 只支持本例的 int8/uint8，digit 在排列后仍保留原位置的方向化键字节，是整数反变换的最后读端，不能提前覆盖。D=2 的 first_order、second_order 与 original_values 保留到组合索引和原值恢复结束；后者保留 NaN payload 与零符号，不以归一化键反推。

连续两字节行用 `cb.mem_copy` 搬入原值，跨步直调用 SIMT gather；一字节路径只构造 digit，不分配原值 Buffer，也不做这次 DMA。各次排列继续使用共用 `radix_permutation` 的有效长度、finalize 和 VST/VLD 依赖。上述表示实现相同的稳定 LSD 映射，不改成按 AscendC 原生 Sort 的输入/输出缓冲推导容量。

例如 uint16 `[257,1,256,1]` 的低字节排列 A=`[2,0,1,3]`，按 A 构造的高字节为 `[1,1,0,0]`，其排列 B=`[2,3,0,1]`；组合 A[B] 得原轴索引 `[1,3,2,0]`。最终从原始 UB 值按这个组合索引取值，不将 B 直接当原轴坐标。
