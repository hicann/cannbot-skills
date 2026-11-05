# DSL 单核 Radix Kernel

## 1 一行的处理骨架

```text
for row = blockIdx; row<M; row+=blocks:
  SIMT 合作加载 N 项：keys[j]=ordered_key(x[row,j])，ids[j]=j
  调用局部稳定 LSD，轮数为输入 dtype 的字节数 D
  D 偶数读 ping_ids，D 奇数读 pong_ids
  SIMT 合作：values[row,rank]=x[row,ids[rank]]
             indices[row,rank]=ids[rank]
```

UB 按 W=align256(N) 分配，所有有效排序、计数和写回只使用 N。保序键、permutation 与每轮 ping/pong 关系见[共用原语](../../local-radix.md)。

## 2 GM 寻址与输出

公开入口内布局为连续 `[M,N]`，源地址 rN+j。低层接口 inner=I 时，行基址为 `(r//I)×N×I+r%I`，沿轴跨度为 I；保留这一映射可以检查直接调用，公开包装仍统一传 I=1。

最终索引始终是原轴 j，values 从原输入按 j 回读，保留原始位模式。原值读取结束前不能释放或覆盖布局规范化后的输入。

生成输出区间时，rank=`first_rank+t`、目标地址 `rL+t`；原值源地址仍用 N 和原轴索引。只输出 indices 时可以省最终原值回读，完整 LSD 键/索引仍不能缩到 L 项。

## 3 同步与核间独立性

digits 的 N..W−1 槽填最大字节，计数、permutation 和最终写回仍只使用 N。SIMD 内 scratch 写后读保留 `vmem_bar("vst_vld")`，不能用 SIMT barrier 替代。各行之间没有数据依赖，不需要全核屏障。局部 VF、字节排列及其缓冲交接保留原来的存储依赖；下一行才能重用同一 UB。groups=1 是物理容量合同，不能只修改 groups 元数据来声称支持 UB 合批。

验证 N 非 256 整数倍、奇偶字节轮数、相等键、原索引回读以及同地址更新输入后重跑。源码：[RadixOneCore](../../example/sort/sort_radix_one_core.py)。

## 4 多字节手算检查

以升序 uint32 `[257,1,256,1]` 为例：最低字节为 `[1,1,0,1]`，一轮后值序列为 `[256,257,1,1]`、原索引 `[2,0,1,3]`；下一字节后为 `[1,1,256,257]`、索引 `[1,3,2,0]`。高两字节为零，后续稳定轮次保持顺序。该例检查低位先排、键和索引同移及重复值稳定性。
