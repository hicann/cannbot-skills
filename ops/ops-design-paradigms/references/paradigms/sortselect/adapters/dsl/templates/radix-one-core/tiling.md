# DSL 单核 Radix Tiling

## 1 输入与容量

M 行、每行 N 项，C 为有效向量核数，U 为扣除 32768 字节 SIMT 保留后的 UB。S 为键存储字节数：int64/uint64 为 8，其余为 4。局部计数为 uint16，物理宽度 `W=align256(N)`。

小轴路线未选中后按下式判断本候选：

```text
W≤65280
required = W×(6 if D==1 else 12 if D==2 else 2S+14)+1024
required≤U
```

D 为输入字节数。1/2 字节使用局部排列特化，32/64 位使用键和索引双缓冲；不满足则进入多核 Radix。系数 6、12、2S+14 来自[实际 DSL Buffer](dag-buffers.md)，局部 Radix 使用静态 Buffer 预算，不查询原生高阶 Sort scratch。

## 2 编译与核分配

```text
width=W，groups=1，方向和 dtype 为编译参数
tasks=M
blocks=min(C,M)
核 c 处理 row=c, c+blocks, c+2blocks, ... < M
```

groups 必须为 1，一次只让一行驻留 UB；任何不等于 1 的 groups 在编译入口拒绝。多行由本核循环逐行处理，不将它们展平后交给单行模板。

## 3 演算

N=513、float32 时 W=768、S=4，required=`768×22+1024=17920` 字节。int64 相同 N 的 required=`768×30+1024=24064`。分别与真实 U 比较；W=65536 即使 UB 足够，也超过本模板的 uint16 计数域。

来源：[PreparedSort](../../example/sort/entry.py) 的单核候选分支。完整计算见 [Kernel](kernel-template.md)，共用字节排序见[局部 Radix](../../local-radix.md)。
