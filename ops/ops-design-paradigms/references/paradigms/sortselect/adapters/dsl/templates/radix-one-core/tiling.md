# DSL 单核 Radix Tiling

## 1 输入与容量

M 行、每行 N 项，C 为有效向量核数，U 为扣除 32768 字节 SIMT 保留后的 UB。S 为键存储字节数：int64/uint64 为 8，其余为 4。局部计数为 uint16，物理宽度 `W=align256(N)`。

小轴路线未选中后按下式判断本候选：

```text
W≤65280
required = W×(2S+15)+1024
required≤U
```

不满足则进入多核 Radix。2S+15 来自[实际 DSL Buffer](dag-buffers.md)，局部 Radix 使用静态 Buffer 预算，不查询原生高阶 Sort scratch。

## 2 编译与核分配

```text
width=W，groups=1，方向和 dtype 为编译参数
tasks=M
blocks=min(C,M)
核 c 处理 row=c, c+blocks, c+2blocks, ... < M
```

groups 必须为 1，一次只让一行驻留 UB；groups=0 或 >1 在编译入口拒绝。多行由本核循环逐行处理，不将它们展平后交给单行模板。

## 3 演算

N=513、float32 时 W=768、S=4，required=`768×23+1024=18688` 字节。int64 相同 N 的 required=`768×31+1024=24832`。分别与真实 U 比较；W=65536 即使 UB 足够，也超过本模板的 uint16 计数域。

来源：[PreparedSort](../../example/sort/entry.py) 的单核候选分支。完整计算见 [Kernel](kernel-template.md)，共用字节排序见[局部 Radix](../../local-radix.md)。
