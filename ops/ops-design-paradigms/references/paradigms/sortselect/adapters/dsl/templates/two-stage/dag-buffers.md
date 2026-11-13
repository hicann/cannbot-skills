# DSL 两阶段排序 Buffer

## 1 UB 分配

W=align256(BN)，S 是键存储字节数。局部 Radix 共用缓冲如下；本例 UB 中没有独立的 inputValues/stage1Values/finalValues 原值侧，也没有输入 Queue。

各区按 `cb.Buffer(cb.MemLoc.UB, dtype, cb.make_layout((E,)))` 声明。ping/pong 用无符号键 dtype；ids、permutation 用 uint32；字节区用 uint8；offsets 用 uint16。E 是对应区的元素容量，shape 通过 `make_layout` 转成显式 Layout。

| 区域 | 字节数 | 内容 |
|---|---|---|
| ping / pong | 2WS | 保序键或第二阶段行号键 |
| ping_ids / pong_ids | 8W | uint32 批内原位置 |
| permutation | 4W | 排序位置→原位置；逆映射分支改作 inverse |
| digits、byte_order | 2W | 当前字节与稳定分组排列 |
| offsets | 512 | 256 个 uint16 局部桶起点 |

线性部分总计 `W(2S+14)`；固定区及对齐由 Host 的 1024 字节余量覆盖，并与实际编译布局核对。SIMT 32768 字节已在 U 中扣除。

## 2 各阶段的索引含义

```text
Stage1：ping_ids/pong_ids = 原始 f=bN+j
inverse：permutation[f] = Stage1 rank
row-group：第二阶段 ping_ids/pong_ids 继续携带原始 f
最终：indices = f%N
```

切换视图前必须结束前一用途的消费者。第二阶段直接携带 f；原输入持续存活到最终回读完成。

## 3 峰值与 GM

整个模板只在 UB 保留当前批次，最终值从原输入按原轴索引回读；没有独立 GM 全批排序结果。公开编译签名的其他 workspace 参数只需合法最小占位。

最终 M×L 输出及实际转置区另计。L=1 或只输出一个字段不会缩减 BN 的第一阶段及恢复容量。

来源：[SmallAxisTwoStage.__init__](../../example/sort/sort_small_axis_two_stage.py)，批次容量和键范围见 [Tiling](tiling.md)。

## 4 阶段角色与预算余量

源码声明总量为 `W(2S+14)+512`；Host 用 `W(2S+14)+1024` 做候选检查。W=4096、S=4 时，两者分别为 90624/91136 字节；512 字节余量用于编译布局对齐。

| 阶段 | 两侧 keys/ids | permutation |
|---|---|---|
| Stage1 LSD | 方向化 key / 批内原位置 f | 当前字节排列 |
| inverse | 最终 ids 提供 f | 改作 inverse，barrier 后扫描 |
| row-group 构键 | ping 改作行号，ping_ids 携带 f | 前轮消费者结束后供 Stage2 复用 |
| Stage2 / 写回 | 最终 ids 提供原位置 f | 排列完成后无消费者 |

连续 32 位构键暂将 ping 解释为原 dtype 的输入 DMA 区，随后 SIMD 原位转为比较键，不增加独立原值区；原输入持续保留到最终恢复。
