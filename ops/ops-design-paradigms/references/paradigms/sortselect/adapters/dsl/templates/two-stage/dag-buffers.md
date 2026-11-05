# DSL 两阶段排序 Buffer

## 1 UB 分配

W=align256(BN)，S 是键存储字节数。局部 Radix 共用缓冲为：

| 区域 | 字节数 | 内容 |
|---|---|---|
| ping / pong | 2WS | 保序键或第二阶段紧凑键 |
| ping_ids / pong_ids | 8W | uint32 批内原位置或 Stage1 名次 |
| permutation | 4W | 排序位置→原位置；逆映射分支改作 inverse |
| digits、byte_order | 2W | 当前字节与稳定分组排列 |
| scratch | W+VECTOR_BYTES | 字节排列临时空间 |
| offsets | 512 | 256 个 uint16 局部桶起点 |
| stage1_ids | 4W | 紧凑键阶段保存 Stage1 的原位置 |

线性部分总计 `W(2S+19)`；固定区、额外向量尾区及对齐由 Host 的 1024 字节余量覆盖，并与实际编译布局核对。SIMT 32768 字节已在 U 中扣除。

## 2 各阶段的索引含义

```text
Stage1：ping_ids/pong_ids = 原始 f=bN+j
inverse：permutation[f] = Stage1 rank
packed：stage1_ids[r] = 原始 f；第二阶段 ping_ids/pong_ids = Stage1 rank r
最终：indices = f%N
```

切换视图前必须结束前一用途的消费者。第二阶段仍需要 stage1_ids，不能因同为 uint32 而覆盖它；原输入持续存活到最终回读完成。

## 3 峰值与 GM

整个模板只在 UB 保留当前批次，最终值从原输入按原轴索引回读；没有独立 GM 全批排序结果。公开编译签名的其他 workspace 参数只需合法最小占位。

最终 M×L 输出及实际转置区另计。L=1 或只输出一个字段不会缩减 BN 的第一阶段及恢复容量。当前实现两种分支都分配 stage1_ids，不能凭该表将某区从实际预算中静默删掉。

来源：[SmallAxisTwoStage.__init__](../../example/sort/sort_small_axis_two_stage.py)，批次容量和键范围见 [Tiling](tiling.md)。

## 4 阶段角色与预算余量

源码声明总量为 `W(2S+19)+VECTOR_BYTES+512`；Host 用 `W(2S+19)+1024` 做候选检查。沿用 W=4096、S=4 的算例，当前 VECTOR_BYTES=256 时两者分别为 111360/111616 字节。实际对齐及向量宽度变化需同步核对固定项。

| 阶段 | 两侧 keys/ids | permutation | stage1_ids |
|---|---|---|---|
| Stage1 LSD | 方向化 key / 批内原位置 f | 当前字节排列 | 已分配未使用 |
| inverse | 最终 ids 提供 f | 改作 inverse，barrier 后扫描 | 仍预留 |
| packed 构键 | ping 改作 key2，ping_ids 改作 Stage1 rank | 前轮用途结束后供 Stage2 复用 | 保存 Stage1 f |
| Stage2 / 写回 | ping_ids 为 Stage1 rank 的排列 | 排列完成后无消费者 | 供 f%N 和原值回读 |

原输入持续存活到恢复 VF 的最后一次回读完成。上述静态区不会因某分支当前未使用而自动释放。
