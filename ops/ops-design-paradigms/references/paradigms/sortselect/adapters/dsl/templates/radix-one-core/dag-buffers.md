# DSL 单核 Radix Buffer

## 1 常驻 UB

W=align256(N)，S 为键存储字节数。

| 区域 | 字节数 |
|---|---|
| ping / pong 键 | 2WS |
| ping_ids / pong_ids | 8W |
| permutation | 4W |
| digits / byte_order | 2W |
| scratch | W+VECTOR_BYTES |
| 256 个 uint16 offsets | 512 |

线性部分为 `W(2S+15)`，Host 使用额外 1024 字节覆盖固定区和当前编译布局的余量。SIMT 32768 字节单独从设备 UB 扣除；局部 uint16 计数要求 W≤65280。

## 2 存活与复用

同一字节轮中，当前键/索引是只读端，另一侧是输出端，不能将 ping/pong 合并。permutation、digits、byte_order、scratch 在每轮按生产消费顺序复用。

最终原值从 GM 回读，因此 UB 不需要两份原 dtype values；但原输入须存活到最后恢复。最终输出不承接本模板的中间键，L 不影响 UB 完整排序容量。

本模板无完整 GM 排序 scratch；运行签名的其他参数使用合法最小占位。分配来源：[RadixOneCore.__init__](../../example/sort/sort_radix_one_core.py)，候选判断见 [Tiling](tiling.md)。

## 3 重排与恢复的存活

| 阶段 | 同时存活的内容 | 最后消费者 |
|---|---|---|
| 字节排列 | 当前键/ids、digits、permutation 与分组辅助区 | 本轮 keys/ids gather |
| 字节 gather | 源侧与目的侧 keys/ids | 本轮重排完成后交换角色 |
| 最终恢复 | 最终 ids、原输入 | 最后一次按索引回读原值 |

两侧是同一排序轮次的源和目标，不能原地乱序覆盖。最终恢复结束后整套区才可供下一行使用；静态分配不会因某阶段只读 ids 而自动减少。
