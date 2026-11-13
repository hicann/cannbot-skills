# DSL 单核 Radix Buffer

## 1 32/64 位的常驻 UB

W=align256(N)，S 为键存储字节数。

长度为 E 的区用 `cb.Buffer(cb.MemLoc.UB, dtype, cb.make_layout((E,)))` 分配。键区用无符号键 dtype，ids/permutation 用 uint32，digits/byte_order 用 uint8，offsets 用 uint16；E 分别取下表对应的元素数，不能把字节数直接用作宽元素区的元素数。

| 区域 | 字节数 |
|---|---|
| ping / pong 键 | 2WS |
| ping_ids / pong_ids | 8W |
| permutation | 4W |
| digits / byte_order | 2W |
| 256 个 uint16 offsets | 512 |

线性部分为 `W(2S+14)`，Host 使用额外 1024 字节覆盖固定区和当前编译布局的余量。SIMT 32768 字节单独从设备 UB 扣除；局部 uint16 计数要求 W≤65280。

## 2 存活与复用

同一字节轮中，当前键/索引是只读端，另一侧是输出端，不能将 ping/pong 合并。permutation、digits、byte_order 在每轮按生产消费顺序复用。

32/64 位路径最终原值从 GM 回读，因此 UB 不需要两份原 dtype values；但原输入须存活到最后恢复。最终输出不承接本模板的中间键，L 不影响 UB 完整排序容量。

本模板无完整 GM 排序 scratch；运行签名的其他参数使用合法最小占位。分配来源：[RadixOneCore.__init__](../../example/sort/sort_radix_one_core.py)，候选判断见 [Tiling](tiling.md)。

## 3 重排与恢复的存活

| 阶段 | 同时存活的内容 | 最后消费者 |
|---|---|---|
| 字节排列 | 当前键/ids、digits、permutation 与分组辅助区 | 本轮 keys/ids gather |
| 字节 gather | 源侧与目的侧 keys/ids | 本轮重排完成后交换角色 |
| 最终恢复 | 最终 ids、原输入 | 最后一次按索引回读原值 |

两侧是同一排序轮次的源和目标，不能原地乱序覆盖。最终恢复结束后整套区才可供下一行使用；静态分配不会因某阶段只读 ids 而自动减少。

## 4 一/二字节局部排列

`NarrowRadixOneCore` 保持稳定 LSD，只改变局部表示。共同区为 digits(W)、first_order(4W)、byte_order(W)、offsets(512)，预算 `6W+1024`。一字节只产生一次排列；恢复整数值时反射方向并撤销符号位变换，按原 dtype bitcast，变换为双射。

两字节增加原始 values(2W) 与 second_order(4W)，预算 `12W+1024`。先按低字节得到 first_order，再按该排列读取原值、生成高字节并排序，最终原位置为 `first_order[second_order[p]]`。两个排列保留到最终恢复；原值保留原 dtype 位模式，不以归一化键反推浮点 payload。

上述两字节路径在连续行用 DMA 搬入原始值，跨步直调用 SIMT gather；最后从 original_values 按组合索引恢复，不增加最终的 GM 原值回读。一字节没有 original_values，digits 保留到整数反变换结束。索引仍为 int64 原轴坐标；所有容量与存活包括原始值、两份排列和固定区。

| 窄类型阶段 | 同时存活 | 最后消费者 |
|---|---|---|
| D=1 排列后 | digits、first_order | 输出按 first_order 读原 digit，撤销方向/符号位变换 |
| D=2 高字节构造 | 原值、first_order；digits 可覆盖低字节 | 按 first_order 读取原值形成高 digit |
| D=2 输出 | 原值、first_order、second_order | 组合 A[B[p]] 并读取原始 UB 值 |

这些区按当前构造函数独立声明，不将排列与原值合并或把两个存活排列相互别名。
