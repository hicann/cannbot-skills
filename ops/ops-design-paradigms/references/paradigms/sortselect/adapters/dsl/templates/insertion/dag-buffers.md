# DSL 插入排序 Buffer

## 1 常驻 UB

W=Ncap，B 为批次行数，D 为值字节数，S 为键存储字节数。所有区在同一批次同时存活。

| Buffer | 字节数 | 内容 |
|---|---|---|
| values | BWD | 原始值及插入后的有序值 |
| keys | BWS | 按目标方向比较的无符号键 |
| ids | 4BW | uint32 原轴索引 |

总量 `BW(D+S+4)≤U`，U 已扣除 SIMT 保留，不能再将保留区计作可分配空间。各 Buffer 的编译布局对齐仍需符合实际 DSL 分配。

## 2 生命周期

```text
合作写入 values/keys/ids → barrier → 行负责人插入
→ barrier → 合作写出 values/ids → barrier → 下一批加载
```

keys 只控制比较，values 保留原始位模式，ids 一直随元素移动。写回结束前不能将 values 或 ids 改作下一批输入。

## 3 GM 与输出

该模板不需要完整 GM 排序 scratch；公开编译签名中的 workspace 参数使用合法的最小占位。最终 M×L 输出与输入/输出布局转换区单独计算。

W 由 dtype 上限确定，N 由真实输入确定，L 由输出配置确定；减小 L 不改变 W 或完整排序容量。来源：[InsertionSort.__init__](../../example/sort/sort_small_axis_insertion.py)，批次演算见 [Tiling](tiling.md)。

## 4 尾批与输出裁剪

尾批仍按 BW 静态分配三块 UB，只对 vN 项加载和排序。三个 barrier 分别保护完整行加载、排序结果可读以及写回读者结束；没有排序行的线程也参与同步。

values 与 keys 同时存活，因为输出原值不能由归一化键恢复；ids 随键和值移动。L 仅改变最终写回范围，不缩减这些排序区。若改变模板以省去某字段，应按真实消费者重新核算，不能只从输出名称推断可删区。
