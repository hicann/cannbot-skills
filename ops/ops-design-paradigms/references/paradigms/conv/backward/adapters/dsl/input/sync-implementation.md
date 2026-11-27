# input 同步与初始化

先按 [通用 DAG](../../../generic/input/compute-flow.md) 明确谁写、谁读、何时能覆盖，再让 DSL 的 buffer 访问表达这些依赖。Ascend C 的 TQue/SetFlag 不需要逐行翻译成同名操作；DSL 自动同步与显式事件的边界以 [公共能力映射](../capability-mapping.md) 为准。

## 1 槽轮换与数据生命周期

| 对象 | 写入 → 读取 | 覆盖前必须完成 |
|---|---|---|
| L1A/L1B | GM→L1 的 MTE2 → Load3D/Load2D 的 MTE1 | 旧父段的最后一次 L0 搬入。 |
| L0A/L0B | MTE1 → MMAD | 使用该槽的 MMAD 读取。 |
| L0C | MMAD → Fixpipe | 旧输出的 Fixpipe 读取。 |
| bias L1/BT | MTE2 → MTE1 → MMAD | 旧 bias 的最后一次消费者。 |

多槽用 `cbd.RingBuffer(..., depth=...)` 或显式地址的 `cbd.make_ring_buffer`，单份资源用 `cbd.Buffer`。`next_write()` / `next_read()` 各自推进槽游标；它们只选地址，不清零、不完成搬运，也不是运行时队列的 EnQue/DeQue。真实的 tensor 读写供编译器建立同步。

取得父段 `next_read()` 的结果后，应保存这个 view 给该父段的多个 L0 子窗口使用。不能每取一个 K 子块就再调用 `next_read()`，否则会提前切到下一父段。读写游标在不同条件分支里推进时，检查每条路径消费的槽序列一致；尾轮没有预取的父段，也不能再消费一个读槽。

开双缓冲只提供两份存储。要重叠搬运和计算，还需按原流水提前提交下一父段的搬入，再执行当前段的消费者。若所有搬入都紧跟本轮等待之后，两个槽仍可能串行。L1 的显式 bank 分配和容量检查见 [Buffer 与类型](../buffer-and-types.md)。

自动同步已覆盖的核内依赖不叠加全核 barrier。L1 插零初始化与紧随其后的 MTE2 搬入有同流水顺序要求，见 [指令映射](instruction-mapping.md)。仅遇到编译器无法表达的依赖时，才针对该边界加入显式事件，并检查生成物。

## 2 l1-resident 的驻留身份

L1B 的驻留范围取决于原模板的 N/K 覆盖、Dk、group 和权重地址。完整父段的 view 保留到复用域结束；下一个 M 块只更新 L0B 的局部窗口，不再次 `next_write()` 覆盖驻留数据。

保留原条件：完整父段覆盖当前任务 N/K，`bl1Pbuffer=1`；跨任务复用还要满足原模板的 `dk=1`、单组及当前 full-load 状态。权重起点或身份变化时结束旧复用域，再装载新父段。不能仅根据 shape 相同跨组复用，也不能每次 M 变化都强制重搬。bias 有独立身份与生命周期，见 [BT 通路](bias-bt-path.md)。

## 3 输出预清零与无贡献区域

保留 [前置处理](../../../generic/preprocessing.md) 和 Tiling 的 `initOutputFlag` 判断。这里的 `L0_INIT` 是历史枚举名，其效果是主计算前清零 GM 输出；它与 MMAD 的 `init=True` 是不同动作。

采用 [融合清零实现](../initialization.md)：AIV 分片清零 GM，经 MTE3 mode 0 汇合后，以 MTE3 mode 2 通知 AIC；AIC 在首次写回前执行 FIXPIPE mode 2 等待。清零与计算在同一个 MIX kernel 内完成。

`initOutputFlag` 同时控制清零、发事件和等事件，所有核取值一致。GM 依赖不能只靠片上 Buffer 自动同步保证；核验生成的 MIX 入口、参与者、事件模式与流水。该通知每次调用只消费一次，不放进输出块循环。

无 bias 的无贡献位置由初始化协议保证为零；带 bias 的无贡献位置仍须写 bias，清零不是最终结果。主计算中没有工作的核，可以跳过局部搬运与 MMAD；若参加融合初始化或其他全核协议，则不能提前跳过相应同步点。

## 4 看生成代码时检查什么

检查父段覆盖、L0 槽回绕和 Fixpipe 之后的重用是否有对应依赖；检查同一次数据交接是否被重复插入等待。比较首轮、稳态与尾轮，不只看中间一轮。出现等待不结束时，先核对实际发出了几次写入/事件、消费了几次，而不是增加更多 `sync_all`。
