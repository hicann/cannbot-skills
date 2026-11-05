# RingBuffer 与 Buffer：明确槽位选择和实际访问

`RingBuffer` 是带独立读写游标的片上槽位池。`next_write()` 选择写游标对应的 Tensor 并推进写游标，`next_read()` 对读游标做同样的事。选择本身不读写数据、不等待，也不检查队列满空；返回值是共享存储的别名。

## 核心写法：先选择，再对 Tensor 操作

以下为设备 DSL 结构伪码；示例每轮各选择一次写槽位与读槽位，编译器分析后续操作的资源访问。

```text
from cannbotdsl import RingBuffer, MemLoc, dtypes, make_layout, mem_copy

tiles = RingBuffer(MemLoc.UB, dtypes.float32,
                   make_layout((TILE_ROWS, TILE_COLS)), depth=DEPTH)
for task in 本核任务:
    write_tile = tiles.next_write()
    mem_copy(write_tile, 当前任务的合法 GM 输入视图)

    read_tile = tiles.next_read()
    对 read_tile 执行计算
    mem_copy(当前任务的合法 GM 输出视图, read_tile)
```

`next_write()` 与 `next_read()` 从各自的零游标开始推进。实现需要保证任务、游标次数和实际访问对应；不能把一次 `next_read()` 当作等待数据的动作。一个已选 Tensor 的多个视图仍指向同一槽位，后续槽位复用会覆盖其内容。

## 槽位、视图与容量

- `depth` 必须是正的编译期 Python 整数，按容量与在途数据的存活区间选择。
- 跨核属性由实际访问推导，不传 `kind`；让实际读写操作可被编译器分析，手工同步的作用域见 [同步](synchronization.md)。
- 先 `next_write()/next_read()` 取得 Tensor，再调用 `tile_slice` 或 `reinterpret`。创建视图不会再次推进游标。
- 游标推进必须在 VF 外完成，再把选出的 Tensor 传入 VF。把选择放到寄存器循环内会被拒绝。
- 动态逻辑 shape 从静态分配派生视图，并保证访问范围不超出分配；见 [Workspace](workspace.md)。

`Buffer(MemLoc.UB, dtype, make_layout(shape))` 声明单个片上 Tensor，适合私有临时量。显式空间视图需要绑定时，`make_buffer(view)` 与 `make_ring_buffer(views)` 接收未绑定的静态片上 space 视图；后者各槽的 dtype、shape、stride 和 layout 必须一致，不能传入已有 Buffer 转移所有权。不要同时在同一内存空间混用互斥的自动分配与显式空间管理方式。

**反例：** 保存 `next_write()` 的结果并认为它保存了数据快照；只推进游标却认为已经发布事件；消费前一任务却使用当前任务的尾块尺寸；把 RingBuffer 直接传给要求 Tensor 的视图 API。

**检查：** 0、1、`DEPTH + 1` 个任务和尾块；流水线启动与排空后读写游标次数一致；重复复用时任务索引、有效范围和数据对应。资源复用安全以实际访问与依赖为依据。
