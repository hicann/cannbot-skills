# Workspace：容量按物理布局算，复用按生命周期算

这里管理计算临时存储。用于分轮、分核和任务描述的调度记录见 [AICPU metadata](aicpu-load-balance-metadata.md)；调度协议不放在 workspace 中重复定义。

## 核心公式：带行 padding 的循环槽位

假定每个槽位显式分配完整的 `rows × row_stride` 存储，列连续，`row_stride >= cols`，包括最后一行的 padding。

```python
def workspace_bytes(rows, cols, row_stride, item_bytes, depth, alignment_bytes):
    if rows < 0 or cols < 0 or row_stride < cols:
        raise ValueError("invalid physical shape")
    if item_bytes <= 0 or depth <= 0 or alignment_bytes <= 0:
        raise ValueError("invalid item size, depth, or alignment")
    raw_slot_bytes = rows * row_stride * item_bytes
    slot_bytes = ((raw_slot_bytes + alignment_bytes - 1) // alignment_bytes) * alignment_bytes
    return depth * slot_bytes
```

槽位 `s` 的字节起点为 `s * slot_bytes`。分配基址也要满足对齐；仅将每个槽位尺寸对齐还不够。该公式不适用于任意非连续视图或分块格式。Host 计算结果在传给固定宽度接口前检查溢出与可用容量。

## 区分 GM 临时张量与片上容量

Host 用 Tensor 分配 GM workspace，并在编译入口为每个必要参数声明 `TensorSpec`。设备内的 `Buffer/Channel` 由片上资源规划管理；二者不能共用一个“总 workspace 字节数”替代各自容量检查。

```text
from cannbotdsl import Buffer, Channel, MemLoc, dtypes

# 在设备 DSL 中，valid_rows 是合法的运行时 i64 尺寸
scratch = Buffer(MemLoc.UB, (valid_rows, TILE_COLS), dtypes.float32,
                 capacity=(MAX_TILE_ROWS, TILE_COLS))
tiles = Channel(MemLoc.L1, (valid_rows, TILE_COLS), dtypes.float16,
                capacity=(MAX_TILE_ROWS, TILE_COLS), depth=DEPTH)
```

`capacity` 是正的编译期整数元组，与逻辑 shape 同 rank；动态 shape 使用静态 backing stride。调用方保证 `0 < valid_rows <= MAX_TILE_ROWS`，空任务走显式空路径；`capacity` 不会自动截断超限访问。

Host 可用 `get_mem_size("ub")`、`get_mem_size("l1")` 等查询单层片上容量，返回单位为字节；不可用时返回 0，应停止该配置的规划。评估占用须计入同时存活的全部资源、Channel 各槽位和布局对齐，查询结果不等于当前剩余可分配字节数。

## 按核 workspace 使用实际启动数量

需要每个 block 独立的临时空间时，Host 按本次 `core_num` 分配，编译描述允许该维度在支持范围内变化：

```text
# 编译描述；elements_per_block 是静态配置
core_count = Dim("core_count", min=1, max=MAX_BLOCK_COUNT)
scratch_spec = TensorSpec((core_count, elements_per_block), dtypes.float32)

# Host 分配；同一个 core_num 继续传给 @host 的 launch 参数
scratch = torch.empty((core_num, elements_per_block), dtype=torch.float32, device=x.device)
```

动态 shape 的 `Dim` 与独立的启动标量不会因名字相同就自动建立相等约束；Host 保证 `scratch.shape[0] == core_num`。只在实现需要按核存储时采用此结构，核数来源见 [编译与启动](compilation-launch.md)。

## 复用条件：最后读者已完成

```text
阶段 A 写 scratch → 阶段 B 读 scratch → B 的最后读取完成
                                            ↓
                               阶段 C 才能覆盖同一存储
```

同一 stream 的有序执行可以建立相应依赖；多 stream 或多角色使用时需要额外证据。运行时提前返回不代表设备已经结束，资源释放须遵循当前执行环境的异步生命周期约定。

**反例：** 使用 `rows * cols` 给带 padding 的搬运分配存储；用总任务数分配本可循环复用的私有槽位；按峰值在途槽位分配后却没有复用同步；不同阶段或不同 stream 同时写同一空间。

**检查：** 最大 shape、尾块、对齐边界、多次调用和并发调用；确认每次读取前有效元素已初始化。零填充只用于算法需要的区域，不用整块清零掩盖未写输出。
