# 布局与尾块：逻辑范围和物理地址分别计算

同一 shape 可以对应不同 stride。以下结构只适用于有明确仿射 stride 的二维视图；ND/NZ 等分块格式须按其真实地址规则处理。

## 核心结构：按输入跨度取数，按有效范围写回

```text
前提：base 指向视图的第一个逻辑元素；stride 单位为元素
row_begin = row_tile * tile_rows
col_begin = col_tile * tile_cols
valid_rows = min(tile_rows, max(0, rows - row_begin))
valid_cols = min(tile_cols, max(0, cols - col_begin))

对 r in [0, tile_rows), c in [0, tile_cols):
    valid = r < valid_rows 且 c < valid_cols
    if valid:
        offset = (row_begin + r) * row_stride + (col_begin + c) * col_stride
        tile[r, c] = input[base + offset]
    else:
        tile[r, c] = 当前运算的填充值

计算后只将有效输出写回；输出地址使用 output 自己的 stride
```

如果底层拿到的是 storage 基址，还需要包含视图的 storage offset，不能重复加或漏加。若硬件搬运要求完整块，必须证明对应物理分配可访问，或选择支持尾部的路径；运算 mask 不能阻止先发生的越界读取。

## 视图与搬运使用不同接口

```text
from cannbotdsl import tile_slice, reinterpret, make_copy_engine, mem_copy

gm_tile = tile_slice(gm_input, (TILE_ROWS, TILE_COLS), (row_tile, col_tile))
# coord 是 tile 编号；实际尾块范围取 gm_tile.shape
slot = l1_tiles.produce()
nd_to_nz = make_copy_engine(format_transform="nd2nz")
mem_copy(slot, gm_tile, engine=nd_to_nz)  # slot 已声明为容量足够的 NZ L1 Tensor
```

- `tile_slice` 保留父视图的跨度，尾块报告实际 extent；调用方仍要保证 tile 坐标在合法范围，不能依赖所有路径都做越界检查。
- `reinterpret(..., offset=offset_bytes)` 的 offset 单位是**字节**，stride 单位是目标 dtype 的**元素**。它只重解释存储；改变 dtype 不做数值 cast，改变 `data_format` 不会转换原有数据。
- `permute(gm_tensor, dims)` 为 GM 的 Identity/ND Tensor 创建零拷贝视图，不是片上 Tensor 或 Channel 的通用转置接口。
- `make_copy_engine` 描述静态搬运方式；`mem_copy` 使用实际 Tensor 视图确定地址。UB 内的格式转换用 VF 实现，不能假定 DMA 支持所有方向的 ND/NZ 转换。
- 分块格式的内侧 tile 边界还需满足 fractal 对齐。使用 ND2NZ 的 `dst_nd_arrangement="stack_m"` 时，合并后的运行时 `batch * rows` 必须不超过目标静态容量，不能只检查单个矩阵的 rows。

| 运算 | 常用填充值 | 边界 |
|---|---|---|
| 求和 | 0 | 累加类型符合精度要求 |
| 最大值 | 负无穷或类型允许的最小值 | 全无效行须有明确定义 |
| 乘积 | 1 | 不应把填充元素算入计数 |

**反例：** 用逻辑 `cols` 作为带 padding 输入的行跨度；最大值归约尾部填 0，导致全负输入结果错误；只 mask 写回，却先完整读取越界尾块。

**检查：** 行跨度大于行宽、视图起点非零、单元素尾块、全负最大值输入、跨格式转换后逐元素的逻辑索引一致。Host 是否接受某种 stride 必须与 Kernel 真正支持的寻址一致。
