---
type: CATLASS DSL Programming Concept
title: Tensor、layout 与本地内存
description: Tensor 视图、layout 标签、地址空间及 C310 本地内存容量的源码约束。
tags: [catlass-dsl, tensor, layout, memory, c310]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
  - {by: process:catlass-dsl-source-audit, at: '2026-08-29T00:00:00Z'}
sources:
  - id: address
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/address_space.py
    title: Address-space definitions
  - id: arch
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/base_dsl/arch.py
    title: Architecture and local-memory metadata
  - id: tensor
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/tla/tensor.py
    title: Tensor implementation
  - id: core
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: Public layout and capacity query API
operator_families: [elementwise, matmul]
arch: [c310]
---

# 接口与概念

`Tensor` 同时携带 shape、dtype、origin shape、coord、stride 与 layout tag；
`tile_view` 创建子视图，`make_tensor_like` 让本地 tensor 复用已有视图的逻辑
描述。[^tensor]

指针地址空间是 `generic`、`gm`、`l1`、`l0a`、`l0b`、`l0c`、`ub`。[^address]
C310 源码容量表给出 L1/cbuf 512 KiB、L0A 64 KiB、L0B 64 KiB、
L0C 256 KiB、UB 248 KiB。[^arch]

公共查询是 `tla.arch.get_capacity_in_bytes(tla.AddressSpace.<scope>)`；参数必须是
`AddressSpace.l1/l0a/l0b/l0c/ub`，不能传旧式 `tla.arch.L1` pipe token。该函数在 Host
直接返回 Python int，在 kernel 内于 trace 期折叠为编译期常量。[^arch][^core]

# 用法

## Tensor 可用信息

kernel 参数和 `tile_view` 返回的 tensor 提供 `shape`、`dtype`、`ptr`、
`addrspace`、`layout_tag` 等结构信息；UB tensor 在 `tla.vec.func` 中使用：

```python
values = ub_tile.load(params=None)
ub_tile.store(values, params=None, mask=None)
offset_ptr = ub_tile.ptr + 8  # 按元素偏移，不是按 byte 偏移
```

`load(params=None)` 返回 `VectorSSA`；使用 `MaskLoadParams` 时返回 `MaskSSA`。
`store(value, params=None, *, mask=None)` 支持 vector predicate store。两者只接受
UB tensor，并且必须位于 `tla.vec.func`。[^tensor]

## Layout 构造

```python
row = tla.make_layout(
    tla.make_shape(rows, cols),
    tla.make_stride(cols, 1),
)
column = tla.make_layout(
    tla.make_shape(rows, cols),
    tla.make_stride(1, rows),
)
gm = tla.make_tensor(mem.ptr, row, tla.make_coord(0, 0))
tile = tla.tile_view(gm, tla.make_shape(16, 32), tla.make_coord(tile_m, tile_n))

ptr = tla.allocate(16 * 32, tla.Float32, tla.AddressSpace.ub, 256)
ub = tla.make_tensor_like(ptr, tile, tla.arch.RowMajor)
```

# 代码模式

## 地址空间与容量

| 地址空间 | C310 容量 | 常见用途 |
| --- | ---: | --- |
| `gm` | 外部设备内存 | kernel 输入输出 |
| `l1` | 512 KiB | Cube 输入 staging |
| `l0a` | 64 KiB | MMAD A operand |
| `l0b` | 64 KiB | MMAD B operand |
| `l0c` | 256 KiB | MMAD accumulator |
| `ub` | 248 KiB | Vector 数据与中间结果 |

容量按物理驻留量计算：

```text
tile_bytes = product(physical_shape) * dtype_bytes
resident_bytes =
    tile_bytes * buffer_count
    + temporary_tensor_bytes
    + alignment_and_layout_padding
```

例如三个 `float32[16, 32]` UB tensor 各占 2,048 bytes，共 6,144 bytes；
双缓冲则为 12,288 bytes。`allocate(shape, dtype, mem_scope, byte_alignment)`
要求 shape 完全静态，内部记录的 `size_bytes` 是元素数乘 dtype bytes。[^tensor]

Packed Cube layout 使用 `tla.arch.zN`、`nZ`、`zZ`、`nN`、`zNUnAlign` 或
`L0Clayout`；不能使用逻辑 M/N/K 元素数替代其物理占用核算。

# 约束

- AIV 使用 `aiv.c310`，AIC/Cube 使用 `aic.c310`。[^arch]
- L0A/L0B/L0C 分别服务 MMAD 操作数与累加结果，不能当作任意通用缓存。
- coord、stride 与 layout 必须描述同一物理存储，否则逻辑 shape 正确也会读错。
- `make_tensor_like` 的 `dst_dtype` 已 deprecated；通过 typed pointer 指定 dtype。
- `allocate` 不接受 `generic` 或 `gm`，alignment 必须是正整数。
- 公共 layout 名严格为 `tla.arch.RowMajor` / `tla.arch.ColumnMajor`；不要使用旧式
  `row_major` / `column_major`。

# 失败表现

- `expected on-chip AddressSpace`：把 `gm/generic` 传给 `allocate`。
- `load/store ... expected addrspace ub`：在 L1/L0/GM tensor 上调用 vector
  register load/store。
- `expected tla.make_layout`：把普通 tuple 当 layout 传入。
- layout/stride 不一致更常表现为结果稳定错位，而不是立即异常。

# 验证方法

静态核对 `get_localmem_capacity_bytes`，对 tensor dump TLAIR，并以非对称 shape
和非默认 layout 构造正确性 oracle。本文未在设备上验证容量边界。

[^address]: 固定提交中的 `AddressSpace` 枚举。
[^arch]: 固定提交中的 C310 target 与 `LOCALMEM_CAPACITY_BYTES`。
[^tensor]: 固定提交中的 tensor 类型与视图实现。
[^core]: 固定提交中的公共 layout token 与 `arch.get_capacity_in_bytes` 参数检查。
