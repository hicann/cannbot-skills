---
type: CATLASS DSL API Reference
title: CATLASS DSL Core API
description: Shape、coord、stride、layout、tensor、tile、copy、MMAD 与 vector API 的开发索引。
tags: [catlass-dsl, api, tensor, vector, cube]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
  - {by: process:catlass-dsl-source-audit, at: '2026-08-29T00:00:00Z'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/docs/en/api/kernel_api_reference.md
    title: Generated CATLASS DSL Kernel API Reference
  - id: core
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: CATLASS DSL Core API implementation
  - id: runtime
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/runtime.py
    title: CATLASS DSL runtime export surface
operator_families: [elementwise, matmul, reduction]
arch: [c310]
---

# 接口与概念

`catlass.core_api.__all__` 与 runtime 的 `_CORE_API_EXPORTS` 共同定义当前 Core
API 面。生成版 Kernel API 文档解释公共接口；下面的清单以源码导出表、实际签名和
参数检查为准。[^api][^core][^runtime]

## 完整 API 清单

### 结构、Tensor 与指针

| API | 签名或对象 | 用途 |
| --- | --- | --- |
| `make_shape` | `make_shape(*components)` | 构造可嵌套 `TlaShape` |
| `make_coord` | `make_coord(*components)` | 构造与 shape 同结构的坐标 |
| `make_stride` | `make_stride(*components)` | 构造物理 stride |
| `make_layout` | `make_layout(shape, stride, *, origin_shape=None, layoutTag=None)` | 组合 shape/stride/layout tag |
| `tile_view` | `tile_view(source, shape, coord)` | 按 tile 坐标裁剪 tensor view |
| `make_tensor` | `make_tensor(ptr, layout, coord=None)` | 从显式 pointer 与 layout 建 tensor |
| `make_tensor_like` | `make_tensor_like(ptr, like, layoutTag=None, dst_dtype=None)` | 复用另一个 view 的结构化元数据 |
| `make_ptr` | `make_ptr(dtype, value, mem_space=AddressSpace.gm, *, assumed_align=None)` | 整数地址转 typed TLA pointer |
| `allocate` | `allocate(shape, dtype, mem_scope, byte_alignment)` | 分配片上本地内存 |
| `recast_ptr` | `recast_ptr(ptr, *, dtype)` | 只改变 pointer 的逻辑 pointee dtype |
| `IndexTree` | 类型别名 | shape/coord/stride 的递归 index 结构 |
| `_Pointer` | pointer wrapper | 支持按元素 offset 的 pointer 运算 |

`make_tensor_like(..., dst_dtype=...)` 在源码中已标记 deprecated；新代码应让
`ptr` 自身携带正确 dtype。[^core]

### 搬运、打印与同步

| API | 签名 | 用途 |
| --- | --- | --- |
| `copy` | `copy(dst, src, params=None)` | GM/L1/L0/UB 间搬运 |
| `print` | `print(value)`、`print(format, *values)` 或 `print(tensor, length=None)` | region 内打印标量、格式化字符串或 tensor 前缀 |
| `flag` | `flag(name, src_pipe=None, dst_pipe=None)` | 创建同核 pipe flag |
| `set_flag` / `wait_flag` | `(...flag_value)` | 生产/消费 flag |
| `cross_flag` | `cross_flag(name, *, mode=2)` | 创建跨核 flag；mode 支持 0/1/2/4 |
| `cross_core_set_flag` / `cross_core_wait_flag` | `(flag, pipe, aiv_id=None)` | 在操作侧指定 pipe；mode 4 用 `aiv_id` 定位 AIV0/AIV1 |
| `pipe_barrier` | `pipe_barrier(pipe)` | 指定 pipe 的 barrier |
| `local_mem_bar` | `local_mem_bar(src, dst)`，参数为 `tla.params.MemType` | Vector region 内的本地内存依赖 barrier |
| `mutex` | `mutex(resource, id=-1)` | 创建语义资源 mutex |
| `mutex_guard` | `mutex_guard(*mutexes)` | 自动推断 body pipe 的上下文管理器 |
| `mutex_lock` / `mutex_unlock` | `(...mutex_value, *, pipe)` | `catlass.core_api` 的显式加解锁入口 |

通常通过 `mutex_value.lock(pipe=...)` 和 `mutex_value.unlock(pipe=...)` 使用显式
mutex；`mutex_lock`/`mutex_unlock` 在 `core_api.__all__` 中，但不在顶层 runtime
转发清单中。[^core]

### 控制流、region 与 Cube

| API | 签名 | 用途 |
| --- | --- | --- |
| `range` | `range(start, end=None, step=None)` | 动态 DSL 循环 |
| `range_constexpr` | `range_constexpr(start, end=None, step=None)` | 前端静态 Python 循环 |
| `cube` | `cube()` | `with tla.cube():` Cube region |
| `vector` | `vector()` | `with tla.vector():` Vector region |
| `vec.func` | `vec.func(*, mode="simd", thread_block_dim=None)` | SIMD vector register 或 SIMT thread region |
| `mmad` | `mmad(acc, lhs, rhs, init_c=None, unit_flag=None, compute_order=M_FIRST, hf32_mode=HF32_DISABLE)` | L0A x L0B 累加到 L0C |

`unit_flag` 仅接受 `0b00`、`0b10`、`0b11`。支持同型 f16/bf16/f32 输入到 f32、任意
f8e4m3fn/f8e5m2 配对到 f32，以及 i8×i8 到 i32；L0C dtype 由这条路由决定。
`compute_order` 接受 `ComputeOrder`，`hf32_mode` 只为 f32 输入选择 HF32 舍入模式。[^core]

### Vector 构造与算术

| API | 签名摘要 |
| --- | --- |
| `full` | `full(value, dtype)` |
| `arange` | `arange(base=0, *, order="increase", dtype=...)` |
| `add`, `sub`, `mul`, `max`, `min`, `div` | `(lhs, rhs, *, mask=None)`；支持 vector/vector 及受支持的 scalar 形式 |
| `exp`, `log`, `sqrt`, `abs`, `neg` | `(operand, *, mask=None)`；其中前四项支持受约束的 SIMT scalar 分派 |
| `where` | `where(mask, x, y)` |
| `squeeze` | `squeeze(src, mask)` |
| `interleave`, `deinterleave` | `(src0, src1) -> (VectorSSA, VectorSSA)` |
| `gather` | `gather(x, y, *, mask=None)`；从 UB tensor 按 vector index 读取 |
| `cmp` | `cmp(lhs, rhs, mode, *, mask=None) -> MaskSSA` |
| `bitwise_not` | `bitwise_not(operand, *, mask=None)` |
| `bitwise_and`, `bitwise_or`, `bitwise_xor` | `(src0_reg, src1_reg, *, mask=None)` |

### Mask、VectorSSA 与支持对象

| API/对象 | 成员 |
| --- | --- |
| `create_mask` | `create_mask(*, pattern, dtype=Float32) -> MaskSSA` |
| `update_mask` | `update_mask(true_shape, dtype=Float32) -> (MaskSSA, remaining)` |
| `mask` | `ALL`, `ALLF`, `VL1`, `VL2`, `VL3`, `VL4`, `VL8`, `VL16`, `VL32`, `VL64`, `VL128`, `M3`, `M4`, `H`, `Q` |
| `VectorSSA` | 算术运算符、`reduce(kind, *, mask, init_value=None, reduction_profile=None)`、`to(dst_type, params, mask=None)` |
| `MaskSSA` | vector predicate SSA wrapper |
| `ReductionOp` | reduction kind 枚举 |
| `vec` | 提供 `vec.func(mode="simd")` 与 `vec.func(mode="simt", thread_block_dim=...)` |
| `arch` | pipe、layout、memory scope 与 block index namespace |
| `LocalmemAllocator` | 片上内存 allocator |
| `TlaCoreAPIError` | 用户 API 前置条件异常 |
| `dsl_user_op` | Core API lowering 装饰器；算子代码通常不直接调用 |

`arch` 当前公开成员为 `CUBE`、`VECTOR`、`FIX`、`SCALAR`、`MTE1`、`MTE2`、
`MTE3`、`L1`、`L0A`、`L0B`、`L0C`、`UB`、`RowMajor`、`ColumnMajor`、
`zN`、`nZ`、`zZ`、`nN`、`zNUnAlign`、`L0Clayout`、`block_idx()`、
`sub_block_idx()`、`block_num()`、`thread_idx()`、`sync_threads()`、
`thread_block_dim()` 和 `get_capacity_in_bytes(mem_scope)`。thread 三项只在 SIMT
`vec.func` 内有效；容量查询接受 `tla.AddressSpace.l1/l0a/l0b/l0c/ub`，在 Host 或
kernel trace 期返回 Python 常量。[^api][^core]

# 用法

以下模式同时展示 layout、tensor、copy、tail mask 和 vector API：

```python
@tla.kernel
def add_one(mem_x: tla.Tensor, mem_y: tla.Tensor, count: tla.Int32):
    ready = tla.flag("ub_ready", tla.arch.MTE2, tla.arch.VECTOR)
    done = tla.flag("vector_done", tla.arch.VECTOR, tla.arch.MTE3)
    ptr_x = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    ptr_y = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    tile_shape = tla.make_shape(64)
    tile_coord = tla.make_coord(tla.arch.block_idx())
    gm_x = tla.tile_view(mem_x, tile_shape, tile_coord)
    gm_y = tla.tile_view(mem_y, tile_shape, tile_coord)
    ub_x = tla.make_tensor_like(ptr_x, gm_x, tla.arch.RowMajor)
    ub_y = tla.make_tensor_like(ptr_y, gm_y, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(ub_x, gm_x)
        tla.set_flag(ready)
        tla.wait_flag(ready)
        with tla.vec.func(mode="simd"):
            tail, _remaining = tla.update_mask(count, tla.Float32)
            result = tla.add(ub_x.load(), 1.0, mask=tail)
            ub_y.store(result, mask=tail)
        tla.set_flag(done)
        tla.wait_flag(done)
        tla.copy(gm_y, ub_y)
```

# 代码模式

Cube 计算的最小调用遵循 `L0A x L0B -> L0C`：

```python
with tla.cube():
    tla.copy(l0_a, l1_a)
    tla.copy(l0_b, l1_b)
    tla.mmad(
        l0_c, l0_a, l0_b,
        init_c=(k_tile == 0),
        unit_flag=0b11 if is_last_k else 0b00,
        compute_order=tla.params.ComputeOrder.M_FIRST,
        hf32_mode=tla.params.HF32Mode.HF32_DISABLE,
    )
```

Tail 循环应把 `update_mask` 返回的 remaining 作为 loop-carried 状态：

```python
    remaining = element_count
    for _ in tla.range(tiles):
        with tla.vec.func(mode="simd"):
            active, remaining = tla.update_mask(remaining, tla.Float32)
            values = inp.load()
            out.store(tla.exp(values, mask=active), mask=active)
```

# 约束

- `shape`、`coord`、`stride` 是结构化 index tree，嵌套深度必须匹配。
- vector 操作必须位于 `tla.vec.func`；mask 的 dtype 决定 lane 数并必须匹配 vector。
- `VectorSSA.to` 只支持有符号 i8/i16/i32/i64 与 f16/bf16/f32 目标类型，并要求
  `CastParams`。
- `gather` 的 source 必须是 UB tensor，index vector 必须是 i32。
- `mmad` 的 A/B/C tensor 地址空间、维度与 dtype 必须满足契约。
- `print`、`allocate`、同步和 region API 都是 lowering-only，不能当普通 Python
  eager 函数使用。
- `tla.Bool` 转为整数遵循 `False -> 0`、`True -> 1`；不要依赖早期错误实现产生的
  `True -> -1`。

# 失败表现

- `tla.<op> is only available in lowered Tla IR`：在 kernel/region 外调用。
- `expected ... TlaShape/TlaCoord/TlaStride`：结构对象类别用错。
- `mask ... expected ... lanes`：mask dtype 与 vector 元素宽度不一致。
- `unsupported cast target dtype`：cast 使用 unsigned、bool 或 f64。
- MMAD operand/layout/unit flag 错误：前端拒绝生成不合法 TLAIR。[^core]

# 验证方法

用 `catlass.core_api.__all__` 和 `catlass.runtime._CORE_API_EXPORTS` 审计导出面；
再调用 kernel 的 `dump_mlir(type_args=...)` 检查 TLAIR，并运行
`tests/test_core_api_preconditions.py`、vector pytest 和 lit lowering 用例。
本文只完成固定提交源码核对，未运行 CATLASS DSL、CANN 或 NPU。

[^api]: 固定提交中生成的 Kernel API 清单。
[^core]: 固定提交中的 `catlass/core_api.py` 参数检查与 lowering 实现。
[^runtime]: 固定提交中的 `catlass/runtime.py` 顶层 Core API 转发清单。
