---
type: CATLASS DSL Debugging Guide
title: debug_print、print_tensor 与最小复现
description: 设备标量/张量打印的适用范围、同步要求和最小复现方法。
tags: [catlass-dsl, debug, print, tensor, minimal-reproduction]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
  - {by: process:catlass-dsl-source-audit, at: '2026-08-29T00:00:00Z'}
sources:
  - id: scalar
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/examples/end_to_end/debug_print/README.md
    title: Scalar debug print guide
  - id: tensor
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/examples/end_to_end/print_tensor/README.md
    title: Tensor print guide
  - id: scalar-code
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/examples/end_to_end/debug_print/debug_print.py
    title: Scalar print executable example
  - id: tensor-code
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/examples/end_to_end/print_tensor/print_tensor.py
    title: Tensor print executable example
arch: [c310]
---

# 接口与概念

公开入口是 `tla.print(*args)` 且只接受位置参数：支持单个 scalar、
`tla.print(format_string, *scalar_values)` 格式化标量，以及
`tla.print(tensor, length=None)` tensor 前缀。scalar 不能传 length；tensor 的
length 是元素数而不是字节数。[^scalar-code][^tensor-code]

标量 `tla.print` 示例支持显式 Cube/Vector region 中的 signless `i32` 或 `f32`，
不把 pointer formatting 作为公共契约。[^scalar] Tensor print 支持受限的 GM/UB
float32 tensor 前缀，并按物理连续前缀读取；它不会按 stride gather。[^tensor]

# 用法

## 标量打印

```python
@tla.kernel
def print_scalar(value: object):
    with tla.vector():
        tla.print(value)

@tla.kernel
def print_expression(lhs: object, rhs: object):
    with tla.cube():
        tla.print(lhs + rhs)
```

离线检查前端生成的 TLAIR：

```python
type_args = (tla.Int32(3),)
print(print_scalar.dump_mlir(type_args=type_args))
```

具备本地 CANN/NPU 环境时，可以直接编译和启动 kernel，不依赖外部示例脚本：

```python
value = tla.Int32(3)
compiled = tla.compile(
    print_scalar,
    value,
    options="--npu-arch 3510",
)
compiled(value, block_num=1)
```

标量输出应匹配 `x=<i32>` 或 `v=<f32>` 的原生 CANN frame；表达式用例应打印
运行时 `lhs + rhs`。[^scalar-code]

## GM Tensor 打印

```python
@tla.kernel
def print_gm(value: tla.Tensor):
    with tla.vector():
        tla.print(value, 16)
```

构造测试输入时使用 `float32[8, 4]` 的 `0..31` 连续值，并只打印前 16 个元素。
再将同一物理 buffer 描述为 column-major layout；两次输出都应遵循物理连续前缀，
而不是按逻辑 stride gather。[^tensor]

## UB Tensor 打印

UB 内容需要显式等待 GM->UB copy 完成：

```python
@tla.kernel
def print_ub(value: tla.Tensor):
    loaded = tla.flag("print_ub_loaded", tla.arch.MTE2, tla.arch.VECTOR)
    ptr = tla.allocate(32, tla.Float32, tla.AddressSpace.ub, 256)
    layout = tla.make_layout(
        tla.make_shape(4, 8),
        tla.make_stride(8, 1),
    )
    gm = tla.make_tensor(value.ptr, layout)
    ub = tla.make_tensor(ptr, layout)
    with tla.vector():
        tla.copy(ub, gm)
        tla.set_flag(loaded)
        tla.wait_flag(loaded)
        tla.print(ub, 16)
```

测试非零对齐地址时，分配 40 个 float32，并使用 `allocation + 8` 构造 UB
tensor；8 个 float32 正好偏移 32 bytes。[^tensor-code]

# 代码模式

## 动态 Shape

当前 pointer-only host tensor 参数不携带 runtime memref extent；动态用例把 rows
和 length 作为独立 scalar kernel 参数，再重建 layout：

```python
@tla.kernel
def print_dynamic(value: tla.Tensor, rows: tla.Int32, length: tla.Int32):
    layout = tla.make_layout(
        tla.make_shape(rows, 4),
        tla.make_stride(4, 1),
    )
    tensor = tla.make_tensor(value.ptr, layout)
    with tla.vector():
        tla.print(tensor, length)
```

## 输出检查

Tensor 示例的稳定公共输出形如：

```text
tla.print dtype=float32 shape=[8,4] count=16 values=[0.0, ..., 15.0]
compile_ok=True
launch_ok=True
output_ok=True
```

缺失、重复、截断、多余或格式错误的 native record 都应判为失败，而不是从 host
数据合成“看似正确”的日志。[^tensor]

# 约束

- scalar：只接受 signless `i32`/`f32`；必须位于显式 Cube 或 Vector region。
- tensor core：单一 `aiv.c310` 或 `aic.c310`，不支持 mixed/regionless。
- launch：示例只声明单 block；多 block 会产生多记录且顺序不稳定。
- storage：GM，或 AIV 上有效地址 32-byte 对齐的 UB；不支持 L1/L0。
- dtype：tensor 只支持 `float32`。
- shape：rank 1/2 的静态或 runtime shape；拒绝 empty、rank > 2 和元数据不符。
- length：1 到 262,112 个元素，且不能大于运行时 tensor 元素数；动态 shape
  必须显式传 length。
- layout：row-major、column-major、padded/strided 与 packed TLA layout；读取
  始终是有效地址起始的物理连续前缀。
- baseline：源码只声明 Ascend950PR、CANN 9.1.0-beta.3 或更新环境。[^tensor]
- 多核输出顺序未定义，不应按日志顺序推断执行顺序。

# 失败表现

- `does not accept keyword arguments`：错误使用 `tla.print(value=...)`。
- `length is only valid when printing a tensor`：scalar 误传第二参数。
- `dynamic-shaped tensors require an explicit length`：动态 tensor 未传长度。
- `length must be between 1 and 262112 elements`：长度越界。
- 无 native record：常见于 UB 地址未对齐、运行时 guard 拒绝或 producer 未完成。
- 值顺序与逻辑 layout 不同：打印的是物理前缀，不是 stride gather。
- UB 内容为旧值：`tla.print` 不会自动插入 producer synchronization。[^tensor-code]

# 验证方法

1. 先用 `kernel.dump_mlir(type_args=...)` 确认 `tla.print` op、region、dtype
   和 length。
2. 将 launch 缩到一个 block，并在项目测试中固定预期 native record 数。
3. 使用已知序列 `0..15` 区分物理前缀、layout 和 offset。
4. 对 UB 在 copy 后显式 set/wait，再移动打印点定位首个错误阶段。
5. 删除打印并重新运行完整正确性；打印会影响日志量和潜在时序。

本文是离线自包含知识，不要求访问 `sources` URL 或取得上游示例脚本。未执行设备
打印；接口、支持矩阵和错误边界来自固定提交示例与实现。

[^scalar]: 固定提交 scalar debug print 的类型与 region 契约。
[^tensor]: 固定提交 tensor print 的 storage、shape、length 和同步约束。
[^scalar-code]: 固定提交 `debug_print.py` 的 kernel、CLI 和输出校验实现。
[^tensor-code]: 固定提交 `print_tensor.py` 的 GM/UB/dynamic kernel 与 CLI。
