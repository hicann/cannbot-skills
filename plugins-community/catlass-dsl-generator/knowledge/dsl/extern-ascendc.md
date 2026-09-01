---
type: CATLASS DSL Programming Concept
title: 用户提供的 Ascend C 外部函数
description: 用 tla.extern 声明、类型检查、编译并从 Cube/Vector region 调用单个内联 Ascend C 入口。
tags: [catlass-dsl, extern, ascend-c, ffi, compile]
status: draft
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-29T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-29T00:00:00Z'}
sources:
  - id: ffi
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/tla/ffi.py
    title: tla.extern declaration API
  - id: core
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: External call frontend validation
  - id: execution
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/execution.py
    title: External Ascend C compilation and cache integration
  - id: tests
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/tests/test_extern_op.py
    title: External function declaration, lowering and compilation tests
operator_families: [elementwise]
arch: [c310]
---

# 接口与概念

`@tla.extern(source=<inline-source>, name=<optional-c-symbol>)` 把一个仅含类型签名的 Python
函数声明转换为 `ExternFunction`。调用声明对象时，函数体不会执行，而是在当前 Cube 或
Vector region 发出 `tla.call_extern`；compile 阶段把内联 Ascend C 源码编译成目标 core
所需 bitcode，并把源码纳入 cache key。[^ffi][^core][^execution]

# 用法

```python
import catlass.tla as tla

ASCENDC_SOURCE = r'''
#include <cstdint>
extern "C" {
[aiv] __attribute__((noinline))
void scale_f32(__gm__ float *src, __ubuf__ float *dst, int32_t count) {
    // 用户实现；签名必须与声明一致。
}
}
'''

@tla.extern(source=ASCENDC_SOURCE, name="scale_f32")
def scale_f32(
    src: tla.Pointer[tla.Float32, tla.AddressSpace.gm],
    dst: tla.Pointer[tla.Float32, tla.AddressSpace.ub],
    count: tla.Int32,
) -> None: ...

@tla.kernel
def kernel(gm: tla.Tensor) -> None:
    ub = tla.allocate(256, tla.Float32, tla.AddressSpace.ub, 256)
    with tla.vector():
        scale_f32(gm.ptr, ub, tla.Int32(256))
```

# 代码模式

声明参数只允许固定位置参数，且每项必须注解为
`tla.Pointer[dtype, AddressSpace]` 或具体 `tla.Numeric` 类型；返回注解必须显式为
`None`。`name` 省略时使用 Python 函数名，并且最终 symbol 必须是合法 C identifier。
[^ffi]

调用侧传 pointer 时，dtype 与 address space 必须逐项匹配声明；scalar 必须是同一具体
Numeric 类型，index 不能代替整数 scalar。一个 kernel v1 最多使用一个不同的
`ExternFunction`，但可多次调用同一对象。相同 extern 同时在 Cube 与 Vector region 调用
时，编译器为两个 core target 构建依赖。[^core][^tests]

前端成功时应出现可审计的调用点：

```mlir
tla.call_extern @scale_f32(...)
```

# 约束

- `source` 必须是非空字符串，不接受文件路径对象；源码文件依赖需由调用方显式读入字符串。
- 声明不能有默认值、可变参数、关键字专用参数或缺失注解，返回值只能是 `None`。
- extern call 必须位于 `with tla.cube()` 或 `with tla.vector()`，且必须在
  `tla.vec.func()` 外。
- 调用参数必须传 `.ptr` 或 allocation pointer，不能用 Tensor 对象代替 pointer。
- 每个 kernel 最多绑定一个 extern 声明；不能靠多次声明同名函数绕过此限制。
- Ascend C 源码、Python 声明、实际 core 属性和 pointer ABI 必须保持一致。
- 源码进入编译缓存 key；修改源码后应产生新 key，不能复用旧 binary。

# 失败表现

- `source must be str` / `source must not be empty`：source 类型或内容非法。
- `name must be a C identifier`：symbol 含连字符等非法字符。
- `parameters must be positional and fixed`：声明使用 `*args`、`**kwargs` 或 keyword-only。
- `expects a Pointer` / `pointer ... address space`：调用传了 Tensor 或错误 memory space。
- `call must be outside tla.vec.func()`：在 SIMD/SIMT register region 内直接调用 extern。
- `at most one external function per kernel`：同一 kernel 引用了两个不同声明对象。
- 后端 undefined symbol：源码 symbol、声明 `name`、core 属性或编译目标不一致。

# 验证方法

先执行 `dump_mlir`，确认 `tla.call_extern @symbol`、pointer dtype/address space 与调用 region；
再运行 `tests/test_extern_op.py` 的签名、错误路径、单/双 core lowering 测试。编译验证应保存
生成 bitcode、cache key 与首个后端 stderr，并在只改变源码后确认 key 改变。设备验证需用
最小 shape 对照独立 oracle，再覆盖 tail、不同 block 数和所有涉及的 core 类型。仅出现
`tla.call_extern` 不证明 Ascend C 函数正确或更快。

[^ffi]: 固定提交 `ffi.py` 的 decorator、签名和 symbol 校验。
[^core]: 固定提交 `core_api.py` 的 region、单 extern、pointer/scalar ABI 校验与 call op 发射。
[^execution]: 固定提交 `execution.py` 的外部源码编译、目标选择和 cache 依赖。
[^tests]: 固定提交测试覆盖声明体不执行、pointer 类型、region、单 extern 与双 core 行为。
