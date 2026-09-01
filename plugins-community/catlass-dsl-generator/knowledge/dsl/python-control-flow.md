---
type: CATLASS DSL Programming Concept
title: Python 语法与动态控制流
description: CATLASS DSL 中自包含静态值、动态分支、循环、短路条件和 loop-carried 状态的语法边界。
tags: [catlass-dsl, python-syntax, control-flow, range, self-contained]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
  - {by: process:catlass-dsl-source-audit, at: '2026-08-29T00:00:00Z'}
sources:
  - id: syntax
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/docs/en/core_concepts/syntax_guide.md
    title: DSL Python Syntax Constraints
  - id: control-tests
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/tests/test_control_flow_analysis.py
    title: Control-flow analysis tests
  - id: lazy-tests
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/tests/test_lazy_conditions_example.py
    title: Lazy condition example tests
  - id: frontend
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/base_dsl/ast_preprocessor.py
    title: Python frontend global-name resolution
operator_families: [elementwise, matmul, reduction]
arch: [c310]
---

# 接口与概念

普通 Python 值驱动的 `if`/`for` 在前端静态执行；TLA 标量或表达式驱动的
`if`/`while`/`tla.range` 会 lower 为动态控制流。`tla.range` 接受一至三个位置参数，
`tla.range_constexpr` 用于明确要求 Python 展开的循环。动态分支支持 `and`/`or`
短路求值，不会无条件计算右侧表达式；`tla.const_expr(value)` 则要求 value 是可在
前端求值的 Python 值，并返回 Python bool。[^syntax][^lazy-tests]

“编译期可求值”只描述执行阶段，不表示模块级全局变量可以作为 kernel 配置。前端能够
解析 `__globals__` 和 closure 中的名字，但这些值不会出现在 kernel ABI 中；生成代码
应只使用 kernel 内定义的 Python 局部常量或显式参数。[^frontend]

kernel 和 `@tla.jit` 可调用裸 Python helper、lambda、嵌套函数、property 与绑定方法；
helper 在调用方的 emission context 中执行并可直接发出 TLA op。裸 helper 本身不会独立
经过控制流变换，因此其中不能包含 TLA 运行时控制流；这类控制流应留在 decorated
caller。`@tla.jit` 可由 kernel 调用，但不能调用或 launch `@tla.kernel`。[^syntax]

# 用法

运行时循环中，跨迭代变化的标量或 SSA 必须形成 loop-carried 状态：

```python
remaining = count
for _ in tla.range(tile_count):
    with tla.vec.func(mode="simd"):
        active, remaining = tla.update_mask(remaining, tla.Float32)
        out.store(tla.exp(inp.load(), mask=active), mask=active)
```

编译期展开使用 `range_constexpr`，适合固定数量 buffer 或 stage：

```python
stage_count = 2
for stage in tla.range_constexpr(stage_count):
    ptr = ptr0 if stage == 0 else ptr1
```

# 代码模式

动态短路条件可保护只在右侧条件成立时才合法的访问：

```python
if index < count and tensor[index] > 0:
    tensor[index] = tensor[index] + 1
```

动态分支两侧对后续继续使用的变量应给出类型兼容的赋值；前端分析会合并状态并拒绝
未定义、类别变化或不可合法携带的值。[^control-tests]

# 约束

- 动态 region 内不支持 Python `break`、`continue`、`return` 或 `raise`。
- 不要把 TLA 动态标量交给 Python `range`、`bool` 或其他 eager-only 控制流。
- 动态分支后的 live value 必须在所有可达路径定义且类型兼容。
- `range_constexpr` 的边界必须可在前端求值；动态边界应使用 `tla.range`。
- `const_expr` 收到 MLIR/TLA 动态值会报错，不能用于把运行时条件强制静态化。
- Python 静态条件、`range_constexpr` 边界和 `const_expr` 输入不得来自模块级用户变量、
  可变全局对象或 closure capture；固定值在 kernel 内定义，运行时值走显式 ABI。
- 在运行时 `if`、`tla.range`、`while` 或 outlined DSL `with` 内更新 Python scalar 前，
  先用 `tla.as_numeric(...)` 提升；普通用户对象只可用于 staging，不能作为动态状态。
- 动态 `for-else` / `while-else` 不支持；裸 helper 的运行时控制流也不支持。

# 失败表现

- “unsupported control flow” 或动态值不能转 Python bool：静态/动态控制流混用。
- 分支后变量未定义或类型不一致：控制流 merge 失败。
- 循环内更新没有带到下一轮：遗漏 loop-carried 状态，结果停留在初值或前端拒绝。
- wrapper 改写全局值后同一 kernel 的静态分支变化：把隐藏模块状态误作编译期接口。
- 裸 helper 内出现动态分支/循环时报 unsupported：helper 没有独立经过 AST 控制流变换。

# 验证方法

先用 `dump_mlir` 确认动态分支/循环保留为 IR，再运行 frontend branching、for/range、
while 和 lazy-condition 测试；同时覆盖零次循环、单次循环和两侧分支。再检查 decorated
kernel 的自由名字，确保静态控制参数全部来自局部值或显式参数。本文结论来自固定提交的
语法指南、前端实现与测试，未推断未覆盖的 Python 语法。

[^syntax]: 固定提交的 Python 语法指南对静态/动态控制流与禁用语句的说明。
[^control-tests]: 固定提交控制流分析测试中的分支合并与 loop-carried 约束。
[^lazy-tests]: 固定提交 lazy condition 测试中的运行时 `and`/`or` 短路行为。
[^frontend]: 固定提交 `ast_preprocessor.py` 对函数全局名字和 closure 的解析方式。
