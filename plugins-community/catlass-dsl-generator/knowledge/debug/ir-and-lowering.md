---
type: CATLASS DSL Debugging Guide
title: TLAIR、MLIR 与 lowering 定位
description: 使用 dump_mlir、pytest 与 lit 将前端、TLAIR 和 NPUIR 问题分层。
tags: [catlass-dsl, debug, tlair, mlir, lowering]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: dump
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/dsl.py
    title: Kernel dump_mlir implementation
  - id: lit
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/tests/lit/tla-compile/mmad-end_to_end.mlir
    title: MMAD end-to-end lowering test
  - id: readme
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/README.md
    title: Test layer documentation
arch: [c310]
---

# 接口与概念

`kernel.dump_mlir(type_args=...)` 可在不 launch 的情况下观察前端生成的 TLAIR。
该入口直接由 kernel wrapper 提供。[^dump] pytest 侧重 Python 前端到 TLAIR，lit 侧重
TLAIR 到预期 NPUIR/lowering。[^readme][^lit]

# 用法

## 最小前端复现

```python
@tla.kernel
def reproduce(x: tla.Tensor, y: tla.Tensor):
    ready = tla.flag("ready", tla.arch.MTE2, tla.arch.VECTOR)
    ptr = tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256)
    tile = tla.tile_view(x, tla.make_shape(64), tla.make_coord(0))
    ub = tla.make_tensor_like(ptr, tile, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(ub, tile)
        tla.set_flag(ready)
        tla.wait_flag(ready)
        with tla.vec.func(mode="simd"):
            ub.store(tla.neg(ub.load()))

mlir = reproduce.dump_mlir(type_args=(x_type, y_type))
print(mlir)
```

先确认 TLAIR 包含预期结构：

```text
tla.func
tla.vector
tla.copy
tla.flag / tla.set_flag / tla.wait_flag
tla.vec.func
tla.load / tla.neg / tla.store
```

# 代码模式

## Pytest 前端断言

```python
def test_reproduce_emits_expected_tlair():
    mlir = reproduce.dump_mlir(type_args=(x_type, y_type))
    assert "tla.copy" in mlir
    assert "tla.neg" in mlir
    assert mlir.index("tla.copy") < mlir.index("tla.neg")
```

## Lit lowering 断言

```mlir
// RUN: TlaCompile %s | FileCheck %s
// CHECK-LABEL: func.func @reproduce
// CHECK: hivm
// CHECK-NOT: tla.neg
```

pytest 失败表示 Python AST/Core API 到 TLAIR 有问题；pytest 通过但 lit 失败表示
TLAIR pass/lowering 问题；lit 通过但生成 `kernel.o` 失败才进入 HIVMC/CANN
后端。[^readme][^lit]

# 约束

- IR 比较要关注 op 顺序、类型、属性和 region，不只搜索 op 名称。
- 动态 shape 问题需保留对应 SSA extent，不能在最小化时静态化掉。
- mixed kernel 需分别观察 Cube/Vector region 和跨核同步。
- 最小化时保留失败所依赖的 dtype、layout、address space、动态 extent 和 params。

# 失败表现

| 表现 | 所属层 |
| --- | --- |
| Python `TypeError` / `TlaCoreAPIError` | API 前置条件或 AST |
| TLAIR 缺 op/attribute/region | frontend lowering |
| lit `CHECK` mismatch | TLA pass 或 NPUIR lowering |
| TLA op 残留在最终 IR | 对应 conversion pattern 未执行 |
| IR 正确但 HIVMC 失败 | backend/toolchain |

# 验证方法

新增最小 pytest 或 lit regression，确认修复前失败、修复后通过；同时保存原始
TLAIR，而不是只保存测试摘要。本文未运行这些测试。

[^dump]: 固定提交 `catlass/dsl.py` 的 kernel `dump_mlir` 实现。
[^lit]: 固定提交 MMAD lowering lit 用例。
[^readme]: 固定提交 README 对 pytest 与 lit 职责的说明。
