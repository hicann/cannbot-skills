---
type: CATLASS DSL Operator Example
title: Vector 算子代码模式
description: Unary、binary、cast、mask、gather、reduction 的离线可复用 API 模式。
tags: [catlass-dsl, operator, vector, mask, reduction]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: unary
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/vector_ops/unary_ops.py
    title: Unary vector examples
  - id: binary
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/vector_ops/binary_op.py
    title: Binary vector examples
  - id: advanced
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/vector_ops/vector_op_harness.py
    title: Vector operation harness
  - id: gather
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/vector_ops/gather_op.py
    title: Gather operation
  - id: reduction
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/vector_ops/reduction_ops.py
    title: Reduction operations
  - id: cast
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/vector_ops/cast_multi.py
    title: Vector cast operations
operator_families: [elementwise, reduction, gather]
arch: [c310]
---

# 接口与概念

## 算子算法

Vector 示例覆盖逐 lane unary/binary、compare+where、gather 以及 `ADD/MAX/MIN`
reduction。[^unary][^binary][^gather][^reduction]

| 模式 | 核心 API |
| --- | --- |
| Unary/Binary | `exp/log/sqrt/add/sub/mul/div` |
| 条件选择 | `cmp`、`where` |
| 索引读取 | `tla.gather` |
| 归约 | `VectorSSA.reduce(tla.ReductionOp.ADD/MAX/MIN)` |
| 类型转换 | `VectorSSA.to(dtype, CastParams, mask)` |

# 用法

## 分核策略与基本块切分

每个 block 处理连续 GM tile，tile 在 UB 内按 dtype 对应的 lane 宽度循环；固定整块
使用 `create_mask`，tail 使用 `update_mask`。gather 的 index vector 与 source tile
使用同一批有效 lane。[^advanced][^gather]

```python
active, remaining = tla.update_mask(remaining, tla.Float32)
values = tla.gather(source_ub, index_ub.load(), mask=active)
```

# 代码模式

## 数据路径与存储层级

常规路径为 `GM -> UB tensor -> VectorSSA -> UB -> GM`。gather 按 UB 中的 index
读取 UB source；reduction 将 vector 压缩成标量后写入单元素 UB tile；
interleave/deinterleave 只改变寄存器排列。[^gather][^reduction]

```python
mask = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
reduced = values.reduce(tla.ReductionOp.ADD, mask=mask)
scalar_tile.store(reduced)
```

## 流水排布、同步关系与数值精度

MTE2、VECTOR、MTE3 分别负责搬入、计算、搬出。mask dtype 决定 lane 数；cast 必须
通过 `CastParams` 指定 register slot、饱和与舍入模式。[^advanced][^cast]

```python
params = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_TRUNC,
)
converted = values.to(tla.Int32, params, mask=mask)
```

# 约束

- mask lane 数必须与 vector dtype 一致。
- gather index 必须落在 source UB tensor 范围内。
- cast 舍入/饱和语义和 reduction dtype 不能被忽略。

# 失败表现

- mask dtype 错误会产生 lane 数不匹配。
- gather index 错误会读到非法位置。
- reduction tail 未屏蔽会把无效 lane 纳入结果。

# 验证方法

分别覆盖整块、tail、边界 index、NaN/Inf，以及每种 cast round/saturation 组合。

[^unary]: 固定提交 unary 示例的逐 lane 算法。
[^binary]: 固定提交 binary 示例的 tensor/scalar 运算。
[^advanced]: 固定提交 vector harness 的 tile、mask 与流水结构。
[^gather]: 固定提交 gather 的 UB source 与 index vector 数据路径。
[^reduction]: 固定提交 reduction kind 与标量写回结构。
[^cast]: 固定提交 VectorSSA cast 与精度控制参数。
