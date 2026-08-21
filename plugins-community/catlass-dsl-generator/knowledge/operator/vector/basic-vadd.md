---
type: CATLASS DSL Operator Example
title: Basic VADD 数据流
description: 从 GM 分块到 UB、vector 加法和回写 GM 的最小完整算子模式。
tags: [catlass-dsl, operator, vadd, elementwise, vector]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: source
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_vadd/basic_vadd.py
    title: Basic VADD source
  - id: guide
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_vadd/README.md
    title: Basic VADD guide
operator_families: [elementwise, vadd]
arch: [c310]
---

# 接口与概念

## 算子算法

逐元素计算 `z[i] = x[i] + y[i]`。Vector region 内依次 load X/Y、执行
`tla.add`、store Z；f32 的基本 vector chunk 为 64 个元素。[^source][^guide]

# 用法

## 分核策略与基本块切分

每个 block 映射一个连续 GM tile，tile coord 应包含 `tla.arch.block_idx()`；tile
内部按 64 元素循环，最后一轮用 `update_mask` 处理 tail。

```python
remaining = tile_elements
for chunk in tla.range((tile_elements + 63) // 64):
    active, remaining = tla.update_mask(remaining, tla.Float32)
```

# 代码模式

## 数据路径与存储层级

数据路径是 `GM X/Y -> UB X/Y -> Vector -> UB Z -> GM Z`。X、Y、Z 各自分配 UB
tensor，并复用对应 GM tile 的 layout。[^source]

```python
@tla.kernel
def vadd(x: tla.Tensor, y: tla.Tensor, z: tla.Tensor):
    x_ub = tla.make_tensor_like(x_ptr, x_gm, tla.arch.RowMajor)
    y_ub = tla.make_tensor_like(y_ptr, y_gm, tla.arch.RowMajor)
    z_ub = tla.make_tensor_like(z_ptr, z_gm, tla.arch.RowMajor)
    with tla.vector():
        tla.copy(x_ub, x_gm)
        tla.copy(y_ub, y_gm)
        with tla.vec.func(mode="simd"):
            z_ub.store(tla.add(x_ub.load(), y_ub.load()))
        tla.copy(z_gm, z_ub)
```

## 流水排布、同步关系与数值精度

MTE2 copy 完成后 Vector 才能读取 UB，Vector store 完成后 MTE3 才能回写 GM；
依赖使用 `loaded`、`computed` flag 或等价 mutex 表达。普通 VADD 保持输入 dtype，
atomic-add 变体改变的是 GM 写回语义。[^source]

```text
MTE2 copy X/Y --loaded--> Vector add --computed--> MTE3 copy Z
```

# 约束

- tile coord、UB 容量和 block 映射必须一致。
- tail mask 的 dtype 必须匹配 vector 数据 dtype。
- atomic add 的输出语义是累加而不是覆盖。

# 失败表现

- 多个 block 写同一 tile：GM 输出竞态。
- 最后一个 chunk 错误：tail mask 或 chunk coord 错误。
- 输出仍为 sentinel：MTE3 回写或同步链未完成。

# 验证方法

使用非整齐长度和多 block case；launch 后同步设备，再以
`torch.allclose(z, x + y, rtol=0.0, atol=1e-5)` 检查结果。

[^source]: 固定提交 `basic_vadd.py` 的 kernel、分块、存储与同步实现。
[^guide]: 固定提交 VADD README 的数据流说明。
