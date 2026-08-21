---
type: CATLASS DSL Programming Concept
title: SIMT Vector Function 与 Tensor 标量访问
description: SIMT thread 索引、thread-block 几何和 UB/GM tensor 标量 load/store 的使用方式。
tags: [catlass-dsl, simt, vector, tensor-indexing]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: core
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/core_api.py
    title: SIMT and tensor indexing Core API
  - id: simt-example
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/simt/basic_vadd_simt.py
    title: Basic SIMT VADD example
  - id: indexing-tests
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/tests/test_tensor_indexing.py
    title: Tensor indexing tests
operator_families: [elementwise, reduction]
arch: [c310]
---

# 接口与概念

`with tla.vec.func(mode="simt", thread_block_dim=...)` 创建 SIMT vector function。
`tla.arch.thread_idx()` 与 `tla.arch.thread_block_dim()` 返回 `(x, y, z)`；它们不同于
launch 级的 `block_idx()` / `block_num()`。tensor 下标表达式支持标量读取和赋值，
可用于 GM 或 UB 的逐元素访问。[^core][^indexing-tests]

# 用法

```python
with tla.vector():
    with tla.vec.func(mode="simt", thread_block_dim=256):
        tid, _, _ = tla.arch.thread_idx()
        width, _, _ = tla.arch.thread_block_dim()
        for i in tla.range(tid, count, width):
            gm_c[i] = gm_a[i] + gm_b[i]
```

该模式直接按线程访问 GM，不要求先搬到 UB；是否优于 SIMD tile 搬运必须实测。
[^simt-example]

# 代码模式

`thread_block_dim` 可给正整数（解释为 x 维）或三个正整数的 tuple/list。SIMT 示例的
thread 数属于单个 block；host launch block 数仍通过 `artifact(..., block_dim=...)`
配置，kernel 内总 block 数用 `tla.arch.block_num()` 获取。

```python
with tla.vec.func(mode="simd"):
    scalar = ub[index]
    ub[index] = scalar + 1
```

# 约束

- `thread_block_dim` 只允许用于 `mode="simt"`，总线程数受实现上限检查。
- `thread_idx`/`thread_block_dim` 只能在 SIMT `vec.func` 内调用。
- `block_num()` 是 launch block 数，不能拿 `thread_block_dim()` 替代。
- 动态 index 必须与 tensor rank/坐标结构兼容，越界仍是调用方责任。

# 失败表现

- `thread_block_dim is only allowed with mode='simt'`：在 SIMD region 传线程几何。
- `arch.thread_idx is only available ... mode='simt'`：调用 region 错误。
- 只处理前一个 thread block：循环步长误用 `block_num()`，或 host `block_dim` 与
  kernel 的 block-stride 映射不一致。
- 标量结果错位：多维 tensor 使用了错误的线性/结构化下标。

# 验证方法

dump TLAIR 检查 SIMT region、thread index op 与标量 load/store；运行 tensor indexing
和 SIMT lowering 测试，并以非整除 thread 数的 tail case 对照 reference。性能结论需
分别 benchmark SIMT 与 SIMD 实现。

[^core]: 固定提交 Core API 的 SIMT region、thread 几何和 tensor indexing 实现。
[^simt-example]: 固定提交 basic SIMT VADD 的 thread-stride 代码与 host launch。
[^indexing-tests]: 固定提交 tensor 标量读取、赋值与前端约束测试。
