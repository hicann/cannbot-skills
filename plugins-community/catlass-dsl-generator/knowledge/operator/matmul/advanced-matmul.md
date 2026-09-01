---
type: CATLASS DSL Operator Example
title: Batched 与 Grouped Matmul 示例
description: 端到端 batched matmul 与按 M 切片 grouped matmul 的动态 GM 和任务映射边界。
tags: [catlass-dsl, operator, matmul, batched-matmul, grouped-matmul]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
  - {by: process:catlass-dsl-source-audit, at: '2026-08-29T00:00:00Z'}
sources:
  - id: batched
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/examples/end_to_end/batched_matmul/batched_matmul.py
    title: Batched matmul end-to-end example
  - id: grouped
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/examples/end_to_end/grouped_matmul_slice_m/grouped_matmul_slice_m.py
    title: Grouped matmul slice-M example
operator_families: [matmul]
arch: [c310]
---

# 接口与概念

## 算子算法

Matmul 样例已从单一 Basic MMAD 扩展到 batched matmul 和按 M 维切片的 grouped
matmul。两者使用动态 GM metadata 从 `origin_shape` 得到运行时 extent，并在 kernel
内完成 batch/group 到 block 的任务映射。[^batched][^grouped]

# 用法

## 分核策略与基本块切分

- Batched matmul：同构 batch 的 A/B/C 绑定后调用 `mark_layout_dynamic()`，kernel
  将 batch、M tile、N tile 展平为任务，并根据运行时 M/N 选择 swizzle 方向。
- Grouped matmul slice-M：多个 group 沿 M 切分，host metadata 与 kernel 的 group
  offset/shape 约定必须一致；block 先定位 group，再定位该 group 的 M/N tile。

# 代码模式

## 数据路径与存储层级

Batched 与 grouped 变体沿用 `GM -> L1 -> L0A/L0B -> MMAD -> L0C -> GM`，差异在
GM base offset、任务映射和动态 origin shape；A/B packed layout 与 Basic MMAD 相同。

```python
ta = tla.from_dlpack(a, layout_tag=tla.arch.RowMajor).mark_layout_dynamic()
tb = tla.from_dlpack(b, layout_tag=tla.arch.RowMajor).mark_layout_dynamic()
tc = tla.from_dlpack(c, layout_tag=tla.arch.RowMajor).mark_layout_dynamic()
artifact = tla.compile(kernel, ta, tb, tc, options="--npu-arch 3510")
artifact(ta, tb, tc, block_num=block_num)
```

不要把两个例子视为可互换模板：batch flatten 与 group offsets 具有不同 ABI 和任务映射。

Grouped host metadata 应被当成 ABI 的组成部分，并在记录中保留：

```text
grouped: group_count, per-group M/N/K, prefix offsets, dtype/layout
```

## 流水排布、同步关系与数值精度

每个任务内部沿用 L1/L0 双缓冲与 MTE2/MTE1/CUBE/FIX 同步，跨 batch/group 不共享
partial sum。输入通常为 f16，L0C 以 f32 累加；动态 GM 只改变运行时 extent，不改变
本地 tile、buffer 数或累加精度。[^batched][^grouped]

# 约束

- Batched A/B/C 的 batch、M/N/K 推导必须相互一致。
- Grouped matmul 的 group shape/offset metadata 必须覆盖所有输入且不重叠输出。
- 示例存在不代表任意 shape/dtype 已支持；以各脚本参数检查与 reference 为准。

# 失败表现

- batch 间串数据：flatten stride 或 batch offset 错误。
- 某些 group 未写/重写：slice-M prefix offset 或 block 映射错误。
- 同一 artifact 不能复用 shape：动态 GM 标记与 kernel runtime extent 使用不完整。

# 验证方法

分别运行样例自带 reference，覆盖 batch>1、非整除 tile 和不同 group M；先检查
compile/TLAIR，再同步设备并比较输出。任何性能结论都需在目标 NPU 上
单独 benchmark/profile，不能从样例结构直接推出。

[^batched]: 固定提交 batched matmul 的 dynamic GM、runtime extent 和 swizzle 实现。
[^grouped]: 固定提交 grouped matmul slice-M 的 group metadata 与分块实现。
