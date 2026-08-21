---
type: CATLASS DSL Programming Concept
title: 动态 Layout 与 DLPack 绑定
description: 用动态 shape/stride 元数据复用编译产物，并保持 DLPack zero-copy 绑定。
tags: [catlass-dsl, dynamic-shape, layout, dlpack, runtime]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: tensor-runtime
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/tla/runtime.py
    title: Runtime tensor and DLPack implementation
  - id: abi-tests
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/tests/test_dynamic_gm_launch_abi.py
    title: Dynamic GM launch ABI tests
  - id: batched
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/batched_matmul/batched_matmul.py
    title: Batched matmul dynamic-GM example
operator_families: [elementwise, matmul, attention]
arch: [c310]
---

# 接口与概念

`from_dlpack` 对 Ascend tensor 做 zero-copy 绑定。默认情况下具体 shape/stride 会进入
kernel type；随后调用 `mark_layout_dynamic(leading_dim=None)` 可把全部 shape 与除连续
leading dimension 外的 stride 标为动态，调用
`mark_compact_shape_dynamic(mode, stride_order=None)` 则只动态化一个紧凑 layout mode
及受其影响的外层 stride。这样同一编译产物可处理多个实际尺寸。[^tensor-runtime][^abi-tests]

# 用法

```python
ta = tla.from_dlpack(a, layout_tag=tla.arch.RowMajor).mark_layout_dynamic()
tb = tla.from_dlpack(b, layout_tag=tla.arch.RowMajor).mark_layout_dynamic()
tc = tla.from_dlpack(c, layout_tag=tla.arch.RowMajor).mark_layout_dynamic()

artifact = tla.compile(kernel, ta, tb, tc, arch_scope="aic.c310")
artifact(ta, tb, tc, block_dim=block_dim)
```

若只有 batch/M 等某一维变化，可用 `mark_compact_shape_dynamic(0)` 缩小动态范围。
kernel 内从 `tensor.origin_shape` 读取本次 launch 的真实工作尺寸。[^batched]

# 代码模式

动态标记必须施加在 root tensor：

```python
root = tla.from_dlpack(storage, layout_tag=tla.arch.RowMajor)
root.mark_compact_shape_dynamic(mode=0)
```

`mark_layout_dynamic` 会保留连续维的 unit stride 和 broadcast stride 0；
`mark_compact_shape_dynamic` 会按 compact stride order 传播需要动态化的 stride。

# 约束

- 仅支持 Ascend/NPU DLPack producer；buffer 生命周期必须覆盖 compile/launch。
- 动态标记要求 root tensor 的 coord 全为 0，不能直接标记带 offset 的 tile view。
- `leading_dim` 必须在 rank 范围内且对应 stride 为 1。
- `stride_order` 若显式提供，必须是 `range(rank)` 的排列。
- 当前 DLPack dtype 覆盖 int、uint、f16、f32、bf16 和 bool；具体 width 仍由类型桥检查。

# 失败表现

- `mark_*_dynamic requires a root tensor with zero coordinates`：对切片 view 做动态标记。
- `leading_dim ... expected 1`：连续维选择与物理 layout 不符。
- 相同逻辑 kernel 反复编译：动态 mode 未标记，具体 extent 仍进入 cache key/type。
- 输出错位：host layout tag、stride/origin shape 与物理 storage 不一致。

# 验证方法

用同一 artifact 连续 launch 至少两个不同 shape，核对 cache 命中、ABI metadata 和
reference；同时测试 row-major、column-major、非法非零 coord 与错误 stride order。
动态复用不等于正确性证明，每个 shape 仍需过 oracle。[^abi-tests]

[^tensor-runtime]: 固定提交 runtime tensor 的 DLPack、动态 shape 与 stride 规则。
[^abi-tests]: 固定提交 dynamic GM launch ABI 的编译和运行参数测试。
[^batched]: 固定提交 batched matmul 对 A/B/C 使用 dynamic GM 的端到端模式。
