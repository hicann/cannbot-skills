---
type: CATLASS DSL Programming Concept
title: MMAD 操作数布局契约与转置表达
description: tla.mmad 强制 acc=L0Clayout、lhs=zN、rhs=nZ；转置只能靠同一块内存的 layout 视图重解释，不能用 transpose 参数。
tags: [catlass-dsl, dsl, mmad, layout, zn, nz, l0clayout, columnmajor, transpose, make-tensor-like]
status: stable
generated: {by: process:catlass-dsl-knowledge-rebuild, at: '2026-09-23T00:00:00-07:00'}
verified:
  - {by: human:maintainer, at: '2026-09-23T00:00:00-07:00'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: CATLASS mmad layout contract and make_tensor_like layout inference
  - id: ref
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/docs/en/api/kernel_api_reference.md
    title: CATLASS layout tag, C0 and stride constraints
  - id: fa
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/attention/flash_attention_infer/flash_attention_infer.py
    title: CATLASS ColumnMajor K operand recipe in the flash-attention example
operator_families: [matmul, sparse-attention]
arch: [c310]
---

# 接口与概念

`tla.mmad(acc, lhs, rhs, ...)` 没有 transpose 开关。lowering 校验形状契约
`lhs(MxK) x rhs(KxN) -> acc(MxN)`，并强制操作数 layout 为 `acc = L0Clayout`、
`lhs = zN`、`rhs = nZ`，否则报
`unsupported tla.mmad operand layout; expected acc L0Clayout, lhs zN, rhs nZ`。[^api]

因此“转置”在 DSL 里不是算子属性，而是同一块内存的 layout 视图属性：想要 `X^T`，就为同一
段数据构造一个转置语义的视图，而不是给 `mmad` 传标志。

# 用法

`make_tensor_like(ptr, like, layoutTag)` 的 layout 推断规则决定了怎么构造视图：[^api]

| 目标地址空间 | 省略 `layoutTag` 时的推断 |
| --- | --- |
| `l0a` | `zN` |
| `l0b` | `nZ` |
| `l0c` | `L0Clayout` |
| `l1` | 取 `like` 的 tag：`RowMajor`/`zN` → `zN`，`ColumnMajor`/`nZ` → `nZ`；其他 tag 报错 |

所以 L0 操作数的 layout 由地址空间决定，L1 的 layout 由参照 tile 决定。要构造 `rhs = nZ`，
通常先让 L1 视图是 `nZ`（或 `ColumnMajor`），再 `make_tensor_like(l0b_ptr, l1_view)`。

# 代码模式

`zN`/`nZ` 的物理 shape/stride 是嵌套结构且 C0 固定为 32 字节，不能直接把普通 2D `shape` 配上
`zN`/`nZ` tag。[^ref] 固定示例给出的转置操作数配方是“GM 用 ColumnMajor 描述同一块 K 内存，
再取 `nZ` 的 L1 视图”：

```python
# 同一块 K 内存，转置语义：逻辑 (D, Skv)，stride (1, kv_row_stride)
gm_k_col = tla.make_tensor(
    mem_k.ptr + k_off,
    tla.make_layout(
        tla.make_shape(HEAD_DIM, kv_len),
        tla.make_stride(1, kv_row_stride),
        layoutTag=tla.arch.ColumnMajor,
    ),
)
k_tile = tla.tile_view(gm_k_col, tla.make_shape(HEAD_DIM, KV_BLOCK), tla.make_coord(c0, c0))
l1_k = tla.make_tensor_like(l1_k_ptr, k_tile, layoutTag=tla.arch.nZ)   # L1 的 nZ 视图
l0_k = tla.make_tensor_like(l0b_ptr, l1_k)                             # l0b 推断为 nZ
```

非转置操作数走 `RowMajor` 视图：`make_tensor_like(l1_ptr, row_view, tla.arch.zN)`。[^fa]

# 约束

- `acc` 必须是 `L0Clayout`，`lhs` 必须是 `zN`，`rhs` 必须是 `nZ`；没有例外或开关。[^api]
- `l0a` 的 layout 由地址空间无条件决定为 `zN`（`l0b` 为 `nZ`），`layoutTag` 省略时不会
  从参照 tile 取；`make_tensor_like` 只把 shape/stride 重映射到目标 layout tree，不搬运也不
  转换数据。所以不能靠“把已有的 L1 `nZ` 视图交给 `l0a`”得到 `zN` 语义——那只是把同一块数据
  按 `zN` 解释。转置的左操作数必须换一个转置语义的 `ColumnMajor` 视图重新构造，或走真正的
  布局转换搬运。[^api]
- 同一块 scratch 内存可以同时被两种视图引用（RowMajor 写、ColumnMajor 读），但两者的
  `shape`/`stride` 必须与各自的 tag 自洽。
- `zN`/`nZ` 的 stride 值必须是 C0 的整数倍，C0 为 32 字节。[^ref]
- 视图只改解释方式，不搬数据；换视图后必须重新确认 buffer 的写者/读者同步。

# 失败表现

- 编译期报 `unsupported tla.mmad operand layout`：某个操作数被构造成非契约 layout。
- 编译期报 layout 推断失败：给 L1 传了 `like` 的 tag 既不是 `RowMajor`/`zN` 也不是
  `ColumnMajor`/`nZ`，且没有显式 `layoutTag`。
- 结果像被转置：视图的 `shape`/`stride` 与 tag 不自洽（例如 `ColumnMajor` 视图配了行主序
  stride）。
- 数值正确但性能异常：`zN`/`nZ` stride 未按 C0 对齐，触发非预期搬运路径。
- 结果与前一次运行不同：换视图后没有重建 buffer 同步，读到了旧内容。
- 出现 MTE 访问异常或 507015：把 `nZ` 数据当 `zN` 解释后搬运到 L0A。

# 验证方法

先用最小 `basic_mmad` 形状分别验证非转置与转置两条路径，再把它接入真实算子。验证转置时用
非方阵（`M != N`）与非对称数据，方阵对称输入无法区分“正确转置”与“根本没转置”。

[^api]: 固定提交 `core_api.py` 中的 `mmad` 形状/layout 校验与 `make_tensor_like` 的 layout 推断分支。
[^ref]: 固定提交 API 文档中 layout tag、C0=32 字节与 stride 对齐约束。
[^fa]: 固定提交 flash-attention 示例中的 ColumnMajor 转置操作数配方与 `make_tensor_like` 用法。
