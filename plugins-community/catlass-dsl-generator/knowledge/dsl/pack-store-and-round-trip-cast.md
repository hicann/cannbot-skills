---
type: CATLASS DSL Programming Concept
title: PACK Store 与设备侧舍入
description: 用 StoreDist.DIST_PACK_B32 直接落窄化结果，并在设备侧完成 scale 与 f32->f16/bf16 舍入，避免 host cast 改变舍入语义。
tags: [catlass-dsl, dsl, store, store-dist, pack-b32, cast, round-mode, reg-slot, low-precision]
status: stable
generated: {by: process:catlass-dsl-knowledge-rebuild, at: '2026-09-23T00:00:00-07:00'}
verified:
  - {by: human:maintainer, at: '2026-09-23T00:00:00-07:00'}
sources:
  - id: params
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/params.py
    title: CATLASS StoreDist, NormalStoreParams, CastParams, RegSlot, SatMode and RoundMode
  - id: example
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/dsl_tutorials/vector_simd_ops/store_pack.py
    title: CATLASS compact vector store example for DIST_PACK_B32
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: CATLASS vector store and cast frontend
operator_families: [matmul, sparse-attention, elementwise]
arch: [c310]
---

# 接口与概念

`NormalStoreParams(store_dist=StoreDist.DIST_PACK_B32)` 是一种紧凑向量 store：源寄存器保持它的
输入 dtype，由 PACK 模式本身完成压缩，`DIST_PACK_B32` 取每个 `i32` lane 的低 16 位
（`DIST_PACK_B16` 取每个 `i16` lane 的低 8 位）。[^params][^example]

这正好匹配“f32 累加器 → 窄 dtype 输出”的落盘：先在寄存器里把 FP32 窄化成 FP16/BF16，再用
`DIST_PACK_B32` 直接把窄结果写进 UB，不需要额外的 deinterleave 或第二次搬运。

窄化本身由 `tla.cast` 的三个参数决定：`reg_slot`、`sat_mode`、`round_mode`。[^params]

| 参数 | 取值 | 含义 |
| --- | --- | --- |
| `reg_slot` | `ZERO` / `ONE` | 2x 宽度 cast（f32→f16、i32↔i16）落在 AVE 的 even/odd part |
| `sat_mode` | `SAT` / `NOSAT` | 溢出时饱和还是环绕 |
| `round_mode` | `CAST_ROUND` | 就近舍入，平局远离零 |

# 用法

低精度输出的舍入是设备侧语义，必须由设备侧完成：

```python
pack_store = tla.params.NormalStoreParams(
    store_dist=tla.params.StoreDist.DIST_PACK_B32
)
cast_zero = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.SAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)

# 设备侧先乘 scale（若该输出需要），再窄化，最后 PACK store
val = tla.mul(src.load(), softmax_scale, mask=f32m)
dst.store(val.to(io_dtype, cast_zero, mask=f32m), pack_store, mask=io_all_mask)
```

把 FP32 中间量读回 host、用 Torch 做 `tensor.to(torch.float16)` 再写回，会改变舍入（Torch 与
`CAST_ROUND` 的舍入/饱和行为不同），而且引入 host 计算路径。[^params][^api]

# 代码模式

```text
L0C/UB(FP32) -> (可选 *scale) -> .to(io_dtype, CastParams) -> NormalStoreParams(DIST_PACK_B32) -> UB
```

`DIST_PACK_B32` 要求源寄存器保留输入 dtype，PACK 只做压缩不做数值转换，所以窄化必须显式写在
store 之前。[^example] 2x 宽度窄化一次只填一个 `reg_slot`，但“完整一行”需要几个 slot
取决于逻辑行宽、单 slot 的有效 lane 数和目标 dtype，不能把“2x 宽度 = 两个 slot”当成
通用规则；先按实际 API 的 lane 布局计算，再用标量/小向量验证。如何合成完整行属于布局/
去交织问题，见
[BSAG zN Cast Passthrough](../operator/sparse-attention/block-sparse-attention-grad/block-sparse-attention-grad.md)。

# 约束

- `DIST_PACK_B32` 只压缩，不转换；源寄存器必须是窄化后的结果，不能是 FP32。
- 低精度舍入必须用设备侧 `tla.cast` 的 `round_mode`/`sat_mode` 表达，不能依赖 host 侧 cast。
- `reg_slot` 的取值取决于 cast 宽度：2x 宽度只用 `ZERO`/`ONE`。
- PACK store 的 mask 用目标 dtype 的 all mask，数值 mask 单独传给窄化；两者不要混用。
- 落盘后若还要被 Cube 当操作数读取，必须重新确认 buffer 的写者/读者同步。

# 失败表现

- 输出出现 65504 之类的饱和值：`sat_mode` 与预期不符，或 host 侧 cast 把大 FP32 值按不同规则
  处理后再写回。
- 数值系统性偏移一个舍入单位：host 与设备侧舍入规则不一致，或漏乘/重复乘 `scale`。
- 输出高低 16 位互换或只写入一半：`reg_slot` 用错，或 PACK 的源不是窄化结果。
- 只有部分 lane 正确：PACK 的 mask 与窄化 mask 用了同一个变量。

# 验证方法

用一个含正负、接近上界、以及无法被 FP16/BF16 精确表示的值的小向量，对比设备侧窄化与 host
cast 的结果差异，确认舍入方向；再把同一逻辑接到真实算子的输出回写路径上，逐 lane 比较。

[^params]: 固定提交 `params.py` 中的 `StoreDist`、`NormalStoreParams`、`CastParams`、`RegSlot`、`SatMode` 与 `RoundMode` 定义。
[^example]: `dsl_tutorials/vector_simd_ops/store_pack.py` 中 `DIST_PACK_B32` 的压缩语义与用法。
[^api]: 固定提交 `core_api.py` 中的向量 store 与 cast 前端接口。
