---
type: CATLASS DSL Programming Concept
title: 在 SIMD 中复现 bf16 的 ties-to-even 舍入
description: SIMD 的 f32→bf16 转换默认 round-half-away-from-zero，与 GEMM 的 round-half-to-even 不一致，需位运算仿真。
tags: [catlass-dsl, precision, bf16, rounding, ties-to-even]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-23T15:20:00+08:00'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-23T15:20:00+08:00'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: SIMD 转换与位运算接口
dsl_version: ascend-catlass-dsl 2.1.0
arch: [c310]
---

# 接口与概念

矩阵乘法路径（ND2NZ / MMAD）对 bf16 采用 round-half-to-even，而 SIMD 的
`to(tla.BFloat16)` 默认是 round-half-away-from-zero。两者在 halfway 值上差 1 ULP，
当代码需要与 GEMM 的数值语义逐位对齐时（例如放大后再做在线统计），必须显式仿真。

# 用法

把 f32 按位解释为 i32，做一次"加偏置 + 奇偶校正"再掩码：

```python
parity  = tla.bitwise_and(tla.shift_right(bits, 16), int_one)   # 保留位的最低位
rounded = tla.bitwise_and(
    tla.add(tla.add(bits, int_bias), parity), int_mask          # bias=32767, mask=~0xFFFF
)
```

等价于 `(x + 0x7FFF + ((x >> 16) & 1)) & ~0xFFFF`。
应在按温度缩放**之前**做，保证后续统计基于与 GEMM 一致的数值。

# 代码模式

见上文三行；`int_bias = 32767`、`int_mask = -65536`，按 32 位整数处理。

# 约束

- 只适用于 f32→bf16；f16 的偏置与掩码位数不同。
- 依赖小端与 IEEE754 位序，不允许在打包/解包路径重复做。
- 会增加一段全 tile 的 SIMD 位运算，需评估对流水的影响。
- 本条目在 2.1.0 上验证；目标版本升级后需重新确认 `to()` 的默认舍入模式是否仍为
  round-half-away-from-zero，再决定是否继续仿真。

# 失败表现

- 与 GEMM 结果存在 1 ULP 级差异，边界用例超差而主体通过。
- 若在缩放后补做，误差会被放大，看起来像"随机"偏差。

# 验证方法

构造 halfway 输入（如 `1 + 2^-8`；1 附近 bf16 相邻间隔为 `2^-7`，中点是 `1 + 2^-8`），
与 numpy 的 `round-half-to-even` 结果逐位比对；再用同一 shape 与 GEMM 输出做 bitwise 比较。

接口行为以固定提交的实现为准。[^api]

[^api]: 固定提交的源码与文档中对应实现。
