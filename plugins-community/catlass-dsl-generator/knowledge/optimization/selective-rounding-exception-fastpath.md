---
type: CATLASS DSL Optimization Guide
title: 舍入差异集合的选择性修正
description: 当硬件 cast 与正确性 oracle 只在稀有位型上不同，先刻画差异集合，再修正例外并保留原生 cast 主路径。
tags: [catlass-dsl, optimization, bf16, rounding, cast, vector, bitwise]
status: stable
generated: {by: process:catlass-dsl-optimize, at: '2026-08-11T16:52:15+08:00'}
verified:
  - {by: process:catlass-dsl-msprof, at: '2026-08-11T16:52:15+08:00'}
sources:
  - id: kernel
    resource: project-evidence:python/tla_dsl/examples/end_to_end/chunk_gla_fwd_kernel_o/chunk_gla_fwd_kernel_o_kernels.py?kernel-sha256=9678ff7347025489d5153413075a577123b4e7f527803def05aadd0d2f31d0b0
    title: BF16 midpoint correction implementation
    kind: implementation
  - id: result
    resource: project-evidence:.catlass-dsl/optimize-runs/chunk-gla-triton-align-20260811/manual/iter-003-midpoint-rne-fastpath/result.json?kernel-sha256=9678ff7347025489d5153413075a577123b4e7f527803def05aadd0d2f31d0b0
    title: Midpoint fast-path correctness and msprof result
    kind: profiling
operator_families: [mixed, attention, kda, elementwise]
arch: [c310]
---

# 何时触发

原生窄化 cast 几乎通过正确性、只有极少数输出越界时，不要立即对所有 lane 实现完整软件
舍入。先定位首个失败输出，反推参与计算的输入，比较硬件 cast 与 oracle cast 的精确位型，
确认差异是否只发生在可识别的稀有集合。[^result]

在已验证的 FP32 到 BF16 路径中，oracle 使用 ties-to-even，而候选硬件路径只在正好位于
中点且保留 BF16 LSB 为偶数时需要向零修正。下面的 mask 同时识别“低 16 位为 `0x8000`”
和“保留位为偶数”：

```text
(bits & 0x1ffff) == 0x08000
```

匹配时将 FP32 位型减一，再走原生 BF16 cast；其它元素不执行完整 bias 加法和低位清零。
该方法使 focused case 从 1109.438 us 降至 998.637 us，并保持最终 12/12 正确。[^result]

# 为什么旧流程容易漏掉

“精确匹配 PyTorch”不等于“每个元素完整实现 RNE”。只有一个输出失败既不能忽略，也不说明
必须重写整条 cast；应先通过位级归因刻画硬件语义与 oracle 语义的差异集合，避免把低频
例外成本扩散到所有 lane。

# 补丁步骤

1. 临时运行原生 cast 候选，只用于定位 mismatch，不得作为最终正确版本。
2. 从首个失败输出反推输入，记录 FP32 bits、oracle BF16、硬件 BF16 和输出贡献差。
3. 写出差异集合谓词，并验证正负数与 parity 条件。
4. 在 Int32 UB 视图中只修正例外；通过独立 vec function 边界重新以 FP32 视图读取，再走
   原生窄化 cast。
5. 对不需要精确 BF16 的 dtype 用 `range_constexpr(0, 0, 1)` 消除修正 pass。

```python
pattern_mask = tla.full(0x1FFFF, tla.Int32)
midpoint_even = tla.full(0x08000, tla.Int32)
one = tla.full(1, tla.Int32)
is_exception = tla.cmp(
    tla.bitwise_and(bits, pattern_mask), midpoint_even, "eq"
)
corrected_bits = tla.where(is_exception, bits - one, bits)
```

完整实现通过 UB 的 Int32/Float32 recast 视图连接修正 pass 和硬件 cast pass。[^kernel]

# 约束与判伪

- 必须实证当前 SoC 上硬件与 oracle 的具体舍入差异，不能把该 BF16 谓词直接套到其它
  SoC 或 dtype。
- NaN、Inf、subnormal、正负 midpoint 和奇偶保留位都要纳入定向测试。
- VectorSSA 没有寄存器 bitcast 时，不要假设同一 vec function 内 UB alias 写后立即可见；
  应使用独立 vec function 边界和正确同步。
- 任一 workload mismatch 都应回退到完整精确舍入。

# 验证

先运行构造出的 midpoint/parity 小测试，再运行完整 workload。性能比较必须同时包含完整
软件舍入基线与选择性修正候选；“原生 cast 更快但失败一个元素”不是可接受结果。[^result]

[^kernel]: 当前 kernel 的 midpoint 位型判断、修正 UB 和原生 cast 路径。
[^result]: focused mismatch 归因、正确性门禁和 msprof latency 结果。
