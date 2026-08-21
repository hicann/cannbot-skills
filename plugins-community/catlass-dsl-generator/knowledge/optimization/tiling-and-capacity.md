---
type: CATLASS DSL Optimization Guide
title: Tile 与本地内存容量预算
description: 依据 C310 L1/L0/UB 容量筛选 tile 候选，并用实测决定是否接受。
tags: [catlass-dsl, optimization, tiling, memory, c310]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: arch
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/base_dsl/arch.py
    title: C310 local-memory capacities
  - id: mmad
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py
    title: MMAD tiling implementation
operator_families: [matmul, elementwise]
arch: [c310]
---

# 接口与概念

C310 源码容量为 L1/cbuf 512 KiB、L0A 64 KiB、L0B 64 KiB、L0C 256 KiB、
UB 248 KiB。[^arch] MMAD 示例按 dtype 字节数计算 A/B/C tile 占用，并分离
L1 K tile 与 L0 K tile。[^mmad]

# 用法

## VADD UB 示例

```text
one_tensor = 400 * 4 = 1600 bytes
three_tensors = 3 * 1600 = 4800 bytes
double_buffered = 2 * 4800 = 9600 bytes
UB headroom = 248 * 1024 - 9600 = 244352 bytes
```

## MMAD 示例

对 f16 A/B、fp32 C，候选 `L1_M=N=256, L1_K=128, L0_K=32`：

```text
L1 A double = 2 * 256 * 128 * 2 = 131072 bytes
L1 B double = 2 * 128 * 256 * 2 = 131072 bytes
L1 total    = 262144 bytes <= 524288
L0A double  = 2 * 256 * 32 * 2 = 32768 bytes <= 65536
L0B double  = 2 * 32 * 256 * 2 = 32768 bytes <= 65536
L0C         = 256 * 256 * 4 = 262144 bytes <= 262144
```

L0C 在此正好触及容量上限，没有额外 accumulator 空间。[^mmad]

# 代码模式

```python
def bytes_for(shape, dtype_bytes, buffers=1):
    elements = 1
    for extent in shape:
        elements *= extent
    return elements * dtype_bytes * buffers

candidate = {
    "l1_a": bytes_for((l1_m, l1_k), 2, 2),
    "l1_b": bytes_for((l1_k, l1_n), 2, 2),
    "l0_a": bytes_for((l0_m, l0_k), 2, 2),
    "l0_b": bytes_for((l0_k, l0_n), 2, 2),
    "l0_c": bytes_for((l0_m, l0_n), 4),
}
```

一次迭代只改变一个 tile 轴，正确性通过后才 benchmark。

# 约束

- 适用：profile 指向搬运、低复用或 tile 形状不匹配。
- 代价：更大 tile 增加本地容量、尾块浪费和寄存器/并行度压力。
- 正确性门禁：覆盖边界 tile、非整齐 shape、dtype 与所有 layout。
- 性能门禁：同配置重复 benchmark，并比较均值与方差。
- tile 候选接受后必须编码为 kernel 函数内的局部常量；不得新增或修改模块级 tile、dtype、
  shape 或开关来驱动编译，也不得通过 closure 捕获候选配置。

# 失败表现

- 单项容量未超但同一 scope 总和超限：只检查了单 buffer。
- L0C 满容量后又增加临时 tensor：静态预算遗漏。
- 正确但变慢：block 数下降、尾块浪费或 copy 粒度不合适。
- 仅整齐 shape 通过：边界 tile/origin shape 没进入门禁。

# 验证方法

记录每个 scope 的公式、候选值和上限；检查 kernel 除 `tla` 和 Python 内建符号外没有
自由名字；运行非整齐 M/N/K 正确性，再比较同一 benchmark 配置的 mean/std。源码不能
证明某 tile 更快。

[^arch]: 固定提交 C310 本地内存容量表。
[^mmad]: 固定提交 MMAD 的元素字节、L1/L0 分块和 buffer 分配。
