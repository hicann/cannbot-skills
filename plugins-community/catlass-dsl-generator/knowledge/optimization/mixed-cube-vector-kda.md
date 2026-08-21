---
type: CATLASS DSL Optimization Guide
title: Mixed CUBE/VECTOR 局部块算子
description: 面向 attention-like 和 KDA 局部块的 CUBE/AIV 分工、片上交接与 solve 优化候选。
tags: [catlass-dsl, optimization, mixed, cube, vector, kda, attention, causal, sparse-tile, load-balance]
status: stable
generated: {by: human:caijianlong, at: '2026-08-06T00:00:00+08:00'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-06T00:00:00+08:00'}
  - {by: process:catlass-dsl-msprof, at: '2026-08-11T16:52:15+08:00'}
sources:
  - id: mixed
    resource: https://gitcode.com/cann/catlass/blob/6ccf88e89723b65461e9921047c7970a71b67b42/python/tla_dsl/examples/end_to_end/basic_mixed/basic_mixed.py
    title: CATLASS DSL mixed Cube and Vector example
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/6ccf88e89723b65461e9921047c7970a71b67b42/python/tla_dsl/catlass/core_api.py
    title: CATLASS DSL copy and synchronization APIs
  - id: sparse-balance-kernel
    resource: project-evidence:python/tla_dsl/examples/end_to_end/chunk_gla_fwd_kernel_o/chunk_gla_fwd_kernel_o_kernels.py?kernel-sha256=9678ff7347025489d5153413075a577123b4e7f527803def05aadd0d2f31d0b0
    title: BT128 triangular tile balancing implementation
    kind: implementation
  - id: sparse-balance-result
    resource: project-evidence:.catlass-dsl/optimize-runs/chunk-gla-triton-align-20260811/manual/iter-005-bt128-triangular-balance/result.json?kernel-sha256=9678ff7347025489d5153413075a577123b4e7f527803def05aadd0d2f31d0b0
    title: Final correctness and msprof result
    kind: profiling
operator_families: [mixed, attention, kda]
arch: [c310]
---

# 接口与概念

适用于 mixed CUBE/VECTOR、attention-like、KDA、block-local solve、L0C 到 UB handoff
和双 AIV 分工。只修改 Python DSL 算子；编译器、lowering、runtime wrapper 和 backend
限制应保留原始错误并回退候选，不能作为本优化轴的修改目标。[^api]

# 用法

先确认 launch 覆盖全部任务，再按 `cube_utilization`、`aicore_time`、`aiv_time`、任务数
扩展性和同步尾部提出单一假设。可用 axis 包括 `mixed-fused-diagonal-subtile`、
`mixed-aiv-subblock-postprocess-split`、`mixed-l0c-to-ub-copy-mode-alignment`、
`mixed-solve-brc-beta-load`、`mixed-packed-output-tail-reuse`、
`mixed-raw-l0c-footprint-reduction` 和 `mixed-redundant-barrier-cleanup`。

# 局部块与分工

若最终只消费 32x32 task 中两个 16x16 diagonal block，优先验证 CUBE 直接产出所需
sub-tile，避免完整 raw matrix 及随后 VECTOR mask/gather。拆分可能减少无效计算和后处理，
但额外 MMAD、flag 和 copy 也可能更慢，必须 profile 判定。[^mixed]

`L0C2UBMode` 必须匹配消费者：AIV0/AIV1 各消费完整 tile 时用对应 `NO_SPLIT_VEC_*`；
按行或列分工时使用 `SPLIT_M` 或 `SPLIT_N`。每个 `raw_ready` 必须覆盖相应 UB 写完成，
`raw_free` 只能在对应消费者完成后释放。双 AIV 分割后处理时，两个 AIV 不得写同一输出
tile，beta、mask 和 row index 必须按 local sub-block 修正。[^mixed]

# 代码模式

```text
task rows 0..31
  sub-block 0: rows 0..15, cols 0..15 -> AIV0
  sub-block 1: rows 16..31, cols 16..31 -> AIV1
```

先确认 CUBE 的 copy mode、`raw_ready` 和 `raw_free` 与消费者一一对应，再让两个 AIV
分别写自己的输出 tile。[^mixed]

# 结构零 Tile 的联合裁剪与 Sub-block 均衡

对 causal、triangular、banded 或 block-diagonal 矩阵，先画 tile 级非零图。只做元素级
`where(mask, value, 0)` 时，结构零 tile 仍可能被 AIV 读取、转换和写回，再被 AIC MMAD
消费。有效候选必须同时考虑 producer AIV 搬运/转换、consumer AIC MMAD，以及裁剪后两个
AIV sub-block 的负载均衡。[^sparse-balance-kernel]

以 128x128 下三角矩阵和 64x64 tile 为例：

```text
diag-0 | structural-zero
-------+----------------
full   | diag-1
```

有效工作只有三个 tile。简单按上下 64 行分工会形成 1 tile 对 2 tiles；kernel 仍由较慢的
sub-block 决定。可把 `full` tile 按 32 行拆半，使每个 sub-block 处理一个 diagonal tile
加半个 full tile，即各 6144 个 FP32 元素。

补丁顺序是：

1. 用 compile-time shape 构造全零、全有效、对角 mask 的 tile 非零图。
2. AIC 仅遍历结构非零 K tile，例如 `tla.range_constexpr(0, row_tile + 1, 1)`。
3. 分别统计两个 AIV 的有效 load、cast、mask 和 store 元素，以最大值而非总和为均衡目标。
4. 必要时拆分一个全有效 tile，与两个 diagonal tile 重新配对。
5. 用 `tla.range_constexpr` 生成专用路径；不要用会在 AST lowering 中引入未初始化 Tensor
   变量的动态 Python `if`。
6. 同步更新 UB 工作区、GM/UB stride、cache key 和尾部同步。

仅跳过 AIC MMAD 可能因 AIV 仍是瓶颈而没有收益；仅裁剪 tile 而不重平衡，也会受最慢
sub-block 限制。在已验证 workload 上，这个联合候选将 focused case 从 784.619 us 降到
692.437 us，最终稳定 profile 为 656.550 us。[^sparse-balance-result]

# 约束

局部下三角 solve 先缩短 live range：beta 可用单元素 broadcast load，连续 `1x16` row
slice 比从 32x32 raw 的逐 lane gather 更稳。将多个 vector region 合并或使用动态 rank1 UB
tensor 可能触发 VF stack 或 lowering 失败，记录并回退，不能宣称为有效经验。[^api]

复用 raw f32 workspace 放置 packed f16/bf16 输出时，压缩输出不能从 raw 起点覆盖仍要读取
的行；在 raw buffer 尾部保留不重叠空间，并用 recast pointer 构造 packed 输出视图。缩小
raw UB、L0C 或 postprocess workspace 时按 raw UB、L0C、large-task profile 的顺序逐项验证。

结构裁剪只适用于恒零 tile；数据相关 mask 不能静态跳过。拆分后的 GM/UB layout stride
必须分别反映原矩阵行宽和紧凑工作区行宽，两个 sub-block 不得重叠写入或遗漏 full tile。
正确性至少覆盖三角边界、BT64/BT128、FP16/BF16、TND/BSND 和多 task。

# 失败表现

- copy mode 与 AIV 消费者不匹配，导致 stale、NaN 或稳定局部 mismatch。
- 将 packed 输出从 raw f32 起点紧凑写入，覆盖尚未读取的 raw 行。
- 动态 rank1 UB tensor 或合并 vector region 触发 VF stack 或 lowering 失败。

# 验证方法

只可删除已由精确 producer/consumer flag 覆盖的冗余 barrier。buffer 生命周期、最后一轮
MTE/FIX/CUBE/VECTOR 完成和重用前的等待必须保留，除非有独立正确性和 profile 证据。[^api]

[^mixed]: 固定提交 mixed 示例中的片上 Cube 到 Vector 数据路径。
[^api]: 固定提交 Core API 的 copy 与同步原语。
[^sparse-balance-kernel]: `chunk_gla_fwd_kernel_o` 的三 tile AIV 紧凑布局、sub-block 分工和 AIC K 范围。
[^sparse-balance-result]: focused 与全量正确性、msprof latency 和最终接受结果。
