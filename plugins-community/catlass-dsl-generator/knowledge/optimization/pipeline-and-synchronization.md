---
type: CATLASS DSL Optimization Guide
title: 计算搬运重叠与同步轴
description: 使用显式 buffer 轮换、unit_flag、flag、mutex 和 barrier 调整流水依赖。
tags: [catlass-dsl, optimization, buffering, synchronization, unit-flag, aiv, mte2, mte3]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
  - {by: process:catlass-dsl-msprof, at: '2026-08-11T16:52:15+08:00'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/catlass/core_api.py
    title: Current range and synchronization implementation
  - id: mmad
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py
    title: MMAD unit_flag and pipeline example
  - id: vadd
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_vadd/basic_vadd.py
    title: VADD flag, mutex and barrier variants
  - id: auto-sync
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul_auto_sync.py
    title: MMAD compiler-inserted synchronization example
  - id: component-pipeline-kernel
    resource: project-evidence:python/tla_dsl/examples/end_to_end/chunk_gla_fwd_kernel_o/chunk_gla_fwd_kernel_o_kernels.py?kernel-sha256=9678ff7347025489d5153413075a577123b4e7f527803def05aadd0d2f31d0b0
    title: Component-wise AIV pipeline implementation
    kind: implementation
  - id: component-pipeline-result
    resource: project-evidence:.catlass-dsl/optimize-runs/chunk-gla-triton-align-20260811/manual/iter-004-aiv-component-pipeline/result.json?kernel-sha256=9678ff7347025489d5153413075a577123b4e7f527803def05aadd0d2f31d0b0
    title: Focused correctness and msprof result
    kind: profiling
operator_families: [matmul, elementwise, mixed, attention, kda]
arch: [c310]
---

# 接口与概念

当前源码通过显式双 buffer、循环索引和同步原语构造搬运/计算重叠；同步 API
提供 flag、cross flag、mutex 和 pipe barrier。生成 API 文档中的
`prefetch_stages`/`pipelining` 关键字并不存在于当前 `range` 实现，不能使用。
[^api] MMAD 使用 `unit_flag` 表达末次 K 计算/搬运协议；VADD 展示 flag、mutex
与 barrier 的可替换实现。[^mmad][^vadd]

当前还提供 `@tla.kernel(auto_sync="v0")`：编译 pass 可为 MMAD/FIX 流水插入核内
mutex 与 unit-flag 协议，示例 kernel 本身不手写 local flag/mutex。它是明确的 v0
模式，不能假设覆盖 cross-core 或所有自定义内存别名。[^auto-sync]

# 用法

一次只改变一个同步轴：

- 调整显式双 buffer 数量、填充/排空和 buffer 轮换；
- 将全 pipe barrier 收窄为精确 flag；
- 在多向资源访问时比较 mutex 与显式依赖；
- 根据 K 循环首末轮调整 `unit_flag`。
- 对结构匹配的 MMAD/FIX kernel 比较显式同步与 `auto_sync="v0"` 生成结果。

# 代码模式

## Barrier 收窄为依赖 flag

修改前：

```python
tla.copy(ub, gm)
tla.pipe_barrier(tla.pipes.ALL)
with tla.vec.func(mode="simd"):
    ub.store(tla.exp(ub.load()))
tla.pipe_barrier(tla.pipes.ALL)
tla.copy(gm_out, ub)
```

候选修改：

```python
loaded = tla.flag("loaded", tla.arch.MTE2, tla.arch.VECTOR)
done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
tla.copy(ub, gm)
tla.set_flag(loaded)
tla.wait_flag(loaded)
with tla.vec.func(mode="simd"):
    ub.store(tla.exp(ub.load()))
tla.set_flag(done)
tla.wait_flag(done)
tla.copy(gm_out, ub)
```

正确性相同且 profile 中无关 pipe 的等待下降，才支持“barrier 过宽”。[^vadd]

## MMAD unit flag

```python
first_k = k_l1 == 0 and k_l0 == 0
unit_flag = 0b10
if k_l1 == k_l1_count - 1 and k_l0 == k_l0_count - 1:
    unit_flag = 0b11
tla.mmad(
    l0_c, l0_a, l0_b,
    init_c=first_k,
    unit_flag=unit_flag,
)
tla.copy(
    gm_c, l0_c,
    tla.params.CopyL0C2DstParams(unit_flag=0b11),
)
```

禁用 unit flag 时，源码使用 Cube→FIX 的 `mmad_done` 与 FIX→Cube 的
`fix_done` 显式握手；两种协议不能只混用其中一半。[^mmad]

## 同一 AIV task 内的独立组件流水

当一个 AIV task 顺序处理多个互不复用 UB 的组件，例如 `Q/G -> QG`、
`A -> masked A`、`H -> cast H`，不要只考虑“同一 tile 的 ping-pong 双缓冲”。如果

```text
aiv_total ~= aiv_mte2_time + aiv_vec_time + aiv_mte3_time
```

且各组件有独立输入、输出和 UB 生命周期，这个关系说明跨 pipe 重叠可能不足。此时可为每个
组件建立独立的 `free`、`loaded`、`done` flag：先向 MTE2 队列提交各组件 load；VECTOR
完成组件 N 后立即通知 MTE3 写回，并继续消费组件 N+1；只有对应 MTE3 写回完成后才能释放
该组件 UB。不要让多个组件复用同一套事件。[^component-pipeline-kernel]

```python
q_free = tla.flag("q_free", tla.arch.MTE3, tla.arch.MTE2)
q_loaded = tla.flag("q_loaded", tla.arch.MTE2, tla.arch.VECTOR)
q_done = tla.flag("q_done", tla.arch.VECTOR, tla.arch.MTE3)
a_free = tla.flag("a_free", tla.arch.MTE3, tla.arch.MTE2)
a_loaded = tla.flag("a_loaded", tla.arch.MTE2, tla.arch.VECTOR)
a_done = tla.flag("a_done", tla.arch.VECTOR, tla.arch.MTE3)

tla.set_flag(q_free)
tla.set_flag(a_free)
for task in tla.range(...):
    tla.wait_flag(q_free)
    tla.copy(q_ub, q_gm)
    tla.set_flag(q_loaded)
    tla.wait_flag(a_free)
    tla.copy(a_ub, a_gm)
    tla.set_flag(a_loaded)

    tla.wait_flag(q_loaded)
    with tla.vec.func(mode="simd"):
        ...
    tla.set_flag(q_done)
    tla.wait_flag(q_done)
    tla.copy(q_out_gm, q_out_ub)
    tla.set_flag(q_free)

    tla.wait_flag(a_loaded)
    with tla.vec.func(mode="simd"):
        ...
    tla.set_flag(a_done)
    tla.wait_flag(a_done)
    tla.copy(a_out_gm, a_out_ub)
    tla.set_flag(a_free)
```

这个候选容易被旧流程漏掉，因为 MTE2、VECTOR、MTE3 子项常被仅作为耗时占比观察，未检查
其和是否接近总时长；检索 `double buffer`、`barrier` 或 GM round trip 也会把注意力限制在
同一 tile 的 buffer 轮换。在已验证 workload 上，组件级流水将 focused case 从
998.637 us 降到 784.619 us。[^component-pipeline-result]

# 约束

- 适用：profile 显示 pipe 等待或同步覆盖过宽。
- 代价：更深预取增加容量；更细同步增加状态；mutex 可能序列化；少 barrier
  可能暴露竞态。
- 正确性门禁：多轮、首末轮、尾 tile、不同 block 和重复运行。
- 性能门禁：fresh profile 加同配置多次 benchmark。
- 使用 auto-sync 时必须审计生成 IR 中的 mutex/unit-flag，并保留与显式同步版本相同
  的正确性 case；自动插入不等于自动证明无竞态。
- 组件流水要求各 UB 物理独立；`free` 必须由最后访问该 UB 的 pipe 产生。存在 alias 或
  生命周期交叠时，不能仅靠不同 flag 宣称独立。

# 失败表现

- `unit_flag=0b11` 过早：FIX 可能读取未完成 accumulator。
- 最后一轮仍用 `0b10`：最终回写协议不闭合。
- set/wait pipe 方向反转：lowering 错误或永久等待。
- 删除 barrier 后偶发错：候选 flag 没有表达 barrier 原本覆盖的全部依赖。
- 正确但无提升：同步不是主瓶颈，或细粒度 flag 成本抵消收益。

# 验证方法

正确性失败的候选不得 benchmark。正确候选也只有超过测量方差与批准最小提升
阈值才更新 best-correct；源码本身不构成性能结论。

[^api]: 固定提交 Core API 中当前 range 和同步实现。
[^mmad]: 固定提交 MMAD 的 unit flag 与循环同步。
[^vadd]: 固定提交 VADD 的 flag、mutex 和 barrier 变体。
[^auto-sync]: 固定提交 `basic_matmul_auto_sync.py` 的 v0 自动同步入口与适用模式。
[^component-pipeline-kernel]: `chunk_gla_fwd_kernel_o` 中每组件独立 flag、load、compute、store 和 release 的实现。
[^component-pipeline-result]: 同环境 focused case 的完整正确性与 msprof 接受结果。
