---
type: CATLASS DSL Optimization Guide
title: Buffering 与数据搬运
description: L1/L0/UB buffering、复用与 GM 流量优化的可证伪策略。
tags: [catlass-dsl, optimization, buffering, copy, memory-traffic]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: mmad
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul.py
    title: MMAD buffering example
  - id: mixed
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mixed/basic_mixed.py
    title: Mixed Cube and Vector data movement
operator_families: [matmul, mixed, elementwise]
arch: [c310]
---

# 接口与概念

MMAD 源码为 A/B 的 L1 与 L0 分配双 buffer，通过交替索引连接搬运和计算。[^mmad]
Mixed 示例展示 L0C->UB 后由 Vector 直接消费，避免先回 GM 再读回的中间路径。[^mixed]

# 用法

从 profile 判断瓶颈后选择单轴假设：增加/减少 stage、扩大复用 tile、消除重复
copy、改变 L0C->UB/GM 路径，或让 Cube/Vector 共享已生成的本地结果。

# 代码模式

## 双缓冲状态

```python
ready0 = tla.flag("buf0_ready", tla.arch.MTE2, tla.arch.VECTOR)
ready1 = tla.flag("buf1_ready", tla.arch.MTE2, tla.arch.VECTOR)
released0 = tla.flag("buf0_released", tla.arch.VECTOR, tla.arch.MTE2)
released1 = tla.flag("buf1_released", tla.arch.VECTOR, tla.arch.MTE2)
tla.set_flag(released0)
tla.set_flag(released1)
buf = 0
for tile_id in tla.range(tile_count):
    if buf == 0:
        tla.wait_flag(released0)
        current = ub0
    else:
        tla.wait_flag(released1)
        current = ub1
    source = tla.tile_view(gm, tile_shape, tla.make_coord(tile_id))
    tla.copy(current, source)
    if buf == 0:
        tla.set_flag(ready0)
        tla.wait_flag(ready0)
    else:
        tla.set_flag(ready1)
        tla.wait_flag(ready1)
    with tla.vec.func(mode="simd"):
        current.store(tla.neg(current.load()))
    if buf == 0:
        tla.set_flag(released0)
    else:
        tla.set_flag(released1)
    buf = 1 - buf
```

这表达资源所有权轮换；是否真正重叠必须通过 profile 验证。[^mmad]

## 消除 GM round trip

```python
with tla.cube():
    tla.mmad(l0_c, l0_a, l0_b, init_c=True)
    tla.copy(shared_ub, l0_c)
    tla.cross_core_set_flag(fix_done, tla.arch.FIX)
with tla.vector():
    tla.cross_core_wait_flag(fix_done, tla.arch.VECTOR)
    with tla.vec.func(mode="simd"):
        shared_ub.store(tla.add(shared_ub.load(), bias_ub.load()))
    tla.copy(gm_out, shared_ub)
```

它少一次 GM 写和读，但增加共享 UB 容量与跨核同步。[^mixed]
# 约束

- 适用：搬运等待明显，且容量允许同时驻留 producer/consumer buffer。
- 代价：双缓冲使容量近似翻倍，并增加 flag/mutex 状态复杂度。
- 正确性门禁：buffer index、首轮填充、末轮排空和尾块全部覆盖。
- 性能门禁：profile 应显示等待/流量变化，benchmark 应达到批准阈值。

# 失败表现

- 首轮未初始化 `released` flag：第一次 wait 阻塞。
- copy 后、compute 前提前切换 index：消费者读取另一 buffer。
- 末轮不等待 compute/release：kernel 返回时仍有未完成访问。
- 更多 buffer 超容量或降低可并行 block 数。

# 验证方法

IR 中检查 `wait released -> copy -> set ready -> wait ready -> compute ->
set released`；正确性覆盖 tile_count=1、2、3 和尾 tile，再比较 MTE/Vector/Cube
等待指标及 benchmark。

[^mmad]: 固定提交 MMAD 的 L1/L0 双缓冲与同步。
[^mixed]: 固定提交 mixed kernel 的 L0C->UB->Vector 数据路径。
