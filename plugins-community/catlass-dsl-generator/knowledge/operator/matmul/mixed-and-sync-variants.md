---
type: CATLASS DSL Operator Example
title: Mixed、atomic add 与 mutex 变体
description: Cube/Vector 混合流水、跨核 flag、atomic add 和 mutex 的源码样例入口。
tags: [catlass-dsl, operator, mixed, atomic-add, mutex]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
sources:
  - id: mixed
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mixed/basic_mixed.py
    title: Basic mixed Cube and Vector kernel
  - id: atomic
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul_atomic_add.py
    title: MMAD atomic-add variant
  - id: mutex
    resource: https://gitcode.com/cann/catlass/blob/7b574fb3547e76bff47c8514b07741d123a2766b/python/tla_dsl/examples/end_to_end/basic_mmad/basic_matmul_mutex.py
    title: MMAD mutex variant
operator_families: [matmul, mixed, elementwise]
arch: [c310]
---

# 接口与概念

## 算子算法

`basic_mixed` 在同一 kernel 中使用 `with tla.cube()` 完成 MMAD 和 L0C->UB，
再用 cross flag 通知 `with tla.vector()` 读取 UB、叠加 addend 并回写 GM。[^mixed]
独立 MMAD 文件展示 atomic-add copy 参数和 mutex 同步变体。[^atomic][^mutex]

# 用法

## 分核策略与基本块切分

Mixed 模式适合确有 Cube 结果后处理的算子；atomic add 适合多个 producer 对同一
输出做累加；mutex 适合需要资源互斥且 flag 所表达的单向依赖不够用的场景。
host block 映射 Cube 任务，两个 Vector sub-block 通过 `sub_block_idx()` 切分同一
输出 tile，通常沿 M 维分别处理上下半区。[^mixed]

# 代码模式

## 数据路径与存储层级

Mixed 路径为 `GM A/B -> L1/L0 -> MMAD -> L0C -> FIX -> shared UB -> Vector -> GM`；
atomic 变体从 L0C 直接累加写 GM，mutex 变体不改变存储路径，只改变本地资源所有权。

### Cube/Fix 到 Vector

```python
done = tla.cross_flag("fix_done", mode=2)
with tla.cube():
    tla.mmad(l0_c, l0_a, l0_b, init_c=True)
    tla.copy(ub_c, l0_c)
    tla.cross_core_set_flag(done, tla.arch.FIX)
with tla.vector():
    tla.cross_core_wait_flag(done, tla.arch.VECTOR)
    with tla.vec.func(mode="simd"):
        result_ub.store(tla.add(ub_c.load(), addend_ub.load()))
    tla.copy(gm_result, result_ub)
```

## 流水排布、同步关系与数值精度

Cube/FIX 与 Vector 通过 cross flag 交接 shared UB；同核 L0A/L0B/L0C 访问由 flag
或 mutex 串联。MMAD 以 f16 输入、f32 L0C 累加，Vector 后处理与最终输出转换遵循
目标 tensor dtype。[^mixed][^mutex]

### Atomic add

```python
tla.copy(
    gm_c,
    l0_c,
    tla.params.CopyL0C2DstParams(
        unit_flag=0b11,
        atomic_mode=tla.params.AtomicMode.ADD,
    ),
)
```

输出必须在 launch 前清零；若先已有 baseline `C0`，oracle 是
`C_final = C0 + contribution`，不是覆盖语义。[^atomic]

### Mutex 资源顺序

```python
mutex_a = tla.mutex(resource="l0a", id=4)
mutex_b = tla.mutex(resource="l0b", id=6)
mutex_c = tla.mutex(resource="l0c", id=8)

with tla.mutex_guard(mutex_a, mutex_b, mutex_c):
    tla.mmad(l0_c, l0_a, l0_b, init_c=first_k)
with tla.mutex_guard(mutex_c):
    tla.copy(gm_c, l0_c)
```

多个 mutex 始终按同一全局次序传入，释放由 guard 逆序完成。[^mutex]
# 约束

- mixed kernel 的共享本地地址、sub-block 映射和跨核协议必须成套设计。
- 当前 mixed 示例的 Vector 侧使用 `sub_block_idx()` 分片，不能让两个 AIV 写同一区域。
- atomic add 要求输出初始化与 reference 采用累加语义。
- mutex 的资源名、ID、加解锁顺序必须稳定，避免环形等待。

# 失败表现

- cross flag source/destination pipe 不符会 lowering 失败或永久等待。
- 共享 UB 分片错误会产生 AIV 间覆盖。
- atomic 输出未清零会叠加历史值。
- mutex 顺序不一致可能形成死锁；漏掉 L0C mutex 会让 FIX 与 Cube 并发访问。

# 验证方法

先分别验证 Cube 与 Vector 子图，再验证跨核组合；对 atomic 重复运行并显式清零；
对 mutex 检查所有退出路径。性能收益不能由这些源码存在性推出。

[^mixed]: 固定提交 mixed kernel 的 Cube/Vector 与 cross flag 数据流。
[^atomic]: 固定提交 MMAD atomic-add copy 变体。
[^mutex]: 固定提交 MMAD mutex 同步变体。
