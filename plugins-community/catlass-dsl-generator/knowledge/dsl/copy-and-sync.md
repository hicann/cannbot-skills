---
type: CATLASS DSL Programming Concept
title: Copy、flag、mutex 与流水同步
description: 数据搬运、同核 flag、跨核 flag、mutex、barrier 和流水循环的使用边界。
tags: [catlass-dsl, copy, synchronization, pipeline]
status: stable
generated: {by: process:catlass-dsl-source-extract, at: '2026-08-10T00:00:00Z'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-10T00:00:00Z'}
  - {by: process:catlass-dsl-source-audit, at: '2026-08-29T00:00:00Z'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/docs/en/api/kernel_api_reference.md
    title: CATLASS DSL API Reference
  - id: vadd
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/examples/end_to_end/basic_vadd/basic_vadd.py
    title: Basic VADD synchronization examples
  - id: core
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: Synchronization implementation
  - id: params
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/params.py
    title: Copy and local-memory parameter types
operator_families: [elementwise, matmul]
arch: [c310]
---

# 接口与概念

`copy(dst, src, params=None)` 表达存储层级间搬运。同步接口包括
`flag`/`set_flag`/`wait_flag`、`cross_flag`/`cross_core_set_flag`/
`cross_core_wait_flag`、`pipe_barrier`、`mutex`、显式 lock/unlock 和
`mutex_guard`。当前源码中的 `tla.range` 只支持 Python range 的一至三个位置
参数；生成 API 文档里的 `prefetch_stages` 与 `pipelining` 已与实现漂移，不能
作为当前接口使用。[^api][^core]

# 用法

## 同核 flag

```python
ready = tla.flag("ready", tla.arch.MTE2, tla.arch.VECTOR)
done = tla.flag("done", tla.arch.VECTOR, tla.arch.MTE3)
with tla.vector():
    tla.copy(ub, gm)       # MTE2
    tla.set_flag(ready)
    tla.wait_flag(ready)   # VECTOR consumes UB
    with tla.vec.func(mode="simd"):
        ub.store(tla.exp(ub.load()))
    tla.set_flag(done)
    tla.wait_flag(done)    # MTE3 consumes UB
    tla.copy(gm_out, ub)
```

## Cross flag

```python
fix_done = tla.cross_flag("fix_done", mode=2)
with tla.cube():
    tla.copy(ub_from_l0c, l0_c)
    tla.cross_core_set_flag(fix_done, tla.arch.FIX)
with tla.vector():
    tla.cross_core_wait_flag(fix_done, tla.arch.VECTOR)
    with tla.vec.func(mode="simd"):
        out.store(ub_from_l0c.load())
```

## Mutex

```python
buffer_mutex = tla.mutex(resource="shared_ub", id=0)
with tla.vector():
    with tla.mutex_guard(buffer_mutex):
        tla.copy(ub, gm)
    with tla.mutex_guard(buffer_mutex):
        with tla.vec.func(mode="simd"):
            ub.store(tla.neg(ub.load()))
```

# 代码模式

## Copy 参数

```python
# UB -> GM 累加而非覆盖
tla.copy(
    gm_out,
    ub,
    tla.params.CopyUbToGmParams(
        atomic_mode=tla.params.AtomicMode.ADD,
    ),
)

# L0C -> UB，按 M 方向 split
tla.copy(
    ub,
    l0_c,
    tla.params.CopyL0C2DstParams(
        unit_flag=0b11,
        l0c2ub_mode=tla.params.L0C2UBMode.SPLIT_M,
    ),
)
```

## Local memory barrier

`local_mem_bar` 只允许在 `tla.vec.func` 中使用，参数是明确的访问类型：

```python
with tla.vec.func(mode="simd"):
    ub.store(values)
    tla.local_mem_bar(
        tla.params.MemType.VEC_STORE,
        tla.params.MemType.VEC_LOAD,
    )
    reloaded = ub.load()
```

`CopyUbToGmParams`、`CopyL0C2DstParams`、`AtomicMode`、`L0C2UBMode` 与
`MemType` 的有效枚举和值由 params 模块约束。[^params]

VADD 源码同时给出 flag、mutex 和 `mutex_guard` 三种同步写法。[^vadd]

# 约束

- flag 的 producer/consumer pipe 必须与实际操作一致。
- cross flag 用于 Cube/Fix/Vector 等跨核域协作；pipe 在 set/wait 调用处指定。mode 4
  还要求各端显式传 `aiv_id=0` 或 `aiv_id=1`，不可替代普通同核依赖。
- `mutex_guard` 的 body 必须包含可推断 pipe 的 `copy` 或 `mmad`。
- 同一个 guard 内不能混入无法唯一推断 pipe 的不兼容访问。
- atomic add 要求输出初值和 reference 都采用累加语义。
- barrier 是强同步；不能把它当作性能默认值。

# 失败表现

- `mutex_guard body must emit ...`：guard 内没有可推断 pipe 的 copy/MMAD/vector。
- `unsupported src and dst`：`local_mem_bar` 使用了未编码的 `MemType` 组合。
- 缺同步可产生旧数据、覆盖或竞态；错误 producer/consumer pipe 可能等待不匹配。
- 过度 barrier/mutex 通常正确但会压缩重叠空间。

# 验证方法

用 TLAIR 检查 set/wait 配对及 pipe，运行同步相关 pytest/lit，再以多 tile
边界 case 检查数据依赖。性能选择必须通过 benchmark/profile 另行确认。

[^api]: 固定提交 API 文档中的 copy、同步与 range 接口。
[^vadd]: 固定提交 VADD 示例中的 flag、mutex、guard 与 barrier 数据流。
[^core]: 固定提交 Core API 中当前同步与 range 实现。
[^params]: 固定提交 params 模块中的 copy 参数与 `MemType`。
