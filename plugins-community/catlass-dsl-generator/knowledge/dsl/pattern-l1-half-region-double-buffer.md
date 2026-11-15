---
type: CATLASS DSL Programming Concept
title: L1 半区偏移与双缓冲 flag 配对
description: 用固定字节偏移把 L1 分成两个半区，每半区放 A 与 B tile，配合两套 flag 实现搬运与计算重叠。
tags: [catlass-dsl, l1, double-buffer, flags, pipeline]
status: stable
generated: {by: process:catlass-dsl-pattern-capture, at: '2026-10-23T15:40:00+08:00'}
verified:
  - {by: process:catlass-dsl-kernel-run-audit, at: '2026-10-23T15:40:00+08:00'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/catlass/core_api.py
    title: flag 的 pipe 配对与 allocate 偏移
dsl_version: ascend-catlass-dsl 2.1.0
arch: [c310]
---

# 接口与概念

L1 双缓冲的常见实现是"一块大 allocation + 固定偏移切半区"，而不是两次 allocate。
每个半区内部再按 A 在前、B 在后的顺序排布，偏移量在编译期算好。

本条描述 flag 配对写法；骨架与融合 kernel 默认改用 mutex，每份缓冲一个 mutex、`lock`/
`unlock` 的 pipe 成对，半区切分方式不变。仅在允许 flag 的 kernel 中使用下面的写法。

# 用法

```python
l1 = tla.allocate(262144, tla.BFloat16, tla.AddressSpace.l1, 512)   # 512KB
a_off = 0
b_off = bm * bk                    # 紧跟 A
half  = 131072                     # 每半区元素数

al = tla.make_tensor_like(l1 + slot * half + a_off, ar, tla.arch.zN)
bl = tla.make_tensor_like(l1 + slot * half + b_off, br, tla.arch.nZ)
```

配套两套 flag，每半区一组，命名与 pipe 一一对应：

    ready0/ready1   : MTE2 -> MTE1   (搬运完成)
    l1free0/l1free1 : MTE1 -> MTE2   (半区可覆盖)

# 代码模式

见上文；每半区独立 set/wait，任一时刻两个半区处于"一个在算、一个在搬"。

# 约束

- half 必须 >= 半区内所有 tile 字节数之和，并按 512B 对齐（allocate 第三参数）。
- 每半区的 flag 必须成对：用了 ready0 就必须有 l1free0，否则半区永不复用。
- 预取距离与 flag 数量匹配：只预取两段就用两套 flag，不要再加第三套。

# 失败表现

- 偏移算错：A/B tile 互相覆盖，数值全错但编译通过。
- flag 不成对：第二轮起半区不被覆盖，读到旧数据，表现为"第一轮对、后面错"。

# 验证方法

把偏移与各 tile 字节数列成表核对总和 <= half；最小 shape 连跑两轮以上，
确认第二轮与第一轮一致。

[^api]: 固定提交的 allocate 与 flag 实现。
