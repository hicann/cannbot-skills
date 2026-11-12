---
type: CATLASS DSL Programming Concept
title: 跨执行引擎 Credit 与 Publication
description: 区分本地流水依赖、buffer credit、跨核 ready 信号和阶段性 GM publication，避免将 barrier 与 flag 混作全局同步。
tags: [catlass-dsl, dsl, credit, publication, cross-flag, pipe-barrier, synchronization, visibility]
status: stable
generated: {by: human:maintainer, at: '2026-09-20T00:00:00-07:00'}
verified:
  - {by: human:maintainer, at: '2026-09-20T00:00:00-07:00'}
sources:
  - id: api
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/docs/en/api/kernel_api_reference.md
    title: CATLASS kernel synchronization API
  - id: mixed
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py
    title: CATLASS mixed Cube Vector example
  - id: sync
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/dsl_tutorials/cross_flag_two_way/cross_flag_two_way.py
    title: CATLASS two-way cross flag example
operator_families: [matmul, sparse-attention, elementwise]
arch: [c310]
---

# 接口与概念

同步需区分四类语义：同核 pipe 依赖、物理 buffer 再使用 credit、Cube/Vector 跨核 ready、
阶段结束后 consumer 可读取 GM workspace 的 publication。它们可以相邻，但不能假定由同一个
flag 或 barrier 自动同时满足。[^api][^mixed][^sync]

# 用法

先声明资源和所有权，再选择原语：

| 需求 | 首选表达 |
| --- | --- |
| 同核 producer 到 consumer | `flag/set_flag/wait_flag` |
| Cube/FIX 与 Vector 交接 | `cross_flag` 配对 |
| 同执行域流水排空 | `pipe_barrier` |
| 全参与核阶段结束 | 经过固定示例或最小测试证明的全核协议 |

# 代码模式

跨核 ready 采用唯一 producer 和明确 consumer pipe；buffer free 另设反向 credit。阶段 post
在最后一个 producer 完成后发布，consumer 只等待一次阶段信号，不在每个输出元素上同步。

# 约束

- `pipe_barrier(ALL)` 不等于任意 AIV/AIC 的全局内存栅栏。
- cross flag 的 mode、pipe 和 `aiv_id` 必须与固定 API/示例一致。
- startup publication 与循环 ready 信号分开设计。
- atomic producer 完成前不得启动读取同一 workspace 的 post。
- 未有固定来源证明的 AIV-to-Cube 或 mode-0 用法必须先做独立最小测试。

# 失败表现

- workspace 最终 round-trip 正确但计算消费错误：publication 或 consumer view 问题。
- 增加 barrier 后死锁：参与核集合、set/wait 次数或执行域假设错误。
- 首轮读取旧值：初始化 store 尚未 publication。
- post 偶发缺少部分累加：atomic/Fixpipe 未 drain。

# 验证方法

分别测试单 producer、双 AIV、单 AIC consumer和多 task；用 sentinel 区分未写、旧值和错误地址。
验证顺序为最小同步测试、focused correctness、重复运行、完整 workload和 timeline。

[^api]: 固定提交 API 文档中的 flag、cross flag 与 barrier 接口。
[^mixed]: 固定提交 mixed 示例中的 Cube/FIX/Vector 同步组合。
[^sync]: 固定提交双向 cross flag 示例中的显式交接模式。
