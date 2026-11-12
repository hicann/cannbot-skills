---
type: CATLASS DSL Debugging Guide
title: Cube Vector 死锁与 Credit 审计
description: 用资源所有权表审计 Cube、FIX、MTE 与 Vector 的 set/wait、启动 credit、循环归还和阶段排空。
tags: [catlass-dsl, debug, deadlock, hang, timeout, credit, cross-flag, cube, vector, publication]
status: stable
generated: {by: human:maintainer, at: '2026-09-20T00:00:00-07:00'}
verified:
  - {by: human:maintainer, at: '2026-09-20T00:00:00-07:00'}
sources:
  - id: mixed
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/tla/kernel/matmul/basic_mixed/basic_mixed.py
    title: CATLASS mixed cross-core flag pattern
  - id: sync
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/dsl_tutorials/cross_flag_two_way/cross_flag_two_way.py
    title: CATLASS two-way cross flag example
operator_families: [sparse-attention, matmul]
arch: [c310]
---

# 接口与概念

flag 是特定资源从 producer 到 consumer 的所有权边，不是抽象的“同步一下”。同核 flag 和
cross flag 都必须对应明确 pipe、物理 buffer、轮次和归还距离。[^mixed][^sync]

# 用法

为每条 flag 建立表：

```text
name | resource/slot | producer pipe | set condition | consumer pipe |
wait condition | startup credit | return distance | final drain
```

审计顺序为 startup、steady state、skip path、drain。第一轮没有前序消费者，所需 credit
必须显式预置或通过首轮特殊路径获得；循环 credit 只能由真实完成的消费者归还。

# 代码模式

双槽资源 `slot=r%2` 的基本不变量：轮次 `r` 写入前，轮次 `r-2` 的最后消费者已经归还该槽。
跨核数据 ready 与本地 buffer free 分开建 flag，不能用同一个事件同时表达两种生命周期。

删除计算阶段时，逐项检查该阶段是否还承担 UB/L1/L0 的 ownership 归还；即使算术结果已由
workspace 取代，资源 credit 仍可能是后续 task 的必要条件。

# 约束

- set/wait 不得位于不对称的运行时分支。
- skip tile 时双方必须同时跳过，或执行对称的空握手。
- `pipe_barrier` 只处理声明执行域内的流水依赖，不自动形成任意跨核 publication。
- mode-0 全核 barrier 只能用于固定提交或最小测试已证明的参与域，不能替代逐槽 credit。
- timeout、无输出和 profiler 停滞只能证明未完成，不能直接证明某条 flag 是根因。

# 失败表现

- 首 task hang：startup credit 或初始化 publication 缺失。
- 第二/第三 task 固定 hang：distance 与物理槽数不一致，或 credit 未归还。
- 稀疏 skip 才 hang：set/wait 分支不对称。
- 计算结束后 hang：最后 Fixpipe/MTE3/atomic 未 drain，或 post phase 等待未发布。

# 验证方法

对 0/1/2/3 个有效 task、连续有效、交替 skip 和最后 task skip 分别核对计数。使用短超时，
一次只改变一条依赖边；通过后再做重复运行、完整正确性和 timeline 等待位置检查。

[^mixed]: 固定提交 mixed 示例中的 Cube/FIX 与 Vector cross-core 交接。
[^sync]: 固定提交双向 cross flag 示例中的 producer/consumer 配对。
