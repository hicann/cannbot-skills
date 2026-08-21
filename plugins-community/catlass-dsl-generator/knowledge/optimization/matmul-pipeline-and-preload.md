---
type: CATLASS DSL Optimization Guide
title: CANN 样例：MatMul 深流水与预取候选
description: 从 CANN Features 样例提取 N-buffer、MMAD/Fixpipe unit flag 和跨 tile MTE2 预取实验。
tags: [catlass-dsl, optimization, matmul, pipeline, n-buffer, unit-flag, preload]
status: draft
generated: {by: process:cann-samples-feature-extract, at: '2026-08-12T00:00:00Z'}
verified:
  - {by: process:cann-samples-source-audit, at: '2026-08-12T00:00:00Z'}
sources:
  - id: n-buffer
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/instruction_optimization/n_buffer/main.asc
    title: N-buffer MatMul pipeline
  - id: unit-flag
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/instruction_optimization/unit_flag/main.asc
    title: MMAD and Fixpipe unit-flag pipeline
  - id: mte2-preload
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/instruction_optimization/mte2_preload/main.asc
    title: MTE2 cross-iteration preload pipeline
operator_families: [matmul]
arch: [c310]
---

# 接口与概念

`n_buffer` 不是“buffer 越多越快”，而是把 stage slot、空闲/就绪事件、首轮填充和末轮排空
组成所有权协议；固定实现分别维护 L1、L0 和 L0C/Fixpipe 阶段。[^n-buffer]
`unit_flag` 在非最终累加和最终累加使用不同值，并让 MMAD 与 Fixpipe 按硬件协议衔接；最后一轮
边界是正确性合同。[^unit-flag]
`mte2_preload` 在当前 K tile 计算前先向备用 L1 slot 提交下一 K tile，且在 tile 域末尾预取
下一个输出 tile 的首个 K 分片，用于测试 MTE2 指令队列堵塞假设。[^mte2-preload]

# 用法

- profile 显示 MTE2/MTE1/Cube 阶段间存在可隐藏空洞时，先扫描 1/2/3 个 stage；
- Fixpipe 与最终 MMAD 串行且当前 DSL/硬件协议匹配时，单独测试 unit flag；
- ping 与 pong 之间指令过多、PONG 发射明显滞后，而缩小 KL1 会伤害性能时，测试跨迭代预取。

实验顺序固定为：建立 baseline → 只改 stage 数 → 恢复 baseline 后只改 unit flag → 恢复后只改
预取距离。组合候选必须在各单轴候选均正确后另开实验。

# 代码模式

```python
slots = [l1_slot(i) for i in range(stage_count)]
free = [flag(f"free_{i}", MTE1, MTE2) for i in range(stage_count)]
ready = [flag(f"ready_{i}", MTE2, MTE1) for i in range(stage_count)]
for event in free:
    set_flag(event)

# fill/steady/drain 都以逻辑 tile id 推导，避免手写错位的 ping/pong 分支。
for logical in range(tile_count + prefetch_distance):
    if logical < tile_count:
        produce_slot = logical % stage_count
        wait_flag(free[produce_slot])
        copy(slots[produce_slot], gm_tile(logical))
        set_flag(ready[produce_slot])
    consume = logical - prefetch_distance
    if consume >= 0:
        consume_slot = consume % stage_count
        wait_flag(ready[consume_slot])
        copy(l0, slots[consume_slot])
        mmad(..., unit_flag=3 if is_final_k(consume) else 2)
        set_flag(free[consume_slot])
```

固定源码表明预取目标不仅可能是 `k+1`，还可能在当前输出 tile 的最后一个 K 分片时切到下一
输出 tile；此时 M/N/K tail layout 和 GM 起点都要重新计算。[^mte2-preload]

若使用 unit flag，Fixpipe 侧必须使用与 MMAD 最终标记配对的参数；不能只删除显式 barrier，
也不能把最终标记提前到非末次累加。[^unit-flag]

# 约束

- 适用范围：`c310` MatMul；dtype/layout 任意变化都需重新验证；前提是 profile 支持流水或队列瓶颈。
- 保持条件：每个 slot 只有一个 owner；消费者等待对应 producer；最后访问者释放 slot；首轮不读
  未填充 slot；末轮不越界预取；最终累加和 Fixpipe 协议一致。
- 资源代价：stage 增加 L1/L0 容量和事件数，可能降低并发 block；预取扩大 live range 和指令体积；
  更复杂的标量地址逻辑可能提高 scalar 或 icache 开销。
- `unit_flag` 值和 pipe 语义必须以当前 CATLASS lowering 为准，不能机械复制 Ascend C 常量。
- 预期观测：空洞/等待或 PONG 发射延迟缩短，且 kernel latency 改善超过噪声；单看某条 pipe
  时间变长或变短不能证明端到端收益。

# 失败表现

- 首轮 hang：空闲事件未初始化或 wait/set 方向错误。
- 第二轮起错误：slot index、预取 tile 和消费 tile 错一位。
- 仅最后 K 错：最终 unit flag 或 accumulator drain 错误。
- 仅尾 tile 错：跨 tile 预取沿用了上一 tile 的 layout/offset。
- 编译膨胀、icache/scalar 上升或并发下降抵消重叠：减少 stage/预取距离或完全回退。
- profile 无队列空洞变化：预取假设被证伪。

# 验证方法

覆盖 K tile 数为 1、2、`stage_count`、`stage_count+1`，输出 tile 数为 1 和多轮，以及所有
M/N/K tail。检查 IR 中的 copy/flag/unit-flag 顺序，重复执行排除竞态。正确后，在同一设备、
shape、频率与采集边界下比较 kernel、MTE2、MTE1、Cube、Fixpipe、scalar 和 icache；用多次
benchmark 的方差定义接受阈值，并 fresh 复测最终 best。

[^n-buffer]: 固定提交中的多级 slot、事件初始化、循环流水与收尾实现。
[^unit-flag]: 固定提交中的中间/最终 MMAD 标记和配对 Fixpipe 参数。
[^mte2-preload]: 固定提交中的 K 内预取及跨输出 tile 首分片预取。
