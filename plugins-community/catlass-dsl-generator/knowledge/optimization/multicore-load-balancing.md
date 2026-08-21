---
type: CATLASS DSL Optimization Guide
title: CANN 样例：多核尾轮与 Split-K 调度候选
description: 从 CANN Features 样例提取尾块再切分、DP+Split-K 混合调度及 workspace 归并候选。
tags: [catlass-dsl, optimization, matmul, multicore, load-balance, split-k, tail]
status: draft
generated: {by: process:cann-samples-feature-extract, at: '2026-08-12T00:00:00Z'}
verified:
  - {by: process:cann-samples-source-audit, at: '2026-08-12T00:00:00Z'}
sources:
  - id: tail-rebalance
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/system_optimization/tail_rebalance/main.asc
    title: Tail-tile rebalance implementation
  - id: stream-k
    resource: https://gitcode.com/cann/cann-samples/blob/928d8dfa322731f576b697c9ec997d34abd810b7/Samples/1_Features/system_optimization/streamk/main.asc
    title: DP and Split-K mixed scheduling implementation
operator_families: [matmul, mixed]
arch: [c310]
---

# 接口与概念

尾轮再平衡把最后一轮不足核数的 M/N tile 再切成二维子块，增加逻辑任务数，并为每个子块
重算 GM 起点、实际 M/N 和 L0C layout。[^tail-rebalance]
Stream-K 则只对不足核数的 M/N 尾部沿 K 再切，使各核产生部分累加；固定 mixed 实现把部分
结果写入 workspace，再由 AIV 按重映射坐标读取和归并。[^stream-k]

两者解决的是“末轮空闲核”，但代价不同：M/N 再切通常不增加归约；Split-K 增加 workspace、
归约和可能的累加顺序变化。它们必须作为互斥的首轮候选比较。

# 用法

先计算每轮活跃任务数及 `tail_tasks / core_count`。只有末轮占总时长不可忽略且 profile 显示明显
核间负载不均时才实验：输出 tile 可继续合法切分时优先 M/N 再切；M/N 已太小、K 足够长且
workspace/归约可接受时再测 Split-K。

# 代码模式

```python
full_round_tasks = (task_count // core_count) * core_count
tail_tasks = task_count - full_round_tasks

# 候选 A：把每个尾 M/N tile 拆成 tail_m * tail_n 个互不重叠输出子块。
subtask = logical_tail_id % (tail_m * tail_n)
m_sub = subtask % tail_m
n_sub = subtask // tail_m
m0, m_len = clipped_partition(original_m, tail_m, m_sub)
n0, n_len = clipped_partition(original_n, tail_n, n_sub)

# 候选 B：沿 K 切分；每个 (m,n,k_part) 有唯一 workspace slot。
k_part = logical_id % split_k
mn_id = logical_id // split_k
partial = mm(a[mn_id, k_part], b[k_part, mn_id], accumulate_fp32=True)
workspace[mn_id, k_part] = partial
barrier_or_cross_core_flag()
out[mn_id] = reduce_k_parts(workspace[mn_id, :])
```

固定实现对线性 `tileIdx` 做 DP 区和 Split-K 尾区的双向映射，并要求 AIC 写入坐标与 AIV
读取坐标一致。[^stream-k] 尾轮再切实现会跳过超出原 tile 边界的空子块，并以实际子块大小
重建张量视图。[^tail-rebalance]

# 约束

- 适用范围：`c310` MatMul/mixed；shape 必须造成显著尾轮且有足够可切工作。
- 保持条件：每个输出元素被 M/N 子块恰好覆盖一次；Split-K 每个部分和 workspace slot 唯一，
  归约发生在所有 producer 完成后；AIC/AIV 使用同一坐标函数。
- 精度：Split-K 改变浮点累加顺序；必须用原算子的 dtype/容差合同判断，不能默认 bitwise 相同。
- 资源代价：M/N 再切增加启动/地址开销并可能生成过小块；Split-K 增加 workspace 字节、GM 往返、
  同步和 AIV 归约，且 K 太短时每份计算不足。
- 可证伪预期：尾轮活跃核数提高、最长核时长下降，新增归约/地址开销未抵消收益。

# 失败表现

- 输出缺口、重叠或只在尾 M/N 错：子块覆盖/裁剪公式错误，恢复原调度。
- Split-K 只在多核重复运行偶发错：producer/consumer 同步或 workspace slot alias。
- 正确但更慢：尾轮占比小、子块过碎或 workspace 往返超过空闲核收益。
- 精度越界：累加顺序或中间写回 dtype 不满足合同；减少 split 或回退 DP。
- AIC/AIV 读取不同块：统一使用单个可测试的坐标映射函数。

# 验证方法

用小尺寸在 host 枚举所有逻辑任务，证明 M/N 覆盖无重叠无缺口、所有 K part 恰好归约一次；再覆盖
`task_count < core_count`、整除、余 1、余 `core_count-1` 和 M/N/K tail。上板先过完整正确性，
再比较 per-core timeline、末轮活跃核、workspace MTE2/MTE3、AIV 归约与总 kernel latency。
对多个相邻 shape 扫描，避免只为单个余数形态过拟合；fresh best 必须复测。

[^tail-rebalance]: 固定提交中的尾 M/N 子块数量、任务重映射、边界裁剪和 L0C 重建。
[^stream-k]: 固定提交中的 DP/Split-K 区分、workspace 布局及 AIC/AIV 坐标重映射。
