---
type: CATLASS DSL Optimization Guide
title: AIV 向量布局与 MTE
description: 面向 AIV/vector-only 算子的布局、连续访问和任务粒度优化候选。
tags: [catlass-dsl, optimization, aiv, mte, vector, layout]
status: stable
generated: {by: human:caijianlong, at: '2026-08-06T00:00:00+08:00'}
verified:
  - {by: process:catlass-dsl-source-audit, at: '2026-08-06T00:00:00+08:00'}
sources:
  - id: dsl
    resource: https://gitcode.com/cann/catlass/blob/6ccf88e89723b65461e9921047c7970a71b67b42/python/tla_dsl/catlass/core_api.py
    title: CATLASS DSL vector examples and APIs
operator_families: [elementwise, scan, reduction]
arch: [c310]
---

# 接口与概念

适用于 scan、cumsum、elementwise、轻量归约和数据重排等主要运行在 AIV/vector 路径的
kernel。先从 profile 区分 `aiv_mte2_time`（GM 到 UB）、`aiv_mte3_time`（UB 到 GM）、
`aiv_vec_time` 和 `aiv_scalar_time`；`aicore_time=0` 或 `cube_util=0` 说明不应把 Cube
当作默认优化对象。[^dsl]

MTE2 高时优先检查输入布局和 load 粒度；MTE3 高时检查输出连续性、store 宽度和 task
切分；vector/scalar 高时检查 unpack、shuffle、分支、循环和地址计算。不要仅因总耗时高
就默认怀疑 Cube。

# 用法

每轮只提出一个可证伪轴，例如 `aiv-mte-compact-input-layout`、
`aiv-mte-wider-contiguous-store`、`aiv-task-coarsening-contiguous-dim`、
`aiv-vector-unpack-cost-reduction` 或 `aiv-scalar-address-simplification`。

从 Triton 或其它实现借鉴时，只提取数据访问形态：grid 对 batch/sequence/head/feature/group
的映射、每个 program/tile 覆盖的连续元素、mask 是否只处理边界、store 的连续程度，以及
scan/reduce/elementwise 的计算轴是否匹配连续内存轴。将这些信息转换为 CATLASS DSL 的
vector 长度、GM stride、task 映射和 UB 布局，不机械复刻 API。

# 候选策略

- GM 输入不要保存 padding、插零 lane 或对齐空洞；在 host/前序阶段压紧连续有效元素，再
  在 AIV 中用 cast、lane 拆分、interleave 或 reshape 恢复计算形态。
- 将多个连续真实小向量合并为更宽的 vector load，确保 GM 读取的都是有效元素。
- 合并 GM 上连续的 head、feature、channel、小 batch 或 group 维度，以减少 kernel task
  数和小块 store。sequence、chunk 或 tile 步长较大时，可在一个 task 中合并少量连续逻辑
  单元；报告时必须区分逻辑工作单元与实际 kernel task。
- 借鉴 Triton 时提取 grid 映射、连续 load/store 宽度、边界 mask 和计算轴，而非照搬 API。
- group/vector 粒度变大后必须比较 MTE、vector、scalar 和 UB 占用。MTE 下降而 vector 或
  scalar 上升导致总时延回退时，应拒绝该候选。

连续维合并的收益应体现在 task 数下降、row store 变宽、`aiv_mte3_time` 小块写回/地址
开销下降，且 device 内 task 循环更接近真实连续访问。只看逻辑二维或三维划分容易误判
实际并行度和搬运粒度。

# 代码模式

```text
axis_id: aiv-mte-wider-contiguous-store
hypothesis: 合并连续输出维度可降低 MTE3 和地址计算。
validation: 完整正确性后比较总时延、MTE3、vector 与 scalar 子项。
```

# 约束

任何改变输入元素数、task 映射或 buffer 大小的方案都必须更新 compile cache key。覆盖
`group=1`、非整除维度、dtype、功能开关、layout、小 shape 与性能目标 shape。先完整
正确性，再在同一配置比较总时延和 MTE/vector/scalar 子项。[^dsl]

增大 group/vector 粒度可能降低 MTE2/MTE3，但也会增加 AIV unpack/interleave/shuffle、
临时 vector 数、寄存器压力、分支与地址计算，或占用更多 UB 并降低并发。group 候选必须
逐个 profile 证伪，不得把局部子项下降当作接受依据。

# 失败表现

- 为满足 vector 宽度而保存 padding 或空洞，导致 MTE2 搬运无效 lane。
- group 增大后 unpack、shuffle 或地址计算抵消 MTE 收益。
- 仅性能目标 shape 通过，非整除边界或 `group=1` 路径错误。
- 忘记把 pack、layout、task 映射或 buffer 大小加入 cache key，复用了旧 kernel。

# 验证方法

单个大 case 的成功不构成通用结论。运行小 shape、非整齐 shape 和性能目标 shape 的完整
正确性，再在同一配置比较总时延和 MTE/vector/scalar 子项。引入新 layout、pack 或 task
映射时，同步更新 meta tensor 大小、host pack/unpack、golden 对比和 benchmark 中的任务
统计口径。[^dsl]

[^dsl]: 固定提交 CATLASS DSL 的向量 API 与端到端示例。
