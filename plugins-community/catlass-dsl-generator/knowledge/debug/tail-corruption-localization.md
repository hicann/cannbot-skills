---
type: CATLASS DSL Debugging Guide
title: 尾块数据损坏定位
description: 按 batch、head、tile、row 和 lane 聚合误差，区分源 workspace、搬运、计算消费和写回阶段的尾块损坏。
tags: [catlass-dsl, debug, tail, corruption, row-16, valid-lane, staging, visibility]
status: stable
generated: {by: human:maintainer, at: '2026-09-20T00:00:00-07:00'}
verified:
  - {by: human:maintainer, at: '2026-09-20T00:00:00-07:00'}
sources:
  - id: layout
    resource: https://gitcode.com/cann/catlass/blob/81da64bca9da5c782f6589541b967456d4fdc4c7/python/tla_dsl/docs/en/core_concepts/tensor_layout.md
    title: CATLASS Tensor and Layout concepts
  - id: simt
    resource: https://gitcode.com/cann/catlass/blob/master/examples/python/tla_dsl/dsl_tutorials/vector_simt_ops/basic_vadd_simt/basic_vadd_simt.py
    title: CATLASS SIMT scalar access example
operator_families: [sparse-attention, matmul, elementwise]
arch: [c310]
---

# 接口与概念

全局 matched ratio 会隐藏结构化尾块错误。诊断必须恢复到逻辑坐标，分别统计 batch、head、
query/key tile、row 和 lane 的 mismatch 数量。固定边界处开始损坏通常比最大误差更有定位价值。

# 用法

先输出每个 tile 的首错行、末错行和逐行 mismatch 数，再按以下指纹分流：

- 整个尾 tile 错：基址、stride、tile layout 或 staging 目标错误。
- 从 16/32/64 行边界开始错：搬运粒度、half split、NZ/row-major 映射或 lane mask。
- 只有最后一行错：有效 extent、unaligned store 或 off-by-one。
- 偶发完整行错：publication、buffer 复写或原子 post 时序。

# 代码模式

使用 kernel-end round-trip 检查 staging workspace，只能证明最终 GM 内容正确；还需单独验证
consumer 读取前的可见性和 L1/UB tensor view。SIMT 标量 copy 适合验证不规则短尾，但必须按
逻辑 row/column 计算源 stride，并在进入下一 pipe 前完成必要同步。[^simt]

测试尾长分桶：`1..15`、`16`、`17..63`、`64`、`65..127`。计算 tile 可保持固定尺寸，
无效 lane 用 mask 或零填充隔离；不要让 storage layout 随有效行数改变。[^layout]

# 约束

- workspace round-trip 正确不证明 Cube/Vector consumer 读取正确。
- 不同时替换 DMA、layout、同步和 post 写回。
- 源 token-major 与目标 grouped layout 必须分别写出 stride 公式。
- 诊断写回不得保留在最终 kernel 中。

# 失败表现

- staging 后误差扩大到整个尾 tile：目标 layout、零填充或 DMA 覆盖范围错误。
- SIMT remainder 改善但未修复：至少存在一个独立的 aligned-copy 或 consumer 问题。
- dK/dV 同行错误：优先检查共享 K/V、P/dS 或共同 post 映射。
- dQ 单独出现完整行错误：检查该 query task 的原子完成与 post 读取时序。

# 验证方法

每次修改保存逐坐标误差摘要；原始尾长通过后扩展到相邻边界和多 batch。最终必须同时验证
workspace、consumer 输入、三个输出和重复运行稳定性。

[^layout]: 固定提交 CATLASS layout 文档对逻辑坐标、shape 和 stride 的地址语义。
[^simt]: 固定提交 SIMT 示例中的线程索引、标量访问与线程同步模式。
