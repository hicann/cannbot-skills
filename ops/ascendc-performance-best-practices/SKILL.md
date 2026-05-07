---
name: ascendc-performance-best-practices
trigger: query
description: Ascend C 算子性能优化最佳实践库。按算子族组织优化经验与参考代码总结，供性能优化实施阶段查询。触发：查询某类算子的性能优化参考实现、实施某项优化时需加载对应优化经验时。
---

# Ascend C 算子性能优化最佳实践

## 算子分类体系

按 **算子族（operator family）** 组织优化知识，同一族内所有变体共享同一份文档（如 `matmul`、`matmul_mxfp4`、`batch_matmul`、`matmul_all_reduce`、`matmul_a16w16` 等均归入 `matmul` 族）。

| 类别 | 典型算子 | 适用架构 | 优化设计指南 |
|------|---------|---------|------------|
| MatMul 矩阵乘类 | MatMul, BatchMatMul, MatMul_MXFP4, MatMul_A16W16, MatMul_AllReduce | DAV_3510 | ✅ [性能优化指南](reference/matmul/guide.md) |
| RadixSort 基数排序类 | TopK, KthValue, Sort, ArgSort, ArgMax/Min | DAV_2201 / DAV_3510 | ✅ [性能优化指南](reference/sort/radix_sort.md) |
| Reduction 归约类 | ReduceSum, Softmax, LayerNorm, ArgMax | — | 📋 规划中 |
| Elementwise 逐元素类 | Sin, Cos, Abs, Exp | — | 📋 规划中 |
| Broadcast 广播类 | Add, Mul, Sub | — | 📋 规划中 |
| Conversion 数据转换类 | Transpose, Concat, Split | — | 📋 规划中 |
| Convolution 卷积类 | Conv2D, DepthwiseConv | — | 📋 规划中 |
| NN 神经网络类 | FlashAttention, GroupNorm | — | 📋 规划中 |
| Random 随机类 | RandomUniform, Dropout | — | 📋 规划中 |

> 未收录的算子族返回「该算子族优化知识暂未收录」。各族详细的优化类型、叠加关系、选型决策见该族 `reference/<family>/` 目录。

## 查询方式

| 输入 | 必需 | 说明 |
|------|------|------|
| 算子名 | 是 | 如 `matmul` / `matmul_mxfp4` / `batch_matmul` |
| 优化类型 | 否 | 如 `pingpong` / `swat` / `streamk` / `fullload` / `scale_coalescing` / `mte2_preload`；不提供则加载全部 |

查询流程：**算子名 → 映射到算子族 → 定位 `reference/<family>/` → 按优化类型筛选文档**。算子族映射规则：精确匹配族名直接命中；以族名为前缀或核心词（如 `matmul_mxfp4`、`batch_matmul`）归入该族；其他形态由调用方按功能显式指定。

## 通用设计文档结构（所有族必须）

每份 `<优化类型>_design.md` 的章节组织：

**必选章节：**

1. 优化目标 —— 效果与量化收益（kernel μs / MTE2 段 / CUBE busy 等）
2. 架构概览 —— 存储层级、数据流、buffer 布局、事件同步模型
3. 关键参数 —— 新增 / 调整字段与 Host 侧计算
4. 核心计算循环 —— 改造前后对照（含事件同步）
5. 优化的关键修改点 —— 表格形式

**可选章节：**

6. 注意事项 / 约束 —— 前置条件、L1/L0 预算、边界与兼容性
7. 实施常见问题与解决方案 —— 高频踩坑与根因
8. 实测性能、选型决策、与其他优化的叠加关系、自检清单

## 扩展新算子族

1. 创建 `reference/<family>/` 目录（以族名命名，非单个变体）
2. 按上述结构编写 `<优化类型>_design.md`
3. 更新本 SKILL.md 分类表格

## 依赖

无外部依赖，所有知识以 Markdown 文档内置。
