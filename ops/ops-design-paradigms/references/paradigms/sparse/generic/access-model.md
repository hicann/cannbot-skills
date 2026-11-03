<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式访存模型

> **层级**：通用知识层（generic/），语言中立。
>
> 核心问题：目标地址由索引值在运行时决定时，如何规划访存路径与冲突处理。

## 1 索引驱动访存

- 目标偏移 `offset[j] = f(indices[j])` 在运行前不可知：无法预取、无法向量化、无法用连续块搬运覆盖
- 主数据面**直访 GM**是本范式的默认选择：离散行/元素访问搬入片上缓冲无收益且增加延迟（sparse_apply_adadelta：行地址离散无法预取；gather：每线程只读单元素）
- 例外——**小体量元数据**（stride 数组、shape、预计算表）经片上共享内存传递给全部线程复用（sparse_reshape：stride ≤ 192B；gather：8 维 stride 数组 128B）

## 2 索引唯一性与冲突模型

设计第一步：判定目标位置是否被多个执行单元竞争。

| 判定来源 | 结论 | 后续路径 |
|---------|------|---------|
| 语义保证唯一（如 TF SparseApply* 文档要求 indices 不重复） | 独占写 | 直写，无原子（sparse_apply_adadelta） |
| 语义允许重复且做累积 | 竞争写 | 原子累积（sparse_bincount / sparse_segment_sum_grad） |
| 语义允许重复但输出为幂等值 | 竞争写 | 直写幂等值替代原子（bincount 的 binary_output：`output[idx] = 1`） |
| 逐元素独立变换，无共享目标 | 无冲突 | 直写（sparse_reshape） |

**唯一性不可静态证明时按重复处理**；把"语义保证唯一"当作前提的算子，须在规格中核对竞品契约并标注风险（kernel 侧不做运行时校验）。

## 3 冲突处理决策树

```
目标位置竞争？
├─ 否 → 直写
└─ 是 → 输出是累积语义？
        ├─ 否（幂等赋值）→ 直写，消除原子
        └─ 是 → 索引 dtype？
                ├─ 整型 → 原子累加（结果精确、顺序无关）
                └─ 浮点 → 原子累加（顺序非确定，与竞品 GPU 行为一致，验证用容差）
                          └─ 要求确定性？→ 分桶/分段预归约后再写（代价：额外 pass 与 workspace）
```

- 浮点原子累加**不改变数学语义**，只改变舍入路径：与竞品 GPU 的 atomicAdd 同为非确定实现，精度对齐以 GPU 为基准而非 CPU 顺序实现
- double 类型无原生原子加时用 CAS 循环位级模拟（见 AscendC 适配层 [atomic-ops.md](../adapters/ascendc/atomic-ops.md)）

## 4 值域与越界策略

| 策略 | 语义 | 实例 | 实现位置 |
|------|------|------|---------|
| **裁剪忽略** | 越界条目跳过，不参与计算 | bincount：`val<0 \|\| val>=size → continue` | kernel 循环内 |
| **报错** | 越界即非法输入 | segment grad：索引范围校验 | tiling/infershape 侧前置校验 |

选择依据是竞品语义：TF Bincount 越界忽略、SegmentGrad 越界报错。报错策略必须在 kernel 之外（host 侧）完成，kernel 内分支只做忽略。

## 5 值依赖输入

索引/形状类张量常在 host 侧 tiling 阶段就需要**读值**（而非仅读 shape）：

- sparse_reshape：`shape`、`new_shape` 需读值预计算 stride、推断 `-1` 维、推导输出 `y_shape`
- 输出 shape 由输入值决定的算子（IndexRemap 族典型），infershape 与 tiling 均值依赖

值依赖的声明契约（声明位置、一致性要求）见公共知识 [../../common/value-depend-process.md](../../common/value-depend-process.md)。

## 6 索引数学

- **线性化**：`linear = Σ indices[i,d] × stride_in[d]`
- **反线性化**：从最高维起 `y_indices[i,d] = linear ÷ stride_out[d]; linear %= stride_out[d]`
- **-1 维推断**：`missing = total_elements ÷ product(other_dims)`（仅允许一个 -1）
- stride 恒在 host 侧预计算后传入，kernel 不做乘积循环
- 索引中间量统一用宽整型（int64）承载，防止 int32 索引在大 shape 下乘加溢出
