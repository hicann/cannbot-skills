<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Sparse 范式场景模型

> **层级**：通用知识层（generic/），语言中立。
>
> 来源：sparse_bincount、sparse_segment_sum_grad、sparse_apply_adadelta、sparse_apply_ftrl、sparse_reshape 五个 AscendC SIMT 算子的实测蒸馏（ascend950 / arch35）。

## 1 准入条件

本范式覆盖的算子需同时满足：

1. **数据面为稀疏表示**——输入/输出以 `(indices, values)` 对、CSR 三元组 `(indptr, indices, values)`、IndexedRows `(index, 行数据)` 或等价结构组织；数据位置由索引张量的**值**决定
2. **计算核心是索引驱动的访存**——gather（按索引取数）、scatter/scatter-add（按索引散写/散加）、计数累积或索引坐标重映射，而非规则向量计算
3. **工作量由 nnz 主导**——循环上界与切分规划以非零元数量（或索引数量）为坐标系，dense shape 只决定值域与输出规模

## 2 术语与坐标系

| 术语 | 含义 |
|------|------|
| **indices** | 索引张量；其**元素值**决定读/写的目标位置 |
| **values** | 与 indices 逐位对应的非零值张量 |
| **indptr** | CSR 行偏移；`indptr[i]..indptr[i+1]` 为第 i 行的索引区间 |
| **N / nnz** | 索引条目总数（循环主坐标） |
| **innerSize** | 行内/分组内元素数（次坐标） |
| **唯一索引** | indices 无重复 → 目标位置独占，可直写 |
| **重复索引** | indices 可能重复 → 目标位置竞争，需原子或预归约 |
| **值域裁剪** | 越界索引的处理策略（忽略 or 报错） |

## 3 算子族分类

按数据流方向与冲突模型分为四族：

| 算子族 | 数学语义 | 冲突模型 | 实例 |
|------|---------|---------|------|
| **Count 计数累积** | `output[val[j]] += w[j]`，val 限值域 `[0, size)` | 重复索引（必然） | sparse_bincount |
| **ScatterAdd 散射累积** | `output[idx[j]] += x[j]` | 重复索引（可能） | sparse_segment_sum_grad |
| **RowApply 行更新** | 行 gather → 行内状态计算 → scatter 回写 | 唯一索引（语义保证） | sparse_apply_adadelta / sparse_apply_ftrl |
| **IndexRemap 索引重映射** | `y_indices = delinearize(linearize(indices, shape), new_shape)` | 无冲突（逐元素独立变换） | sparse_reshape |

- Count 与 ScatterAdd 同为散射累积，差异在值域契约：Count 对越界索引**裁剪忽略**，ScatterAdd 通常**报错**（tiling 侧校验）
- RowApply 是"稀疏优化器"标准形态：`var[index[i]] = f(var[index[i]], grad[i], 状态变量)`，索引唯一是语义前提（竞品文档明确要求，kernel 不做运行时校验）
- IndexRemap 不搬运 values，只做索引坐标变换；输出 shape 由输入张量**值**决定，是典型的值依赖算子

## 4 数据流骨架

四族共享"索引读入 → 值驱动访存 → 输出"三段式，差异在中间段：

```
[段 1: 索引读入]     读 indices[j]（及伴随的 values[j] / segment_ids[j]）
[段 2: 值驱动访存]   Count/ScatterAdd: 计算目标偏移 → 原子累积
                     RowApply:        gather 行 → 行内顺序计算 → scatter 回写
                     IndexRemap:      线性化 → 反线性化坐标变换
[段 3: 输出]         Count/ScatterAdd 依赖输出零初始化；RowApply 直写；IndexRemap 逐元素直写
```

- 散射累积族的输出初始值必须为 0，初始化与散射两阶段间需要全核同步（见 [sync-model.md](sync-model.md)）
- RowApply 的行内计算存在数据依赖（如 adadelta 的 accum/accum_update 状态），一行必须由同一执行单元顺序处理

## 5 能力边界

1. **稀疏 × 稠密矩阵乘**（sparse_tensor_dense_mat_mul 类）不在本范式 standard 路径——其主体是 Contraction，稀疏索引只影响左矩阵取行，按 Contraction 范式设计、用本范式的访存模型处理 gather 段
2. **注意力稀疏跳过**（sparse_flash_attention 类）归 FA 范式稀疏 trait——其稀疏 pattern 服务于 KV 块跳过，不改变本范式的数据面组织
3. **dense 张量的索引取放**（gather / scatter_update / index_put）归 IndexGather / ScatterUpdate 范式——数据面主体是 dense 张量；本范式以稀疏数据结构为主体。两类的实现技术（索引驱动访存、原子操作、grid-stride）互通，设计时可交叉参考
4. **原子语义本身**（AtomicUpdate 范式）不展开——本范式把原子操作当作重复索引冲突的解决手段，语义细节见对应范式

## 6 依赖谓词

| 谓词 | 用途 |
|------|------|
| `CAP_SIMT_GM_DIRECT` | 数据面直访 GM、不经片上缓冲转译的前提 |
| `CAP_SIMT_ATOMIC_ADD_DTYPE_SET` | 原子累加支持的 dtype 集合（决定浮点冲突路径） |
| `CAP_SIMT_ATOMIC_CAS_WIDTH` | CAS 位宽（double 原子模拟的前提） |
| `CAP_SYNC_ALL` | 零初始化与散射两阶段间的全核同步 |
| `CAP_SIMT_THREAD_MAX` / `CAP_STACK_PER_THREAD` | 线程数与栈预算 |

谓词定义见 [capability-contract.md](capability-contract.md)；各语言真值见对应适配层 capability-mapping。
