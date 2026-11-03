<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# sparse_bincount_simt_draft —— DRAFT 参考实现说明

> **状态：DRAFT（未编译、未上板、未实测）**。Sparse 范式 Count 算子族的 DSL 结构参考实现，**与 AscendC 侧同算子对齐**（[../../../ascendc/example/sparse_bincount_package/](../../../ascendc/example/sparse_bincount_package/)，语义契约一致，实现载体按语言）；cannbot-dsl 仓当前没有 Sparse 类算子样例，本文件填补"从文档到首例"之间的结构空白。

## 结构展示点

| 文件位置 | 展示内容 | 依据文档 |
|---------|---------|---------|
| `sparse_bincount`（@host） | 路径分派：计数 / 加权 / binary_output 三路径独立编译实例（TilingKey 的 DSL 对等物） | atomic-ops.md §2 |
| 两段顺序 launch（zero → scatter） | 零初始化 + 散射累积拆两次 launch，同 stream 排序免 SyncAll | kernel-entrance.md §4、sync-model.md |
| `_bincount_count` | grid-stride + `cb.simt.atomic_add(output.ptr(off), v)` 标量点加轨 | atomic-ops.md §1 |
| `_bincount_binary` | binary_output 幂等直写（全核同值），消除原子 | atomic-ops.md §4 |
| `size_t[0]` / `dense_shape_t[0]` | 值依赖输入设备侧运行期读取（host 构图期不可读 GM） | kernel-entrance.md §5 |
| `dtypes.int64(...)` 全程承载 | 索引中间量 int64 统一 | index-mapping.md |

## 语义覆盖范围

覆盖算子 README 两公式的公共骨架（1D `output[v] += w_i`；multi-dim `output[batch][v] += w_i`，v/batch 越界丢弃），batch 取 indices 首维折叠示意。**不含**：多维 stride 完整展开、-1 维推断、weights 空/非 fp32 dtype 分档——以 AscendC 样例包为准。

## 待验证点（首例接入时逐项闭环）

| # | 待验证项 | 说明 |
|---|---------|------|
| C1 | `cb.simt.atomic_add` 返回值可靠性 | 本草稿只用写语义；返回值语义须单测复核后方可依赖 |
| C2 | 计数 +1 的 dtype 分派 | int32/int64/float 输出的 `dtypes.*` 匹配（atomic value 类型须与 ptr 元素一致） |
| C3 | `indices[i * rank]` 的 int64 读取路径 | values 为 int32 时 `dtypes.int64(values[i])` 的符号扩展正确性 |
| C4 | 零初始化与散射段的可见性顺序 | stream 顺序 launch 的 GM 可见性；异常时补显式同步并登记 |
| C5 | 线程档与核数 | THREADS/CORES 示意值按目标核数与 N 重核定（splitting.md） |

## 首例验证义务（verification.md 双轨）

1. 重复索引用例：计数守恒（`sum(output) == 有效元素数`）
2. 越界丢弃：v ≥ size / batch ≥ dense_shape[0] 不越界不计数
3. binary_output 幂等：重复索引下输出恒 1，重跑一致
4. 与 AscendC 样例包 golden 对拍（同 seed/同输入逐元素一致，整型精确）
