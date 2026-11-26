<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
You may refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# three_interpolate_simt_draft —— DRAFT 参考实现说明

> **状态：DRAFT（未编译、未上板、未实测）**。Interpolation 范式 IndexedInterpolate 算子族的 DSL 结构参考实现，**与 AscendC 侧同算子对齐**（[../../../ascendc/example/three_interpolate_package/](../../../ascendc/example/three_interpolate_package/)，语义契约一致，实现载体按语言）；cannbot-dsl 仓当前没有 Interpolation 类算子样例，本文件填补"从文档到首例"之间的结构空白。

## 结构展示点

| 文件位置 | 展示内容 | 依据文档 |
|---------|---------|---------|
| `_three_interpolate_simt`（@kernel） | grid-stride 输出元素级遍历 + `(b, n, c)` 整除/取模坐标分解（int64 全程承载） | sampling-ops.md §1 |
| `raw → int64 → 双向 clamp` | 显式索引防御（int64 负值/大值防 OOB，契约保证合法仍保留） | sampling-ops.md §2 |
| `p0/p1/p2 分步绑定 + (p0+p1)+p2` | 加权和分步舍入（FMA 收缩未实测的保守写法）；项序与 golden 一致 | sampling-ops.md §4 |
| `features[feat_off + safe*cs + c]` | 邻居离散直读 GM（值驱动，运行期读 idx） | sampling-ops.md §2 |
| `three_interpolate`（@host） | 空输出不 launch（`dim3(0)` 非法）；cores 经 Constexpr、核数按 `vector_core_num` 查询 | kernel-entrance.md §6 |
| 正向无零初始化段 | 逐位直写；grad 反向才有两段顺序 launch | kernel-entrance.md §4 |

## 语义覆盖范围

覆盖算子 README 公式（`y[b,n,c] = Σ_k weight[b,n,k] × features[b, idx[b,n,k], c]`，布局 `(B, M, C) → (B, N, C)`）的公共骨架。**不含**：多 dtype 分档编译实例（lru_cache 键设计）、grad 反向（两段 launch + atomic_add 骨架）——以 AscendC 样例包与 [../../kernel-entrance.md](../../kernel-entrance.md) §3/§4 为准。

## 待验证点（首例接入时逐项闭环）

| # | 待验证项 | 说明 |
|---|---------|------|
| C1 | `w0*f0 + w1*f1 + w2*f2` 单表达式的 FMA 收缩行为 | 以恒定 ULP 偏差用例（issue #35 类）实测；未判定前维持分步绑定 |
| C2 | `//` / `%` 坐标分解的除法开销档位 | 大 totalOut 热点实测；必要时 params Tensor 手工 magic/shift |
| C3 | half 输入升 fp32 累加的出口落型 | `output[i]` 按 output dtype 赋值的隐式 cast 路径核对 |
| C4 | `idx` int32/int64 两种输入的读取路径 | `dtypes.int64(idx[...])` 符号扩展正确性 |
| C5 | 线程档与核数 | THREADS/CORES 示意值按目标核数与 totalOut 重核定（splitting.md §4） |

## 首例验证义务（verification.md 双轨）

1. 越界 clamp 一致性：与 AscendC 样例包同防御口径（clamp 后逐元素一致）
2. 权重不归一：和 ≠ 1 直乘直加，与 golden 一致
3. NaN/Inf 传播：含权重恰为零 × inf 样本分支
4. 与 AscendC 样例包 golden 对拍（同 seed/同输入，fp32 逐位一致——双方同为分步舍入路径）
