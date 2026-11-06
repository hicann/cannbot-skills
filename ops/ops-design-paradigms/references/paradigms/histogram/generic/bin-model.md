<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram bin 规格与 range 推导

> 语言中立。定义 bins 的两种形态、range/min-max 来源，以及 `nbins` / `bin_edges` 的推导规则。

## 1 bins 形态

| 形态 | 输入 | `nbins` | `bin_edges` |
|------|------|---------|-------------|
| **EqualWidth（等宽）** | `bins` 为 int | `nbins = bins` | `linspace(lo, hi, nbins+1)`，宽度 `width = (hi - lo) / nbins` |
| **CustomEdges（自定义边界）** | `bins` 为 tensor 递增边界 | `nbins = bins.size - 1` | 直接取 `bins`，各桶宽度可不等 |

两形态共享同一套 [bin index 计算](bin-index.md)：等宽用线性公式，自定义边界用二分定位 `lower_bound`。

## 2 range / min-max 来源

`lo = min`、`hi = max` 的来源有两种，对应两个算子的核心差异：

| 来源 | 描述 | 前置动作 |
|------|------|----------|
| **显式 value_range** | host 传入 `value_range=[min, max]` 标量，kernel 直接读 GM 标量  | 无（直接读标量） |
| **缺省 input min/max** | range 未给，取输入数组的 `min(x)` / `max(x)` | 需前置 min/max 归约（额外归约 kernel 或 host 预处理） |

前置 min/max 归约的能力依赖 `` `PREFIX_MINMAX_REDUCE` ``；其 SIMT 落地（`` `asc_atomic_min/max` `` 每线程原子归约）已上板实测通过（见 [../adapters/ascendc/capability-mapping.md](../adapters/ascendc/capability-mapping.md)）。

## 3 bin_edges 推导

- **等宽**：`bin_edges[k] = lo + k * width`，`width = (hi - lo) / nbins`。当 `hi == lo` 时退化为特判（见 [bin-index.md](bin-index.md) §2）。
- **自定义边界**：`bin_edges = bins`（递增），`nbins = bins.size - 1`。

## 4 分批加载与单批驻留判定

局部直方图模板的 UB 局部直方图**每核一份、按批次加载**（见 [accumulation.md](accumulation.md) §2），单批桶数受 `` `ONCHIP_HIST_CAPACITY` `` 约束：

```
(BATCH_BINS + ALIGN_GRANULARITY) × sizeof(HistT) × depth ≤ ONCHIP_HIST_CAPACITY   （depth = useDoubleBuffer ? 2 : 1）
```

- `nbins ≤ BATCH_BINS`（单批装得下）→ 一批处理完，无分批循环。
- `nbins > BATCH_BINS` → 按批次切分为 `ceil(nbins / BATCH_BINS)` 批，每批清 0 → 核内统计 → `` `LOCAL_HIST_FLUSH` `` 归并，循环至所有 bins 处理完。
- **DOUBLE BUFFER 条件**（tiling 字段 `` `useDoubleBuffer` ``，host 侧 tiling 按 `` `totalBins × sizeof(HistT)` `` 动态计算，**不是恒 0/恒 1**）：
  - `` `totalBins × sizeof(HistT) > 1024` `` 字节 → `` `useDoubleBuffer = 1` ``：UB 局部直方图用队列 **depth 2 双缓冲**（每批 `分配 → 计数 → 入队 → 出队 → 归并 → 释放`，下一批计数与本批归并重叠），并按 `` `BATCH_BINS` `` 分批循环。
  - `` `totalBins × sizeof(HistT) ≤ 1024` `` 字节 → `` `useDoubleBuffer = 0` ``：bins 数据量过少，全部驻片**单批一次性**计算完成（depth 1 单缓冲或不建队列，无分批循环、无双缓冲）。
  - 两条件的**触发**相互独立，但 `depth` 参与单批容量计算：`` `BATCH_BINS` `` 由片上容量 `` `ONCHIP_HIST_CAPACITY` `` 决定（决定是否分批），`` `useDoubleBuffer` `` 由 1024 字节阈值决定（决定是否双缓冲）；双缓冲（depth 2）下片上占用翻倍、`BATCH_BINS` 相应减半，故上式 `depth` 因子不可省略；D 维时 `totalBins` 取总桶数 `` `∏ nbins_axis` ``。

`` `ONCHIP_HIST_CAPACITY` `` 真值以命中语言适配层的 capability-mapping 为准（Ascend C 需预留 DCache、DSL 需预留 SIMT kernel 上下文，见各适配层 capability-mapping.md §1）。`sizeof(HistT)` 为输出 dtype 字节数（int32 计数 = 4B；float 权重 = 4B；int64 计数 = 8B）。行宽填充依赖 `` `ALIGN_GRANULARITY` ``。D 维时上式中的 `nbins` 替换为总桶数 `` `∏ nbins_axis` ``，见 §5。

## 5 二维 / 多维 bins 模型

D 维直方图的 bins 规格是 1D 的**逐轴独立推广**：每个特征轴 `d ∈ [0, D)` 拥有独立的 bins 形态、`nbins_axis[d]` 与 range 来源，三者可逐轴不同（如一个轴等宽 float + 一个轴离散类标）。

### 5.1 每轴独立 bins 形态与 range 来源

| 轴形态 | `nbins_axis[d]` | `bin_edges_d` | range 来源 | bin index（见 [bin-index.md](bin-index.md)） |
|------|------|-------------|-----------|------|
| 等宽 | 显式 int `bins[d]` | `linspace(lo_d, hi_d, n_d+1)` | `value_range[d]` 或缺省 `[min,max]` | 线性公式 |
| 自定义边界 | `bins[d].size - 1` | tensor `bins[d]`（递增） | 边界本身蕴含 | 二分 `lower_bound` |
| 离散类标 | `num_classes_d`（= 显式给定或 `max+1`） | `[0, 1, …, num_classes_d]` 整数格点 | 天然 `[0, num_classes_d)` | 类标直接作 index（无需 floor/二分） |

- 各轴 range 来源独立：`value_range[d]` 显式给定，或缺省取该轴输入的 `min/max`（需逐轴前置 min/max 归约，同 §2，依赖 `` `PREFIX_MINMAX_REDUCE` ``）。
- 离散类标轴无 range/边界计算：类标取值天然落入 `[0, num_classes_d)`，越界类标（如 `≥ num_classes_d`）按 [bin-index.md](bin-index.md) §越界丢弃处理。

### 5.2 总桶数

```
N_BINS = ∏_{d=0}^{D-1} nbins_axis[d]        // D=1 时 N_BINS = nbins
```

输出 `hist` 为长度 `N_BINS` 的展平一维数组，语义上是 D 维直方图（host 按形状 `[nbins_axis0, nbins_axis1, …]` 解读）。

### 5.3 行优先展平布局

D 维 bin 下标 `(i0, i1, …, i_{D-1})` 按**行优先**（row-major）展平，第一轴为最高位：

```
idx = Σ_{d=0}^{D-1} i_d × stride_d，  stride_d = ∏_{k=d+1}^{D-1} nbins_axis[k]（stride_{D-1} = 1）
D=2：  idx = i0 × nbins_axis1 + i1
```

### 5.4 多维单批驻留判定

§4 的单批驻留判定以总桶数替换 `nbins`：

```
(BATCH_TOTAL_BINS + ALIGN_GRANULARITY) × sizeof(HistT) × depth ≤ ONCHIP_HIST_CAPACITY   （depth = useDoubleBuffer ? 2 : 1）
```

`∏ nbins_axis` 随轴数/每轴桶数乘积增长，比 1D 更快触顶；设计时先算总桶数再定单批容量与批次数（见 [scene.md](scene.md) §4 能力边界）。模板选择按边界条件 `bins/inputsize < GM_ATOMIC_ADD_FACTOR`（见 [accumulation.md](accumulation.md) §1），与 nbins 是否整体驻片无关。
