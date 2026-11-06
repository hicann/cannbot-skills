<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 累加策略

> 语言中立。定义三种累加模板——局部直方图模板（UB 分批 bins + DMA 原子归并）、无局部直方图模板（GM 直接原子累加）与确定性模板（无原子串行累加，浮点累加需确定时选用）；前两种按边界条件 `bins/inputsize < GM_ATOMIC_ADD_FACTOR` 选择，另有 weight/density 变体。

## 1 冲突来源与模板划分

多个元素可映射到同一 bin；并行下同核多线程或跨核会并发写同 bin 地址。两种模板的差异在于**累加去向**：

- **无局部直方图模板**：每元素直接 GM 原子累加（`` `GLOBAL_MERGE_ATOMIC` ``），同 bin 并发写在 GM 处串行化；
- **局部直方图模板**：先在本核片上 UB 分批 bins 上累加（`` `LOCAL_HIST_ATOMIC` `` 收敛核内冲突），再经 DMA 原子归并（`` `LOCAL_HIST_FLUSH` ``）合入 GM，把 GM 原子次数从「每元素一次」降为「每核每 bin 一次」。

**边界条件**：`bins/inputsize < GM_ATOMIC_ADD_FACTOR`（阈值 `GM_ATOMIC_ADD_FACTOR` 默认 100，需按性能测试结果调整；`inputsize` 为输入元素数）：

- **成立**（bins 在 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍以内，atomicadd 冲突较大）→ **局部直方图模板**；
- **否则**（bins 超过 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍，atomicadd 冲突较小）→ **无局部直方图模板**。

语义：bins 在 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍以内时，元素集中到有限的 bin，atomicadd 冲突较大，用局部直方图模板先把冲突收敛在片上再归并；bins 超过 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍时，元素分散到大量 bin，冲突较小，直接 GM 原子即可。

## 2 模板一：局部直方图模板

> 前提：输出 GM `hist` 在计数/归并前清零（host 预清零优先，或 kernel 内 每核分块清零），可见性约束见 §2.2。

1. **inputsize 平均分核**：把 `N` 个输入元素按 每核分块平均分配到每个核（每核 `ceil(N / 核数)` 个元素）。
2. **根据 UB 大小分批加载 bins**：把 `[0, nbins)` 的桶按片上容量 `` `ONCHIP_HIST_CAPACITY` `` 切分为若干 bin 批，每批装入 UB 局部直方图（突破单次片上容量对 `nbins` 的限制）。
3. **核内统计**：每个核使用 SIMT 线程 每核分块 + 核内跨步统计本核所有 input 落在当前 UB bin 批范围的元素，计算 bin index，用 `` `LOCAL_HIST_ATOMIC` `` 累加到 UB 局部直方图。
4. **DMA 原子归并**：用 `` `LOCAL_HIST_FLUSH` `` 把当前 UB bin 批原子累加（归并）到输出 GM。
5. **循环 2-4**：回到步骤 2 处理下一 bin 批，直到每个核把所有 bins 处理完。

### 2.1 局部直方图布局与分批

- 局部直方图布局：UB 上每核一份 `localHist[BATCH_BINS + ALIGN_GRANULARITY]`，`BATCH_BINS` 为该核单批加载的桶数（受 `` `ONCHIP_HIST_CAPACITY` `` 约束），行宽填充 `` `ALIGN_GRANULARITY` `` 避免非对齐访问冲突/越界。
- **分批**：`nbins` 超过单批容量时，按 `BATCH_BINS` 切为 `ceil(nbins / BATCH_BINS)` 批；每批覆盖桶范围 `[batchStart, batchStart + BATCH_BINS)`。
- **队列 + 双缓冲（条件开启）**：`bins × sizeof(HistT) > 1024` 字节时局部直方图 UB 采用队列 + 双缓冲（depth 2），每批 `分配 → 计数 → 入队 → 出队 → 归并 → 释放`，使下一批计数与本批归并重叠；`bins × sizeof(HistT) ≤ 1024` 字节时 bins 数据量过少，全部驻片一次性计算完成（单批，无分批循环、无双缓冲）。队列/双缓冲的具体形态与异步交接能力由适配层给真值（AscendC 为队列 depth 2 双缓冲；DSL 通常串行分批、片上 buffer 分批复用），语言落地见对应适配层。
- **清零**：每批局部直方图在使用前用 `` `LOCAL_HIST_CLEAR` `` 清零（批量清零，真值见适配层：AscendC 为 SIMD 批量填零、DSL 为 SIMT 线程跨步清 0），计数内核只计数不清 0。语言落地见对应适配层。

### 2.2 归并与可见性

- **归并**（步骤 4）：`` `LOCAL_HIST_FLUSH` `` 把 UB 局部直方图的 `` `BATCH_BINS` `` 个计数原子累加到 `hist[batchStart .. batchStart+BATCH_BINS)`，跨核收敛由归并原语的原子语义保证（非栅栏）。
- **分批边界不重叠**：相邻批覆盖的桶范围相邻且不重叠，故 `` `LOCAL_HIST_FLUSH` `` 之间无重复累加；每批 flush 前该批局部直方图只含本批桶内的计数。
- **输出清 0 可见性**（前提）：host 预清零优先；kernel 内清零须在计数/归并前对所有线程/核可见（必要时 `` `MEMORY_FENCE` `` 与跨 block 栅栏），否则清零与归并并发写 `hist` 产生数据竞争。

## 3 模板二：无局部直方图模板

1. **输出清 0**：输出 GM `hist` 清零（host 预清零，或 kernel 内分核 每核分块清零，可见性约束同 §2.2）。
2. **inputsize 平均分核**：把 `N` 个输入元素按 每核分块平均分配到每个核。
3. **核内统计**：每个核的 SIMT 线程 每核分块 + 核内跨步遍历本核 input，计算 bin index（等宽公式 / 二分 / 离散类标 + `` `FLAT_INDEX_CALC` `` 展平），用 `` `GLOBAL_MERGE_ATOMIC` `` **直接原子累加到输出 GM**（不分配片上局部直方图，无归并段，无需阶段间 `` `THREAD_BARRIER` ``）。

## 4 确定性模板

> 与 §2/§3 两模板并列的第三模板：**bins 并行、input 串行、无原子累加**。由确定性开关选择，不参与 §1 的 `bins/inputsize < GM_ATOMIC_ADD_FACTOR`
> 边界条件分派。目标：每个 bin 只被唯一一个线程以固定顺序串行累加，浮点累加结果确定、可复现。

1. **bins 分核**：把 `[0, nbins)` 的桶按核数平均分配到每个核（每核 `ceil(nbins/核数)` 个桶，末核为尾块）。
   D 维时先把 D 维输出按行优先展平为 1D `[0, N_BINS)`（`N_BINS = ∏ nbins_axis`），再对展平 index 分核。
2. **核内 bins stride 排布**：本核的 bins 段**不再切分**，核内 `` `THREAD_COUNT_PER_BLOCK` `` 个线程以 stride
   方式遍历本核所有 bins（`` `for (bin = coreBinStart + thread; bin < coreBinEnd; bin += blockDim.x)` ``），
   每个 bin 恰好被一个线程命中。
3. **线程对命中的 bin 遍历全部 input**：线程对每个命中的 bin，按固定顺序遍历**全部**输入元素，计算 bin index；
   若 `idx` 等于该 bin，则累加计数（`+1`）或累加权重（`+weight[n]`）到 `hist[bin]`。
4. **无原子依赖**：每个 bin 只被 strided 命中的唯一线程累加，累加顺序即该线程遍历 input 的固定顺序，无并发原子写 → 确定。

- **生成条件（仅 float 累加）**：确定性模板**只在累加值为 float 数据类型时生成**（如 `` `weight` `` 加权累加、输出 `HistT=float`）。
  原因：整数（int32/int64）原子累加可交换、可结合，无精度损失，任意并发顺序结果一致；而 float 原子累加（`` `GLOBAL_MERGE_ATOMIC` `` / `` `LOCAL_HIST_ATOMIC` ``）的浮点舍入随累加顺序变化，并发顺序不确定 → 结果不确定。因此**只有 float 累加才存在确定性问题**。
- **累加值不支持 float 时无需生成**：累加值为整数（int32/int64 计数）时原子累加天然确定，不生成确定性模板、也不暴露确定性开关。输出为 int 时直接走 §2/§3；输出为 float 时额外补 int→float cast 后处理，见 §4.1。

- 语义（bin index / 越界丢弃 / min==max 特判 / 边界钳制）与 §2/§3 完全一致，仅**并行维度从 input 换成 bins**、
  累加去向从「原子」换成「单线程串行」。
- 能力依赖：**无**（不需要 `` `LOCAL_HIST_ATOMIC` `` / `` `LOCAL_HIST_FLUSH` `` / `` `GLOBAL_MERGE_ATOMIC` ``）。
- 代价：每线程遍历全部 input（访存量 `N × 参与分线程数` 量级放大），性能通常低于 §2/§3；仅在确定性优先时选用。
  落地见 [../adapters/ascendc/kernel-skeleton.md](../adapters/ascendc/kernel-skeleton.md) §9。

### 4.1 int 计数累加 + float 输出（无需确定性模板）

当**累加值为 int（计数 `+1`）而输出数据类型为 float** 时，计数累加本身是确定的整数原子加，**无需生成确定性模板**；
仅需在计数完成后补一步 int→float cast 后处理写回输出。两模板处理步骤：

**原地 cast 优先（免 workspace）**：若输出能够原地从 int 数据类型 cast 到 float 数据类型（int32 计数与 float 输出同宽），则直接在输出 GM 上进行原地 cast 操作——计数目标即输出的 int 视图，**无需申请 workspace**；下述步骤中的「workspace」在该形态下均替换为「输出的 int 视图」。

**局部直方图模板（§2）：**

1. 申请 int 数据类型 workspace（或原地形态：直接以输出的 int 视图为累加目标，见上）；
2. 将局部累加结果 atomic add 到所有 bins 的 workspace 中；
3. 全核同步；
4. 使用 SIMT VF 读取 workspace 中所有 bins 的 int 累加值，cast 成 float 后写入输出 GM；本步 SIMT VF 线程采用全局排布，不再对每个核做 bins 切分。

**无局部直方图模板（§3）：**

1. 申请 int 数据类型 workspace（或原地形态：直接以输出的 int 视图为累加目标，见上）；
2. SIMT VF 直接累加 int 到 workspace 中；
3. 全核同步；
4. 使用 SIMT VF 读取 workspace 中所有 bins 的 int 累加值，cast 成 float 后写入输出 GM；本步 SIMT VF 线程采用全局排布，不再对每个核做 bins 切分。

> 语言落地注意：无局部直方图模板 float 输出路径的 DCache 一致性强制约束（计数/cast SIMT VF 的 GM 参数须加 `volatile`）属 Ascend C 语言细节，见 [../adapters/ascendc/kernel-skeleton.md](../adapters/ascendc/kernel-skeleton.md) §8.4。

## 5 两模板对比

| 维度 | 局部直方图模板（§2） | 无局部直方图模板（§3） |
|------|---------------------|------------------------|
| GM 原子次数 | 每核每 bin 一次（少） | 每元素一次（`N` 次，竞争激烈） |
| 片上占用 | UB 分批 bins（受 `` `ONCHIP_HIST_CAPACITY` `` 约束，分批突破容量） | 无大数组 |
| 累加去向 | 片上 UB 分批 bins → DMA 归并 GM | 直接 GM 原子 |
| 能力依赖 | `` `LOCAL_HIST_ATOMIC` `` + `` `LOCAL_HIST_FLUSH` ``（+ `` `VECTOR_MTE3_SYNC` ``，清 0→计数无需 `` `THREAD_BARRIER` ``） | 仅 `` `GLOBAL_MERGE_ATOMIC` `` |
| 触发条件 | `bins/inputsize < GM_ATOMIC_ADD_FACTOR`（bins 在 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍以内，atomicadd 冲突较大） | `bins/inputsize ≥ GM_ATOMIC_ADD_FACTOR`（bins 超过 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍，atomicadd 冲突较小） |
| 处理次数 | 每批重扫本核 input（`ceil(nbins/BATCH_BINS)` × `N/核数`） | 单次遍历本核 input |

## 6 变体

| 变体 | 计数 | 累加值 | 说明 |
|------|------|--------|------|
| 计数（默认） | `+1` | 整数 | 输出 `HistT=int32` |
| 计数（float 输出） | `+1` | 整数 | 输出 `HistT=float`；累加走 int、末步 cast 成 float（见 §4.1，无需确定性模板） |
| 加权 `weight` | `+weight[i]` | 浮点 | 输出 `HistT=float`；累加值非 1 |
| `density=True` | `+1` | 整数 | 完成后对每个 bin 归一化（后处理 pass，非本范式内） |

`density` 归一化因子 `Σ(hist × width)` 需在计数完成后另算，不属于并行计数骨架，作为后处理。

## 7 能力依赖

- 局部直方图模板：步骤 3 依赖 `` `LOCAL_HIST_ATOMIC` ``（片上原子）；步骤 4 依赖 `` `LOCAL_HIST_FLUSH` ``（UB→GM DMA 原子归并，支持 int32/int64/float 取决于语言真值）；每批内依赖 `` `VECTOR_MTE3_SYNC` ``（计数→归并跨引擎同步；清 0→计数无需 `` `THREAD_BARRIER` ``，见 [sync-model.md](sync-model.md) §1/§2），必要时 `` `MEMORY_FENCE` ``。
- 无局部直方图模板：仅依赖 `` `GLOBAL_MERGE_ATOMIC` ``（GM 逐元素原子，支持 int32/int64/float 取决于语言真值）。

真值见 [../adapters/ascendc/capability-mapping.md](../adapters/ascendc/capability-mapping.md)。

## 8 多维局部直方图布局

D 维直方图的累加路径与 1D 完全同构，唯一变化是「1D idx」换「展平 index」、「nbins」换「总桶数 `N_BINS = ∏ nbins_axis`」。

### 8.1 布局

局部直方图仍是展平一维数组，按总桶数分批展开：

```
localHist[BATCH_TOTAL_BINS + ALIGN_GRANULARITY]
BATCH_TOTAL_BINS = min(N_BINS, 片上容量单批上限)，  N_BINS = ∏_{d=0}^{D-1} nbins_axis[d]
```

- 每核一份、行宽填充、`` `LOCAL_HIST_CLEAR` `` 清零、分批加载均同 §2（`BATCH_BINS` 取 `BATCH_TOTAL_BINS`）。
- 单个样本的 D 维 bin 下标经 `` `FLAT_INDEX_CALC` `` 展平为 `idx ∈ [0, N_BINS)`；每批只统计落在当前批展平 index 范围内的样本。

### 8.2 累加与归并

- **核内统计**：`` `LOCAL_HIST_ATOMIC`(localHist + (idx - batchStart), v) ``，`idx` 为展平 index，`v = 1`（计数）或 `weight[n]`（加权）。
- **归并**：`` `LOCAL_HIST_FLUSH` `` 把 `` `BATCH_TOTAL_BINS` `` 个计数原子累加到 `hist[batchStart .. batchStart+BATCH_TOTAL_BINS)`。
- 输出 `hist`（长度 `N_BINS`）仍由 host 启动前清零；展平顺序即 [bin-model.md](bin-model.md) §5.3 的行优先顺序，host 按形状 `[nbins_axis0, …]` 解读。
