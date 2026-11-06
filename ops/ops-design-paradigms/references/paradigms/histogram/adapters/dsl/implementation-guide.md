<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 范式 CANNBotDSL 实现指引

> **层级**：CANNBotDSL 语言适配层叶子文档。通用层算法见 [../../generic/index.md](../../generic/index.md)；谓词真值见 [capability-mapping.md](capability-mapping.md)。
>
> 本指引结合 DSL 公开入口、Host、Kernel 的职责和 Histogram 通用层模型（三模板 + 边界条件）。具体 API 签名与设备能力以调用方提供的 DSL 资料为准，本文不复制 API 签名。

## 1 三模板数据流骨架 → DSL 三层结构映射

按 [../../generic/accumulation.md](../../generic/accumulation.md) §1 的边界条件 `` `bins/inputsize < GM_ATOMIC_ADD_FACTOR` `` 选前两个模板，确定性模板（第三模板）由确定性标志独立选择（见 [../../generic/accumulation.md](../../generic/accumulation.md) §4），三者均落到 DSL 的入口/Host/Kernel：

| 模板 | 通用层阶段 | DSL 落地 | 关键约束 |
|------|-----------|---------|---------|
| 局部直方图（`bins/inputsize < GM_ATOMIC_ADD_FACTOR`） | 分批清 0 → 计数 → DMA 归并，循环 | Kernel 内：片上局部直方图 Buffer 按 `` `ONCHIP_HIST_CAPACITY` `` 分批（每批清 0 → bin index + 片上原子累加 → 搬运归并到 GM 输出），循环至所有 bins | 片上原子累加依赖 `` `LOCAL_HIST_ATOMIC` ``（已证实）；归并依赖 `` `LOCAL_HIST_FLUSH` ``（已证实）；计数→归并交接若跨执行阶段用 RingBuffer 的 `head_next()`（写游标）/`tail_next()`（读游标）（独立游标，同步由数据访问推导；据仓内源码证据，待上板实测确认） |
| 无局部直方图（`bins/inputsize ≥ GM_ATOMIC_ADD_FACTOR`） | 直接 GM 原子累加 | Kernel 内：读 input → bin index → GM 输出逐元素原子累加 | 依赖 `` `GLOBAL_MERGE_ATOMIC` ``（已证实）；不分配片上局部直方图 Buffer |
| 确定性（第三模板，确定性标志触发） | bins 分核 + 核内 stride 串行遍历全部 input | Kernel 内：bins 分核 → 核内线程 stride 命中本核 bins → 每线程按固定顺序遍历全部 input，命中的 bin 用 GM 点赋值 `hist[bin] += 1/weight` | **无原子依赖**；不用 `cb.simt.atomic_add`；GM 点赋值不受限（[capability-mapping.md](capability-mapping.md) §4 规则 7）；累加顺序确定 |

> **确定性模板选择与落地**：DSL 无 `GetDeterministic()`/`TilingKey`（见 [index.md](index.md) §4），确定性模板由 Host launcher 按确定性标志分支决定（传参或编译分支），不参与 `bins/inputsize < GM_ATOMIC_ADD_FACTOR` 边界条件分派。其内核不含片上局部直方图、不含任何原子原语：bins 分核 + 核内 stride 串行遍历 input + GM 点赋值累加，算法见 [../../generic/accumulation.md](../../generic/accumulation.md) §4。

## 2 代码组织（三层职责）

| 层 | 职责 | 边界 |
|------|------|------|
| 公开入口 | 检查公开 dtype / shape / layout / device 参数组合；校验 `num_classes` / `bins` / 可选 `weights` / `value_range`；准备输出并委托 Host/Kernel | 不把设备计算藏在入口或可达辅助函数 |
| Host launcher | 读 shape/dtype/device 元数据、分配输出/workspace、建立零拷贝视图、计算切分参数（核数、每核分块、单批 bins 容量）后发起 DSL kernel | 只做元数据 / 分配 / 零拷贝视图；torch 调用若触发设备算子或复制即越界（见 [capability-mapping.md](capability-mapping.md) §4 规则 1） |
| Kernel | 按 bin index 计算、原子累加、分批归并、越界丢弃拆阶段方法 | 模板选择、边界条件、越界丢弃、min==max 特判均在 kernel 内，不因 DSL 而改变语义 |

## 3 局部直方图 Buffer/RingBuffer 规划

1. 片上局部直方图是**每核一份、分批复用**的临时量：单批 `` `BATCH_BINS × sizeof(HistT)` `` 受 `` `ONCHIP_HIST_CAPACITY` `` 约束（[../../generic/bin-model.md](../../generic/bin-model.md) §4），`` `nbins` 超单批容量时按批切分循环
2. 阶段内反复读写的局部直方图用 **Buffer**（计数阶段内清 0 → 累加，不跨阶段交接）；仅当「计数」与「归并」真属两个执行阶段需多槽位流水交接时用 **RingBuffer**，`head_next()`（写游标）/`tail_next()`（读游标）独立推进游标、无 FIFO 满/空检查（据仓内源码证据，待上板实测确认）
3. 单批 bins 数量由 Host 按运行时片上容量 + dtype 字节数计算后传入；对齐粒度以 [capability-mapping.md](capability-mapping.md) `` `ALIGN_GRANULARITY` `` 真值为准（= 32 字节；行宽填充元素数 = 32 / sizeof(HistT)）
4. **Buffer 静态 shape**（capability-mapping.md §4 规则 5）：DSL Buffer 仅编译期静态 shape，分批 Buffer 建成 `` `batchTotalBins = min(totalBins, MAX_BATCH_BINS) ≤ MAX_BATCH_BINS` `` 不变量，Buffer 物理 shape 统一编译期 `` `MAX_BATCH_BINS` ``，运行时只用前 `` `batchTotalBins` `` 槽位；Host 负责 clamp
5. 双缓冲（本范式 ascendc 侧为 `` `TQue` `` depth 2）在 DSL 侧对应「分批复用 + 计数/归并并行」：若 DSL 的 RingBuffer/搬运支持异步交接，可重叠下一批计数与本批归并；是否支持以 DSL 搬运/同步能力为准（待验证），不支持则串行分批

## 4 原子累加、归并与同步

1. **片上原子累加**（局部直方图计数步）：在 `` `cb.vf(mode="simt")` `` 内用 `` `cb.simt.atomic_add(local_hist.ptr(off), v)` ``（UB 地址，`` `LOCAL_HIST_ATOMIC` `` 已证实）。SIMT 多线程写同 bin 由原子加收敛；**不在** SIMT VF 内对 UB Buffer 做普通点赋值（capability-mapping.md §4 规则 7）
2. **归并**（局部直方图归并步）：`` `mem_copy(hist[start:start+bins,], local_hist[0:bins,], atomic_add=True)` ``（`` `LOCAL_HIST_FLUSH` `` 已证实）；1-D 切片用单元素元组（规则 9）
3. **无局部直方图 GM 原子**：`` `scalar.atomic_add(hist.ptr(idx), v)` ``（`` `GLOBAL_MERGE_ATOMIC` `` 已证实）；越界元素丢弃在原子加之前判定
4. **同步**：SIMT VF 内清 0 与累加之间用 `` `cb.simt.syncthreads()` ``；计数 → 归并的跨引擎同步由 SIMT VF 边界 + DSL 编译器自动处理（无需显式 SetFlag/WaitFlag，`` `VECTOR_MTE3_SYNC` `` 已证实）

## 5 bin index、越界与边界

1. bin index 计算（等宽公式 / 二分 lower_bound / 离散类标）与 `` `FLAT_INDEX_CALC` `` 展平是纯标量，在 kernel 内用普通表达式完成（[../../generic/bin-index.md](../../generic/bin-index.md)）
2. 越界丢弃（`val < lo || val > hi`；离散类标 `class < 0 || class ≥ num_classes`）、`min==max` 特判、上界钳制（`idx == nbins → nbins-1`）均在 kernel 内完成，语义不变
3. 前置 `` `PREFIX_MINMAX_REDUCE` ``（range 缺省时）真值已证实：每线程局部 min/max + `` `cb.simt.atomic_min/atomic_max(gm.ptr(0), v)` `` 归约到 GM 暂存（float32 跨核原子收敛，见 [capability-mapping.md](capability-mapping.md) §1）；D 维各轴独立
4. 尾块/空输入：N=0 空输入早退；每核分块的末核尾块按实际长度处理，不把无效元素计入

## 6 产出与验证义务

每个 DSL 实现决策按如下格式回答（micro-patterns 纪律）：

```
适用前提 → 最小代码位置/伪代码 → 生命周期或同步不变量 → 对应 Unit 测试
```

- 数学语义（hist[i][j] = count/sum 落桶）、shape/dtype 以当前算子规格为准；Buffer/RingBuffer 容量、同步关系按实际数据流及资源预算推导
- 引用待验证谓词的结论，按 [capability-mapping.md](capability-mapping.md) §3 声明假设并给出验证义务
- 用独立 Golden 覆盖支持的 dtype、shape、模板边界（`bins/inputsize=GM_ATOMIC_ADD_FACTOR`）、越界丢弃、min==max、N=0/N=1 极值、双输出 bin_edges
