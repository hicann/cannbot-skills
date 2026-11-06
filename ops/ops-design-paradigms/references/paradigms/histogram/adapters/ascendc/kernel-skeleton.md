<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# Histogram 范式 Ascend C SIMT kernel 骨架与原子累加落地（混合 VF）

> 给出可编译骨架（节选自 `simt_histogram_story/src/1_basic_simt.asc` 内核，做范式化注释，并改写为 `__simt_vf__` 函数形态），以及 bin index 分支、两种累加模板的 Ascend C 落地。
> 本范式含两个模板，按边界条件 `bins/inputsize < GM_ATOMIC_ADD_FACTOR` 选择：bins 在 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍以内（atomicadd 冲突较大）用局部直方图模板，否则（冲突较小）用无局部直方图模板。
>
> - **局部直方图模板**（`bins/inputsize < GM_ATOMIC_ADD_FACTOR`，bins 在 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍以内、atomicadd 冲突较大）：§2
> - **无局部直方图模板**（`bins/inputsize ≥ GM_ATOMIC_ADD_FACTOR`，bins 超过 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍、atomicadd 冲突较小）：§8
>
> §3~§4 为 bin index 分支，§5 为关键落地约束，§6 为二维（多维）落地，§7 为混合 VF 的 regbase 工程框架。API 真值见 [capability-mapping.md](capability-mapping.md)。

## 1 头文件与常量

```cpp
#include "kernel_operator.h"                        // SetAtomicAdd / DataCopyPad / SetAtomicNone（归并段）
#include "simt_api/device_sync_functions.h"         // asc_syncthreads / asc_threadfence
#include "simt_api/device_atomic_functions.h"       // asc_atomic_add

constexpr int32_t ALIGN_GRANULARITY = 8;                                // 行宽填充
constexpr int32_t SIMT_THREADS_PER_BLOCK = 512;                         // 每 block 线程数（默认 512，按性能测试结果调整）
constexpr int32_t BATCH_BINS_RAW = /* 单批桶数，受 ONCHIP_HIST_CAPACITY 约束 */;
constexpr int32_t BATCH_BINS = BATCH_BINS_RAW + ALIGN_GRANULARITY;      // 局部直方图行宽（padded）
```

计数段（SIMT VF）只包含 `simt_api/` 头文件；归并段（SIMD DMA 原子）用 `kernel_operator.h` 的 `SetAtomicAdd`/`DataCopyPad`/`SetAtomicNone`。

## 2 局部直方图模板（模板一，`bins/inputsize < GM_ATOMIC_ADD_FACTOR`）

> 每核一份 UB 局部直方图，按片上容量分批加载 bins（突破总 `nbins` 限制）；计数段 SIMT VF 每核分块 + 核内跨步统计本核 input，归并段 SIMD DMA 原子把本批 UB 累加到输出 GM，循环处理所有 bins。对应通用层 [../../generic/accumulation.md](../../generic/accumulation.md) §2。

### 2.1 计数段（SIMT VF，`__simt_vf__`）

```cpp
template <typename XType, typename OutType>
__simt_vf__ __aicore__ __launch_bounds__(SIMT_THREADS_PER_BLOCK) inline void HistogramSimtCount(
    __gm__ XType* x, __gm__ XType* min, __gm__ XType* max,
    __ubuf__ OutType* localHist,                    // 本批局部直方图（每核一份，padded 行宽）
    int32_t blockIdx, int32_t needXCoreNum,
    int64_t xIndexBase, int64_t coreDataLength,
    int32_t batchStart, int32_t batchBins, int32_t bins)
{
    if (blockIdx >= needXCoreNum) return;           // 多余 core 早退

    // ── 每核分块 + 核内跨步统计本核 input，只计落在当前批桶范围的元素 ──
    //    清 0 由 §2.3 regbase 侧 SIMD Duplicate 完成，SIMT VF 只做计数
    float minVal = *min, maxVal = *max;
    float range = maxVal - minVal, binsF = static_cast<float>(bins);
    for (int64_t index = threadIdx.x; index < coreDataLength; index += blockDim.x) {
        float val = x[xIndexBase + index];
        if (val >= minVal && val <= maxVal) {
            int32_t idx = BinIndex(val, minVal, maxVal, range, binsF, bins);
            if (idx >= batchStart && idx < batchStart + batchBins)
                asc_atomic_add(localHist + (idx - batchStart), 1);   // UB 原子累加（LOCAL_HIST_ATOMIC）
        }
    }
    // 计数结束无需 asc_syncthreads：SIMT VF→MTE3 归并前同步由 §2.3 的 EnQue/DeQue 携带
}
```

### 2.2 归并段（SIMD，regbase `__aicore__`）

```cpp
// regbase 入口 Process 内，每批计数完成后执行（非 SIMT VF 内）
// 注意：目的必须是 GlobalTensor（y_tensor 由 y_gm 经 SetGlobalBuffer 绑定，y_gm[batchStart] 为标量不可作 DataCopyPad 目的）
SetAtomicAdd<OutType>();                             // 置 DMA 原子加模式（dtype 由模板参数指定）
DataCopyPad(y_tensor[batchStart], localHist, dataCopyParams);  // hist[batchStart..] += localHist[0..]
SetAtomicNone();                                     // 关闭原子模式（DisableDmaAtomic 等价）
```

> 归并段把 `` `BATCH_BINS_RAW` `` 个 UB 计数原子累加到 `hist[batchStart .. batchStart+BATCH_BINS_RAW)`，跨核收敛由 DMA 原子的硬件原子性保证。精确签名与用法见 [capability-mapping.md](capability-mapping.md) §2.8。
>
> **归并前跨引擎同步**：计数段（SIMT VF / VECTOR）写 UB 完成到归并段（MTE3 `DataCopyPad`）读 UB 之间的 VECTOR→MTE3 同步，由 `` `TQue<TPosition::VECOUT>` `` 的 `` `EnQue` ``/`` `DeQue` `` 携带（见 §2.3 与 [capability-mapping.md](capability-mapping.md) §2.9），**无需显式 `SetFlag<HardEvent::V_MTE3>`**；`` `asc_syncthreads()` `` 只同步 block 内线程，不替代队列的跨引擎同步。

### 2.3 分批循环（regbase 入口 `Process`，QUEUE + DOUBLE BUFFER）

> UBUF 采用 `` `TQue` ``（QUEUE）+ **DOUBLE BUFFER**：每批 `` `AllocTensor` `` 取一块空闲 UB → SIMD `` `Duplicate` `` 清 0（`__aicore__` 上下文，SIMT VF 只计数不清 0）→ SIMT VF 计数写入 → `` `EnQue` `` 交 MTE3（携带 VECTOR→MTE3 同步）→ `` `DeQue` `` 取回由 MTE3 读、归并 → `` `FreeTensor` `` 归还。
>
> **depth 由 tiling 动态决定**（`` `useDoubleBuffer` `` 字段）：`` `totalBins × sizeof(HistT) > 1024` `` 字节 → depth=2（双缓冲分批）；`` `≤ 1024` `` 字节 → 单批一次性（depth=1 或不建队列，无分批循环），见 [../../generic/bin-model.md](../../generic/bin-model.md) §4。
>
> **双缓冲并行机理**：`` `TQue<TPosition::VECOUT, 2>` `` 的 depth=2 允许连续 `` `AllocTensor` `` 两次拿到两块不同 UB——第 1 批 `` `DataCopyPad` ``（MTE3）异步搬运时，第 2 批的 `` `Duplicate` ``+`` `asc_vf_call` ``（VECTOR）已在写第 2 块 UB，达到 VECTOR（计数）与 MTE3（归并）并行。

```cpp
template <typename XType, typename OutType>
__aicore__ inline void Process(GM_ADDR x, GM_ADDR min, GM_ADDR max, GM_ADDR y,
                               const HistogramTilingData* tiling)
{
    __gm__ XType* x_gm = (__gm__ XType*)x;
    __gm__ XType* min_gm = (__gm__ XType*)min;
    __gm__ XType* max_gm = (__gm__ XType*)max;
    __gm__ OutType* y_gm = (__gm__ OutType*)y;
    GlobalTensor<OutType> y_tensor;                          // 归并段 DataCopyPad 目标须为 GlobalTensor（裸 GM 指针的下标是标量，不可用）
    y_tensor.SetGlobalBuffer(y_gm);

    // ── 每核分块：host tiling 已把 input 切为每核连续块 ──
    int32_t blockIdx = static_cast<int32_t>(GetBlockIdx());
    int64_t xIndexBase = blockIdx * tiling->formerLength;
    int64_t coreDataLength = (blockIdx == tiling->needXCoreNum - 1) ? tiling->tailLength : tiling->formerLength;

    // ── 局部直方图：QUEUE + DOUBLE BUFFER（depth 由 tiling->useDoubleBuffer 决定）──
    // useDoubleBuffer=1（totalBins×sizeof > 1024 字节）→ depth=2 双缓冲分批；
    // useDoubleBuffer=0 → 单批一次性（depth=1 或不建队列，无分批循环，此处以 depth=2 双缓冲路径示意）
    AscendC::TPipe pipe;                                      // ⚠ 必须显式构造：InitBuffer/FetchEventID/GetTPipePtr() 依赖 TPipe 构造赋值
    int32_t histDepth = tiling->useDoubleBuffer ? 2 : 1;
    TQue<TPosition::VECOUT, 2> histQue;                       // 模板 depth 为编译期常量，双缓冲路径取 2
    pipe.InitBuffer(histQue, histDepth, BATCH_BINS * sizeof(OutType));

    for (int32_t batchStart = 0; batchStart < tiling->bins; batchStart += BATCH_BINS_RAW) {
        int32_t batchBins = min(BATCH_BINS_RAW, tiling->bins - batchStart);

        // (2) AllocTensor：从 queue 取一块空闲 UB
        LocalTensor<OutType> localHist = histQue.AllocTensor<OutType>();

        // (2) 清 0：SIMD Duplicate 把本批 batchBins 个 bins 清零（regbase __aicore__ 上下文，SIMD 向量指令）
        Duplicate(localHist, static_cast<OutType>(0), batchBins);

        // (3) 计数段：SIMT VF 每核分块 + 核内跨步统计本核 input（只计数、不清 0，见 §2.1）
        asc_vf_call<HistogramSimtCount<XType, OutType>>(dim3(SIMT_THREADS_PER_BLOCK),
            x_gm, min_gm, max_gm, (__ubuf__ OutType*)localHist.GetPhyAddr(), blockIdx, tiling->needXCoreNum,
            xIndexBase, coreDataLength, batchStart, batchBins, tiling->bins);

        // EnQue：计算出的结果写入 queue（携带 VECTOR→MTE3 跨引擎同步语义）
        histQue.EnQue(localHist);

        // DeQue：MTE3 部分开始读 queue 数据
        LocalTensor<OutType> mergeHist = histQue.DeQue<OutType>();

        // (4) 归并段：SIMD DMA 原子把本批累加到输出 GM
        SetAtomicAdd<OutType>();
        DataCopyPad(y_tensor[batchStart], mergeHist, dataCopyParams);
        SetAtomicNone();

        // FreeTensor：归还 UB
        histQue.FreeTensor(mergeHist);
    }
}
```

## 3 bin index 分支落地

```cpp
// 等宽线性映射；自定义边界见 §4
__inline__ int32_t BinIndex(float val, float minVal, float maxVal, float range,
                            float binsF, int32_t bins)
{
    int32_t idx;
    if (minVal == maxVal) {
        idx = bins / 2;                                        // min==max 特判
    } else {
        idx = static_cast<int32_t>((val - minVal) * binsF / range);  // floor（截断）
    }
    if (idx == bins) idx = bins - 1;                           // max 精确值钳制
    return idx;
}
```

与 [../../generic/bin-index.md](../../generic/bin-index.md) §1~§3 一一对应：等宽公式、min==max 特判、边界钳制、越界丢弃（`val < minVal || val > maxVal` 在上游 `if` 中丢弃）。

## 4 自定义边界

```cpp
// 逐线程标量二分定位；edges 为递增边界 [0..nbins]（共 nbins+1 条），val 落在 [edges[j], edges[j+1])
int32_t lo = 0, hi = nbins + 1;             // 上界含 edges[nbins]，使 val > edges[nbins] 得 idx = nbins 走丢弃分支
while (lo < hi) {
    int32_t mid = (lo + hi) >> 1;
    if (edges[mid] <= val) lo = mid + 1;
    else hi = mid;
}
int32_t idx = lo - 1;                       // lower_bound - 1
if (idx < 0) return;                        // val < edges[0]：越界丢弃
if (idx >= nbins) {                         // idx == nbins，即 val >= edges[nbins]
    if (val == edges[nbins]) idx = nbins - 1;   // 末桶右闭：val == edges[nbins] 归 nbins-1（bin-index.md §3.1）
    else return;                                // val > edges[nbins]：越界丢弃
}
// 落在当前批 [batchStart, batchStart+batchBins) 才累加：
if (idx >= batchStart && idx < batchStart + batchBins)
    asc_atomic_add(localHist + (idx - batchStart), 1);
```

参考样例未覆盖该形态，属「待验证」；`edges` 常驻 GM 逐线程读取。

## 5 关键落地约束

- **输出清零**：GM `hist` 由 host 在启动前清零（`SetAtomicAdd`/`asc_atomic_add` 均不初始化），否则归并结果错误。
- **`__ubuf__` 静态数组**：编译期确定大小，`BATCH_BINS = BATCH_BINS_RAW + ALIGN_GRANULARITY`；`BATCH_BINS_RAW` 由 `` `ONCHIP_HIST_CAPACITY` `` 决定（见 [../../generic/bin-model.md](../../generic/bin-model.md) §4）。总 `nbins` 超过单批容量时按批切分（§2.3 循环），不再要求 `nbins` 整体驻片上。
- **原子累加选择**：计数段用 `__ubuf__` 重载的 `` `LOCAL_HIST_ATOMIC` ``（`asc_atomic_add(__ubuf__ ...)`）；归并段用 SIMD DMA 原子 `` `LOCAL_HIST_FLUSH` ``（`SetAtomicAdd`+`DataCopyPad`+`SetAtomicNone`），非 SIMT 原子（见 [capability-mapping.md](capability-mapping.md) §2.1/§2.8）。
- **计数段 vs 归并段上下文**：计数段在 `__simt_vf__` 内（SIMT VF / VECTOR）；归并段在 regbase `__aicore__` 内（SIMD MTE3）。两者由 `asc_vf_call` 的返回衔接；计数段内无 `asc_syncthreads()`，同 bin 并发写由 `asc_atomic_add(__ubuf__)` 原子收敛，**归并（MTE3 读 UB）前的 VECTOR→MTE3 跨引擎同步由 `` `TQue<TPosition::VECOUT>` `` 的 `EnQue`/`DeQue` 携带，无需显式 `SetFlag<HardEvent::V_MTE3>`**（见 §2.3 与 [../../generic/sync-model.md](../../generic/sync-model.md) §2/§3）。
- **float4 向量化**：计数段阶段 B 的标量读可替换为 `float4` 向量读，不改 bin 语义与同步点。

## 6 二维（多维）落地

> 本节为 §2 骨架的 D 维推广，与通用层 [../../generic/bin-index.md](../../generic/bin-index.md) §6、[../../generic/accumulation.md](../../generic/accumulation.md) §8 对应。
> 参考样例 `simt_histogram_story` 仅覆盖 1D，以下 D 维骨架为**范式化推导**，落地前需上板验证（标「待验证」，见 [capability-mapping.md](capability-mapping.md) §3）。

与 1D 骨架的差异仅三点：局部直方图大小取总桶数 `N_BINS = ∏ nbins_axis` 的**分批**（`` `BATCH_TOTAL_BINS` ``）；计数段每样本读 D 个特征、各轴算 bin index 后行优先展平，再判断是否落在当前批展平 index 范围并累加；归并段按展平 index 分批归并（与 §2.2 同构，仅桶数换成 `` `BATCH_TOTAL_BINS_RAW` ``）。

### 6.1 一般 D 维推广

- **读 D 个特征**：D 个 GM 数组 `feat[0..D-1][n]`（或单一 N×D 布局，按 `feat[n*D + d]` 访问）。D 为规格参数，由 host 传入。
- **每轴 bin index**：对每轴独立复用 §3 / §4 的 `BinIndex` 分支（等宽公式 / 二分 / 离散类标），得到 `i_d`。
- **展平**（`` `FLAT_INDEX_CALC` ``）：行优先 Horner 递推，纯标量整数乘加，无专用 API：

  ```cpp
  int32_t flatIdx = i0;
  for (int32_t d = 1; d < D; d++)
      flatIdx = flatIdx * nbinsAxis[d] + i_d;      // nbinsAxis[d] 为各轴桶数（标量/常量数组）
  ```

- **局部直方图容量**：`` `BATCH_TOTAL_BINS = BATCH_TOTAL_BINS_RAW + ALIGN_GRANULARITY` ``，`BATCH_TOTAL_BINS_RAW × sizeof(HistT) ≤ ONCHIP_HIST_CAPACITY`（见 [../../generic/bin-model.md](../../generic/bin-model.md) §5.4）。
- **归并**：归并段与 §2.2 同构，以 `batchStart` 遍历展平 index 分批归并（`totalBins = ∏ nbins_axis`）。
- 加权变体（`weights`）将 `asc_atomic_add(localHist + (flatIdx - batchStart), 1)` 的 `1` 换成 `weights[n]`（`float`/`int` 由输出 dtype 决定），其余不变。

### 6.2 与 1D 的同构关系

§6.1 与 §2 骨架逐行对应：`bins → totalBins`、`BATCH_BINS → BATCH_TOTAL_BINS`、`x[i] → feat[d][n]`、`BinIndex → 每轴 BinIndex + 展平`、同步点数量与位置不变（计数段内无 `asc_syncthreads()`，计数→归并的 VECTOR→MTE3 跨引擎同步由 `TQue<TPosition::VECOUT>` 的 `EnQue`/`DeQue` 携带，同 §2.1/§2.3）。因此 1D 的全部结论（寄存器选型、GridDim 扫描、float4 向量化）可直接复用到 D 维。

## 7 混合 SIMT VF 工程框架

> 混合 SIMT VF = regbase 工程框架 + SIMT VF 内核。SIMT 内核即 §2 计数段，由 regbase 入口的 `asc_vf_call` 启动、归并段在 regbase `__aicore__` 内 SIMD 完成；tiling 由 `op_host` TilingFunc 计算。
> 参考样例 `examples/add_example/`（本范式内置）演示该工程框架（逐元素加法）；API 真值见 [capability-mapping.md](capability-mapping.md) §4。

### 7.1 op_host TilingFunc（`op_host/arch35/*_tiling.cpp`）

```cpp
#include "register/op_impl_registry.h"          // IMPL_OP_OPTILING
#include "../../op_kernel/arch35/add_example_tiling_data.h"

static ge::graphStatus AddExampleTilingFunc(gert::TilingContext* context)
{
    // 1. 校验输入 → 2. 取平台信息（coreNum/ubSize）→ 3. 计算 tiling（totalNum/needCoreNum）
    AddExampleTilingData* tiling = context->GetTilingData<AddExampleTilingData>();
    // ...
    context->SetBlockDim(tiling->needCoreNum);             // grid = core 数
    // ✓ 必须为 DCache 预留空间：SetLocalMemorySize 传入 ubSize − DCACHE_SIZE，禁止 SetLocalMemorySize(0)。
    //   置 0 = DCache 预留 0，__ubuf__ 静态直方图会与 DCache 抢占同一片上空间，有越界风险；
    //   「纯 SIMT 无动态 UB」不是置 0 的理由——该接口除声明动态 UB 外还承担预留 DCache 职责。
    //   依据：add_example_tiling.cpp（DCACHE_SIZE=32*1024，SetLocalMemorySize(ubSize − DCACHE_SIZE − STATIC_UB_ESTIMATE)，
    //   且校验 ubSize > DCACHE_SIZE + STATIC_UB_ESTIMATE，L36/L195-201）；ascendc-simt-tiling-design/guide.md L63-71（DCACHE_SIZE=128*1024）。
    if (ubSize <= DCACHE_SIZE) { /* 报错：UB 不足以预留 DCache（≥32KB） */ }
    context->SetLocalMemorySize(ubSize - DCACHE_SIZE);     // dyn UB 预算 = __ubuf__ 局部直方图单批可用上限
    context->SetTilingKey(GET_TPL_TILING_KEY(SCH_MODE));   // tiling key（模板分派）
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(AddExample)
    .Tiling(AddExampleTilingFunc)
    .TilingParse<AddExampleCompileInfo>(TilingParseForAddExample);
```

> **容量核对**：`SetLocalMemorySize` 的传入值 `ubSize − DCACHE_SIZE` 即 `__ubuf__` 局部直方图**单批**可用上限，容量预算按 `ubSize − DCACHE_SIZE` 口径而非 `ubSize`。LOCAL_HIST 模板用 `` `TQue<TPosition::VECOUT, 2>` `` **DOUBLE BUFFER**（depth=2）同时持有两批，且每批行宽为 `` `BATCH_TOTAL_BINS = BATCH_TOTAL_BINS_RAW + ALIGN_GRANULARITY` ``（padded 对齐），故容量约束为 `2 × (BATCH_TOTAL_BINS_RAW + ALIGN_GRANULARITY) × sizeof(HistT) ≤ ubSize − DCACHE_SIZE`。int32 计数（sizeof=4、ALIGN_GRANULARITY=8）且预算约 216KB 时，`BATCH_TOTAL_BINS_RAW ≤ 216KB / (2×4) − 8 = 27640`，即单批 raw 桶数约 27640（padded 行宽 27648），而非整预算单份的 55296。

### 7.2 regbase 入口（`op_kernel/*_apt.cpp`）

```cpp
#include "arch35/add_example.h"

template <uint32_t schMode>
__global__ __aicore__ void add_example(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    // ⚠ 混合 SIMT VF + TQue（SIMT VF 访问 UB）必须声明混合核类型并让 AIC 子核早退，
    //    否则 asc_vf_call 编译为空体、GetPhyAddr() UB 偏移越界 → error 271。
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    if (g_coreType == AIC) { return; }
    REGISTER_TILING_DEFAULT(AddExampleTilingData);                        // 声明默认 tiling 结构
    GET_TILING_DATA_WITH_STRUCT(AddExampleTilingData, tilingData, tiling); // 解析 tiling
    if constexpr (schMode == /* TILING_KEY_* */) {
        NsAddExample::Process<T>(x, y, z, &tilingData);
    }
}
```

### 7.3 SIMT VF 内核 + `asc_vf_call`（`op_kernel/arch35/*.h`）

```cpp
#include "kernel_operator.h"
#include "simt_api/common_functions.h"      // asc_vf_call / dim3

namespace NsAddExample {
template <typename T>
__simt_vf__ __aicore__ __launch_bounds__(512) inline void OpComputeSimt(
    uint64_t count, __gm__ T* x1, __gm__ T* x2, __gm__ T* y)
{
    // grid-stride：blockIdx.x/blockDim.x 即 core 号/每 core 线程数
    for (uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += blockDim.x * gridDim.x) {
        y[i] = x1[i] + x2[i];
    }
}

template <typename T>
__aicore__ inline void Process(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const AddExampleTilingData* tilingData)
{
    __gm__ T* x1_gm = (__gm__ T*)x1;
    __gm__ T* x2_gm = (__gm__ T*)x2;
    __gm__ T* y_gm  = (__gm__ T*)y;
    asc_vf_call<OpComputeSimt<T>>(dim3(512), tilingData->totalNum, x1_gm, x2_gm, y_gm);
}
} // namespace NsAddExample
```

### 7.4 直方图内核接入混合 VF

将 §2 的计数段 `HistogramSimtCount` 由 regbase 入口 `Process` 内 `asc_vf_call` 启动、归并段在 `Process` 内 SIMD 完成即可，仅三点改动：

1. **启动**：regbase 入口 `Process` 内先按 §2.3 计算每核分块参数（`blockIdx`/`xIndexBase`/`coreDataLength`），再 `asc_vf_call<HistogramSimtCount<XType, OutType>>(dim3(SIMT_THREADS_PER_BLOCK), x_gm, min_gm, max_gm, localHist, blockIdx, tiling->needXCoreNum, xIndexBase, coreDataLength, batchStart, batchBins, bins)`；`grid` 由 `SetBlockDim` 的 core 数决定，`block` 固定为 `dim3(SIMT_THREADS_PER_BLOCK)`。
2. **`__ubuf__` 局部直方图（QUEUE + 双缓冲）**：`Process` 内声明 `TQue<TPosition::VECOUT, 2> histQue` + `InitBuffer(histQue, 2, BATCH_BINS * sizeof(OutType))`，每批 `AllocTensor` 取 buffer → SIMD `Duplicate` 清 0 → `asc_vf_call` 计数（`asc_atomic_add(__ubuf__)` 语义不变，SIMT VF 不清 0、无 `asc_syncthreads()`）→ `EnQue` → `DeQue`。
3. **归并段**：`DeQue` 取回后，在 `Process` 内用 `SetAtomicAdd<OutType>()` + `DataCopyPad` + `SetAtomicNone()` 把本批累加到输出 GM，再 `FreeTensor` 归还。

其余（bin index 分支、分批循环、float4 向量化、GridDim 扫描）与 §2~§6 结论完全一致，可直接复用。

## 8 无局部直方图模板（模板二，`bins/inputsize ≥ GM_ATOMIC_ADD_FACTOR`）

> 与 §2 局部直方图模板并列的模板：不在 `__ubuf__` 上分配局部直方图，不做分批与 DMA 归并；每核分块 + 核内跨步遍历输入直接 `asc_atomic_add(__gm__ ...)` 累加到输出 GM。适用 `bins/inputsize ≥ GM_ATOMIC_ADD_FACTOR`（bins 超过 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍、atomicadd 冲突较小），或大 `nbins`（局部直方图超 `ubSize − DCACHE_SIZE` 预算）。对应通用层 [../../generic/accumulation.md](../../generic/accumulation.md) §3。

### 8.1 SIMT VF 内核（无 `__ubuf__` 局部直方图）

```cpp
template <typename XType, typename OutType>
__simt_vf__ __aicore__ __launch_bounds__(SIMT_THREADS_PER_BLOCK) inline void HistogramSimtNoLocal(
    __gm__ XType* x, __gm__ XType* min, __gm__ XType* max, __gm__ OutType* y,
    int32_t blockIdx, int32_t clearYCoreNum, int64_t clearYIndexBase, int64_t clearYDataLength,
    int32_t needXCoreNum, int64_t xIndexBase, int64_t coreDataLength,
    int32_t bins)
{
    // ── 阶段 A（可选）：每核分块 + 核内跨步清零输出 GM（若 host 未预清零；见 8.2 时序约束）──
    if (blockIdx < clearYCoreNum) {
        for (int64_t index = threadIdx.x; index < clearYDataLength; index += blockDim.x)
            y[clearYIndexBase + index] = 0;      // GM 直接清零
    }

    // ── 阶段 B：每核分块 + 核内跨步计数，直接 GM 原子累加（无 __ubuf__）──
    if (blockIdx >= needXCoreNum) return;
    float minVal = *min, maxVal = *max;
    float range = maxVal - minVal, binsF = static_cast<float>(bins);
    for (int64_t index = threadIdx.x; index < coreDataLength; index += blockDim.x) {
        float val = x[xIndexBase + index];
        if (val >= minVal && val <= maxVal) {
            int32_t idx = BinIndex(val, minVal, maxVal, range, binsF, bins);
            asc_atomic_add(y + idx, 1); // GM 原子累加（GLOBAL_MERGE_ATOMIC，y + idx 为 __gm__ 地址）
        }
    }
}
```

### 8.2 关键落地约束

- **无局部直方图/无归并**：VF 内不分配 `__ubuf__` 数组，无 §2 的清零与归并段；仅（可选）GM 清零段 + 每核分块 + 核内跨步计数段。
- **bin index 分支复用**：§3 等宽 / §4 自定义边界二分完全复用，仅累加目标由 `localHist + (idx - batchStart)` 变为 `y + idx`。
- **清零时序与可见性**（二选一，见 8.3）：
  1. **host 预清零（推荐）**：host 侧 `memset` 在 kernel 启动前将 `hist` 清零，VF 内跳过阶段 A 直接计数；
  2. **kernel 内每核分块清零**：VF 内先跑阶段 A 清零，但跨 block 的清除写须在计数前对所有线程/block 可见，否则线程「读到旧 `hist` + 原子写新值」产生数据竞争，需 `asc_threadfence()`（`` `MEMORY_FENCE` ``）与跨 block 栅栏（`SyncAll()` + tiling 侧 `SetScheduleMode(1)`）保证。本形态无 `__ubuf__` 共享数据故不依赖块内 `asc_syncthreads()`，默认优先 host 预清零。
- **加权变体**：`asc_atomic_add(y + idx, 1)` 的 `1` 换成 `weights[n]`，输出 `OutType` 相应为 float；`asc_atomic_add` 的 `__gm__` 重载支持 int32/int64/float，见 [capability-mapping.md](capability-mapping.md) §2.1。
- **性能预期**：每元素一次 GM 原子，同 bin 并发写在 GM 串行化；`bins/inputsize ≥ GM_ATOMIC_ADD_FACTOR`（bins 超过 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍、atomicadd 冲突较小）或大 `nbins` 下可接受，`bins/inputsize < GM_ATOMIC_ADD_FACTOR`（bins 在 inputsize 的 `GM_ATOMIC_ADD_FACTOR` 倍以内、atomicadd 冲突较大）时优先 §2 局部直方图模板。

### 8.3 接入混合 VF 工程框架

与 §7.4 同构（regbase 入口 `Process` 内 `asc_vf_call` 启动、`grid` 由 `SetBlockDim` 的 core 数决定），差异如下：

1. **启动**：regbase 入口 `Process` 内先计算每核分块参数（输入分块的 `blockIdx`/`needXCoreNum`/`xIndexBase`/`coreDataLength`，及输出清零分块的 `clearYCoreNum`/`clearYIndexBase`/`clearYDataLength`），再 `asc_vf_call<HistogramSimtNoLocal<XType, OutType>>(dim3(SIMT_THREADS_PER_BLOCK), x_gm, min_gm, max_gm, y_gm, blockIdx, clearYCoreNum, clearYIndexBase, clearYDataLength, needXCoreNum, xIndexBase, coreDataLength, bins)`。
2. **UB 预算与 DCache 预留**：VF 无 `__ubuf__` 局部直方图，`SetLocalMemorySize` 传入值不再受 `` `BATCH_BINS_RAW × sizeof(HistT)` `` 约束，但**仍须传 `ubSize − DCACHE_SIZE`、禁止 `SetLocalMemorySize(0)`**——DCache 预留职责不变（见 §7.1）。
3. **输出清零**：`hist` 由 host 预清零（推荐）或 VF 内每核分块清零（见 8.2）。

### 8.4 无局部直方图模板 float 输出（int 计数 + float cast）强制约束

> 对应通用层 [../../generic/accumulation.md](../../generic/accumulation.md) §4.1。累加值为 int 计数而输出 dtype 为 float 时，计数走 int 原子加（无需确定性模板），末步 int→float cast 写回输出；无局部直方图模板的该路径须满足下列 `volatile` 强制约束（第 1、2 点为必须执行项，MANDATORY）：

1. **（必须执行）计数 SIMT VF 的输出 GM 参数必须增加 `volatile` 关键字修饰**（形如 `volatile __gm__ int32_t*` 计数目标）——以避免计数写通过 DCache 写出后，输出 GM 内容会被 DCache 中内容（过期脏行回写）覆盖；
2. **（必须执行）cast SIMT VF 的输入 GM 参数必须增加 `volatile` 关键字修饰**（形如 `volatile __gm__ int32_t*` 计数源）——直接从输入 GM 中读取数据，以免读取到 DCache 中过期数据。

> 强制性说明：上两点为无局部直方图模板 float 输出路径的**必须执行项**，不可省略、不可用其他同步原语替代。缺失 `volatile` 时，跨核计数结果对 kernel 内后继读（cast 段）存在被 DCache 过期内容覆盖/污染的确定性行为（实测表现为 bin 计数丢失）。

## 9 确定性模板（模板三）

> 与 §2/§8 两模板并列的第三模板：**bins 并行 + input 串行 + 无原子累加**，确保浮点累加顺序确定。
> 由 tiling 侧 `` `GetDeterministic()` `` 选择，不参与 `bins/inputsize < GM_ATOMIC_ADD_FACTOR` 边界条件分派。
> 对应通用层 [../../generic/accumulation.md](../../generic/accumulation.md) §4。

### 9.1 tiling 侧：确定性判定 + tilingKey + bins 分核

```cpp
// op_host/arch35/*_tiling.cpp
// 1. 确定性开关：GetDeterministic()==1 → 进入确定性模板
tiling->useDeterministic = (context_->GetDeterministic() == 1);

// 2. tilingKey 补充确定性模板枚举值（见 *_tiling_key.h / *_tiling.cpp）
//    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, ..., SCH_MODE_ELEMWISE, SCH_MODE_DETERMINISTIC)
//    SetTilingKey(tiling->useDeterministic ? GET_TPL_TILING_KEY(SCH_MODE_DETERMINISTIC)
//                                          : GET_TPL_TILING_KEY(SCH_MODE_ELEMWISE));

// 3. 输出 bins 分核：确定单核/尾核处理的 bins 段；二维输出先展平为 1D。
//    饱和计算避免 nbins < coreNum 时 tailBins 为负：coreStart/coreEnd 以 nbins 封顶，允许尾核为空段。
int64_t binsPerCore  = (nbins + coreNum - 1) / coreNum;        // 单核 bins 数（ceil）
int64_t coreBinStart = min(coreIdx * binsPerCore, nbins);       // 本核 bins 起始（饱和）
int64_t coreBinEnd   = min(coreBinStart + binsPerCore, nbins);  // 本核 bins 终止（饱和）
```

### 9.2 kernel 侧：核内 stride 排布遍历 bins + 遍历全部 input

```cpp
// 每核负责 bins 段 [coreBinStart, coreBinEnd)；本核 bins 不再切分，
// 核内线程以 stride 方式遍历本核所有 bins（每个 bin 恰被一个线程命中）。
for (int32_t bin = coreBinStart + threadIdx.x; bin < coreBinEnd; bin += blockDim.x) {
    // 对命中的每个 bin，按固定顺序遍历全部 input
    for (int64_t n = 0; n < totalNum; n++) {
        float val = static_cast<float>(x[n]);
        if (val < minVal || val > maxVal) {
            continue;                                          // 越界丢弃
        }
        int32_t idx = BinIndex(val, minVal, maxVal, range, binsF, bins);
        if (idx == bin) {
            y[bin] += 1.0f;                                    // 计数；加权换成 += weights[n]
        }
    }
}
```

- **无原子**：`y[bin] += ...` 是普通 GM 点赋值/自增（`__simt_vf__` 内 GM 点赋值不受限），因每个 bin 只被 strided
  命中的本线程累加，无并发写，累加顺序即本线程遍历 input 的固定顺序。
- **输出清零**：`hist` 由 host 预清零（优先）或 VF 内每核分块清零（见 §8.2）；本形态无原子累加但仍需初值 0。
- **bin index / min==max 特判 / 越界丢弃 / 边界钳制**：复用 §3。
- **二维/多维**：先把展平 `idx ∈ [0, N_BINS)` 当 1D 处理（同 §6 展平），再按 §9.1 分核、按 §9.2 核内 stride 排布。
- **确定性语义**：本形态是计数/加权浮点累加中唯一保证累加顺序确定的路径；`` `GetDeterministic()` `` 为 1 时
  优先于 §2/§8 两模板。
