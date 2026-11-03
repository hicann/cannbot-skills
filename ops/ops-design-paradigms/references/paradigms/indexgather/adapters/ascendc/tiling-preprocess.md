# IndexGather Tiling 预处理：合轴与元信息

> dim 归一化 / 广播校验 / MergeAxis 合轴算法（宽定义）/ 元信息计算 / use64Bit 判定。设计文档 §3 的真值源。

---

## 1 dim 归一化与静态校验

```cpp
// dim ∈ [-R, R)，负值 +R 归一化到 [0, R)
int64_t dimNorm = (dimRaw < 0) ? dimRaw + static_cast<int64_t>(rank) : dimRaw;
OP_CHECK_IF(dimNorm < 0 || dimNorm >= static_cast<int64_t>(rank),
    OP_LOGE(context, "dim %ld out of range [0, %zu)", dimNorm, rank),
    return ge::GRAPH_FAILED);
```

校验清单（infershape 与 tiling 双端，动态维由 tiling 兜底；逐条标注适用型）：

- 【C 型】`input.rank == index.rank`，`rank ∈ [1, MAX_RANK]`（A 型 index 任意 rank、B 型 indices 任意 rank，均无此相等约束）
- 【A/C 型】dim 负值 +R 归一化后 ∈ [0, R)（B 型无 dim 属性，轴语义由 batch_dims 承载）
- 【C 型】广播约束：`d != dim` 时 `index.size(d) <= input.size(d)`
- 【B 型】`K = indices.shape[-1]`，语义约束 `0 <= K <= rank(params) - batch_dims`；具体合法范围由 spec/需求确定
- 【A/B 型】batch_dims 组合：前缀逐维相等 `index.shape[:b] == input.shape[:b]`（相等非广播）+ 归一化后 `axis >= b`（batch 前缀不参与负值归一化目标域）

## 2 MergeAxis 合轴（宽定义切段 reshape 合并）

**目的**：把 dim 之外可合并的连续维折叠成单根 prod 轴，等效解码除法次数 `rank-1 → mergeRank-1`，对所有场景无退化（无段可合时 mergeRank = rank，行为不变）。

### 2.1 切段规则

- **dim 维单独保留**（其坐标被 index 值 j 替换，两空间步长不同构，禁止折叠）
- 其余维切成**极大可合并段**：段 `[a..b]` 可合并 ⟺ 段内除**段头 a**（行主序编号最小维）外 `index.size[d] == input.size[d]`（段身必须逐维相等）
- **段头广播豁免**：`index.size[a] < input.size[a]` 允许（段头 size 不进任何步长，只影响坐标范围）
- 段长 ≥ 2 才合并，单根段原样保留

**不可合轴判定**：除 dim 外不存在任何相邻维对 `(d, d+1)` 同时满足 `index.size==input.size` 且 `d+1 != dim`（全广播型 / 相等维被广播维隔断的孤岛型）。

### 2.2 实现代码

```cpp
static ge::graphStatus MergeAxis(gert::TilingContext* context,
                                 const std::array<int64_t, MAX_RANK>& xShape,
                                 const std::array<int64_t, MAX_RANK>& idxShape,
                                 int32_t rank, int32_t& dim,
                                 std::array<int64_t, MAX_RANK>& xMerge,
                                 std::array<int64_t, MAX_RANK>& idxMerge,
                                 int32_t& mergeRank)
{
    int32_t outIdx = 0;
    int32_t dimShift = 0;    // dim 前合并段累计减少的维数
    int32_t i = 0;
    while (i < rank) {
        if (i == dim) {
            xMerge[outIdx] = xShape[i];
            idxMerge[outIdx] = idxShape[i];
            ++outIdx;
            ++i;
            continue;
        }
        int32_t j = i;
        while (j + 1 < rank && j + 1 != dim && idxShape[j + 1] == xShape[j + 1]) {
            ++j;
        }
        if (j > i) {
            // 合并段 [i..j]：两空间分别 prod（段头广播时可不同）
            int64_t xProd = 1;
            int64_t idxProd = 1;
            for (int32_t d = i; d <= j; ++d) {
                OP_CHECK_IF(xShape[d] != 0 && xProd > INT64_MAX / xShape[d],
                    OP_LOGE(context, "input merge prod overflow at dim %d", d),
                    return ge::GRAPH_FAILED);
                OP_CHECK_IF(idxShape[d] != 0 && idxProd > INT64_MAX / idxShape[d],
                    OP_LOGE(context, "index merge prod overflow at dim %d", d),
                    return ge::GRAPH_FAILED);
                xProd *= xShape[d];
                idxProd *= idxShape[d];
            }
            xMerge[outIdx] = xProd;
            idxMerge[outIdx] = idxProd;
            if (i < dim) {
                dimShift += j - i;
            }
            ++outIdx;
        } else {
            xMerge[outIdx] = xShape[i];   // 单根段原样保留
            idxMerge[outIdx] = idxShape[i];
            ++outIdx;
        }
        i = j + 1;
    }
    dim -= dimShift;
    mergeRank = outIdx;
    return ge::GRAPH_SUCCESS;
}
```

### 2.3 不变量（合轴后必须保持）

| 量 | 不变原因 |
|----|---------|
| `outSize = prod(index.shape)` | 每维恰入一段/单根，prod 重组合不变 |
| `gatherDimSize = input.shape[dim]` | dim 单独保留 |
| `inStrideDim = prod(input.shape[dim+1:])` | 低段合并轴 prod = 同值 |

调用位置：ValidateShape 校验通过后、元信息计算前；后续全部按合并 shape 计算。

## 3 元信息计算与 use64Bit

```cpp
// 合并后视图的累积步长（行主序）
for (int32_t d = mergeRank - 1; d >= 0; --d) {
    outStrides[d] = oStride;  oStride *= idxMerge[d];   // index 空间
    inStrides[d]  = iStride;  iStride  *= xMerge[d];    // input 空间
}
inStrideDim = inStrides[dim];   // 合轴不变量
inputRank = mergeRank;

// use64Bit：input/output/index/task 四项计数任一超过 INT32_MAX 即切 64 位；
// indexDtype 不参与（范式强制规则 5）
int64_t inputNum = input.GetShapeSize();
int64_t outSize  = output.GetShapeSize();
int64_t indexNum = indexOrIndices.GetShapeSize();   // A/C: index；B: indices
int64_t taskNum  = outSize;                         // 每输出元素一任务
bool use64Bit = (inputNum > INT32_MAX) || (outSize > INT32_MAX) ||
                (indexNum > INT32_MAX) || (taskNum > INT32_MAX);
```

**覆盖性说明（C 型）**：任意 `outStrides[d] ≤ outSize`、`inStrides[d] ≤ inputNum`、`gatherDimSize ≤ inputNum`，均被四项计数判定覆盖，无需逐项判定。

**A 型 batch 组合覆盖性**：`paramsBatchStride = prod(input.shape[b:]) ≤ inputNum`、`innerTotal ≤ outSize`（outSize = batchNum × innerTotal）、`batchNum ≤ outSize`，四项计数判定仍完备，无需扩展。

**B 型覆盖性**：`sliceSize ≤ inputNum`、`innerTotal ≤ outSize`、`paramsBatchStride ≤ inputNum`，同样被四项计数判定覆盖，无需再逐项判定。

## 4 多核切分（核数两步法）

算法与公式真值源在 [../../generic/thread-layout.md](../../generic/thread-layout.md) §2。本文仅载 AscendC 侧落地。

```cpp
int64_t blockFactor = (outSize + coreNum - 1) / coreNum;
if (blockFactor < PER_CORE_MIN) { blockFactor = PER_CORE_MIN; }   // 下限保护 1024
int64_t needCoreNum = (outSize + blockFactor - 1) / blockFactor;
if (needCoreNum > coreNum) { needCoreNum = coreNum; }
if (needCoreNum <= 0) { needCoreNum = 1; }   // 空输出兜底
context->SetBlockDim(needCoreNum);
```

以 outSize（输出元素总数）为切分对象；VF 内建网格（gridDim×blockDim）给出全局步长，无需 per-core 区间计算。

## 5 Local Memory / DCache 预留

DCache 预留的硬件级规则（范围 / 收益 / 兜底）真值源在 [common/simt-hardware-model.md](../../../common/simt-hardware-model.md) §4。本文仅载本范式 AscendC 侧落地。

本范式纯 GM 直访，UB 仅作 TilingData 数组参数中转（~数百 B，kernel 内 `LocalMemAllocator` 动态分配，不占此预留），真正需要在 host 侧预留的是 **DCache**。

```cpp
constexpr uint32_t DCACHE_MIN = 32 * 1024;
constexpr uint32_t DCACHE_MAX = 128 * 1024;

// 剩余 UB = ubSize - STATIC_UB_ESTIMATE（本范式 STATIC_UB_ESTIMATE = 0）
uint64_t dcache = ubSize - STATIC_UB_ESTIMATE;
if (dcache > DCACHE_MAX) { dcache = DCACHE_MAX; }        // 上限裁剪
OP_CHECK_IF(dcache < DCACHE_MIN,
    OP_LOGE(context, "remaining UB %lu less than DCache min %u", dcache, DCACHE_MIN),
    return ge::GRAPH_FAILED);                             // 下限兜底
context->SetLocalMemorySize(
    static_cast<uint32_t>(ubSize - dcache - STATIC_UB_ESTIMATE));
```

> 说明：DCACHE_MIN=32KB 为下限示例值（example 取该值），实际按剩余 UB 在 [32KB, 128KB] 间取、越大性能越好；`SetLocalMemorySize` 传入的是扣除 DCache 预留后剩余的 local memory 容量。
