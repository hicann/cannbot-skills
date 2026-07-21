# P2: Adaptive Tiling (自适应分块策略)
## Overview
专家实现的Tiling不是固定值，而是根据平台UB大小和数据类型动态计算。CalMaxFormerNum函数计算UB能容纳的最大行数，确保数据尽量驻留在UB中，减少GM访问。计算公式：availableUb = ubSize - RESERVED_UB_SIZE - idxAlignNum * sizeof(int) * USE_IDX_NUM_IN_UB；maxFormerNum = (availableUb / (gradAlignNum * sizeof(float) * useGradNum)) * gradAlignNum。其中RESERVED_UB_SIZE (20KB)预留栈空间，USE_IDX_NUM_IN_UB (3)是索引队列数量，useGradNum根据数据类型变化（FP32=3, FP16=4）。

## When to Use
- Variable row/column dimensions
- 针对不同输入形状选择最优策略，最大化UB利用率和计算效率，预期性能提升20-50%
- 根据数据特征选择最优策略，大数据量用向量化，小数据量用scatter，重叠场景用atomic add，覆盖全场景性能最优
- 针对不同数据规模自动选择最优策略，最大化UB利用率和计算效率

## Trade-off
- Tiling逻辑复杂，需要维护三种Kernel实现
- tiling逻辑复杂，需要维护多个kernel路径
- 增加tiling计算复杂度和代码量

**Source operators**: adaptive_avg_pool3d, adaptive_max_pool3d_grad, add_rms_norm_dynamic_quant, apply_adagrad_d, ascend_quant_v2, conv2d_v2, dynamic_block_quant, embedding_dense_grad_v2, flash_attention_score, foreach_abs, foreach_add_scalar, foreach_addcdiv_list, grouped_dynamic_mx_quant, grouped_matmul, inplace_add_rms_norm, layer_norm_v3, layer_norm_v4, max_pool_v3, max_pool_with_argmax_v3, modulate, multi_scale_deformable_attention_grad, norm_common, rms_norm_grad, scaled_masked_softmax_v2

---

## Variant A: 三模式UB Tiling策略
Source: adaptive_avg_pool3d

根据输入输出形状关系设计三种UB Tiling策略：MODE_SPLIT_C(通道维度分块)适合C大但输出空间小的情况；MODE_SPLIT_W(宽度维度分块)适合空间下采样较大情况；MODE_MULTI_W(多窗口并行)同时计算多个输出窗口最大化数据复用。三种策略根据UB容量计算动态选择。

**Expert implementation:**
```cpp
// 动态策略选择
if (doubleC > tileLen) {
    mode = MODE_SPLIT_C;
    params.cTileLength = alignC > tileLen ? tileLen : alignC;
} else if (inputTileNum < params.maxWindowWLength) {
    mode = MODE_SPLIT_W;
} else {
    mode = MODE_MULTI_W;
}
```

**vs. baseline:**
```cpp
// baseline单一策略，仅按输出元素分块
uint32_t elems_per_core = (total_output_elems + BLOCK_DIM - 1) / BLOCK_DIM;
```

Benefit: 针对不同输入形状选择最优策略，最大化UB利用率和计算效率，预期性能提升20-50%
Trade-off: Tiling逻辑复杂，需要维护三种Kernel实现

---

## Variant B: 多策略Tiling系统
Source: adaptive_max_pool3d_grad

专家实现的核心性能优化是三策略Tiling系统，根据数据特征动态选择最优策略：1)Normal策略（tiling_key=0/100）: 针对大数据量场景，采用向量化并行计算，在D/H/W维度进行并行切分，使用Vector指令进行批量计算，通过Transpose优化内存访问模式；2)Scatter策略（tiling_key=2）: 针对小数据量场景，直接根据argmax索引scatter梯度，避免生成完整的kernel mask，减少计算冗余；3)Overlap策略（tiling_key=102）: 针对kernel重叠场景，使用atomic add处理多个输出位置映射到同一输入位置的情况，通过workspace保证精度。

**Expert implementation:**
```cpp
class AdaptiveMaxPool3DGradNormalTiling : public AdaptiveMaxPool3DGradTilingBase {
    bool IsCapable() override {
        uint64_t normalTensorSizeMin = CalUBTotalSize(1UL, 1UL, 1UL);
        return normalTensorSizeMin <= maxPoolGradParams.maxUbSize;
    }
};

class AdaptiveMaxPool3DGradScatterTiling : public AdaptiveMaxPool3DGradTilingBase {
    bool IsCapable() override { return true; }
};

if (TILING_KEY_IS(0)) {
    GENERAL_OP_IMPL(AdaptiveMaxPool3DGradNormal, DTYPE_X, DTYPE_GRAD, DTYPE_ARGMAX, DTYPE_Y, false);
} else if (TILING_KEY_IS(2)) {
    GENERAL_OP_IMPL(AdaptiveMaxPool3DGradScatter, DTYPE_X, DTYPE_GRAD, DTYPE_ARGMAX, DTYPE_Y);
}
```

**vs. baseline:**
```cpp
const uint32_t BLOCK_DIM = 16;
uint32_t maxTileC = 512;
uint32_t tileC = maxTileC < channels ? maxTileC : channels;
while (channels % tileC != 0 && tileC > 1) {
    tileC = tileC / 2;
}
uint32_t tasksPerCore = totalTasks / BLOCK_DIM;
```

Benefit: 根据数据特征选择最优策略，大数据量用向量化，小数据量用scatter，重叠场景用atomic add，覆盖全场景性能最优
Trade-off: tiling逻辑复杂，需要维护多个kernel路径

---

## Variant C: 三层次Tiling策略自适应选择
Source: add_rms_norm_dynamic_quant

专家实现实现了三种UB Tiling策略：NORMAL策略(一次性处理多行，rowStep<=16)、SINGLE_ROW策略(逐行处理完整的hidden维度)、SLICE_D策略(列方向切片处理，slice长度8864)。三种策略通过CheckUbNormalTiling()->CheckUbSingleRowTiling()->CheckUbSliceDTiling()的优先级顺序进行判断，确保总是选择最优策略。

**Expert implementation:**
```cpp
bool AddRmsNormDynamicQuantTilingHelper::DoUbTiling() {
    OP_TILING_CHECK(CheckUbNormalTiling(), return true);
    OP_TILING_CHECK(CheckUbSingleRowTiling(), return true);
    OP_TILING_CHECK(CheckUbSliceDTiling(), return true);
    return false;
}
bool CheckUbNormalTiling() {
    int64_t rowStep = ubAvaliable / rowCommons;
    this->firstDimPerLoop_ = (rowStep <= MAX_ROW_STEP) ? rowStep : MAX_ROW_STEP;
}
constexpr uint32_t SLICE_COL_LEN = 8864;
```

**vs. baseline:**
```cpp
const uint32_t MAX_TILE_LEN = 4096;
uint32_t tile_length = MAX_TILE_LEN;
if (hidden_size < static_cast<int32_t>(MAX_TILE_LEN)) {
    tile_length = hidden_size;
}
uint32_t n_tiles = (hidden_size + tile_length - 1) / tile_length;
```

Benefit: 针对不同数据规模自动选择最优策略，最大化UB利用率和计算效率
Trade-off: 增加tiling计算复杂度和代码量

---

## Variant D: 统一Buffer管理与多级内存优化
Source: apply_adagrad_d

专家实现使用ElementwiseSch自动管理UB (Unified Buffer) 的使用，相比baseline的手动队列管理有以下优势：自动内存分配根据DAG分析自动决定buffer分配策略；内存复用通过数据依赖分析复用不再需要的buffer空间；访问模式优化自动对齐和padding以最大化内存带宽利用率；Level-2内存配置MemOptCfg<MemLevel::LEVEL_2>启用更激进的内存优化策略。baseline实现手动管理多个TQue和TBuf，虽然直观但容易出错且难以优化。

**Expert implementation:**
```cpp
// 专家实现 - 内存配置
using MemCfg = MemOptCfg<MemLevel::LEVEL_2>;
using OpDag = DAGSch<Outputs, void, MemCfg>;

// Kernel中使用ElementwiseSch自动管理
ElementwiseSch<schMode, ApplyAdagradDOp::ApplyAdagradDUpdateSlots<DTYPE_VAR>::OpDag> sch(&(tilingData.baseTiling), &pipe);
sch.Init(var, accum, lr, grad, var_out, accum_out);
sch.Process();
```

**vs. baseline:**
```cpp
// baseline手动管理
AscendC::TQue<AscendC::TPosition::VECIN, 1> varQueue;
AscendC::TQue<AscendC::TPosition::VECIN, 1> accumQueue;
AscendC::TQue<AscendC::TPosition::VECIN, 1> gradQueue;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> varOutQueue;
AscendC::TQue<AscendC::TPosition::VECOUT, 1> accumOutQueue;
AscendC::TBuf<AscendC::TPosition::VECCALC> gradPowerBuf;
// ... 多个队列和buffer的手动初始化
pipe.InitBuffer(varQueue, 1, tileSize * sizeof(float));
pipe.InitBuffer(gradPowerBuf, tileSize * sizeof(float));
```

Benefit: 自动最优内存分配、智能内存复用、优化的访问模式，减少内存相关bug
Trade-off: 对内存使用的控制粒度降低，需要理解调度器的内部行为

---

## Variant E: 三级Tiling分层策略
Source: ascend_quant_v2

专家实现采用Block级（多核）、UB级（缓冲区）、Loop级（计算）三级分层Tiling。Block级支持单轴切分和双轴切分两种模式，双轴切分在shape0和shape1都较大时更充分利用核心。UB级根据可用UB大小和cache line对齐要求动态计算最优baseN和baseLen。Loop级通过分离CopyInScaleAndOffset和CopyXAndCompute设计，使scale/offset只需加载一次而被多个x数据块复用，显著减少内存访问。

**Expert implementation:**
```cpp
// 智能核心分配
int64_t cacheLineNum = Ops::Base::CeilDiv(shape1, cacheLine_ / dtypeSize);
uint32_t actCoreNum0 = GetCoreNum(shape0, coreNum_);
uint32_t actCoreNum1 = GetCoreNum(cacheLineNum, coreNum_);

// 双轴切分尝试
if (actCoreNum_ < coreNum_ && shape0 > 1 && shape1 > g_BlockSize) {
    uint32_t actCoreNum2 = GetCoreNumDoubleCut(shape0, cacheLineNum, coreNum_);
    if (actCoreNum2 > actCoreNum_) useDoubleCut = true;
}
```

**vs. baseline:**
```cpp
// 简单固定切分
const uint32_t BLOCK_DIM = 16;
uint32_t tileSize = 2048;
uint32_t innerLoops = elementsPerCore / tileSize;
```

Benefit: 最大化核心利用率，减少内存访问，适应不同输入shape和硬件平台
Trade-off: Tiling逻辑复杂，需要考虑多种切分策略的选择

---

## Variant F: 自动Tiling算法
Source: dynamic_block_quant

专家实现的自动tiling算法是其核心性能优化之一。算法首先计算理论可用核数（根据数据量和block size），然后使用FindUniqueCut函数枚举所有可能的二维切分组合。对于每种切分(m, n)，计算负载均衡指标delta值（表示tail core需要额外处理的数据量），选择delta最小的切分方案。这种设计解决了固定切分策略在数据分布不均匀时的负载不均衡问题。算法还区分normal core和tail core，通过rowNormalCoreNum、rowTailCoreNum等参数精确控制每个核的处理范围，确保所有核的负载尽可能均衡。

**Expert implementation:**
```cpp
static void AutoTiling(DynamicBlockQuantTilingParam& tilingParam) {
    tilingParam.usedCoreNum = std::min(tilingParam.totalCoreNum, 
        tilingParam.rowBlockLoopNum * tilingParam.colBlockLoopNum);
    std::set<int64_t> cutSet = FindUniqueCut(tilingParam.usedCoreNum);
    for (int64_t m : cutSet) {
        int64_t n = tilingParam.usedCoreNum / m;
        int64_t rowNormalBlock = Ops::Base::CeilDiv(tilingParam.rowBlockLoopNum, m);
        int64_t delta = rowNormalBlock * colNormalBlock;
        // 选择最优切分...
    }
}
```

**vs. baseline:**
```cpp
const uint32_t BLOCK_DIM = 16;
uint32_t rowsPerCore = (M + BLOCK_DIM - 1) / BLOCK_DIM;
context->SetBlockDim(BLOCK_DIM);
```

Benefit: 最大化多核利用率，负载均衡，适应不同shape输入
Trade-off: tiling计算开销，增加Host端复杂度

---

## Variant G: 动态Tiling与UB感知分块
Source: embedding_dense_grad_v2

专家实现的Tiling不是固定值，而是根据平台UB大小和数据类型动态计算。CalMaxFormerNum函数计算UB能容纳的最大行数，确保数据尽量驻留在UB中，减少GM访问。计算公式：availableUb = ubSize - RESERVED_UB_SIZE - idxAlignNum * sizeof(int) * USE_IDX_NUM_IN_UB；maxFormerNum = (availableUb / (gradAlignNum * sizeof(float) * useGradNum)) * gradAlignNum。其中RESERVED_UB_SIZE (20KB)预留栈空间，USE_IDX_NUM_IN_UB (3)是索引队列数量，useGradNum根据数据类型变化（FP32=3, FP16=4）。

**Expert implementation:**
```cpp
inline void CalMaxFormerNum(uint64_t ubSizeLeft)
{
    uint64_t idxAlignNum = BLOCK_SIZE / sizeof(int);
    uint64_t gradAlignNum = BLOCK_SIZE_GRAD / sizeof(float);
    ubSizeLeft -= RESERVED_UB_SIZE + idxAlignNum * sizeof(int) * USE_IDX_NUM_IN_UB;
    uint64_t availableUbForGrad = ubSizeLeft > 0UL ? ubSizeLeft : 0UL;
    uint64_t useGradNum = dataTypeSize_ == sizeof(float) ? USE_GRAD_NUM_IN_UB : USE_GRAD_NUM_IN_UB_WITH_CACHE;
    maxFormerNum = (availableUbForGrad / (gradAlignNum * sizeof(float) * useGradNum)) * gradAlignNum;
}

// 使用动态计算的maxFormerNum
formerEmbeddingDim_ = embeddingDim_ <= maxFormerNum ? embeddingDim_ : maxFormerNum;
```

**vs. baseline:**
```cpp
// Fixed tile size
uint32_t tileSize = 32;
```

Benefit: 最大化UB利用率，减少GM访问次数20-40%
Trade-off: Tiling计算稍微增加了Host端开销，但可忽略

---

## Variant H: 多核细粒度负载均衡
Source: foreach_abs

专家实现的Tiling策略更加复杂和精细。通过ForeachCommonTiling类实现基于硬件平台信息的自适应Tiling。关键Tiling参数包括tensorDataCountList（每个Tensor的数据量）、tensorStartList/tensorEndList（每个核处理的起止Tensor索引）、tensorStartOffsetList/tensorEndOffsetList（每个核处理的起止偏移）。这种策略实现了细粒度的多核负载均衡。

**Expert implementation:**
```cpp
BEGIN_TILING_DATA_DEF(ForeachCommonTilingData)
TILING_DATA_FIELD_DEF(uint64_t, inputsTensorUbSize);
TILING_DATA_FIELD_DEF_ARR(int64_t, MAX_TENSOR_CONT, tensorDataCountList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorStartList);
TILING_DATA_FIELD_DEF_ARR(uint16_t, MAX_CORE_CONT, tensorEndList);
END_TILING_DATA_DEF;
```

**vs. baseline:**
```cpp
BEGIN_TILING_DATA_DEF(ForeachAbsCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalElements);
  TILING_DATA_FIELD_DEF(uint32_t, tileSize);
  TILING_DATA_FIELD_DEF(uint32_t, innerLoops);
END_TILING_DATA_DEF;
```

Benefit: 细粒度负载均衡，多Tensor并行处理，提升大规模数据处理效率
Trade-off: Tiling逻辑复杂，调试难度增加

---

## Variant I: 分块计算与Tiling策略
Source: foreach_add_scalar

专家实现采用了精细的分块（Tiling）策略，将大规模tensor计算分解为适合UB大小的小块处理。通过inputsTensorUbSize参数控制每块处理的数据量，自动计算copyTimes和余数处理。这种策略允许算子处理任意大小的tensor，而不受UB容量限制。更重要的是，专家实现支持多tensor场景（TensorList），通过tensorStart、tensorEnd、tensorStartOffset、tensorEndOffset等参数实现跨tensor的负载均衡。baseline实现的Tiling策略相对简单，仅支持单tensor场景，且tileSize为固定的2048，缺乏灵活性。

**Expert implementation:**
```cpp
uint32_t copyTimes = dataCount / Base::maxDataCount;
uint32_t copyTimesRemainder = dataCount % Base::maxDataCount;
for (uint32_t i = 0; i < copyTimes; i++) {
    bool isRemainder = (i == copyTimes - 1 && copyTimesRemainder > 0);
    uint32_t tempDataCount = isRemainder ? copyTimesRemainder : Base::maxDataCount;
    CopyIn(i, tempDataCount, isRemainder);
    Compute(i, tempDataCount, float32Tensor, isRemainder);
    CopyOut(i, tempDataCount, isRemainder);
}
```

**vs. baseline:**
```cpp
const uint32_t TILE_SIZE = 2048;  // 固定值
uint32_t innerLoops = elementsPerCore / tileSize;
for (uint32_t i = 0; i < this->innerLoops; i++) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
}
```

Benefit: 支持任意大小tensor、多tensor负载均衡、充分利用UB容量
Trade-off: Tiling逻辑复杂度增加，需要额外的边界处理

---

## Variant J: Regbase架构优化（针对Ascend910_95）
Source: foreach_add_scalar

专家实现针对Ascend910_95平台提供了专门的regbase架构实现，通过foreach_add_scalar_regbase.h实现了更高效的寄存器级优化。ForeachAddScalarRegbase类继承自ForeachRegbaseUnary，利用硬件特性进行更精细的内存访问控制。该架构使用DataCopyPadExtParams进行灵活的内存拷贝，支持padding和边界处理。此外，regbase架构支持通过TPipe进行更细粒度的流水线控制，包括显式的pipeOp.Destroy()调用来释放资源。这种架构优化专门针对新硬件平台的特性，可以显著提升内存访问效率和计算吞吐量。

**Expert implementation:**
```cpp
template <typename T, typename ScalarT, typename Tiling>
class ForeachAddScalarRegbase : public ForeachRegbaseUnary<T, Tiling, ForeachAddScalarRegbase<T, ScalarT, Tiling>> {
public:
    __aicore__ inline void CopyIn(int64_t index, int64_t dataCount) {
        DataCopyPadExtParams<T> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = dataCount * sizeof(T);
        DataCopyPad(dataLocal, inTensorGM[index * maxDataCount], copyInParams, dataCopyPadExtParams);
    }
};
```

Benefit: 充分利用新硬件特性，提升内存访问效率和计算吞吐量
Trade-off: 代码维护复杂度增加，需要针对不同硬件维护多份实现

---

## Variant K: 分块处理与边界优化
Source: foreach_addcdiv_list

专家实现通过maxDataCount和maxCastDataCount参数实现了精细的分块处理。在SingleTensorProcess函数中，数据被划分为多个批次处理，每个批次处理maxDataCount个元素。对于剩余元素（copyTimesRemainder），单独标记为isRemainder进行特殊处理。针对剩余数据，使用DataCopyPad代替DataCopy，后者可以处理非对齐的内存访问，避免访问越界。baseline实现中虽然也有分块处理（tile_size），但对边界情况的处理比较简单，没有使用专门的Pad指令。

**Expert implementation:**
```cpp
// 专家实现: DataCopyPad边界处理
if (isRemainder) {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad(inLocal_2, inTensorsGM_2[index * Base::maxDataCount], copyParams, padParams);
} else {
    DataCopy(inLocal_2, inTensorsGM_2[index * Base::maxDataCount], dataCount);
}
```

**vs. baseline:**
```cpp
// baseline: 简单边界处理
uint32_t current_tile_size = (tile_offset + this->tile_size <= this->elements_per_core) ? 
                              this->tile_size : (this->elements_per_core - tile_offset);
AscendC::DataCopy(inputLocal, inputGm[tile_offset], current_tile_size);
```

Benefit: 安全的边界处理，避免内存越界，支持任意大小的输入
Trade-off: 增加了代码复杂度

---

## Variant L: 多级分块与并行策略
Source: grouped_dynamic_mx_quant

专家实现采用了多级的数据分块策略，在Host端Tiling阶段就确定了各级并行粒度。核级并行通过totalCoreNum和usedCoreNum实现，按[group, cacheline]维度划分任务。UB级分块通过maxUbCol确定单次UB计算能处理的最大行数，ubFactor确定每行的处理粒度。tailBlockFactor和tailUbFactor专门处理不能被整除的边界情况，确保负载均衡。

**Expert implementation:**
```cpp
int64_t spliteCoreData = tilingParam.uo * tilingParam.groupSize;
int64_t coreData = CeilDiv(spliteCoreData, tilingParam.totalCoreNum);
tilingParam.usedCoreNum = CeilDiv(spliteCoreData, coreData);
tilingParam.blockFactor = CeilDiv(spliteCoreData, tilingParam.usedCoreNum);
tilingParam.tailBlockFactor = spliteCoreData - (tilingParam.usedCoreNum - 1) * tilingParam.blockFactor;

bool isTailBlock = blockIdx_ == (usedCoreNum_ - 1);
int64_t loopNum = isTailBlock ? this->tailBlockFactor_ : this->blockFactor_;
blockLoopOffset_ = this->blockIdx_ * this->blockFactor_;
```

**vs. baseline:**
```cpp
int64_t spliteCoreData = tilingParam.uo * tilingParam.groupSize;
int64_t coreData = CeilDiv(spliteCoreData, tilingParam.totalCoreNum);
tilingParam.usedCoreNum = CeilDiv(spliteCoreData, coreData);
```

Benefit: 多级并行充分利用硬件资源；尾块处理保证负载均衡；可扩展性强
Trade-off: Tiling逻辑复杂度增加；需要精确计算各级分块参数

---

## Variant M: 自适应Tiling策略
Source: inplace_add_rms_norm

专家实现采用了五种自适应Tiling模式来应对不同的数据规模：NORMAL（常规模式）、SPLIT_D（列切分模式）、MERGE_N（行合并模式）、SINGLE_N（单行模式）、MULTI_N（多行并行模式）。这种自适应策略的核心思想是根据输入数据的shape（numRow和numCol）以及硬件资源（UB大小、核数）动态选择最优计算模式。例如，当numCol > ubFactor时选择SPLIT_D模式，将列方向的数据切分到多次计算中；当blockFactor==1时选择SINGLE_N模式以减少循环开销。

**Expert implementation:**
```cpp
// 专家实现 - 自适应模式选择
constexpr uint32_t MODE_NORMAL = 0;
constexpr uint32_t MODE_SPLIT_D = 1;
constexpr uint32_t MODE_MERGE_N = 2;
constexpr uint32_t MODE_SINGLE_N = 3;
constexpr uint32_t MODE_MULTI_N = 4;

if (numCol > ubFactor) {
    modeKey = MODE_SPLIT_D;
} else if (blockFactor == 1) {
    modeKey = MODE_SINGLE_N;
} else if (numColAlign <= SMALL_REDUCE_NUM) {
    modeKey = MODE_MERGE_N;
} else if (isPerformance == 1) {
    modeKey = MODE_MULTI_N;
}
```

**vs. baseline:**
```cpp
// baseline - 简单行分块
uint32_t rowsPerCore = (nRows + BLOCK_DIM - 1) / BLOCK_DIM;
tiling.set_rowsPerCore(rowsPerCore);
tiling.set_cols(nCols);
```

Benefit: 在各种输入规模下都能实现接近峰值的性能
Trade-off: Tiling逻辑复杂，需要精细的参数调优

---

## Variant N: 输入维度自适应分块（Adaptive Tiling）
Source: layer_norm_v3

专家实现的tiling系统根据输入维度自动选择最优分块策略。LayerNormV3TilingDataWelford针对Welford算法计算tileLength、welfordUpdateTimes等参数；LayerNormV3TilingDataRegBaseTwoPass针对Two-Pass计算aFactor、aBlockFactor等；LayerNormV3TilingDataRegBaseTwoPassPerf针对高性能版本优化aUbFactorAlignB32对齐。通过CanFitInBuffer函数检查UB容量约束，自动调整分块大小以最大化UB利用率。

**Expert implementation:**
```cpp
// 自适应分块
BEGIN_TILING_DATA_DEF(LayerNormV3TilingDataWelford)
  TILING_DATA_FIELD_DEF(int64_t, M);
  TILING_DATA_FIELD_DEF(int64_t, N);
  TILING_DATA_FIELD_DEF(int64_t, tileLength);
  TILING_DATA_FIELD_DEF(int64_t, welfordUpdateTimes);
  TILING_DATA_FIELD_DEF(int64_t, welfordTempSize);
END_TILING_DATA_DEF;

bool CanFitInBuffer(int64_t curA, int64_t largeBufferMemPerA, int64_t baseMemSize, int64_t& tmpBufferUse);
```

**vs. baseline:**
```cpp
// 简单固定分块
const uint32_t BLOCK_DIM = 8;
uint32_t rowsPerCore = (totalRows + BLOCK_DIM - 1) / BLOCK_DIM;
tiling.set_tileLength(normalizedDim);
tiling.set_rowsPerCore(rowsPerCore);
```

Benefit: 根据输入尺寸和硬件自动优化分块，UB利用率提升20-40%
Trade-off: tiling逻辑复杂，需要针对不同策略分别实现

---

## Variant O: 自适应多策略Tiling系统
Source: layer_norm_v4

专家实现的核心创新是自适应tiling策略选择。Host端根据输入shape、数据类型、硬件资源动态选择最优kernel策略。通过IsCapable()方法判断适用性，例如SingleRead要求rowAlign≤rowMax，Transpose要求rowSize≤64。baseline使用固定32核并行，不考虑输入大小或硬件限制，导致资源浪费或内存溢出。

**Expert implementation:**
```cpp
bool LayerNormV4SingleReadTiling::IsCapable() {
    uint32_t rowMax = ((commonParams.ubSizePlatForm - NUM_EIGHT * BLOCK_SIZE) / FLOAT_SIZE) / NUM_TWO;
    if (commonParams.rowAlign > rowMax) return false;
}
bool LayerNormV4TransposeTiling::IsCapable() {
    if (commonParams.rowSize > TRANSPOSE_ROW_LIMIT) return false;
}
REGISTER_TILING_TEMPLATE("LayerNormV4", LayerNormV4SingleReadTiling, 1000);
```

**vs. baseline:**
```cpp
const uint32_t BLOCK_DIM = 32;
context->SetBlockDim(BLOCK_DIM);
uint32_t rowsPerCore = (nRows + BLOCK_DIM - 1) / BLOCK_DIM;
```

Benefit: 不同场景下都能达到接近最优性能，小输入避免核浪费，大输入避免内存溢出
Trade-off: tiling逻辑复杂，需要维护多个策略类和优先级系统

---

## Variant P: Smart Tiling 与多维分块
Source: max_pool_with_argmax_v3

专家实现tiling算法在N、H、W、C四个维度尝试分块，优先顺序N→H→W→C。当kernel尺寸过大时支持kernel分块。使用二分查找确定最优分块大小，满足UB容量和核数利用率约束。baseline仅使用固定tileC和tasksPerCore。

**Expert implementation:**
```cpp
// 专家实现 - 智能Tiling搜索
void SearchBestTiling() {
    if (TrySplitN()) return;
    if (TrySplitH()) return;
    if (TrySplitW()) return;
    SplitC();
    if (!IsMeetUBSize()) {
        splitData_.isSplitKernel = 1;
        SplitKernel();
    }
}

void BinarySearch(int64_t start, int64_t end, int64_t* value, int64_t rate) {
    int64_t left = start, right = end, bestSplit = 1;
    while (left <= right) {
        int64_t mid = left + (right - left) / DOUBLE;
        *value = mid * rate;
        if (IsMeetUBSize() && IsMeetTargetCoreNum()) {
            bestSplit = mid;
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }
}
```

**vs. baseline:**
```cpp
// baseline - 简单固定分块
uint32_t tileC = (MAX_TILE_C < channels) ? MAX_TILE_C : channels;
uint32_t totalTasks = batchSize * outH * outW;
uint32_t tasksPerCore = totalTasks / BLOCK_DIM;
```

Benefit: 在各种输入规模下都能获得接近理论峰值的性能
Trade-off: Host端tiling计算开销增加，需要精确的平台信息

---

## Variant Q: 智能Tiling策略选择
Source: modulate

专家实现采用三维度（B/L/D）自适应Tiling策略。策略选择逻辑基于：TilingB策略当Batch维度足够大时按B维度分块；TilingL策略当Batch较小但Batch*L较大时按L维度分块；TilingD策略当Batch*L很小但D很大时按D维度分块。通过MIN_TILE_SIZE判断确保每个分块足够大以充分利用硬件。相比单一维度分块，智能策略选择可使并行效率提升30-60%。

**Expert implementation:**
```cpp
// 专家实现: 智能策略选择
TilingStrategy SelectStrategy() {
    if (this->tilingData.inputB * this->tilingData.inputL < this->coreNum) {
        int64_t tileD = this->tilingData.inputD / this->coreNum;
        int64_t tileDSize = this->tilingData.inputB * this->tilingData.inputL * tileD * this->dataTypeSize;
        return tileDSize < MIN_TILE_SIZE ? TilingStrategy::TilingB : TilingStrategy::TilingD;
    }
    if (this->tilingData.inputB >= this->coreNum) {
        return TilingStrategy::TilingB;
    }
    if (this->tilingData.inputB >= this->coreNum / HALF_CORE_NUM) {
        int64_t tileL = this->tilingData.inputB * this->tilingData.inputL / this->coreNum;
        int64_t tileLSize = this->tilingData.inputD * tileL * this->dataTypeSize;
        return tileLSize < MIN_TILE_SIZE ? TilingStrategy::TilingB : TilingStrategy::TilingL;
    }
    return TilingStrategy::TilingL;
}
```

**vs. baseline:**
```cpp
// baseline: 简单的B维度分块
int64_t totalRows = inputB;
int64_t rowsPerCore = totalRows / GetBlockNum();
int64_t remainRows = totalRows % GetBlockNum();
this->startB = coreIdx * rowsPerCore + (coreIdx < remainRows ? coreIdx : remainRows);
this->endB = this->startB + rowsPerCore + (coreIdx < remainRows ? 1 : 0);
```

Benefit: 并行效率提升30-60%，适应不同shape分布
Trade-off: Host端策略选择需要额外计算开销，但相对于Kernel执行时间可忽略

---

## Variant R: 基于UB容量的智能Tiling计算
Source: multi_scale_deformable_attention_grad

专家实现不是简单地使用固定核心数，而是根据实际UB容量动态计算max_ub_num。通过公式(ub_size / dtype_size - NUM_LEVEL_BUFFER * num_levels_align - NUM_EMEBDDIM_BUFFER * channels) / (NUM_QUERIE_BUFFER + NUM_CHANNEL_BUFFER * channels)精确计算每个循环能处理的query数量。这种方法确保了UB内存的高效利用，避免了内存浪费或溢出。常量NUM_LEVEL_BUFFER=3、NUM_EMEBDDIM_BUFFER=8、NUM_QUERIE_BUFFER=15、NUM_CHANNEL_BUFFER=13反映了专家对各种中间buffer大小的精确估算。

**Expert implementation:**
```cpp
uint64_t num_levels_align = (num_levels + data_align - 1) / data_align * data_align;
max_ub_num = (ub_size / dtype_size - NUM_LEVEL_BUFFER * num_levels_align - NUM_EMEBDDIM_BUFFER * channels) /
              (NUM_QUERIE_BUFFER + NUM_CHANNEL_BUFFER * channels);
max_ub_num = max_ub_num / data_align * data_align;
uint64_t taskNum = ((num_query + max_ub_num - 1) / max_ub_num) * batch_size * num_heads * num_levels * num_point;
core_used = std::min(core_num, taskNum);
```

**vs. baseline:**
```cpp
// 简单固定策略
const uint32_t nCores = 32;
uint32_t bhPerCore = (totalBH + nCores - 1) / nCores;
```

Benefit: 最大化UB利用率，减少内存访问次数，预期性能提升2-3倍
Trade-off: Tiling计算复杂，需要精确估算各buffer大小

---

## Variant S: 多策略自适应Tiling框架
Source: norm_common

专家实现设计了5种不同的Tiling策略（SingleRead/Transpose/RegBaseTwoPass/Welford/RegBaseTwoPassPerf），根据输入shape特征、数据类型、硬件平台自动选择最优策略。SingleRead策略适用于小rowSize场景，通过一次读取多行数据提升数据复用；Transpose策略适用于大rowSize场景，通过转置内存布局优化访问模式；RegBase策略针对高维数据优化；Welford策略采用Welford算法提升数值稳定性。baseline实现采用单一的简单tiling策略，固定tile_length=4096，未考虑不同shape下的最优策略选择。

**Expert implementation:**
```cpp
// 专家实现：多策略Tiling框架
class LayerNormV4TilingBase : public Ops::NN::Optiling::TilingBaseClass {
public:
    virtual bool IsCapable() = 0;
    virtual uint64_t GetTilingKey() const = 0;
    virtual ge::graphStatus DoOpTiling() = 0;
};

class LayerNormV4SingleReadTiling : public LayerNormV4TilingBase {
    bool IsCapable() override;
    uint64_t GetTilingKey() const override { return 100 + dtypeKey_; }
};

class LayerNormV4TransposeTiling : public LayerNormV4TilingBase {
    bool IsCapable() override;
    uint64_t GetTilingKey() const override { return 200 + dtypeKey_; }
};

static ge::graphStatus Tiling4LayerNormV4(gert::TilingContext* context)
{
    return Ops::NN::Optiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}
```

**vs. baseline:**
```cpp
// baseline：单一简单Tiling策略
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    uint32_t n_cores = 32;
    context->SetBlockDim(n_cores);
    uint32_t max_tile_len = 4096;
    uint32_t tile_length = (max_tile_len < norm_size) ? max_tile_len : norm_size;
    uint32_t n_tiles = (norm_size + tile_length - 1) / tile_length;
}
```

Benefit: 针对不同shape和数据类型自动选择最优策略，在各种场景下都能获得接近最优的性能
Trade-off: 需要维护多种Tiling策略实现，增加开发和维护复杂度

---

## Variant T: 平台自适应与CompileInfo预解析
Source: norm_common

专家实现通过TilingParseContext在编译期预解析平台信息（Core数量、UB大小、SOC版本等），存储到LayerNormV4CompileInfo结构中。这些信息在Tiling阶段重复使用，避免重复查询平台API。同时，根据不同平台（910B/310P/910_95等）选择不同的配置策略，如310P使用不同的数据类型限制。baseline实现硬编码核数（32）和tile长度（4096），未进行平台自适应。

**Expert implementation:**
```cpp
// 专家实现：平台信息预解析
ge::graphStatus TilingPrepare4CompileInfo(gert::TilingParseContext* context, LayerNormV4CompileInfo* compileInfo)
{
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    compileInfo->coreNum = ascendcPlatform.GetCoreNumAiv();
    compileInfo->isAscend310P = ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND310P;
    compileInfo->isRegBase = (ascendcPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_95);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    compileInfo->ubSizePlatForm = ubSizePlatForm;
}

// Tiling阶段使用预解析的信息
ge::graphStatus LayerNormV4TilingBase::GetPlatformInfo()
{
    const LayerNormV4CompileInfo* compileInfo = context_->GetCompileInfo<LayerNormV4CompileInfo>();
    commonParams.coreNum = compileInfo->coreNum;
    commonParams.ubSizePlatForm = compileInfo->ubSizePlatForm;
}
```

**vs. baseline:**
```cpp
// baseline：硬编码参数
uint32_t n_cores = 32;
context->SetBlockDim(n_cores);
uint32_t max_tile_len = 4096;
```

Benefit: 自适应不同硬件配置，在各种Ascend平台上都能获得最优性能，避免硬编码带来的平台依赖
Trade-off: 需要处理多平台的差异性，增加代码复杂度

---

## Variant U: 自适应Tiling策略(Split-N vs Split-D)
Source: rms_norm_grad

专家实现根据输入shape自动选择最优的tiling策略。Split-N策略：当col <= buffer_size时使用，按行切分数据，每核处理完整的列。Split-D策略：当col > buffer_size时使用，同时按行和列切分，适应大宽度输入。这种自适应策略确保小宽度输入充分利用UB空间，向量化计算；大宽度输入避免UB溢出，合理分块。baseline实现仅支持简单的按行切分，无法有效处理大宽度输入。

**Expert implementation:**
```cpp
if (col_val <= buffer_size) {
    tiling_key = ub_key;  // Split-N策略
    LargeNSmallD(context, tiling, buffer_size, row_val, col_val_align, core_num);
} else {
    ub_key = UB_SPLIT_D;
    tiling_key = ub_key;  // Split-D策略
    LargeNLargeD(context, tiling, buffer_size, row_val, col_val, core_num);
}
```

**vs. baseline:**
```cpp
uint32_t rowsPerCore = rows / BLOCK_DIM;
tiling.set_rowsPerCore(rowsPerCore);
```

Benefit: 适应各种输入shape；最大化UB利用率；避免大宽度输入的内存溢出
Trade-off: tiling逻辑复杂度增加；需要根据输入动态选择策略

---

## Variant V: 基于UB大小的动态分块策略
Source: scaled_masked_softmax_v2

专家实现的Tiling策略完全基于实际可用的UB（Unified Buffer）大小进行动态计算。首先通过PlatformAscendC::GetCoreMemSize获取当前平台UB总大小，然后考虑Softmax高阶API需要的临时缓冲区（32K或64K），计算出可用于数据缓冲的空间。接着基于每行数据大小（包含输入x、输出y、mask的padding后大小）计算每次迭代能处理的行数。这种动态分块策略确保了对不同输入shape和不同硬件平台都能达到最优的UB利用率。

**Expert implementation:**
```cpp
// 专家实现动态分块
ascendCPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, totalUbSize);
int64_t availableUbSize = totalUbSize;

uint64_t padedWidth = tiling.get_padLineNum();
uint64_t maskPaddedWidth = tiling.get_alignedMaskWidth();
uint64_t maxByteLine = padedWidth * XY_PARAMS * xDtypeSize + padedWidth * FP32_SIZE + maskPaddedWidth * BOOL_SIZE;

uint64_t softmaxBuffSize = (ascendCPlatform.GetSocVersion() == platform_ascendc::SocVersion::ASCEND910_95) ? SOFTMAX_BUF_SIZE_D : SOFTMAX_BUF_SIZE;
uint64_t availableLinePerIter = (availableUbSize - softmaxBuffSize) / maxByteLine;
```

**vs. baseline:**
```cpp
// baseline 固定分块
uint32_t blockDim = 32;
context->SetBlockDim(blockDim);
uint32_t rowsPerCore = totalRows / blockDim;
uint32_t tileLength = seqLen;
```

Benefit: UB利用率接近100%，避免溢出或利用率不足，性能提升10-20%
Trade-off: Tiling计算复杂度增加，Host端开销略增

---

## Variant W: Conv2D M-Mode/HW-Mode自适应空间分块
Source: conv2d_v2

Conv2D根据输出空间分辨率自适应选择两种分块模式：M-Mode将Ho*Wo合并为单一M维度分块（适合小空间分辨率，减少分块维度），HW-Mode将Ho和Wo独立分块（适合大空间分辨率，更好利用空间局部性）。选择在编译期通过constexpr模板参数OutputOrder决定，每种模式有独立的地址计算和SetSingleOutputShape路径。尾块通过isNDimTail/isMDimTail/isHoDimTail/isWoDimTail/isBatchDimTail多维度独立标记。

**Expert implementation:**
```cpp
constexpr static bool isMMode = CONV_CFG::outputOrder == static_cast<int8_t>(ConvOutputOrder::M_MODE);

if constexpr (isMMode) {
    // M-Mode: Ho*Wo合并为M维度
    uint64_t totalM = conv2dRunInfo->hout * conv2dRunInfo->wout;
    DimDataToFill mToFill(singleCoreM, mIdxStart, isMDimTail);
    convCommon.CalcDimData(blockPerMDim, conv2dRunInfo->mDim,
        convCommon.AlignB(totalM, M0), totalM, mToFill);
    conv.SetFmapStartPosition(mIdxStart);
} else {
    // HW-Mode: Ho和Wo独立分块
    DimDataToFill hoToFill(singleCoreHo, hoIdxStart, isHoDimTail);
    convCommon.CalcDimData(blockPerHoDim, conv2dRunInfo->hoDim,
        conv2dRunInfo->hout, conv2dRunInfo->hout, hoToFill);
    DimDataToFill woToFill(singleCoreWo, woIdxStart, isWoDimTail);
    convCommon.CalcDimData(blockPerWoDim, conv2dRunInfo->woDim,
        conv2dRunInfo->wout, conv2dRunInfo->wout, woToFill);
    conv.SetFmapStartPosition(singleCoreHiStartPos, singleCoreWiStartPos);
}
```

**vs. baseline:**
```cpp
// 单一线性分块，不区分空间维度
uint32_t totalElements = batch * cout * hout * wout;
uint32_t elementsPerCore = (totalElements + numCores - 1) / numCores;
```

Benefit: 编译期选择空间分块模式，零运行时开销；多维尾块独立处理确保精确的边界计算
Trade-off: 两种模式需要独立的地址计算和输出配置；模板参数增加编译时间

---

## Variant X: Flash Attention S2维度自适应分块
Source: flash_attention_score

Flash Attention在S2(Key/Value序列长度)维度进行分块迭代，分块大小根据UB容量、Softmax对齐要求(M对齐8、K对齐64)和IBShare配置动态计算。每个S2块内完成BMM1+Softmax+BMM2全流程。对于不同的稀疏掩码模式(Causal/Band/Prefix)，有效S2范围不同——Causal模式下只有对角线以下的块需要计算，通过预计算偏移跳过无效块。

**Expert implementation:**
```cpp
constexpr static int32_t SOFTMAX_M_ALIGNED_SIZE = 8;
constexpr static int32_t SOFTMAX_K_ALIGNED_SIZE = 64;

__aicore__ inline bool IsBasicBlockInSoftMax(int32_t srcM, int32_t srcK) {
    return srcM % SOFTMAX_M_ALIGNED_SIZE == 0 && srcK % SOFTMAX_K_ALIGNED_SIZE == 0;
}

// S2-block loop: skip invalid blocks based on sparse mode
for (int s2Block = 0; s2Block < numS2Blocks; s2Block++) {
    // Compute mask offset based on sparse mode
    int64_t offset = ComputeOffsetForCausal(delta, s1BaseSize, s2BaseSize, attenMaskS2Size);
    if (isInvalidBlock(offset)) continue;  // Skip fully masked block
    // BMM1 -> ApplyMask -> OnlineSoftmax -> BMM2
}
```

**vs. baseline:**
```cpp
// 固定S2分块，不跳过无效块
uint32_t s2BlockSize = 128;
for (int s2 = 0; s2 < seqLen; s2 += s2BlockSize) {
    compute_attention_block(s2, s2BlockSize);  // 包括全mask的无效块
}
```

Benefit: 对齐约束确保Softmax最优性能；稀疏模式下跳过无效块减少计算量
Trade-off: 对齐约束可能导致padding增加；稀疏判断增加控制流开销

---

## Variant Y: Grouped MatMul窗口化M维度自适应分块
Source: grouped_matmul

ASWT调度器使用窗口化(Window-based)的M维度分块策略，窗口大小为4。M维度被划分为多个窗口，每个窗口内的tile按顺序映射。主窗口(mainWindow)处理大部分M范围，尾窗口(tailWindow)处理剩余部分。偶数行的N维度扫描方向反转(nTileIdx = nTileNum - 1 - nTileIdx)，这种Z字形扫描模式优化了L3缓存的数据复用。

**Expert implementation:**
```cpp
const int64_t WINDOW_LEN = 4;
mainWindow_ = WINDOW_LEN < mTileNum_ ? WINDOW_LEN : mTileNum_;
mainRow_ = mTileNum_ / mainWindow_ - 1;
tailWindow_ = mTileNum_ - mainWindow_ * mainRow_;

__aicore__ inline BlockShape GetTileIdx(int64_t curBlock, int64_t count) {
    uint64_t index = curBlock - count;
    uint64_t rowIdx = index / nTileNum_ / mainWindow_;
    if (rowIdx < mainRow_) {
        mTileIdx = rowIdx * mainWindow_ + index % mainWindow_;
        nTileIdx = (index / mainWindow_) % nTileNum_;
    } else {
        uint64_t tailIndex = index - mainRow_ * mainWindow_ * nTileNum_;
        mTileIdx = mainRow_ * mainWindow_ + tailIndex % tailWindow_;
        nTileIdx = (tailIndex / tailWindow_) % nTileNum_;
    }
    // Z-scan: reverse N direction for even rows
    if (rowIdx % 2 != 0) {
        nTileIdx = nTileNum_ - 1 - nTileIdx;
    }
    return {mTileIdx, nTileIdx, k_, b_};
}
```

**vs. baseline:**
```cpp
// 简单行优先映射
int mTileIdx = tileIdx / nTileNum;
int nTileIdx = tileIdx % nTileNum;
```

Benefit: 窗口化+Z字形扫描优化L3缓存复用率；主/尾窗口分离处理避免边界复杂度
Trade-off: 索引映射逻辑比简单行优先复杂；窗口大小为固定值4可能不适合所有场景
