# P6: Multi-Algorithm Adaptive Selection (多算法自适应选择)
## Overview
Welford算法是BatchNorm训练场景的核心算法，相比传统的两遍计算（先算均值再算方差），Welford算法单次遍历即可同时得到均值和方差，且数值稳定性更好。专家实现针对昇腾硬件特点进行了深度优化：1）并行Welford更新：WelfordParallelUpdate函数使用向量化指令同时更新多个通道的均值和M2（方差累计量）；2）分阶段归约：对于大规模数据，先在各分块上进行Welford更新，再通过WelfordParallelFinalize系列函数进行跨分块归约，支持R0/R1两个维度的灵活切分；3）二分累加优化：FullAichotomizeAdd实现了非二次幂长度的二分累加，先处理差值部分再进行标准二分归约，充分利用Vector Unit的并行能力。

## When to Use
- Normalization, reduction ops
- 大数据量场景下性能提升显著，减少同步开销
- 单次遍历减少50%的内存访问；数值稳定性避免大数吃小数问题；二分归约充分利用Vector并行度
- 根据数据规模选择最优算法，最大化数据复用

## Trade-off
- 代码复杂度增加，需要处理非64对齐的边界情况
- 算法理解难度较高；Welford更新需要更多中间变量
- 需要维护两套实现代码

**Source operators**: add_rms_norm_dynamic_quant, batch_norm_v3, dequant_bias, dynamic_block_quant, dynamic_quant_update_scatter_v2, gather_elements_v2, gemma_rms_norm, layer_norm_v3, multi_scale_deformable_attention_grad, rms_norm_quant, scatter_elements_v2

---

## Variant A: 高效的Reduce操作实现
Source: add_rms_norm_dynamic_quant

专家实现提供了高度优化的ReduceSumHalfInterval和ReduceMaxInplace函数。这些函数采用分治策略，首先使用Vector的Max/Add指令在UB内部进行归约(每次处理64个元素)，最后使用WholeReduceMax/WholeReduceSum指令完成最终的归约。相比通用ReduceSum，专家实现的自定义归约充分利用了数据局部性，减少了同步开销。

**Expert implementation:**
```cpp
__aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& srcLocal, int32_t count) {
    uint64_t repsFp32 = count >> 6;       // count / 64
    uint64_t remsFp32 = count & 0x3f;     // count % 64
    if (likely(repsFp32 > 1)) {
        Max(srcLocal, srcLocal[ELEM_PER_REP_FP32], srcLocal, ELEM_PER_REP_FP32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0)) {
        Max(srcLocal, srcLocal[offsetsFp32], srcLocal, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    WholeReduceMax(srcLocal, srcLocal, mask, 1, 8, 1, 8);
    PipeBarrier<PIPE_V>();
}
```

**vs. baseline:**
```cpp
AscendC::ReduceSum(tempLocal, tempLocal, sharedLocal, tileLength);
float tileSum = tempLocal.GetValue(0);
```

Benefit: 大数据量场景下性能提升显著，减少同步开销
Trade-off: 代码复杂度增加，需要处理非64对齐的边界情况

---

## Variant B: Welford在线算法优化
Source: batch_norm_v3

Welford算法是BatchNorm训练场景的核心算法，相比传统的两遍计算（先算均值再算方差），Welford算法单次遍历即可同时得到均值和方差，且数值稳定性更好。专家实现针对昇腾硬件特点进行了深度优化：1）并行Welford更新：WelfordParallelUpdate函数使用向量化指令同时更新多个通道的均值和M2（方差累计量）；2）分阶段归约：对于大规模数据，先在各分块上进行Welford更新，再通过WelfordParallelFinalize系列函数进行跨分块归约，支持R0/R1两个维度的灵活切分；3）二分累加优化：FullAichotomizeAdd实现了非二次幂长度的二分累加，先处理差值部分再进行标准二分归约，充分利用Vector Unit的并行能力。

**Expert implementation:**
```cpp
// Welford在线算法，一趟完成
__aicore__ inline void WelfordParallelUpdate(
    float& count, LocalTensor<float>& meanTensor, LocalTensor<float>& m2Tensor, 
    LocalTensor<float>& xTensor, LocalTensor<float>& deltaTensor, const uint32_t& calcMask)
{
    count += 1;
    Sub(deltaTensor, xTensor, meanTensor, calcMask);
    PipeBarrier<PIPE_V>();
    Muls(xTensor, deltaTensor, 1 / count, calcMask);
    PipeBarrier<PIPE_V>();
    Add(meanTensor, meanTensor, xTensor, calcMask);
    Mul(deltaTensor, deltaTensor, deltaTensor, calcMask);
    PipeBarrier<PIPE_V>();
    Muls(deltaTensor, deltaTensor, (count - 1) / count, calcMask);
    PipeBarrier<PIPE_V>();
    Add(m2Tensor, m2Tensor, deltaTensor, calcMask);
}

// 二分累加归约
if (dichotomizeAddDiffSize != 0) {
    Add(calcTensor, calcTensor, calcTensor[sumNum - dichotomizeAddDiffSize], dichotomizeAddDiffSize);
    sumNum = sumNum - dichotomizeAddDiffSize;
}
while (sumNum > ELEM_PER_REP_FP32) {
    sumNum = sumNum / TWO_NUM;
    Add(calcTensor, calcTensor, calcTensor[sumNum], sumNum);
}
```

**vs. baseline:**
```cpp
# baseline两趟算法
# Pass 1: 计算均值
channel_sum = 0.0
while elements_processed < elements_per_channel:
    tl.reduce_sum(shared_ub, data_ub, shared_ub)
    channel_sum += tl.extract_scalar(shared_ub, 0)
mean_val = channel_sum / elements_per_channel

# Pass 2: 计算方差（再次遍历）
channel_sq_diff_sum = 0.0
while elements_processed < elements_per_channel:
    # 再次加载数据计算方差
```

Benefit: 单次遍历减少50%的内存访问；数值稳定性避免大数吃小数问题；二分归约充分利用Vector并行度
Trade-off: 算法理解难度较高；Welford更新需要更多中间变量

---

## Variant C: 场景感知的算法分派（N<=8192 vs N>8192）
Source: dequant_bias

专家实现根据N维度的大小采用不同的实现策略。N<=8192场景一次性加载完整weight_scale和bias，每行数据复用；N>8192场景将N分块处理，按列优先顺序避免UB溢出。8192这个阈值对应float32类型下约32KB数据量，是UB大小的合理分配。

**Expert implementation:**
```cpp
// 专家实现：场景感知的算法选择
if (tilingData.N <= 8192) {
    DequantBias::DequantBiasImpl<...> op;
    op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
    op.Process();
} else { 
    DequantBias::DequantBiasMultiImpl<...> op;
    op.Init(x, weight_scale, activate_scale, bias, y, &tilingData);
    op.Process();
}
```

**vs. baseline:**
```cpp
// baseline：无场景分派
// 单一实现处理所有N值
```

Benefit: 根据数据规模选择最优算法，最大化数据复用
Trade-off: 需要维护两套实现代码

---

## Variant D: 差异化Reduce策略
Source: dynamic_block_quant

专家实现针对fp16和bf16采用了不同的reduce优化策略。BF16路径：先将数据cast到float（保护精度），然后Abs，使用Max指令进行block内横向reduce（通过配置stride参数），最后WholeReduceMax跨block reduce。FP16路径：直接在fp16上进行Abs，WholeReduceMax reduce，然后通过Brcb广播结果，再cast到float进行除法。这种差异化设计的原因在于：bf16精度较低，需要在float上进行关键计算；fp16有专门的硬件加速，但除法仍需在float上进行以避免精度损失。两种路径都使用了PipeBarrier和SetFlag/WaitFlag确保数据依赖正确。

**Expert implementation:**
```cpp
// BF16路径
Cast(xLocalTmp, xLocal, RoundMode::CAST_NONE, calcNum);
Abs(xLocalAbs, xLocalTmp, static_cast<int32_t>(calcNum));
Max(xLocalAbs, xLocalAbs, xLocalAbs[NUM_EIGHT], NUM_SIX_FOUR, scaleCount, 
    {1, NUM_TWO, NUM_TWO, NUM_EIGHT, NUM_ONE_SIX, NUM_ONE_SIX});
WholeReduceMax(scaleLocal, xLocalAbs, NUM_SIX_FOUR, scaleCount, 1, 1, NUM_EIGHT, 
    ReduceOrder::ORDER_ONLY_VALUE);
// FP16路径
WholeReduceMax(scaleLocalT, xLocalAbs, NUM_ONE_TWO_EIGHT, scaleCount, 1, 1, NUM_EIGHT, 
    ReduceOrder::ORDER_ONLY_VALUE);
Brcb(quantScaleLocalT, scaleLocalT, CeilDiv(calcNum / NUM_ONE_TWO_EIGHT, NUM_EIGHT), {1, NUM_EIGHT});
```

**vs. baseline:**
```cpp
AscendC::Abs(absLocal, inputLocal, blockSize);
AscendC::ReduceMax(sharedLocal, absLocal, sharedLocal, blockSize);
```

Benefit: 针对不同数据类型优化，平衡性能和精度
Trade-off: 代码分支增加，维护复杂度提高

---

## Variant E: 分层量化流水线(Hierarchical Quantization Pipeline)
Source: dynamic_quant_update_scatter_v2

专家实现构建了一个高效的分层量化流水线，将FP16/BF16数据转换为INT4。流水线包含以下阶段：(1) Cast到FP32进行高精度统计；(2) ReduceMax/ReduceMin计算动态范围；(3) 计算scale和offset；(4) Muls/Adds进行归一化；(5) Cast到INT32进行量化；(6) SetDeqScale设置反量化参数；(7) Cast到half再Cast到目标格式。每个阶段之间使用PipeBarrier<PIPE_V>()确保Vector Pipeline的指令顺序。这种分层设计允许每个阶段使用最合适的数据精度，既保证了量化精度又最大化了指令吞吐。

**Expert implementation:**
```cpp
// 分层量化流水线
Cast(tempFp32, inLocal[i * sizeHalfLen], RoundMode::CAST_NONE, tilingData_.rowLen);
PipeBarrier<PIPE_V>();
ReduceMax(temp, tempFp32, temp, tilingData_.rowLen, false);
PipeBarrier<PIPE_V>();
maxValue = temp.GetValue(0);
ReduceMin(temp, tempFp32, temp, tilingData_.rowLen, false);
PipeBarrier<PIPE_V>();
minValue = temp.GetValue(0);
GetScaleAndOffset(maxValue, minValue, scale, offset);
backScale = 1 / scale;
Muls(tempFp32, tempFp32, backScale, tilingData_.rowLen);
PipeBarrier<PIPE_V>();
Adds(tempFp32, tempFp32, offset, tilingData_.rowLen);
PipeBarrier<PIPE_V>();
Cast(tempInt32, tempFp32, RoundMode::CAST_RINT, tilingData_.rowLen);
PipeBarrier<PIPE_V>();
SetDeqScale(static_cast<half>(1.0));
PipeBarrier<PIPE_V>();
Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, tilingData_.rowLen);
PipeBarrier<PIPE_V>();
Cast(outLocal, tempHalf, RoundMode::CAST_TRUNC, tilingData_.rowLen);
```

**vs. baseline:**
```cpp
// baseline可能使用简化流水线
float max = ReduceMax(inLocal);
float min = ReduceMin(inLocal);
float scale = (max - min) / 15.0;
for (int i = 0; i < len; i++) {
    outLocal[i] = (inLocal[i] / scale) + 7;
}
```

Benefit: 每个阶段使用最优数据精度，平衡精度与性能
Trade-off: 流水线阶段多，需要更多的UB缓冲区

---

## Variant F: 多模式自适应执行策略
Source: gather_elements_v2

专家实现的核心创新在于提供了三种不同的执行模式，通过tiling阶段的分析自动选择最优路径：Scalar模式适用于UB容量不足以支持向量化处理时；Transpose模式通过先transpose将gather维度转换为最后一维，然后利用连续的内存访问模式进行向量化gather；LastDim模式专门优化最后一维gather场景。模式选择逻辑位于tiling阶段，通过判断是否为最后一维以及CalcMaxBufferSize判断是否需要使用Transpose模式。

**Expert implementation:**
```cpp
ge::graphStatus TilingGatherElementsV2(gert::TilingContext* context) {
    auto dim = *(attr->GetAttrPointer<uint64_t>)(0);
    auto dimNum = context->GetInputShape(X_IDX)->GetStorageShape().GetDimNum();
    if ((dim + dimNum) % dimNum != dimNum - 1) {
        GatherElementsV2Tiling tilingObject(context);
        return tilingObject.SetKernelTiling();
    } else {
        GatherElementsV2LastDimTiling tilingObject(context);
        return ge::GRAPH_SUCCESS;
    }
}

extern "C" __global__ __aicore__ void gather_elements_v2(...) {
    if (TILING_KEY_IS(0)) {
        AscendC::GatherElementsV2ScalarKernel<DTYPE_X, int32_t> op(...);
    } else if (TILING_KEY_IS(1)) {
        AscendC::GatherElementsV2TransposeKernel<DTYPE_X, int32_t> op(...);
    } else if (TILING_KEY_IS(2)) {
        AscendC::GatherElementsV2LastDim<DTYPE_X, int32_t> op(...);
    }
}
```

**vs. baseline:**
```cpp
class KernelGatherElementsV2 {
    // 单一处理流程
    __aicore__ inline void Process() {
        for (loopIdx = 0; loopIdx < tileLoops; loopIdx++) {
            CopyIn(loopIdx, currentRows);
            Compute(loopIdx, currentRows);
            CopyOut(loopIdx, currentRows);
        }
    }
}
```

Benefit: 不同场景选择最优执行路径，获得接近最优的性能
Trade-off: 增加了代码复杂度和tiling计算开销

---

## Variant G: 多模式自适应计算策略
Source: gemma_rms_norm

专家实现根据输入数据的形状特征（num_row, num_col）自动选择四种计算模式之一：MODE_NORMAL（标准模式）、MODE_SPLIT_D（列切分模式）、MODE_MERGE_N（行合并模式）、MODE_SINGLE_ROW（单行模式）。选择策略基于 UB 容量计算和形状启发式，通过 CalMixDtypeTiling 函数实现自动决策。baseline实现只有简单的两层分块，无法适应不同规模的输入。

**Expert implementation:**
```cpp
// 专家实现 - 多模式自适应
bool CalMixDtypeTiling(uint32_t& modeKey, uint64_t& rowFactor, uint64_t& ubFactor, const RMSNormTilingInfo& rmsTilInfo) {
    // 1. Mode MergeN：小 reduce 维度优化
    if (numColAlign <= SMALL_REDUCE_NUM && rmsTilInfo.isSoc910B) {
        modeKey = MODE_MERGE_N;
        rowFactor = ubSize / oneRowBufSize;
        ubFactor = rowFactor * numColAlign;
        return true;
    }
    // 2. Mode Normal / 3. Mode SingleRow / 4. Mode SplitD
}
```

**vs. baseline:**
```cpp
// baseline - 简单 Tiling
uint32_t rowsPerCore = (batchSize + BLOCK_DIM - 1) / BLOCK_DIM;
uint32_t tileLength = (MAX_TILE_LEN < normSize) ? MAX_TILE_LEN : normSize;
uint32_t nTiles = (normSize + tileLength - 1) / tileLength;
```

Benefit: 四种模式适应不同输入规模，确保在各种 shape 下都能高效运行
Trade-off: Tiling 计算更复杂，Host 端开销略有增加

---

## Variant H: 自定义 ReduceSum 算法（半区间归约）
Source: gemma_rms_norm

专家实现没有使用 AscendC 内置的 ReduceSum API，而是自定义了 ReduceSumHalfInterval 算法。该算法采用二分法（Half Interval）策略，通过不断将数据对半相加，充分利用 Vector 单元的并行能力。这种实现比标准 ReduceSum 更高效，特别是在数据量不是 2 的幂次时。算法通过 findPowerTwo 找到小于等于 count 的最大 2 的幂次，然后递归折半求和，最后使用 WholeReduceSum 完成最终归约。

**Expert implementation:**
```cpp
// 专家实现 - 自定义 ReduceSum
__aicore__ inline float ReduceSumHalfInterval(const LocalTensor<float>& src_local, int32_t count) {
    if (likely(count > ELEM_PER_REP_FP32)) {
        int32_t bodyCount = findPowerTwo(count);  // 找最大 2 的幂次
        int32_t tailCount = count - bodyCount;
        if (tailCount > 0) {
            Add(src_local, src_local, src_local[bodyCount], tailCount);
            PipeBarrier<PIPE_V>();
        }
        while (bodyCount > ELEM_PER_REP_FP32) {
            bodyCount = bodyCount / HALf_INTERVAL;
            Add(src_local, src_local, src_local[bodyCount], bodyCount);
            PipeBarrier<PIPE_V>();
        }
        // ... WholeReduceSum
    }
}
```

**vs. baseline:**
```cpp
// baseline - 使用内置 ReduceSum
AscendC::ReduceSum(sharedLocal, tempLocal, sharedLocal, this->tileLength);
float tileSqSum = sharedLocal.GetValue(0);
```

Benefit: 比内置 ReduceSum 更高效，特别是非 2 幂次数据量场景，复杂度 O(logN)
Trade-off: 代码复杂度增加，需要手动管理 mask 设置

---

## Variant I: 多算法策略动态选择
Source: layer_norm_v3

专家实现提供4种算法策略，通过tiling key动态选择。Welford算法（400-422）单次遍历计算mean和variance，数值稳定性最优，适合大维度reduce；Two-Pass算法（300-322）先算mean再算variance，实现简单，适合中等维度；Two-Pass Perf（500-522）优化的Two-Pass版本，使用更高精度的rsqrt计算，适合小批量场景；No-Reduce（600-622）针对normalize dim=1的特殊情况，直接pass through。每种策略通过IsCapable()函数在tiling阶段判断是否适用，实现自动策略选择。

**Expert implementation:**
```cpp
enum class LayerNormV3TilingKey : int64_t {
    LAYER_NORM_REGBASE_TWO_PASS_FLOAT32_FLOAT32 = 300,
    LAYER_NORM_REGBASE_TWO_PASS_FLOAT16_FLOAT32 = 310,
    LAYER_NORM_REGBASE_WELFORD_FLOAT32_FLOAT32 = 400,
    LAYER_NORM_REGBASE_TWO_PASS_PERF_FLOAT32_FLOAT32 = 500,
    LAYER_NORM_REGBASE_NO_REDUCE_FLOAT32_FLOAT32 = 600,
};

// 根据场景选择最优算法
if (TILING_KEY_IS(LNV3_REGBASE_WELFORD_FLOAT_FLOAT)) {
    RegbaseWelfordImpl<float, float, DTYPE_MEAN, IsOutRstd>(...);
} else if (TILING_KEY_IS(LNV3_REGBASE_TWO_PASS_FLOAT_FLOAT)) {
    RegbaseTwoPassImpl<float, float, DTYPE_MEAN, IsOutRstd>(...);
}
```

**vs. baseline:**
```cpp
// 单一算法：简单的Two-Pass
// Step 1: Compute mean
AscendC::ReduceSum(sharedLocal, xLocal, sharedLocal, this->tileLength);
float rowSum = sharedLocal.GetValue(0);
float meanVal = rowSum / this->tileLength;

// Step 2: Compute variance
AscendC::Adds(sharedLocal, xLocal, -meanVal, this->tileLength);
AscendC::Mul(sharedLocal, sharedLocal, sharedLocal, this->tileLength);
AscendC::ReduceSum(sharedLocal, sharedLocal, sharedLocal, this->tileLength);
```

Benefit: 针对不同输入尺寸和精度要求选择最优算法，整体性能提升20-50%
Trade-off: tiling逻辑复杂，需要维护多种算法实现

---

## Variant J: Welford算法的数值稳定实现
Source: layer_norm_v3

Welford算法是统计计算中数值最稳定的在线mean/variance算法，避免了大数求和的精度损失。专家实现的LayerNormV3RegbaseWelford类使用WelfordInitialize初始化累加器，通过WelfordUpdate逐tile更新mean和M2（方差累加器），使用WelfordFinalize计算最终variance，支持分块更新（welfordUpdateTimes），适应任意长的reduce维度。相比baseline简单的ReduceSum后除法，Welford在大维度（如hidden_dim=8192）场景下可避免catastrophic cancellation，精度提升1-2个数量级。

**Expert implementation:**
```cpp
// Welford算法，数值稳定
WelfordInitialize(mean_, variance_, td_->tileLength);
for (int64_t welfordUpdateCount = 0; welfordUpdateCount < td_->welfordUpdateTimes; welfordUpdateCount++) {
    int64_t offset = i * td_->N + welfordUpdateCount * td_->tileLength;
    ProcessWelfordUpdate(offset, td_->tileLength);
}
// Finalize with epsilon
WelfordFinalize<true>(meanTensor[paramAddr + cacheCount], varianceTensor[paramAddr + cacheCount], 
                      mean_, variance_, shared_, para);
```

**vs. baseline:**
```cpp
// 简单Two-Pass，存在数值稳定性问题
AscendC::ReduceSum(sharedLocal, xLocal, sharedLocal, this->tileLength);
float rowSum = sharedLocal.GetValue(0);
float meanVal = rowSum / this->tileLength;

AscendC::Adds(sharedLocal, xLocal, -meanVal, this->tileLength);
AscendC::Mul(sharedLocal, sharedLocal, sharedLocal, this->tileLength);
AscendC::ReduceSum(sharedLocal, sharedLocal, sharedLocal, this->tileLength);
float varSum = sharedLocal.GetValue(0);
float varVal = varSum / this->tileLength;
```

Benefit: 大维度reduce场景下数值稳定性提升1-2个数量级
Trade-off: 算法复杂度高，需要额外临时buffer存储中间结果

---

## Variant K: Binary Reduction树优化
Source: layer_norm_v3

对于大维度reduce，专家实现使用Binary Reduction树（二叉规约树）优化性能。将reduce维度分解为binaryAddQuotient和binaryAddRemainder，通过两两相加的方式，将对数级减少指令数量（从O(n)到O(log n)）。使用LocalMemBar确保中间结果同步。例如，对于reduce_dim=4096，传统线性求和需要4096次累加，而二叉规约仅需约12层迭代。

**Expert implementation:**
```cpp
// Binary add reduction
for (uint16_t i = 0; i < binaryAddKLoop; i++) {
    curBinaryAddLoopMean = curBinaryAddLoopMean / 2;
    for (uint16_t j = 0; j < curBinaryAddLoopMean; j++) {
        DataCopy(binaryAddQ, ...);
        DataCopy(binaryAddR, ...);
        Add(binaryAddQ, binaryAddQ, binaryAddR, pregMain);
        DataCopy(..., binaryAddQ, pregMain);
    }
    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
}
```

**vs. baseline:**
```cpp
// 未优化的reduce
AscendC::ReduceSum(sharedLocal, xLocal, sharedLocal, this->tileLength);
```

Benefit: 大维度reduce场景下指令数减少50-80%
Trade-off: 需要额外临时buffer存储中间结果

---

## Variant L: Gamma/Beta可选参数支持
Source: layer_norm_v3

专家实现支持gamma和beta为nullptr的场景（即只做normalize，不做scale/shift）。通过nullptrGamma/nullptrBeta标志在tiling中传递，在Kernel端使用if constexpr在编译期优化掉不必要的计算分支，使用LayerNormConfig模板参数配置不同组合。这种设计避免了运行期分支判断的开销，同时支持更灵活的使用模式。

**Expert implementation:**
```cpp
// 可选gamma/beta
if constexpr (hasGammaFlag && hasBetaFlag) {
    FusedMulDstAdd(y1, gamma, beta, pregLoop);
} else if constexpr (hasGammaFlag) {
    Mul(y1, y1, gamma, pregLoop);
} else if constexpr (hasBetaFlag) {
    Add(y1, y1, beta, pregLoop);
}
```

**vs. baseline:**
```cpp
// 固定使用gamma/beta
AscendC::Mul(yLocal, yLocal, weightLocal, this->tileLength);
AscendC::Add(yLocal, yLocal, biasLocal, this->tileLength);
```

Benefit: 避免运行期分支开销，支持更灵活的使用模式
Trade-off: 需要生成更多模板实例，代码体积增加

---

## Variant M: 双路径梯度计算策略
Source: multi_scale_deformable_attention_grad

专家实现针对双线性插值的边界条件设计了两条计算路径：ComputeGradTogether当采样点完全在特征图内部时使用，一次性加载4个角点的value值，通过向量化指令并行计算所有梯度；ComputeGradSeparate当采样点靠近边界时使用，逐个角点处理，避免越界访问。这种设计的核心优势在于：内部区域(大部分数据)使用最快路径，边界区域使用安全路径。通过条件判断动态选择路径，既保证了性能又确保了正确性。

**Expert implementation:**
```cpp
if (hLow >= 0 && wLow >= 0 && hLow < h - 1 && wLow < w - 1) {
    ComputeGradTogether(distH, distW, w1, w2, w3, w4, attenWeight);
} else {
    Duplicate(wv1Local, (DTYPE_VALUE)0, embedDims);
    ComputeGradSeparate(1 - distH, 1 - distW, distH, distW, w1, w2, w3, w4, attenWeight);
}
```

**vs. baseline:**
```cpp
// baseline简单循环处理
for (uint32_t p = 0; p < nPoints; p++) {
    // 直接计算，无路径选择
    ComputeGradAttnWeight();
}
```

Benefit: 内部区域使用向量化快速路径，预期向量化效率提升4倍
Trade-off: 代码复杂度增加，需要维护两套计算逻辑

---

## Variant N: FastCompute vs SliceCompute双模式
Source: rms_norm_quant

专家实现根据数据规模自动选择两种计算模式：当列数num_col <= slice_size时使用FastCompute模式（单核可处理完整行），否则使用SliceCompute模式（分片处理）。FastCompute模式避免了多次数据搬运，将gamma加载和计算流水线化。SliceCompute模式通过num_slice_和tail_copy_精确控制分片大小，避免内存浪费。

**Expert implementation:**
```cpp
__aicore__ inline void Launch() {
    if constexpr (FastComputeMode) {
        FastCompute();
    } else {
        SliceCompute();
    }
}

__aicore__ inline void FastCompute() {
    // 预加载gamma
    AscendC::LocalTensor<T> fp16_g = fp32_xy_buf_.Get<T>(num_col_align_f32);
    DataCopyCustom<T>(fp16_g, gm_g_, num_col_);
    while (pid < row_work_) {
        CopyIn(offset, num_col_);
        Compute();
        CopyOut(offset, num_col_);
        ++pid;
    }
}
```

**vs. baseline:**
```cpp
for (uint32_t row = rowStart; row < rowEnd; row++) {
    float varianceSum = ComputeRowVariance(row);
    float variance = varianceSum / colsFloat;
    float rms = sqrt(variance + eps);
    float rmsInv = 1.0f / rms;
    ProcessRow(row, rmsInv);
}
```

Benefit: 短序列使用FastCompute性能更优，长序列使用SliceCompute内存效率更高
Trade-off: 需要维护两套计算逻辑

---

## Variant O: 架构特定的ReduceSum优化
Source: rms_norm_quant

专家实现针对不同AICORE架构提供了不同的ReduceSum实现。对于AICORE 100（旧架构），使用自定义的ReduceSumCustom实现，通过手动控制mask和repeat times处理非对齐数据；对于AICORE 220和3003（新架构），直接使用AscendC内置的ReduceSum接口。

**Expert implementation:**
```cpp
#if __CCE_AICORE__ == 100
    ReduceSumCustom(sum, sqx, work, numel);
#else
    ReduceSum(sum, sqx, work, numel);
#endif
```

**vs. baseline:**
```cpp
AscendC::ReduceSum(accumLocal, xSqLocal, sharedLocal, count);
```

Benefit: 充分利用新老硬件的指令特性，达到最优性能
Trade-off: 需要维护多套实现代码

---

## Variant P: 双模式动态处理策略
Source: scatter_elements_v2

专家实现创新性地设计了两种处理模式：Small Mode和Scatter Mode。Small Mode针对'小包'场景优化——当总数据量可以一次性装入UB时，采用极简的数据流，避免复杂的分块循环。Scatter Mode针对大数据量场景，采用精细的tiling策略，将数据划分为input piece和indices piece，通过双循环实现高效处理。模式选择通过Host端tiling计算自动决定，通过modeFlag传递到Kernel端。

**Expert implementation:**
```cpp
// Expert: 双模式动态选择
if (totalSize <= max_ub) {
    modeFlag = SMALL_MODE;
    // 简化tiling计算
} else {
    modeFlag = 0;
    // 复杂tiling计算
}

// Kernel端模式分发
if (tilingDevice->modeFlag == 1) {
    op.ProcessSmall();
} else {
    op.ProcessScatter();
}
```

**vs. baseline:**
```cpp
// Baseline: 单模式顺序处理
for (uint64_t i = 0; i < numIterations; ++i) {
    DataCopy(varLocal, inputGm[varOffset], inputOneTime);
    DataCopy(indicesLocal, indicesGm[indicesOffset], indicesOneTime);
    DataCopy(updatesLocal, updatesGm[updatesOffset], updatesOneTime);
    // 处理scatter...
}
```

Benefit: 小包场景避免复杂分块开销，大包场景充分利用UB容量，全场景性能最优
Trade-off: 需要维护两套处理逻辑，代码复杂度增加
