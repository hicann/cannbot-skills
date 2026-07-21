# P4: Multi-Core Load Balancing (多核负载均衡)
## Overview
专家实现将sparse_to_dense操作分解为两个独立的阶段：Default Value填充阶段和Sparse Value写入阶段。这种分解的精妙之处在于：1) 任务并行：第一阶段使用defaultValueUsedCoreNum个核并行将输出tensor填充为默认值，第二阶段使用sparseUsedCoreNum个核并行写入稀疏值；2) 负载均衡：两个阶段的核心数可能不同，通过std::max(defaultValueUsedCoreNum_, sparseUsedCoreNum_)确定最终使用的核心数；3) 同步机制：两个阶段之间通过SyncAll()进行全局同步，确保所有核完成第一阶段后再进入第二阶段。这种设计充分利用了NPU的多核架构，避免了单个核处理全部数据导致的性能瓶颈。Host端在DoOpTiling()中精心计算每个核处理的数据量，包括正常核和尾核的数据分配，确保负载均衡。

## When to Use
- Uneven data distributions
- 最小化负载不均衡带来的性能损失，在非整除场景下性能提升10-30%
- 动态负载均衡可充分利用所有core，避免部分core空闲，提升多核并行效率20-50%
- 避免某些核心空闲等待，提高整体并行效率，适应任意batch size

## Trade-off
- Tiling参数增加，逻辑稍复杂
- tiling计算复杂度增加
- Tiling计算稍微复杂

**Source operators**: adaptive_avg_pool3d, adaptive_max_pool3d_grad, add_rms_norm_cast, add_rms_norm_dynamic_quant, apply_adam_w_v2, ascend_quant_v2, conv2d_v2, deep_norm, dequant_bias, dynamic_quant_update_scatter_v2, fake_quant_affine_cachemask, gather_elements_v2, gemma_rms_norm, grouped_matmul, linear_index, masked_scatter_with_position, mat_mul_v3, max_pool_v3, modulate, moe_token_permute, norm_common, scaled_masked_softmax_grad_v2, scaled_masked_softmax_v2, scatter_elements_v2, sparse_to_dense

---

## Variant A: ArithProgression序列生成
Source: linear_index

在合轴场景中，需要生成一个等差数列（arange）来辅助索引计算。专家实现使用ArithProgression指令直接生成序列，而不是通过循环计算每个元素。这是一个专用的硬件指令，可以在一个cycle内生成多个元素。配合Cast指令将float转换为int，实现了高效的序列生成。随后通过浮点乘法和floor取整来实现整数除法和取模操作，这看似反直觉，但实际上利用了Vector单元强大的浮点计算能力来替代整数除法（整数除法在大多数架构上都很慢）。

**Expert implementation:**
```cpp
if constexpr (MODE == 1) {
    ArithProgression(arangeLocal, static_cast<float>(offset), static_cast<float>(1), indicesAlign);
    PipeBarrier<PIPE_V>();
    Cast(arangeIntLocal, arangeLocal, RoundMode::CAST_FLOOR, indicesAlign);
    Muls(arangeLocal, arangeLocal, 1 / static_cast<float>(indicesStride), indicesAlign);
    PipeBarrier<PIPE_V>();
    Cast(indicesTemp, arangeLocal, RoundMode::CAST_FLOOR, indicesAlign);
    PipeBarrier<PIPE_V>();
    Muls(indicesTemp, indicesTemp, static_cast<int>(indicesStride), indicesAlign);
    PipeBarrier<PIPE_V>();
    Sub(indicesTemp, arangeIntLocal, indicesTemp, indicesAlign);
    Muls(indices32Local, indices32Local, selfStride, indicesAlign);
    PipeBarrier<PIPE_V>();
    Add(indices32Local, indices32Local, indicesTemp, indicesAlign);
}
```

**vs. baseline:**
```cpp
if (isCombineDim0 == 1) {
    int32_t arange_val = static_cast<int32_t>(tileStart + i);
    int32_t quotient = arange_val / static_cast<int32_t>(indicesStride);
    int32_t remainder = arange_val - quotient * static_cast<int32_t>(indicesStride);
    idx_val = idx_val * static_cast<int32_t>(selfStride) + remainder;
}
```

Benefit: ArithProgression指令生成序列比循环快数十倍，浮点除法+floor比整数除法快，整体性能提升5-10倍; 浮点乘法比整数除法快5-10倍，显著提升合轴场景性能
Trade-off: 需要额外的UB缓冲区存储arange序列，增加了内存占用; 浮点精度限制，对于极大的indicesStride可能产生误差（但实际场景中indicesStride通常较小）

---

## Variant B: 多核负载均衡策略
Source: adaptive_avg_pool3d

采用former/tail模式处理输出点数量不能被核心数整除的情况。前formerNum个核心处理formerLength个输出点，其余核心处理tailLength个输出点(通常是formerLength-1)。当输出点少于核心数时动态减少核心使用，避免空转。

**Expert implementation:**
```cpp
// 动态负载均衡
if (outputNum < params.coreNum) {
    params.formerNum = outputNum;
    params.formerLength = 1UL;
    usedCoreNum = static_cast<int32_t>(outputNum);
} else if (outputNum % params.coreNum == 0) {
    params.formerNum = params.coreNum;
    params.formerLength = outputNum / params.coreNum;
} else {
    params.formerNum = outputNum % params.coreNum;
    params.tailNum = params.coreNum - params.formerNum;
    params.formerLength = outputNum / params.coreNum + 1UL;
    params.tailLength = outputNum / params.coreNum;
}
```

**vs. baseline:**
```cpp
// baseline固定16核简单划分
const uint32_t BLOCK_DIM = 16;
uint32_t elems_per_core = (total_output_elems + BLOCK_DIM - 1) / BLOCK_DIM;
```

Benefit: 最小化负载不均衡带来的性能损失，在非整除场景下性能提升10-30%
Trade-off: Tiling参数增加，逻辑稍复杂

---

## Variant C: 多核并行与任务分配策略
Source: adaptive_max_pool3d_grad

专家实现采用了多维分层并行策略：第一层是NC维度切分（batch和channel合并），第二层是D/H/W输出空间维度切分。通过ncCntRound和preCoreNum实现任务均匀分配。关键设计是ncRound和ncRoundTail机制：计算每个core需要处理的NC round数（向上取整），前preCoreNum个core处理ceil(NC_CNT / CORE_NUM)个NC，剩余core处理floor(NC_CNT / CORE_NUM)个NC。这种设计确保了在多核场景下的负载均衡，避免了某些core空闲的情况。

**Expert implementation:**
```cpp
maxPoolGradParams.ncCnt = Ops::Base::CeilDiv(maxPoolGradParams.ncDim, maxPoolGradParams.singleCoreNc);
maxPoolGradParams.doCnt = Ops::Base::CeilDiv(maxPoolGradParams.doDim, maxPoolGradParams.singleCoreDo);
maxPoolGradParams.hoCnt = Ops::Base::CeilDiv(maxPoolGradParams.hoDim, maxPoolGradParams.singleCoreHo);
maxPoolGradParams.woCnt = Ops::Base::CeilDiv(maxPoolGradParams.woDim, maxPoolGradParams.singleCoreWo);
maxPoolGradParams.totalCnt = maxPoolGradParams.ncCnt * maxPoolGradParams.doCnt * 
                              maxPoolGradParams.hoCnt * maxPoolGradParams.woCnt;
maxPoolGradParams.usedCoreNum = std::min(maxPoolGradParams.totalCnt, maxPoolGradParams.totalCoreNum);

params_.ncCntRound = tiling->ncRound;
params_.preCoreNum = tiling->preCoreNum;
if (params_.preCoreNum == 0 || blockId < params_.preCoreNum) {
    params_.ncIndex = blockId * params_.ncCntRound;
    params_.ncRealRound = params_.ncCntRound;
} else {
    params_.ncIndex = params_.preCoreNum * params_.ncCntRound + 
                      (blockId - params_.preCoreNum) * tiling->ncRoundTail;
    params_.ncRealRound = tiling->ncRoundTail;
}
```

**vs. baseline:**
```cpp
const uint32_t BLOCK_DIM = 16;
uint32_t tasksPerCore = totalTasks / BLOCK_DIM;
context->SetBlockDim(BLOCK_DIM);
```

Benefit: 动态负载均衡可充分利用所有core，避免部分core空闲，提升多核并行效率20-50%
Trade-off: tiling计算复杂度增加

---

## Variant D: 多核负载均衡
Source: add_rms_norm_cast

专家实现在Tiling阶段精确计算每核处理的数据量，确保所有核心的负载均衡。特别是处理last core时，会根据剩余行数动态调整。策略为：前GetBlockNum() - 1个核心处理blockFactor行，last core处理剩余行数numRow - (GetBlockNum() - 1) * blockFactor。使用CeilDiv确保计算的正确性。

**Expert implementation:**
```cpp
// 负载均衡
blockIdx_ = GetBlockIdx();
if (blockIdx_ < GetBlockNum() - 1) {
    this->rowWork = blockFactor;
} else if (blockIdx_ == GetBlockNum() - 1) {
    this->rowWork = numRow - (GetBlockNum() - 1) * blockFactor;
}
```

**vs. baseline:**
```cpp
// 固定行数分配
const uint32_t BLOCK_DIM = 32;
context->SetBlockDim(BLOCK_DIM);
uint32_t rowsPerCore = (batchSize + BLOCK_DIM - 1) / BLOCK_DIM;
uint32_t rowStart = rowsPerCore * AscendC::GetBlockIdx();
```

Benefit: 避免某些核心空闲等待，提高整体并行效率，适应任意batch size
Trade-off: Tiling计算稍微复杂

---

## Variant E: 多核负载均衡优化
Source: add_rms_norm_dynamic_quant

专家实现的多核分配策略比baseline实现更加精细。它不仅计算firstDimPerCore(每个核心处理的行数)，还计算firstDimPerCoreTail(最后一个核心处理的行数)，并通过firstDimPerLoop控制每轮处理的行数。这种设计确保当总行数不能整除核心数时，最后一个核心不会过载或欠载。

**Expert implementation:**
```cpp
this->firstDimPerCore_ = Ops::Base::CeilDiv(this->numFirstDim_, this->socCoreNums_);
this->useCore_ = Ops::Base::CeilDiv(this->numFirstDim_, this->firstDimPerCore_);
this->firstDimPerCoreTail_ = this->numFirstDim_ - this->firstDimPerCore_ * (this->useCore_ - 1);
if (blockIdx != this->numCore - 1) {
    this->rowWork = this->firstDimPerCore;
} else {
    this->rowWork = this->firstDimPerCoreTail;
}
this->rowTail_ = (this->rowWork % this->rowStep == 0) ? this->rowStep : (this->rowWork % this->rowStep);
```

**vs. baseline:**
```cpp
const uint32_t N_CORES = 32;
uint32_t rows_per_core = total_tokens / N_CORES;
context->SetBlockDim(N_CORES);
```

Benefit: 充分利用所有核心，避免最后一个核心成为瓶颈
Trade-off: 增加边界处理逻辑复杂度

---

## Variant F: 精细化的Tiling策略与负载均衡
Source: apply_adam_w_v2

专家实现的Tiling策略体现了对昇腾硬件的深度理解。首先，每个核单次处理2432个元素（对于FP32约为2432*4=9728字节），这个数值经过精心计算以充分利用UB容量。其次，采用动态负载均衡策略：计算总循环数`loopNum`，将循环分配给所有核心，前`handleExtraLoopCoreNum`个核心多处理一次循环。这种设计确保在数据量不能被核心数整除时，各核心的负载差异最多为1个循环，实现了近乎完美的负载均衡。

**Expert implementation:**
```cpp
const size_t numPerLoop = 2432;  // Elements per loop
int64_t loopNum = (totalDataNum + numPerLoop - 1) / numPerLoop;
int64_t loopNumPerCore = loopNum / tilingParam.totalCoreNum;
int64_t handleExtraLoopCoreNum = loopNum % tilingParam.totalCoreNum;
int64_t usedCoreNum = loopNumPerCore > 0 ? tilingParam.totalCoreNum : handleExtraLoopCoreNum;
```

**vs. baseline:**
```cpp
// Simple partitioning
int64_t elementsPerCore = (totalElements + coreNum - 1) / coreNum;
int64_t usedCoreNum = (totalElements + elementsPerCore - 1) / elementsPerCore;
```

Benefit: 近乎完美的负载均衡；最大化UB利用率；减少尾处理开销
Trade-off: Tiling计算复杂度增加

---

## Variant G: Per-Head量化模式优化
Source: ascend_quant_v2

针对Transformer模型Multi-Head Attention结构的专门优化，scale和offset沿特定head维度广播，每个head使用独立量化参数。通过模板参数SQRT_MODE和HAS_OFFSET在编译期确定计算路径，避免运行时分支。三层嵌套循环结构（ProcessLoop/ProcessParamLoop/ProcessInputLoop）配合blockUnion计算，实现4D张量[batch,seq_len,num_heads,head_dim]的高效并行处理。

**Expert implementation:**
```cpp
// Per-Head三层循环
template <bool SQRT_MODE, bool HAS_OFFSET>
void ProcessLoop() {
    for (int64_t i = 0; i < blockS_; ++i) {
        ProcessParamLoop<SQRT_MODE, HAS_OFFSET>();
        gmXOffset_ += tilingData_.dim1 * tilingData_.dim2;
    }
}

// scale/offset复用
for (int64_t i = 0; i < nLoopNum; ++i) {
    ProcessParamOneLoop<SQRT_MODE, HAS_OFFSET>(nLoopLen, baseSOffset, baseXOffset);
    baseXOffset += nLoopLen * tilingData_.dim2;
    baseSOffset += tilingData_.baseN;
}
```

Benefit: 针对Transformer模型优化，提高大模型推理效率
Trade-off: 代码复杂度增加，需要额外的维度计算

---

## Variant H: 多核负载均衡
Source: deep_norm

专家实现的 Tiling 策略中考虑了多核负载均衡问题。通过 CEIL_DIV 计算最优核数，并将行数据分配给各核心处理。对于最后一个核心，可能处理较少的行，代码中通过条件判断正确处理这种情况。这种负载均衡策略确保了在所有核心完成任务之前没有核心处于空闲状态。

**Expert implementation:**
```cpp
// 专家实现 - 精细负载均衡
uint32_t numCore = CEIL_DIV(numRow, CEIL_DIV(numRow, maxCoreNum));
uint32_t rowWork = CEIL_DIV(static_cast<uint32_t>(numRow), numCore);
uint32_t lFirstdimPerCoreNum = static_cast<uint32_t>(numRow) - rowWork * (numCore - 1U);

// Kernel 端处理不同负载
if (block_idx < num_core - 1) {
    row_work = nl_first_dim_per_core;
} else {
    row_work = l_first_dim_per_core;
    row_step = MIN(first_dim_per_times, row_work);
}
```

**vs. baseline:**
```cpp
// baseline - 简单负载分配
context->SetBlockDim(BLOCK_DIM);
uint32_t rowsPerCore = (batchSize + BLOCK_DIM - 1) / BLOCK_DIM;
```

Benefit: 最大化多核并行效率，避免核心空闲
Trade-off: 需要额外的逻辑处理边界情况

---

## Variant I: Tiling策略的动态计算
Source: dequant_bias

专家实现的Tiling策略基于UB容量感知、行列平衡、核心负载均衡、循环展开优化等因素动态计算。关键计算公式包括perCoreRow = CeilDiv(M_, aivNum)、releaseUbSizeForX = ubSize - (weightScaleBuffSize + asBuffSize + biasBuffSize)。

**Expert implementation:**
```cpp
// 专家实现：动态Tiling计算
uint32_t perCoreRow = Ops::Base::CeilDiv(M_, aivNum);
uint32_t needCoreNum = Ops::Base::CeilDiv(M_, perCoreRow);
uint32_t tailCoreRow = M_ - (needCoreNum - 1) * perCoreRow;

// UB容量感知的数据分块
uint32_t releaseUbSizeForX = aicoreParams_.ubSize - (weightScaleBuffSize + asBuffSize + biasBuffSize);
perCoreLoopRow = std::min(releaseUbSizeForX / inBuffSize, perCoreRow);
```

**vs. baseline:**
```cpp
// baseline：固定分块策略
const uint32_t nCores = 8;
uint32_t elementsPerCore = totalElems / nCores;
uint32_t tileSize = 128;
uint32_t innerLoops = elementsPerCore / tileSize;
```

Benefit: 自适应不同输入规模，最大化多核利用率
Trade-off: Tiling计算复杂，增加Host端开销

---

## Variant J: 精细化的多核负载均衡(Fine-grained Load Balancing)
Source: dynamic_quant_update_scatter_v2

专家实现采用head/tail core策略实现精细化的多核负载均衡。当总行数不能被核数整除时，前headCoreNum个核(head cores)每个处理rowPerHeadCore行，剩余的tail core处理rowPerTailCore行（比head core少一行）。这种分配方式确保最多只有一个核的负载与其他核不同，最大负载差异控制在一行以内。通过计算rowPerHeadCore = (rowNum + vectorCoreNum - 1) / vectorCoreNum实现向上取整，保证所有行都被处理。

**Expert implementation:**
```cpp
// Tiling阶段: 计算负载均衡参数
rowPerHeadCore = static_cast<uint32_t>(rowNum + vectorCoreNum - static_cast<uint32_t>(1)) / vectorCoreNum;
coreNum = static_cast<uint32_t>(rowNum + rowPerHeadCore - static_cast<uint32_t>(1)) / rowPerHeadCore;
headCoreNum = static_cast<uint32_t>(coreNum - static_cast<uint32_t>(1));
rowPerTailCore = rowNum - headCoreNum * rowPerHeadCore;  // 最后一核处理的行数

// Kernel阶段: 根据blockIdx确定处理的行范围
if (blockIdx < tilingData_.headCoreNum) {
    inGm.SetGlobalBuffer((__gm__ xDtype*)x + blockIdx * lenHead, lenHead);
    rowPerCore = rowPerHeadCore;
} else {
    inGm.SetGlobalBuffer(
        (__gm__ xDtype*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail,
        lenTail);
    rowPerCore = rowPerTailCore;
}
```

**vs. baseline:**
```cpp
// baseline可能使用简单均匀分配
uint32_t rowPerCore = rowNum / coreNum;
for (uint32_t i = 0; i < coreNum; i++) {
    ProcessRows(i * rowPerCore, rowPerCore);
}
```

Benefit: 最大化多核利用率，减少空闲等待时间，处理非对齐数据时性能提升显著
Trade-off: Tiling参数增加，代码逻辑稍微复杂

---

## Variant K: 多核负载均衡与Tiling优化
Source: fake_quant_affine_cachemask

专家实现采用了精细化的多核并行策略，通过headNum（第一维大小）进行任务划分，而非简单的元素均分。这种策略特别适用于per-channel量化的场景，其中scale和zero_point是针对每个channel（head）独立的。Tiling算法首先计算loopNum = headNum / coreNum（每个core处理的基本head数）和remainNum = headNum % coreNum（剩余的head数）。前remainNum个core处理loopNum + 1个head，其余core处理loopNum个head，实现了负载的均衡分配。此外，专家实现还考虑了不同芯片版本的UB大小差异（ASCEND910 vs ASCEND910B），在ASCEND910B上预留了SELECT_MODE_GE_ZERO_TMP_UB（8000字节）的空间，用于select指令的临时存储。

**Expert implementation:**
```cpp
// 专家实现多核负载均衡
uint32_t GetNeedCoreNum(const uint32_t coreNumPlatform, ge::DataType dType) {
    if (dType == ge::DT_FLOAT16) {
        dataPerRepeat = BYTE_REPEAT / SIZE_OF_FP16;
    } else {
        dataPerRepeat = BYTE_REPEAT / SIZE_OF_FP32;
    }
    return headNum < coreNumPlatform ? headNum : coreNumPlatform;
}

// 负载均衡计算
loopNum = headNum / coreNum;
remainNum = headNum % coreNum;
if (GetBlockIdx() < remainNum) {
    blockLength = totalLengthAligned * (loopNum + 1);
    circleNum = loopNum + 1;
} else {
    blockLength = totalLengthAligned * loopNum;
}

// 芯片版本感知的tileSize计算
if (socVersion == platform_ascendc::SocVersion::ASCEND910) {
    tileLength = ubSizePlatForm / bytesPerData / dataPerRepeat * dataPerRepeat;
} else {
    tileLength = (ubSizePlatForm - SELECT_MODE_GE_ZERO_TMP_UB) / bytesPerData / dataPerRepeat * dataPerRepeat;
}
```

**vs. baseline:**
```cpp
// baseline简单单核处理
const uint32_t BLOCK_DIM = 1;
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
  context->SetBlockDim(BLOCK_DIM);
  uint32_t elementsPerCore = total_elems;
  uint32_t tileSize = elementsPerCore < 2048 ? elementsPerCore : 2048;
}
```

Benefit: 充分利用多核并行能力，通过负载均衡提高整体吞吐量；芯片版本感知确保在不同硬件上都能获得最优性能
Trade-off: Tiling逻辑复杂度增加，需要考虑headNum、remainNum等因素；对输入shape有一定要求（需要有head维度）

---

## Variant L: 精细的多核并行分块策略
Source: gather_elements_v2

专家实现的多核并行策略远比baseline的单核处理复杂和高效。实现了两种分核策略：行分核（当pre维度大于可用核数时，直接按pre维度分核）和行列联合分核（当pre维度较小，在pre维度分组内再按post维度分核）。这种分块策略通过formerGroupNum/tailGroupNum、formerGroupCoreNum/tailGroupCoreNum等参数处理不均衡分配，确保负载均衡。

**Expert implementation:**
```cpp
inline void GatherElementsV2Tiling::Tiling4GatherElementsV2() {
    usedCoreNum_ = std::min(idxPreDim_ * idxPostDim_, coreNum_);
    if (idxPreDim_ > usedCoreNum_) {
        coreGroupNum_ = usedCoreNum_;
        tailGroupNum_ = (coreGroupNum_ - idxPreDim_ % coreGroupNum_) % coreGroupNum_;
        formerGroupNum_ = coreGroupNum_ - tailGroupNum_;
        formerGroupPreDim_ = (idxPreDim_ + usedCoreNum_ - 1) / usedCoreNum_;
        tailGroupPreDim_ = idxPreDim_ / usedCoreNum_;
    } else {
        coreGroupNum_ = idxPreDim_;
        formerGroupCoreNum_ = (usedCoreNum_ + coreGroupNum_ - 1) / coreGroupNum_;
        tailGroupCoreNum_ = usedCoreNum_ / coreGroupNum_;
        formerGroupTailNum_ = (formerGroupCoreNum_ - idxPostDim_ % formerGroupCoreNum_) % formerGroupCoreNum_;
        // ...
    }
}
```

**vs. baseline:**
```cpp
// Core partitioning - use single core for correctness
uint32_t n_cores = 1;
uint32_t rows_per_core = total_rows;
context->SetBlockDim(n_cores);
```

Benefit: 充分利用多核并行能力，性能提升可达10-100倍
Trade-off: 显著增加了tiling逻辑的复杂度

---

## Variant M: 合轴优化
Source: gather_elements_v2

专家实现在LastDim模式下实现了合轴（MergeAxis）优化，将相邻且相同大小的轴合并为一个轴。这种优化减少了维度处理的复杂度，特别是在处理高维张量时。通过减少有效维度数，可以简化后续的tiling计算和kernel执行逻辑。

**Expert implementation:**
```cpp
ge::graphStatus GatherElementsV2LastDimTiling::MergeAxis(...) {
    while (i < xDimNum) {
        if (xOriginShape[i] != indexOriginShape[i] || i == xDimNum - 1) {
            xShape_.AppendDim(xOriginShape[i]);
        } else {
            int j = i;
            while (j < xDimNum && xOriginShape[j] == indexOriginShape[j] && j != xDimNum - 1) {
                j++;
            }
            if (j - i > 1) {
                int64_t val = 1;
                for (int k = i; k < j; k++) val *= xOriginShape[k];
                xShape_.AppendDim(val);
            }
            i = j - 1;
        }
        i++;
    }
}
```

Benefit: 减少维度处理复杂度，简化tiling计算
Trade-off: 增加了初始化阶段的计算量

---

## Variant N: 多核负载均衡与尾部处理
Source: gemma_rms_norm

专家实现采用精细的多核负载均衡策略。通过 block_factor 和 latsBlockFactor 区分普通核与尾核的处理量，确保所有核的负载尽可能均衡。对于不能被核数整除的 row 数，最后一个核处理剩余行，通过条件判断避免越界。baseline实现采用简单的平均分配，最后一个核可能处理极少数据，造成资源浪费。

**Expert implementation:**
```cpp
// 专家实现 - 精细负载均衡
blockIdx_ = GetBlockIdx();
if (blockIdx_ < GetBlockNum() - 1) {
    this->row_work = block_factor;  // 普通核处理量
} else if (blockIdx_ == GetBlockNum() - 1) {
    this->row_work = num_row - (GetBlockNum() - 1) * block_factor;  // 尾核处理剩余行
}
```

**vs. baseline:**
```cpp
// baseline - 简单平均分配
uint32_t rowsPerCore = (batchSize + BLOCK_DIM - 1) / BLOCK_DIM;
```

Benefit: 最大化多核利用率，避免尾核空闲，提升整体吞吐量
Trade-off: Init 逻辑略微复杂，需要计算尾核处理量

---

## Variant O: 精细的Tiling与负载均衡
Source: linear_index

专家实现采用了一套精细的tiling策略来最大化硬件利用率。首先根据数据量动态计算实际使用的核数（usedCoreNum），避免核数浪费。然后将数据划分为eachCount（每个核的基础数据量）和lastCount（最后一个核的数据量，可能较少）。进一步地，根据UB大小计算maxSize，将每个核的数据再细分为多个loop，每个loop处理eachNum个元素，最后一个loop处理eachTail个元素。这种多级tiling策略确保了：1) 多核负载均衡；2) 单核内部数据可以完全放入UB；3) 处理尾部数据时不会浪费太多计算资源。

**Expert implementation:**
```cpp
uint32_t times = (indicesCount + DATA_ALIGN - 1) / DATA_ALIGN;
usedCoreNum = times < coreNum ? times : coreNum;

max_ub = ubSizePlatForm / max_ub * max_ub / BUFFER_NUM;
maxSize = max_ub / indicesSize;

eachCount = (indicesCount + usedCoreNum - 1) / usedCoreNum;
usedCoreNum = (indicesCount + eachCount - 1) / eachCount;
lastCount = indicesCount - eachCount * (usedCoreNum - 1);

eachNum = eachCount;
eachLoop = 1;
eachTail = eachCount;
if (eachCount > maxSize) {
    eachNum = maxSize;
    eachLoop = (eachCount + maxSize - 1) / maxSize;
    eachTail = eachCount - (eachLoop - 1) * eachNum;
}
```

**vs. baseline:**
```cpp
const uint32_t BLOCK_DIM = 30;
context->SetBlockDim(BLOCK_DIM);
uint32_t elementsPerCore = (total_elems + BLOCK_DIM - 1) / BLOCK_DIM;
uint32_t tileSize = 2048;
uint32_t innerLoops = (elementsPerCore + tileSize - 1) / tileSize;
```

Benefit: 动态核数分配避免核数浪费，多级tiling确保数据局部性，UB大小感知避免spill到GM
Trade-off: 增加了tiling逻辑的复杂度，需要更精细的参数管理

---

## Variant P: PipeBarrier流水线同步
Source: linear_index

专家实现精心安排了PipeBarrier<PIPE_V>的位置来控制指令流水线。在Ascend C架构中，Vector单元是异步执行的，PipeBarrier确保前一条指令完成后才执行下一条，避免数据竞争。但专家实现的barrier放置非常讲究：在负数索引转换和合轴计算的每个阶段后都放置了barrier，确保数据一致性；但在数据传输（MTE）和计算（V）之间使用的是SetFlag/WaitFlag机制，这是更细粒度的同步方式。这种混合同步策略既保证了正确性，又最大化了流水线并行度。

**Expert implementation:**
```cpp
int32_t eventIDMTE2ToV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);

if constexpr (IS_CAST_INT) {
    Cast<int, T>(indices32Local, indicesLocal, RoundMode::CAST_NONE, indicesAlign);
    PipeBarrier<PIPE_V>();
}

ShiftRight(indicesTemp, indices32Local, INT32_OFFSET, static_cast<int>(indicesAlign));
PipeBarrier<PIPE_V>();
Muls(indicesTemp, indicesTemp, static_cast<int>(target), static_cast<int>(indicesAlign));
PipeBarrier<PIPE_V>();
Sub(indices32Local, indices32Local, indicesTemp, static_cast<int>(indicesAlign));
```

**vs. baseline:**
```cpp
__aicore__ inline void Process() {
    for (uint32_t i = 0; i < innerLoops; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}
```

Benefit: 正确的同步机制保证数据一致性，细粒度同步最大化流水线并行
Trade-off: 过多的barrier会降低并行度，需要精心安排位置

---

## Variant Q: 动态核数分配
Source: linear_index

专家实现根据数据量动态计算实际使用的核数，避免核数浪费。baseline固定使用30核，当数据量很小时（如只有1000个元素），每个核只处理几十个元素，核数开销（launch overhead）会远大于计算本身。专家实现动态计算usedCoreNum，如果数据量小于核数能力，则减少使用的核数。

**Expert implementation:**
```cpp
uint32_t times = (indicesCount + DATA_ALIGN - 1) / DATA_ALIGN;
usedCoreNum = times < coreNum ? times : coreNum;
eachCount = (indicesCount + usedCoreNum - 1) / usedCoreNum;
usedCoreNum = (indicesCount + eachCount - 1) / eachCount;
lastCount = indicesCount - eachCount * (usedCoreNum - 1);
context->SetBlockDim(usedCoreNum);
```

**vs. baseline:**
```cpp
const uint32_t BLOCK_DIM = 30;
context->SetBlockDim(BLOCK_DIM);
uint32_t elementsPerCore = (total_elems + BLOCK_DIM - 1) / BLOCK_DIM;
```

Benefit: 小数据量场景下避免核数浪费，减少launch overhead
Trade-off: 增加了tiling逻辑复杂度

---

## Variant R: 动态核心数分配与负载均衡
Source: masked_scatter_with_position

专家实现采用动态核心数分配策略，根据数据规模计算最优AI Core使用数量。核心数计算：usedCoreNum = min(ceil(xEleNums / THREAD_NUM), totalCoreNum)，THREAD_NUM=1024。这种策略确保资源利用率最大化（小数据量时减少核心使用，避免空闲浪费）、负载均衡（每个核心处理约1024个元素）、可扩展性（自动适应不同规模输入）。baseline使用固定N_CORES=16，在小数据场景下大部分核心空闲，大数据场景下负载不均衡。

**Expert implementation:**
```cpp
// 专家实现 - 动态核心数分配
static constexpr int64_t THREAD_NUM = 1024;

ge::graphStatus MaskedScatterWithPositionTiling::DoOpTiling() {
    usedCoreNum_ = Ops::Base::CeilDiv(xEleNums_, THREAD_NUM);
    usedCoreNum_ = std::min(usedCoreNum_, totalCoreNum_);
}

ge::graphStatus MaskedScatterWithPositionTiling::PostTiling() {
    context->SetBlockDim(usedCoreNum_);  // 动态设置核心数
}
```

**vs. baseline:**
```cpp
// baseline - 固定核心数
const uint32_t N_CORES = 16;
const uint32_t TILE_SIZE = 128;

uint32_t nCores = N_COPES;
uint32_t elementsPerCore = totalElems / nCores;
```

Benefit: 资源利用率最大化，负载均衡，自动适应不同数据规模
Trade-off: 需要获取平台信息，增加少量Host端计算开销

---

## Variant S: 均衡分块算法
Source: modulate

专家实现采用均衡分块算法处理不规则的数据划分。传统简单分块会导致最后一个core处理的数据量可能远小于其他core。均衡分块算法使用tailLength作为基础每个core处理的元素数，frontLength=tailLength+1作为前frontNum个core多处理的元素数，frontNum=totalElements%coreNum为需要多处理的core数量。这种算法的优势是最大不均衡度为1，计算简单高效，适用于任意数量。

**Expert implementation:**
```cpp
// 专家实现: 均衡分块算法
void CalcTilingParam(int64_t TilingDim, int64_t totalElements, bool useDtiling = false) {
    this->tilingData.tailLength = totalElements / this->coreNum;
    this->tilingData.frontLength = this->tilingData.tailLength + 1;
    this->tilingData.frontNum = totalElements % this->coreNum;
    this->tilingData.tailNum = this->tilingData.tailLength == 0 ? 0 : this->coreNum - this->tilingData.frontNum;
}
__aicore__ inline int64_t GetNum() {
    return GetBlockIdx() >= this->frontNum ? this->tailLength : this->frontLength;
}
```

**vs. baseline:**
```cpp
// baseline: 简单取模分块
int64_t rowsPerCore = totalRows / GetBlockNum();
int64_t remainRows = totalRows % GetBlockNum();
this->startB = coreIdx * rowsPerCore + (coreIdx < remainRows ? coreIdx : remainRows);
this->endB = this->startB + rowsPerCore + (coreIdx < remainRows ? 1 : 0);
```

Benefit: 减少10-30%的等待时间，特别是当totalElements%coreNum较大时
Trade-off: 需要在Kernel端维护更多的状态变量

---

## Variant T: 精细的多核负载均衡
Source: norm_common

专家实现通过loopCount、tailLoop、tailNRow等参数实现精细的多核负载均衡。对于不能整除的行数分配，采用'整核+尾核'的策略——部分核处理loopCount+1个block，其他核处理loopCount个block，最后一个核处理剩余的tailNRow行。这种策略确保所有核的计算量差异不超过一个block，最大化多核利用率。baseline实现采用简单的平均分配（rows_per_core = batch_size / n_cores），未处理不能整除的情况，导致尾核可能空闲或负载不均。

**Expert implementation:**
```cpp
// 专家实现：精细负载均衡
__aicore__ inline void Init(...)
{
    if (GetBlockIdx() < tailLoop) {
        xGmOffset = (loopCount + 1) * nRow * rowSize * GetBlockIdx();
        xGmSize = (loopCount + 1) * blockLength;
    } else {
        xGmOffset = loopCount * nRow * rowSize * GetBlockIdx() + nRow * rowSize * tailLoop;
        xGmSize = loopCount * blockLength;
    }
    xGm.SetGlobalBuffer((__gm__ Tfm*)x + xGmOffset, xGmSize);
}

__aicore__ inline void Process()
{
    uint32_t Count = loopCount;
    if (GetBlockIdx() < tailLoop) {
        Count += 1;
    }
    for (uint32_t loopIdx = 0; loopIdx < Count; ++loopIdx) {
        ProcessBasicBlock(nRow, currentBlockOffset, currentParamOffset);
    }
    if (tailNRow > 0 && GetBlockIdx() == (blockDim - 1)) {
        ProcessBasicBlock(tailNRow, currentBlockOffset, currentParamOffset);
    }
}
```

**vs. baseline:**
```cpp
// baseline：简单平均分配
uint32_t n_cores = 32;
context->SetBlockDim(n_cores);
uint32_t rows_per_core = batch_size / n_cores;
tiling.set_rows_per_core(rows_per_core);
```

Benefit: 最大化多核利用率，避免尾核空闲，在不能整除的shape下性能提升可达10-30%
Trade-off: Tiling逻辑复杂化，需要计算和传递更多参数（tailLoop/tailNRow等）

---

## Variant U: 多核负载均衡与尾核优化
Source: scaled_masked_softmax_grad_v2

专家实现采用精细的多核任务分配策略，考虑了尾核（tail core）处理量差异。通过headCoreNum和totalLinePerHeadCore/totalLinePerTailCore的计算，实现负载均衡，避免某些核空闲等待，提高整体利用率。当总行数不能被核数整除时，尾核处理更少的行，避免性能瓶颈。

**Expert implementation:**
```cpp
uint64_t totalLine = batch * channel * seqLength;
uint64_t usedCoreNum = std::min(totalLine, static_cast<uint64_t>(coreNum));
totalLinePerHeadCore = Ops::Base::CeilDiv(totalLine, usedCoreNum);
uint64_t totalLinePerTailCore = totalLine / usedCoreNum;
uint64_t headCoreNum = totalLine % usedCoreNum;

if (currentCoreIdx < tilingData.headCoreNum) {
    lineOffset = currentCoreIdx * tilingData.totalLinePerHeadCore;
    loopTimes = CeilDiv(tilingData.totalLinePerHeadCore, maxLinePerLoop_);
    minLine = tilingData.tailLinePerHeadCore;
} else {
    lineOffset = tilingData.headCoreNum * tilingData.totalLinePerHeadCore +
                 (currentCoreIdx - tilingData.headCoreNum) * tilingData.totalLinePerTailCore;
    loopTimes = CeilDiv(tilingData.totalLinePerTailCore, maxLinePerLoop_);
    minLine = tilingData.tailLinePerTailCore;
}
```

**vs. baseline:**
```cpp
const uint32_t BLOCK_DIM = 16;
uint32_t elemsPerCore = (totalElems + BLOCK_DIM - 1) / BLOCK_DIM;
uint32_t innerLoops = (elemsPerCore + TILE_SIZE - 1) / TILE_SIZE;
context->SetBlockDim(BLOCK_DIM);
```

Benefit: 提高多核利用率，避免尾核成为性能瓶颈，适应不同shape的输入
Trade-off: Tiling参数增加，Host端和Kernel端需要更多计算

---

## Variant V: 大小核负载均衡调度
Source: scaled_masked_softmax_v2

专家实现采用了精细的大小核负载均衡策略。首先获取实际可用的AIV核心数，然后将核心分为大核（headCore）和小核（tailCore）两组：前headCoreNum个核心每核处理lineHeadCore行，剩余核心每核处理lineTailCore行（lineTailCore = lineHeadCore - 1）。这种分配方式确保了当总行数不能被核心数整除时，负载尽可能均匀分布。同时，每个核心内部还分为多次迭代处理，每次迭代处理lineHeadIter行，最后一次迭代处理剩余行数。

**Expert implementation:**
```cpp
// 专家实现大小核调度
uint64_t lineTailCore = totalLine / coreNum;
uint64_t headCoreNum = totalLine % coreNum;
uint64_t lineHeadCore = lineTailCore + 1;

uint64_t iterHeadCore = Ops::Base::CeilDiv(lineHeadCore, availableLinePerIter);
uint64_t lineLastHeadIter = lineHeadCore - (iterHeadCore - 1) * lineHeadIter;

// Kernel端
this->coreOffset = (this->blockIdx >= tilingData.headCoreNum ? tilingData.headCoreNum : this->blockIdx) * (tilingData.lineHeadCore * tilingData.width) +
                   (this->blockIdx >= tilingData.headCoreNum ? this->blockIdx - tilingData.headCoreNum : 0) * (tilingData.lineTailCore * tilingData.width);
uint64_t loop = this->blockIdx >= tilingData.headCoreNum ? tilingData.iterTailCore : tilingData.iterHeadCore;
```

**vs. baseline:**
```cpp
// baseline 简单均分
uint32_t rowStart = blockIdx * rowsPerCore;
if (blockIdx == GetBlockNum() - 1) {
    rowEnd = totalRows;
} else {
    rowEnd = rowStart + rowsPerCore;
}
```

Benefit: 负载均衡度提升，多核并行效率提高15-25%，特别是在shape不能被核心数整除时
Trade-off: Tiling数据结构复杂，需要传输更多参数到Kernel

---

## Variant W: 多核并行与任务调度
Source: scatter_elements_v2

专家实现采用灵活的多核并行策略，根据数据规模和硬件核心数动态调整任务分配。对于小规模数据（times < coreNum），采用任务分割策略，将单个任务分配给多个核心并行处理；对于大规模数据，每个核心独立处理一个或多个完整任务。通过eachNum、extraTaskCore、eachPiece等参数精确控制任务分配，确保负载均衡。

**Expert implementation:**
```cpp
// Expert: 动态多核调度
if (times < coreNum) {
    eachPiece = coreNum / times;
    inputOnePiece = (inputOneTime + eachPiece - 1) / eachPiece;
    usedCoreNum = (inputOneTime + inputOnePiece - 1) / inputOnePiece * times;
} else {
    usedCoreNum = coreNum;
    eachNum = times / coreNum;
    extraTaskCore = times - eachNum * coreNum;
}

// Kernel端任务索引计算
if (eachNum == 0) {
    start = coreId / eachPiece;
    currentPiece = coreId % eachPiece;
} else {
    currentNum = coreId < extraTaskCore ? (eachNum + 1) : eachNum;
    start = coreId * eachNum + (coreId < extraTaskCore ? coreId : extraTaskCore);
}
```

**vs. baseline:**
```cpp
// Baseline: 单核执行
tilingContext->SetBlockDim(1);
```

Benefit: 最大化多核利用率，负载均衡，适应各种数据规模
Trade-off: 任务调度逻辑复杂，需要处理边界情况

---

## Variant X: 双阶段多核并行流水线
Source: sparse_to_dense

专家实现将sparse_to_dense操作分解为两个独立的阶段：Default Value填充阶段和Sparse Value写入阶段。这种分解的精妙之处在于：1) 任务并行：第一阶段使用defaultValueUsedCoreNum个核并行将输出tensor填充为默认值，第二阶段使用sparseUsedCoreNum个核并行写入稀疏值；2) 负载均衡：两个阶段的核心数可能不同，通过std::max(defaultValueUsedCoreNum_, sparseUsedCoreNum_)确定最终使用的核心数；3) 同步机制：两个阶段之间通过SyncAll()进行全局同步，确保所有核完成第一阶段后再进入第二阶段。这种设计充分利用了NPU的多核架构，避免了单个核处理全部数据导致的性能瓶颈。Host端在DoOpTiling()中精心计算每个核处理的数据量，包括正常核和尾核的数据分配，确保负载均衡。

**Expert implementation:**
```cpp
defaultValueUsedCoreNum_ = Ops::Base::CeilDiv(outSize_, MIN_THREAD_NUM);
defaultValueUsedCoreNum_ = std::min(totalCoreNum_, defaultValueUsedCoreNum_);
sparseUsedCoreNum_ = Ops::Base::CeilDiv(numValues_, MIN_THREAD_NUM);
useCoreNum_ = std::max(defaultValueUsedCoreNum_, sparseUsedCoreNum_);
context->SetBlockDim(useCoreNum_);

// Kernel:
SetDefaultValue();
SyncAll();
SparseToDenseSimtCompute(...);
```

**vs. baseline:**
```cpp
uint32_t nCores = 1;  // Force single core
context->SetBlockDim(nCores);
```

Benefit: 充分利用多核并行，两个阶段可以独立调优核心数
Trade-off: 需要全局同步，增加了一定的同步开销

---

## Variant Y: StreamK混合DP/SK尾块负载均衡
Source: mat_mul_v3

StreamK调度器将M×N tile grid分为Data-Parallel(DP)区域和Stream-K(SK)区域，解决标准Block Tiling中tile数不能被核心数整除时的负载不均问题。DP区域的tile每个核心独立计算完整K维度；SK区域的尾部tile在K维度分割到多个核心，通过Atomic操作合并。尾部tile数量 = (mTileNum * nTileNum) % usedCoreNum，这些tile被均匀分配到所有核心。

**Expert implementation:**
```cpp
template <class ProblemShape_, class L1TileShape_, class L0TileShape_>
class BlockSchedulerStreamKBuiltIn {
    __aicore__ inline BlockSchedulerStreamKBuiltIn(const ProblemShape& shape, const Params& params) {
        mTileNum_ = CeilDiv(shape.m, mL1_);
        nTileNum_ = CeilDiv(shape.n, nL1_);
        int64_t tailMNTileNum = (mTileNum_ * nTileNum_) % usedCoreNum_;
        // DP tiles: complete K per core; SK tiles: split K across cores
        tileNum_ = (mTileNum_ * nTileNum_ - tailMNTileNum) + tailMNTileNum * skKTileNum_;
        totalMNTileNumInDP_ = mTileNum_ * nTileNum_ - tailMNTileNum;
    }

    __aicore__ inline void UpdateMNTileIdx(int64_t tileIdx) {
        if (CheckIsSkScene(tileIdx)) {
            kTileIdx_ = (tileIdx % usedCoreNum_) % curKTileNum_;  // K slice index
            mnIdx = (tileIdx % usedCoreNum_) / curKTileNum_ + totalMNTileNumInDP_;
        } else {
            kTileIdx_ = 0;  // Full K computation
            mnIdx = tileIdx / curKTileNum_;
        }
    }
};
```

**vs. baseline:**
```cpp
// 标准分块：尾部核心空闲
int tileIdx = GetBlockIdx();
if (tileIdx >= mTileNum * nTileNum) return;  // 尾部核心idle
```

Benefit: 消除尾部核心空闲，在tile数不整除时性能提升10-30%
Trade-off: SK区域需要Atomic累加操作

---

## Variant Z: Grouped MatMul尾块分裂
Source: grouped_matmul

MoE场景下不同Expert的矩阵大小不同，最后一轮计算的tile可能较小。ASWT调度器支持尾块分裂(Tail-Split)：当tailSplit标志启用时，将最后一轮的尾块在M和N维度进一步分割，分配到mTailCnt×nTailCnt个核心。每个核心通过blockIdx在tailCnt网格中的位置确定自己处理的子块范围，超出范围的核心直接返回空块{0,0,0,0}。

**Expert implementation:**
```cpp
class BlockSchedulerGroupedMatmulAswt {
    __aicore__ inline BlockShape GetBlockShape(int64_t mTileIdx, int64_t nTileIdx,
                                              int64_t cureBlock, ...) {
        if (cureBlock / blockNum_ != (perCoreBlockNum_ - 1) || tailCnt_ == 1) {
            return {blockShapeM, blockShapeN, 0, 0};
        }
        // Tail block split
        int64_t singleCoreMSplit = CeilDiv(blockShapeM, static_cast<int64_t>(mTailCnt_));
        int64_t singleCoreNSplit = CeilDiv(blockShapeN, static_cast<int64_t>(nTailCnt_));
        int64_t mSplitIdx = (blockIdx_ % tailCnt_) % mTailCnt_;
        int64_t nSplitIdx = (blockIdx_ % tailCnt_) / mTailCnt_;
        if (mSplitIdx * singleCoreMSplit >= blockShapeM) return {0, 0, 0, 0};
        return {tailL1M, tailL1N, mSplitAddrOffset, nSplitAddrOffset};
    }
};
```

**vs. baseline:**
```cpp
// 简单按Expert数均分核心，小Expert的核心空闲
for (int expert = 0; expert < num_experts; expert++) {
    if (expert % num_cores == GetBlockIdx()) matmul(x[expert], w[expert], y[expert]);
}
```

Benefit: 避免尾块计算时大量核心空闲；支持变长分组的自适应负载均衡
Trade-off: 尾块分裂需要额外的地址偏移计算；NZ格式要求N维度对齐

---

## Variant 27: Pooling尾块感知核心分配
Source: max_pool_v3

MaxPool使用精确的尾块感知负载分配：将输出元素总数除以核心数，余数部分分配给前blockTail个核心（每个多处理一个元素），其余核心处理标准数量。这种"+1"分配策略确保最大负载差异仅为1个元素，实现近乎完美的负载均衡。

**Expert implementation:**
```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y) {
    if (GetBlockIdx() < tilingData_->blockTail) {
        // 前blockTail个核心多处理1个元素
        beginIdx_ = GetBlockIdx() * (tilingData_->blockFactor + 1);
        endIdx_ = beginIdx_ + tilingData_->blockFactor + 1;
    } else {
        // 其余核心处理标准数量
        beginIdx_ = GetBlockIdx() * tilingData_->blockFactor + tilingData_->blockTail;
        endIdx_ = beginIdx_ + tilingData_->blockFactor;
    }
}
```

**vs. baseline:**
```cpp
// 简单均分，最后一个核心处理所有余量
uint32_t elemPerCore = totalElements / numCores;
uint32_t start = blockIdx * elemPerCore;
uint32_t end = (blockIdx == numCores - 1) ? totalElements : start + elemPerCore;
```

Benefit: 最大负载差异仅为1个元素；无核心空闲
Trade-off: 需要Host端预计算blockFactor和blockTail
