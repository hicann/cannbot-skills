# P8: UB Memory Partitioning (UB内存分区管理)
## Overview
针对BF16（BFloat16）数据类型，专家实现采用了高精度中间计算策略。由于BF16只有8位指数和7位尾数，直接计算可能导致精度损失。专家实现通过InnerComputer模板的特化版本，在BF16场景下将数据先Cast到float32进行计算，然后再Cast回bfloat16_t。这种策略的核心优势在于：1) 计算过程使用FP32保证数值稳定性；2) 存储和传输使用BF16节省内存带宽。代价是增加了两次Cast操作的开销和额外的UB内存占用。

## When to Use
- Kernels with multiple UB tensors
- 减少Kernel中复杂的除法/取模运算，预期性能提升5-15%
- 最大化UB利用率；灵活的分块大小调整；清晰的内存布局
- BF16场景下保证数值精度和稳定性，避免低精度计算的累积误差

## Trade-off
- 占用更多UB空间存储索引buffer
- 需要维护偏移常量；代码可读性降低
- 增加两次Cast操作开销和额外UB内存占用

**Source operators**: adaptive_avg_pool3d, apply_adam_w_v2, foreach_abs, foreach_add_list, foreach_addcdiv_list, gather_elements_v2, linear_index, rms_norm_quant, scaled_masked_softmax_grad_v2

---

## Variant A: IndexBuffer预计算与复用
Source: adaptive_avg_pool3d

通过IndexBuffer预计算并缓存输入索引信息，避免重复索引计算。IndexBuffer包含六个buffer存储每个输出点对应的输入区域起止索引。以indexBufLen为批次预计算一批输出点的索引，然后在ReduceMean阶段复用。

**Expert implementation:**
```cpp
// 预计算索引到buffer
for (int64_t i = offset, j = 0; i < offset + len; ++i, ++j) {
    OutputOffsetToInputIndex(i, outputShape, inputShape, index);
    startDIndexLocal.SetValue(j, index.dstart);
    // ...
}

// 从buffer获取复用
index.dstart = startDIndexLocal.GetValue(start);
```

**vs. baseline:**
```cpp
// baseline运行时计算索引
uint32_t d_start = d_out * D_in / D_out;
uint32_t d_end = (d_out + 1) * D_in / D_out;
if ((d_out + 1) * D_in % D_out != 0) d_end += 1;
```

Benefit: 减少Kernel中复杂的除法/取模运算，预期性能提升5-15%
Trade-off: 占用更多UB空间存储索引buffer

---

## Variant B: UB内存分区与数据复用优化
Source: apply_adam_w_v2

专家实现通过精细的UB内存分区实现数据复用和计算优化。在`apply_adam_w_v2_base.h`中定义了输入/输出张量在LocalTensor中的偏移顺序，这种布局使得多个输入数据可以复用同一个LocalTensor的不同区域，减少了UB分配压力。

**Expert implementation:**
```cpp
constexpr int32_t VAR_ORDER_IN_LOCAL_TENSOR = 0;
constexpr int32_t EXP_AVG_ORDER_IN_LOCAL_TENSOR = 1;
constexpr int32_t EXP_AVG_SQ_ORDER_IN_LOCAL_TENSOR = 2;
constexpr int32_t MAX_GRAD_NORM_ORDER_IN_LOCAL_TENSOR = 3;
// ...
varOffset_ = VAR_ORDER_IN_LOCAL_TENSOR * numPerLoop_;
expAvgOffset_ = EXP_AVG_ORDER_IN_LOCAL_TENSOR * numPerLoop_;
```

**vs. baseline:**
```cpp
// Simple buffer layout
dataLocal[0] // var
dataLocal[MAX_ELEMENTS_PER_LOOP] // m
dataLocal[2 * MAX_ELEMENTS_PER_LOOP] // v
dataLocal[3 * MAX_ELEMENTS_PER_LOOP] // grad
```

Benefit: 最大化UB利用率；灵活的分块大小调整；清晰的内存布局
Trade-off: 需要维护偏移常量；代码可读性降低

---

## Variant C: BF16高精度中间计算
Source: foreach_abs

针对BF16（BFloat16）数据类型，专家实现采用了高精度中间计算策略。由于BF16只有8位指数和7位尾数，直接计算可能导致精度损失。专家实现通过InnerComputer模板的特化版本，在BF16场景下将数据先Cast到float32进行计算，然后再Cast回bfloat16_t。这种策略的核心优势在于：1) 计算过程使用FP32保证数值稳定性；2) 存储和传输使用BF16节省内存带宽。代价是增加了两次Cast操作的开销和额外的UB内存占用。

**Expert implementation:**
```cpp
template <TriangleOp<float>* op, uint8_t paramsCount>
class InnerComputer<bfloat16_t, float, op, paramsCount>
{
    __aicore__ inline void ComputePerCast(...) {
        PipeBarrier<PIPE_V>();
        Cast(float32Tensor, x1Local[index * maxCastDataCount], RoundMode::CAST_NONE, dataCount);
        PipeBarrier<PIPE_V>();
        op(float32Tensor[offset], float32Tensor, dataCount);
        PipeBarrier<PIPE_V>();
        Cast(yLocal[index * maxCastDataCount], float32Tensor[offset], RoundMode::CAST_RINT, dataCount);
    }
};
```

**vs. baseline:**
```cpp
// baseline不支持BF16
```

Benefit: BF16场景下保证数值精度和稳定性，避免低精度计算的累积误差
Trade-off: 增加两次Cast操作开销和额外UB内存占用

---

## Variant D: 内存复用与原地操作支持
Source: foreach_add_list

专家实现通过物理地址比较支持原地操作（In-Place）。当输入和输出的物理地址相同时，避免不必要的数据拷贝。通过dstLocal.GetPhyAddr() != srcLocal1.GetPhyAddr()判断，原地操作可以显著减少内存带宽使用。

**Expert implementation:**
```cpp
// 专家实现: 原地操作检测
Axpy<T, T>(srcLocal1, srcLocal2, scalarVal, uValue);
if (dstLocal.GetPhyAddr() != srcLocal1.GetPhyAddr()) {
    PipeBarrier<PIPE_V>();
    DataCopy(dstLocal, srcLocal1, uValue);
}
```

**vs. baseline:**
```cpp
// baseline: 总是分配独立输出
AscendC::GlobalTensor<float> outputGm;
outputGm.SetGlobalBuffer((__gm__ float *)output + coreOffset, elementsPerCore);
```

Benefit: 减少内存带宽使用，支持链式算子优化
Trade-off: 需要运行时地址比较

---

## Variant E: 统一UB大小管理
Source: foreach_add_list

专家实现通过ForeachCommonTilingData统一管理Unified Buffer (UB)大小。Tiling阶段计算inputsTensorUbSize，Kernel阶段根据数据类型计算maxDataCount = inputsTensorUbSize / sizeof(T)。对于BF16类型，使用COPY_SPACE_MULTIPLE = 9的扩展因子。

**Expert implementation:**
```cpp
// 专家实现: 动态UB管理
constexpr uint8_t COPY_SPACE_MULTIPLE = 9;
if (std::is_same_v<T, bfloat16_t>) {
    totalTensorUbSize = inputsTensorUbSize * COPY_SPACE_MULTIPLE;
    maxDataCount = totalTensorUbSize / sizeof(T);
    maxCastDataCount = inputsTensorUbSize / sizeof(float);
} else {
    maxDataCount = inputsTensorUbSize / sizeof(T);
}
```

**vs. baseline:**
```cpp
// baseline: 固定tile size
const uint32_t DEFAULT_TILE_SIZE = 2048;
uint32_t tileSize = DEFAULT_TILE_SIZE;
```

Benefit: UB使用高效且安全，支持不同数据类型的最优tile大小
Trade-off: Tiling数据结构设计复杂

---

## Variant F: 原地修改检测与避免冗余拷贝
Source: foreach_addcdiv_list

专家实现通过dstLocal.GetPhyAddr() != tensor1Local.GetPhyAddr()检测是否为原地修改（即输出与输入指向同一块内存）。如果是原地修改，则跳过最终的DataCopy操作，避免冗余的内存拷贝。这不仅提高了性能，也减少了可能的精度损失（因为减少了一次内存读写）。baseline实现没有这种优化，总是执行完整的拷贝流程。

**Expert implementation:**
```cpp
// 专家实现: 原地修改检测
if (dstLocal.GetPhyAddr() != tensor1Local.GetPhyAddr()) {
    PipeBarrier<PIPE_V>();
    if (uValue * sizeof(T) % ADDCDIV_LIST_BYTE_PER_BLOCK == 0) {
        DataCopy(dstLocal, tensor1Local, uValue);
    } else {
        // ...
    }
}
```

**vs. baseline:**
```cpp
// baseline: 总是执行拷贝
AscendC::Add(outLocal, inputLocal, divLocal, current_tile_size);
// 总是执行DataCopy
AscendC::DataCopy(outputGm[tile_offset], outLocal, current_tile_size);
```

Benefit: 避免冗余拷贝，提升性能，减少精度损失风险
Trade-off: 需要额外的地址比较操作

---

## Variant G: UB内存精细管理
Source: gather_elements_v2

专家实现对UB内存的管理远超baseline的简单容量计算。关键优化包括：保留RESERVED_UB_SIZE字节作为安全边界、所有buffer大小都按照BLOCK_SIZE字节对齐确保向量指令正确执行、同时考虑Transpose模式和Gather模式的buffer需求取最大值、对于大tensor在gather维度上进行切片控制单次处理的数据量。

**Expert implementation:**
```cpp
inline bool GatherElementsV2Tiling::CalcMaxBufferSize() {
    carryNumAlign_ = CACHELINE / xDtypeSize_;
    uint64_t xAlign = BLOCK_SIZE / xDtypeSize_;
    uint64_t availableUb = ubSize_ - RESERVED_UB_SIZE;
    
    uint64_t gatherInBufferSize = Ops::Base::CeilAlign(xGatherDim_, xAlign) * xDtypeSize_ +
                                  Ops::Base::CeilAlign(minIdxGatherDimSlice, idxAlign) * sizeof(int32_t) * NUM_TWO;
    uint64_t transInBufferSize = TRANSPOSE_WS_LEN * CACHELINE;
    inBufferSize_ = std::max(gatherInBufferSize, transInBufferSize);
    outBufferSize_ = std::max(gatherOutBufferSize, transOutBufferSize);
    // ...
}
```

**vs. baseline:**
```cpp
uint32_t dtype_x_size = 4;   // float32
uint32_t mem_per_row = x_gather_dim * dtype_x_size + idx_gather_dim * dtype_idx_size + idx_gather_dim * dtype_x_size;
uint32_t tile_rows = (mem_per_row > 0) ? (UB_CAPACITY / mem_per_row) : 1;
```

Benefit: 最大化UB利用率，支持更大的数据规模
Trade-off: 增加了内存计算的复杂度

---

## Variant H: UB大小感知的数据分块
Source: linear_index

专家实现基于实际UB大小动态计算maxSize，确保数据不会spill到GM。baseline固定tileSize=2048，在不同数据类型和场景下可能导致UB溢出或利用率不足。专家实现从platform info获取实际UB大小，根据indices数据类型大小动态计算maxSize。

**Expert implementation:**
```cpp
auto ubSizePlatForm = compileInfo->ubSizePlatForm;
max_ub = ubSizePlatForm / max_ub * max_ub / BUFFER_NUM;
maxSize = max_ub / indicesSize;
if (eachCount > maxSize) {
    eachNum = maxSize;
    eachLoop = (eachCount + maxSize - 1) / maxSize;
    eachTail = eachCount - (eachLoop - 1) * eachNum;
}
```

**vs. baseline:**
```cpp
uint32_t tileSize = 2048;
uint32_t innerLoops = (elementsPerCore + tileSize - 1) / tileSize;
```

Benefit: 确保数据完全驻留在UB中，避免昂贵的GM spill
Trade-off: 需要获取platform信息，增加了tiling复杂度

---

## Variant I: Buffer池化复用
Source: rms_norm_quant

专家实现通过BUF_FACTOR和offset管理策略实现Buffer的高效复用。在同一个calc_buf_中，通过OFFSET_GAMMA（0）、OFFSET_SQX（1）、OFFSET_SUM（2）等偏移量复用内存，将原本需要3个独立buffer的存储需求压缩到1个。这种设计在UB有限的场景下尤为重要。

**Expert implementation:**
```cpp
static constexpr uint32_t BUF_FACTOR = 3;
static constexpr uint32_t OFFSET_GAMMA = 0;
static constexpr uint32_t OFFSET_SQX = 1;
static constexpr uint32_t OFFSET_SUM = 2;

pipe.InitBuffer(calc_buf_, BUF_FACTOR * num_col_align_f32 * sizeof(float) + 32);

AscendC::LocalTensor<float> g = buf[OFFSET_GAMMA * num_col_align_f32];
AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * num_col_align_f32];
AscendC::LocalTensor<float> work = buf[OFFSET_SUM * num_col_align_f32];
```

**vs. baseline:**
```cpp
AscendC::TBuf<AscendC::TPosition::VECCALC> xSqBuf;
AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuf;
AscendC::TBuf<AscendC::TPosition::VECCALC> sharedBuf;
AscendC::TBuf<AscendC::TPosition::VECCALC> outBuf;
AscendC::TBuf<AscendC::TPosition::VECCALC> quantBuf;
```

Benefit: UB内存利用率提升3倍以上，可处理更大规模数据
Trade-off: 代码可读性略有下降，需要仔细管理offset

---

## Variant J: UB内存精细化管理
Source: scaled_masked_softmax_grad_v2

专家实现展示了UB内存的精细化管理，通过详细的计算确定buffer大小。使用SoftMaxGradTilingFunc计算softmaxGrad所需的临时buffer大小，根据dataType（2B/4B）和lineNum动态计算buffer需求，sharedBuffer被多个阶段复用，paddedHeadDim确保64字节对齐。

**Expert implementation:**
```cpp
uint64_t oneLineSoftmaxGradSize = AscendC::GetSoftMaxGradMaxTmpSize(srcShape, SIZE_4, false, false);
usedUbSize = paddedHeadDim * ((REQUIRED_INPUT_NUM + REQUIRED_OUTPUT_NUM) * dataSize +
                              SIZE_4 * (dataType == SIZE_2 ? REQUIRED_INPUT_NUM : 0)) +
             paddedHeadDim + oneLineSoftmaxGradSize;
maxLinePerLoop = ubSize / usedUbSize;
AscendC::SoftMaxGradTilingFunc(srcShape, SIZE_4, selectSize, tiling.softmaxGradTilingData, false);
```

**vs. baseline:**
```cpp
pipe.InitBuffer(inQueueGradOutput, 1, tileSize * sizeof(float));
pipe.InitBuffer(inQueueSoftmaxOutput, 1, tileSize * sizeof(float));
pipe.InitBuffer(outQueueGradInput, 1, tileSize * sizeof(float));
pipe.InitBuffer(gradSoftmaxBuf, tileSize * sizeof(float));
pipe.InitBuffer(tmpBuf, tileSize * sizeof(float));
```

Benefit: 最大化UB利用率，支持更大的tiling块，减少循环次数
Trade-off: Host端Tiling计算复杂，需要理解内部API的buffer需求
