# P1: Double Buffering (双缓冲机制)
## Overview
专家实现充分运用了昇腾处理器的多级流水线架构，通过PipeBarrier指令精确控制不同流水阶段的同步，并在关键路径上启用双缓冲机制：1）PipeBarrier精细控制：在Vector指令之间插入PipeBarrier<PIPE_V>()确保数据依赖，在跨单元操作时使用PipeBarrier<PIPE_ALL>()进行全局同步；2）双缓冲机制：xQueue和yQueue使用DOUBLE_BUFFER（=2）配置，使得数据搬入、计算、搬出可以流水线并行，隐藏数据传输延迟；3）事件同步机制：在FullReduce模板中，使用HardEvent::MTE2_S和HardEvent::V_S等事件机制实现MTE2（数据搬入）与Scalar、Vector之间的异步流水线；4）队列资源精细管理：输入队列（VECIN）和输出队列（VECOUT）分开管理，根据数据流向精确配置队列深度。

## When to Use
- Any memory-bound kernel
- 避免一种实现应对所有场景的问题，每种模式可以针对特定场景进行深度优化
- 内存访问与计算重叠，提高流水线利用率，特别适合大数据量场景
- 隐藏内存访问延迟，吞吐量提升20-30%

## Trade-off
- 代码量增加，需要维护多个kernel变体
- UB内存占用增加
- UB占用翻倍；代码复杂度增加

**Source operators**: add_rms_norm_cast, apply_adam_w_v2, ascend_quant_v2, batch_norm_v3, clipped_swiglu, conv2d_v2, deep_norm, dequant_bias, dynamic_block_quant, dynamic_mx_quant, dynamic_quant_update_scatter_v2, embedding_dense_grad_v2, fake_quant_affine_cachemask, flash_attention_score, foreach_abs, foreach_add_list, foreach_add_scalar, foreach_add_scalar_list, foreach_addcdiv_list, fused_floyd_attention, grouped_dynamic_mx_quant, inplace_add_rms_norm, layer_norm_v3, layer_norm_v4, modulate, multi_scale_deformable_attn_function, rms_norm_grad, scaled_masked_softmax_grad_v2, scatter_elements_v2, trans_quant_param_v2

---

## Variant A: 双缓冲（Double Buffering）机制
Source: foreach_abs, foreach_add_list, foreach_add_scalar, foreach_add_scalar_list, foreach_addcdiv_list

专家实现采用了双缓冲机制来隐藏数据传输延迟。通过设置BUFFER_NUM = 2，在UB中同时维护两个输入缓冲区和两个输出缓冲区。这样可以在计算当前数据块的同时，预取下一个数据块到另一个缓冲区，实现计算与数据传输的流水线并行。TQue<QuePosition::VECIN, bufferNum>和TQue<QuePosition::VECOUT, bufferNum>的声明明确了双缓冲的使用。相比之下，baseline实现仅使用了单缓冲（TQue<..., 1>），无法有效隐藏访存延迟，在数据密集型场景下会成为性能瓶颈。

**Expert implementation:**
```cpp
static constexpr int32_t BUFFER_NUM = 2;
TQue<QuePosition::VECIN, bufferNum> dataQueue;
TQue<QuePosition::VECOUT, bufferNum> outQueue;
Base::pipe.InitBuffer(dataQueue, bufferNum, Base::inputsTensorUbSize);
if (needCopyOut) {
    Base::pipe.InitBuffer(outQueue, bufferNum, Base::inputsTensorUbSize);
}
```

**vs. baseline:**
```cpp
AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueue;   // buffer数量为1
AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueue; // buffer数量为1
pipe.InitBuffer(inQueue, 1, this->tileSize * sizeof(float));
pipe.InitBuffer(outQueue, 1, this->tileSize * sizeof(float));
```

Benefit: 隐藏数据搬运延迟，提高指令级并行性，预计性能提升10-20%; 数据搬运与计算并行执行，提升吞吐量约20-30%; 隐藏数据传输延迟，提高计算单元利用率，显著提升内存密集型算子性能; 流水线并行可将性能提升30-50%，特别是在数据量大、计算相对简单的情况下; 隐藏内存传输延迟，实现计算与数据传输并行，提升整体吞吐量
Trade-off: 增加UB内存占用（需要两个缓冲区）; UB使用量增加一倍; UB内存占用增加一倍，需要更精细的内存管理; 需要更多的UB内存; 增加了UB内存使用量（约2倍）

---

## Variant B: 双缓冲流水线设计
Source: dynamic_mx_quant, grouped_dynamic_mx_quant

专家实现采用经典的双缓冲（Double Buffering）技术，通过DB_BUFFER = 2定义缓冲区数量。核心优势在于计算与数据传输重叠：当Vector单元在计算当前数据块时，DMA可以同时搬运下一个数据块。通过预取机制将数据搬运延迟隐藏在计算过程中。N_BUFFER / EXIST_NODE_NUM的计算方式合理分配UB空间给输入、输出和中间缓冲区。三个队列（inQueue_, mxScaleQueue_, outQueue_）都初始化为双缓冲模式，实现了高效的流水线执行。

**Expert implementation:**
```cpp
constexpr int64_t N_BUFFER = 2;
constexpr int64_t EXIST_NODE_NUM = 3;

TPipe pipe_;
TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue_;
TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;

int64_t maxUbAvailable = tilingParam.ubSize / N_BUFFER / EXIST_NODE_NUM;
tilingParam.maxUbCol = static_cast<int64_t>(maxUbAvailable / static_cast<int64_t>(tilingParam.vfLen) / 
    (tilingParam.blockSize*DIGIT_TWO) * (tilingParam.blockSize*DIGIT_TWO));

this->pipe_.InitBuffer(this->inQueue_, DB_BUFFER, bufferSize_);
this->pipe_.InitBuffer(this->mxScaleQueue_, DB_BUFFER, bufferSize_);
this->pipe_.InitBuffer(this->outQueue_, DB_BUFFER, bufferSize_);
```

**vs. baseline:**
```cpp
TPipe pipe_;
TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;

int64_t bufferSize_ = this->ubFactor_ * this->maxUbCol_ * sizeof(T);
this->pipe_.InitBuffer(this->inQueue_, DB_BUFFER, bufferSize_);
```

Benefit: 隐藏内存传输延迟，提升 10-20% 性能; 计算与传输重叠，隐藏内存延迟；最大化Vector单元利用率；高效UB内存分配
Trade-off: UB 内存占用翻倍，每个 buffer 需要分配两份空间; UB内存需求翻倍；代码复杂度增加

---

## Variant C: 多模式自适应计算策略
Source: add_rms_norm_cast

专家实现根据输入数据的规模特征，设计了四种计算模式：Normal模式适用于中等规模行数；SplitD模式适用于大hidden size（列数超过UB容量），列方向分块处理；SingleN模式适用于小batch场景（每核只处理1行），单核单行的极致优化；MultiN模式适用于小batch但需要处理多行的场景，使用双缓冲最大化吞吐量。选择逻辑为：numCol > ubFactor时选择SplitD，blockFactor == 1时选择SingleN，其他情况选择Normal或MultiN。

**Expert implementation:**
```cpp
// 多模式选择
if (numCol > ubFactor) {
    modeKey = MODE_SPLIT_D;
    ubFactor = (data_type == ge::DT_FLOAT) ? UB_FACTOR_B32_CUTD : UB_FACTOR_B16_CUTD;
} else if (blockFactor == 1 && socVersion != ASCEND310P) {
    modeKey = MODE_SINGLE_N;
} else if (data_type == ge::DT_FLOAT16 && numCol == numColAlign) {
    modeKey = MODE_NORMAL;
    ubFactor = UB_FACTOR_B16;
}

// kernel端分发
if (TILING_KEY_IS(10)) {
    GENERAL_OP_IMPL(KernelAddRmsNormCast, half);
} else if (TILING_KEY_IS(13)) {
    GENERAL_OP_IMPL(KernelAddRmsNormCastSingleN, half);
}
```

**vs. baseline:**
```cpp
// 简单Tiling策略
uint32_t tileLength = (MAX_TILE_LEN < hiddenSize) ? MAX_TILE_LEN : hiddenSize;
uint32_t nTiles = (hiddenSize + tileLength - 1) / tileLength;

for (uint32_t rowIdx = 0; rowIdx < this->rowsPerCore; rowIdx++) {
    for (uint32_t tileId = 0; tileId < this->nTiles; tileId++) {
        CopyInPass1(rowIdx, tileId);
        ComputePass1();
    }
}
```

Benefit: 避免一种实现应对所有场景的问题，每种模式可以针对特定场景进行深度优化
Trade-off: 代码量增加，需要维护多个kernel变体

---

## Variant D: 双缓冲(Doble Buffering)优化
Source: add_rms_norm_cast

在MultiN模式中，专家实现使用了双缓冲技术（DOUBLE_BUFFER_NUM = 2），允许在计算当前数据块的同时，预取下一个数据块。输入队列inQueueX深度设为2，可以容纳两个数据块；输出队列outQueueY深度也设为2。通过PipeBarrier和事件同步确保数据依赖正确。这种设计有效地隐藏了内存访问延迟，内存访问与计算重叠，提高流水线利用率。

**Expert implementation:**
```cpp
// MultiN模式双缓冲
Ppipe->InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
Ppipe->InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, ubFactor * sizeof(T));
```

**vs. baseline:**
```cpp
// 单缓冲
pipe.InitBuffer(inQueueX, 1, this->tileLength * sizeof(half));
pipe.InitBuffer(inQueueResidual, 1, this->tileLength * sizeof(half));
pipe.InitBuffer(outQueue, 1, this->tileLength * sizeof(half));
```

Benefit: 内存访问与计算重叠，提高流水线利用率，特别适合大数据量场景
Trade-off: UB内存占用增加

---

## Variant E: 双缓冲(Double Buffer)与流水线并行
Source: apply_adam_w_v2

专家实现全面采用双缓冲机制(`BUFFER_NUM = 2`)，实现了数据搬运与计算的流水线并行。在`CopyIn`、`Compute`、`CopyOut`三个阶段，通过乒乓缓冲(Queue0/Queue1交替使用)使得当前循环的计算可以与下一个循环的数据搬运重叠执行。这种设计显著隐藏了内存访问延迟，提升了整体吞吐量。

**Expert implementation:**
```cpp
constexpr int32_t BUFFER_NUM = 2;  // Double buffer
pipe_.InitBuffer(inQueue_, BUFFER_NUM, IN_BUFFER_NUM * numPerLoop_ * sizeof(T));
// Process中的流水线处理
for (int64_t n = 0; n < curLoopCount; n++) {
    CopyIn(n, numPerLoop_);      // Data copy (parallel with prev compute)
    Compute(numPerLoop_);         // Compute
    CopyOut(n, numPerLoop_);      // Write back (parallel with next compute)
}
```

**vs. baseline:**
```cpp
constexpr int32_t BUFFER_NUM = 1; // Single buffer
pipe_.InitBuffer(inQueue_, BUFFER_NUM, 4 * MAX_ELEMENTS_PER_LOOP * sizeof(float));
```

Benefit: 隐藏内存访问延迟，吞吐量提升20-30%
Trade-off: UB占用翻倍；代码复杂度增加

---

## Variant F: 流水线优化与双缓冲
Source: ascend_quant_v2

大量使用PipeBarrier<PIPE_V>()进行指令流水线同步，确保Vector引擎数据依赖性正确处理，相比全局同步有更低开销。使用DataCopyPad代替DataCopy，支持自动填充和非对齐访问。对于FP16类型设计专门的cast缓冲区，配合MTE2_V事件同步，实现DMA搬运和Vector类型转换的流水线并行。

**Expert implementation:**
```cpp
// 流水线同步优化
DataCopyPad(sLocal, scaleGm_[sInOffset], copyParams, {false, 0, 0, 0});
event_t eventID = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
SetFlag<HardEvent::MTE2_V>(eventID);
WaitFlag<HardEvent::MTE2_V>(eventID);
Cast(castSLocal, sLocal, RoundMode::CAST_NONE, sLen);
PipeBarrier<PIPE_V>();

// Vector计算流水线
Mul(castXLocal, castXLocal, sLocal, dataCount);
PipeBarrier<PIPE_V>();
if constexpr (SQRT_MODE) {
    Mul(castXLocal, castXLocal, sLocal, dataCount);
    PipeBarrier<PIPE_V>();
}
```

**vs. baseline:**
```cpp
// 简单顺序执行
AscendC::DataCopy(inputLocal, inputGm[offset], this->tileSize);
AscendC::Adds(tempSubLocal, inputLocal, -zeroPointVal, this->tileSize);
AscendC::Muls(tempDivLocal, tempSubLocal, invScale, this->tileSize);
```

Benefit: 提高指令级并行度，减少流水线气泡，最大化硬件利用率
Trade-off: 需要精确的同步点设计，调试复杂度增加

---

## Variant G: 流水线与双缓冲优化
Source: batch_norm_v3

专家实现充分运用了昇腾处理器的多级流水线架构，通过PipeBarrier指令精确控制不同流水阶段的同步，并在关键路径上启用双缓冲机制：1）PipeBarrier精细控制：在Vector指令之间插入PipeBarrier<PIPE_V>()确保数据依赖，在跨单元操作时使用PipeBarrier<PIPE_ALL>()进行全局同步；2）双缓冲机制：xQueue和yQueue使用DOUBLE_BUFFER（=2）配置，使得数据搬入、计算、搬出可以流水线并行，隐藏数据传输延迟；3）事件同步机制：在FullReduce模板中，使用HardEvent::MTE2_S和HardEvent::V_S等事件机制实现MTE2（数据搬入）与Scalar、Vector之间的异步流水线；4）队列资源精细管理：输入队列（VECIN）和输出队列（VECOUT）分开管理，根据数据流向精确配置队列深度。

**Expert implementation:**
```cpp
// 双缓冲配置
constexpr static uint32_t DOUBLE_BUFFER = 2;
this->pipe_->InitBuffer(xQueue, DOUBLE_BUFFER, r0UbFactor * FLOAT_SIZE);
this->pipe_->InitBuffer(yQueue, DOUBLE_BUFFER, r0UbFactor * FLOAT_SIZE);

// 流水线同步
Adds(xTensor, xTensor, -finalMean, r0ProcNum);
PipeBarrier<PIPE_V>();
Muls(xTensor, xTensor, weightMulInvstd, r0ProcNum);
PipeBarrier<PIPE_V>();

// 事件同步
TEventID eventIdVtoS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
SetFlag<HardEvent::V_S>(eventIdVtoS);
WaitFlag<HardEvent::V_S>(eventIdVtoS);
```

**vs. baseline:**
```cpp
# baseline无显式同步
with tl.copyin():
    tl.load(input_ptr + offsets, data_ub)
with tl.compute():
    tl.vsub_scalar(temp_ub, data_ub, mean_val)
with tl.copyout():
    tl.store(output_ptr + offsets, data_ub)
```

Benefit: 双缓冲隐藏数据传输延迟，提升整体吞吐约30%；事件机制减少不必要的同步等待
Trade-off: UB内存占用翻倍；同步逻辑复杂易出错

---

## Variant H: 双缓冲与软件流水线
Source: clipped_swiglu

专家实现采用双缓冲（Double Buffering）机制，通过设置DB_BUFFER = 2，使得数据搬运（CopyIn）和计算（Compute）可以并行执行。当Core正在计算当前tile的数据时，DMA可以并行搬运下一个tile的数据到另一个buffer，实现计算-搬运的流水线重叠。关键设计包括：1)双Buffer声明TQue<QuePosition::VECIN, DB_BUFFER> xQueue_；2)Ping-Pong机制：AllocTensor自动在buffer 0和buffer 1之间切换；3)PipeBarrier同步：在Compute函数的关键位置插入PipeBarrier<PIPE_V>()，确保向量指令的顺序执行。

**Expert implementation:**
```cpp
constexpr static int64_t DB_BUFFER = 2;
TQue<QuePosition::VECIN, DB_BUFFER> xQueue_;
pipe_->InitBuffer(xQueue_, DB_BUFFER, xQueSpace_);

// Compute中的PipelineBarrier使用
Mins(tmpB, tmpB, tl_->gluLimit, calPairNum_);
PipeBarrier<PIPE_V>();
Maxs(tmpB, tmpB, -1 * tl_->gluLimit, calPairNum_);
PipeBarrier<PIPE_V>();
```

**vs. baseline:**
```cpp
// 单缓冲
AscendC::TQue<AscendC::TPosition::VECIN, 1> x1Queue;
pipe.InitBuffer(x1Queue, 1, tileSize * sizeof(float));

__aicore__ inline void Process() {
    for (uint32_t i = 0; i < innerLoops; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}
```

Benefit: 双缓冲可隐藏数据搬运延迟，理想情况下性能提升接近2倍
Trade-off: 增加UB内存使用（需要2倍的buffer空间）

---

## Variant I: UB内存精确计算与自适应Tiling
Source: clipped_swiglu

专家实现不是使用固定的TILE_SIZE，而是根据实际硬件的UB大小、数据类型、双缓冲需求等因素，精确计算UB可以容纳的最大pair数（ubMaxPair_）。计算公式：ubMaxPair_ = ((numerator / (xBuffer + yBuffer + tmpBuffer1 + tmpBuffer2) / BLOCK_SIZE * BLOCK_SIZE) - (BLOCK_SIZE - 1)) / SIZE_OF_BF16_FP16。其中numerator是UB总大小减去预留空间和group_index buffer，最后除以2得到pair数。

**Expert implementation:**
```cpp
ge::graphStatus CountMaxPair() {
    ubMaxPair_ = 1;
    int64_t xBuffer = DB_BUFFER * SWI_FACTOR * DTYPE_FACTOR;
    int64_t yBuffer = DTYPE_FACTOR;
    int64_t numerator = static_cast<int64_t>(ubSize_) - UB_RESERVE - groupindexBuffer;
    ubMaxPair_ = ((numerator / (xBuffer + yBuffer + tmpBuffer1 + tmpBuffer2) / BLOCK_SIZE * BLOCK_SIZE) - (BLOCK_SIZE - 1)) / SIZE_OF_BF16_FP16;
    isLongH_ = ubMaxPair_ * SWI_FACTOR < dim2H_ ? 1 : 0;
    return ge::GRAPH_SUCCESS;
}
```

**vs. baseline:**
```cpp
const uint32_t TILE_SIZE = 2048;
uint32_t tileSize = TILE_SIZE;
uint32_t innerLoops = elementsPerCore / tileSize;
```

Benefit: UB利用率最大化，适应不同硬件平台和数据规模
Trade-off: Tiling逻辑复杂度增加

---

## Variant J: 流水线与双缓冲优化
Source: deep_norm

专家实现大量使用 PipeBarrier、SetFlag、WaitFlag 等指令进行精细的流水线控制。这些指令用于同步 Vector 单元、Scalar 单元和 MTE 之间的执行，确保数据依赖正确的同时最大化指令级并行。通过 BUFFER_NUM = 1 配置和队列的 Ping-Pong 机制，实现了数据传输和计算的重叠，有效隐藏内存访问延迟。

**Expert implementation:**
```cpp
// 专家实现 - 精细流水线控制
Cast(local_y_fp32, x_local, RoundMode::CAST_NONE, stepSize);
PipeBarrier<PIPE_V>();
Axpy(local_x_fp32, local_y_fp32, alphaVal, stepSize);
PipeBarrier<PIPE_V>();
// 标量同步
SetFlag<HardEvent::V_S>(EVENT_ID0);
WaitFlag<HardEvent::V_S>(EVENT_ID0);
float mean_local_temp = ReduceSumCustom(local_y_fp32[offset], num_last_dim);
event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(event_v_s);
WaitFlag<HardEvent::V_S>(event_v_s);
```

**vs. baseline:**
```cpp
// baseline - 无显式流水线控制
AscendC::DataCopyPad(inputLocal, inputGm[offset], ...);
inQueueX.EnQue(inputLocal);
AscendC::LocalTensor<float> inputTensor = inQueueX.DeQue<float>();
AscendC::ReduceSum(sharedTensor, inputTensor, sharedTensor, actualTileLen);
```

Benefit: 最大化指令级并行，隐藏内存延迟
Trade-off: 代码复杂度显著增加，需要深入理解硬件流水线

---

## Variant K: 双缓冲（Ping-Pong Buffer）流水线
Source: dequant_bias

专家实现采用了经典的双缓冲技术，通过pingBuf_和pongBuf_两个缓冲区交替工作，实现数据传输（MTE2）和计算（VECTOR）的流水线并行。核心机制包括事件标志同步、交替执行、Ping-Pong ID管理。这种设计能够显著隐藏数据传输延迟，在计算密集型场景下可提升20-30%的有效吞吐。

**Expert implementation:**
```cpp
// 专家实现：双缓冲流水线
event_t pongId_{EVENT_ID7};
event_t pingId_{EVENT_ID6};

for (uint32_t idx = 0; idx < curCoreLoops_; idx++) {
    auto pipeId = (idx % 2 == 0) ? pingId_ : pongId_;
    LocalTensor<XTYPE> xLocal = (idx % 2 == 0) ? pingBuf_.Get<XTYPE>() : pongBuf_.Get<XTYPE>();
    
    WaitFlag<HardEvent::MTE3_MTE2>(pipeId);  // 等待数据就绪
    CopyIn(xLocal, groupRowOffset, inRows);
    SetFlag<HardEvent::MTE2_V>(pipeId);       // 标记数据已载入
    
    WaitFlag<HardEvent::MTE2_V>(pipeId);      // 等待计算完成
    ComputeDequant(xLocal, inRows, idx * curCoreLoopRow_);
    SetFlag<HardEvent::V_MTE3>(pipeId);       // 标记计算完成
}
```

**vs. baseline:**
```cpp
// baseline：单缓冲串行执行
for (uint32_t i = 0; i < innerLoops; i++) {
    CopyIn(tileStart);    // 串行：数据拷贝
    Compute();            // 串行：计算
    CopyOut(tileStart);   // 串行：结果写回
}
```

Benefit: 隐藏数据传输延迟，提升20-30%吞吐
Trade-off: 代码复杂度增加，需要精确的事件管理

---

## Variant L: 流水线同步与双缓冲
Source: dynamic_block_quant

专家实现通过精细的事件同步机制实现了计算-数据传输的流水线化。核心设计使用四个事件：MTE2_V（数据加载完成）、V_MTE2（计算完成，准备下一次加载）、V_MTE3（计算完成，准备写回）、MTE3_V（写回完成）。通过SetFlag/WaitFlag指令对，可以实现CopyIn、Compute、CopyOut三个阶段的重叠执行。例如，当当前块在Compute时，可以并行加载下一块数据；当Compute完成写回当前结果时，下一块已经在计算。这种双缓冲设计隐藏了内存访问延迟，使得向量单元利用率最大化。

**Expert implementation:**
```cpp
SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
Compute(perLoopRow * colPadNum, rowIdx * colBlockTotalNum);
SetFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
WaitFlag<HardEvent::V_MTE3>(eventIdVToMTE3);
CopyOutY(rowIdx, perLoopRow);
```

**vs. baseline:**
```cpp
CopyIn(rowOffset, blk);
Compute(blk);
CopyOut(rowOffset, blk);
```

Benefit: 隐藏内存延迟，最大化向量单元利用率，显著提升性能
Trade-off: 同步指令开销，增加代码复杂度，需要精细设计避免死锁

---

## Variant M: UB大小精确计算
Source: dynamic_block_quant

专家实现根据输入数据类型和block size精确计算UB需求。对于fp16输入：需要考虑input buffer（fp16）、tmp buffer（fp32用于计算）、output buffer（int8/fp8）、scale buffer（fp32）。对于bf16输入：计算类型是float，需要更大的tmp buffer。计算还考虑了block size的组合（row * col），以及双缓冲（N_BUFFER=2）的需求。通过精确计算，可以确定单次UB循环可以处理多少个block（maxUbAvailable），从而优化数据传输次数。这种精确计算避免了UB溢出或利用率不足的问题。

**Expert implementation:**
```cpp
inline static int64_t CalcPerBlockUbSize(DataType inputType, DynamicBlockQuantTilingParam& tilingParam) {
    int64_t perBlockTmpUbSize = 0;
    if (tilingParam.blockSizeRow == 1) {
        perBlockTmpUbSize += tilingParam.blockSizeRow * tilingParam.blockSizeCol 
            * (BYTES_OF_INPUT_TYPE + BYTES_OF_OUTPUT_TYPE);
        perBlockUbSize = perBlockTmpUbSize + BLOCK_SIZE;
    } else if (inputType == DT_FLOAT16) {
        perBlockTmpUbSize += tilingParam.blockSizeRow * tilingParam.blockSizeCol 
            * (BYTES_OF_INPUT_TYPE * DIGIT_TWO + BYTES_OF_OUTPUT_TYPE);
        perBlockUbSize = perBlockTmpUbSize + BLOCK_SIZE / BYTES_OF_INPUT_TYPE * BYTES_OF_FLOAT_TYPE + BLOCK_SIZE;
    } else {
        perBlockTmpUbSize += tilingParam.blockSizeRow * tilingParam.blockSizeCol 
            * (BYTES_OF_INPUT_TYPE + BYTES_OF_OUTPUT_TYPE + BYTES_OF_FLOAT_TYPE);
        perBlockUbSize = perBlockTmpUbSize + BLOCK_SIZE * DIGIT_TWO;
    }
    return perBlockUbSize;
}
int64_t maxUbAvailable = (tilingParam.ubSize - RESERVED_UB_SIZE)/ N_BUFFER / perBlockUbSize;
```

**vs. baseline:**
```cpp
// 简单固定大小
pipe.InitBuffer(inQueue, 1, blockSize * sizeof(float));
pipe.InitBuffer(outQueueQuantized, 1, blockSize * sizeof(int8_t));
```

Benefit: 最大化UB利用率，避免溢出，优化数据传输
Trade-off: 计算复杂，需要理解各种数据类型的buffer需求

---

## Variant N: 自适应双缓冲(Adaptive Double Buffering)
Source: dynamic_quant_update_scatter_v2

专家实现采用自适应双缓冲策略，根据UB内存容量和数据特征动态决定是否启用双缓冲。核心思想是通过比较'单次处理多行(无DB)'和'分批次处理(有DB)'的内存占用，选择更高效的策略。当UB能够一次性容纳所有行数据时(maxRow > rowPerHeadCore)，选择无DB模式以减少同步开销；否则启用DB模式以隐藏内存传输延迟。这种自适应策略避免了固定的双缓冲带来的额外内存开销，同时确保了大数据量场景下的流水线效率。

**Expert implementation:**
```cpp
// Tiling阶段: 自适应双缓冲决策
bool useDb = true;
uint32_t multiRowNum = maxRowDB;

// 如果UB足够大，可以一次性处理所有行，则不使用双缓冲
if (maxRow > rowPerHeadCore) {
    useDb = false;
    multiRowNum = maxRow;
}
SetTilingKey(context, xDtype, useDb);

// Kernel阶段: 根据Tiling Key选择实现
if (TILING_KEY_IS(0) || TILING_KEY_IS(1)) {
    DynamicQuantUpdateScatterV2<DTYPE_X, DTYPE_VAR> op(&pipe);  // 无DB版本
    op.Init(x, indices, var, varScale, varOffset, workSpace, &tilingData);
    op.Process();
} else if (TILING_KEY_IS(2) || TILING_KEY_IS(3)) {
    DynamicQuantUpdateScatterV2DbOpt<DTYPE_X, DTYPE_VAR> op(&pipe);  // DB版本
    op.Init(x, indices, var, varScale, varOffset, workSpace, &tilingData);
    op.Process();
}
```

**vs. baseline:**
```cpp
// baseline可能固定使用单缓冲或双缓冲
constexpr uint32_t BUFFER_NUM = 2;
TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
```

Benefit: 根据实际内存情况选择最优策略，避免不必要的同步开销或内存浪费
Trade-off: 需要维护两套实现代码，增加开发和维护成本

---

## Variant O: 异步流水线同步(Asynchronous Pipeline Synchronization)
Source: dynamic_quant_update_scatter_v2

专家实现充分利用Ascend C的异步流水线机制，通过显式的事件标志(Event Flag)控制计算与数据传输的并行执行。在双缓冲版本中，使用HardEvent::MTE3_S(MTE3到Scalar)和HardEvent::MTE3_V(MTE3到Vector)事件实现Copy与Compute的并行。通过SetFlag/WaitFlag机制，确保数据依赖的正确性同时最大化流水线并行度。例如，在计算当前行量化参数的同时，可以并行启动下一行的数据拷贝。这种细粒度的同步控制相比全局同步屏障(PipeBarrier)具有更低的同步开销。

**Expert implementation:**
```cpp
// 双缓冲版本中的流水线同步
event_t eventMTE3S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
SetFlag<HardEvent::MTE3_S>(eventMTE3S);
event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
SetFlag<HardEvent::MTE3_V>(eventMTE3V);

for (uint64_t i = 0; i < multiRow; i++) {
    // ... 计算逻辑 ...
    
    if (unlikely(i == 0)) {
        WaitFlag<HardEvent::MTE3_S>(eventMTE3S);  // 首次迭代等待初始化完成
    }
    
    event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event_v_mte3);
    WaitFlag<HardEvent::V_MTE3>(event_v_mte3);
    
    // Copy out...
}
```

**vs. baseline:**
```cpp
// baseline可能使用全局同步
PipeBarrier<PIPE_ALL>();
DataCopy(inLocal, inGm, len);
PipeBarrier<PIPE_ALL>();
Compute(inLocal);
```

Benefit: 最大化流水线并行度，隐藏数据传输延迟，提升整体吞吐
Trade-off: 编程复杂度增加，需要仔细处理事件依赖关系

---

## Variant P: 多核并行与原子操作批处理
Source: embedding_dense_grad_v2

baseline采用单核策略避免写冲突，但这无法利用多核NPU的并行能力。专家实现的核心创新是'延迟原子操作'：不立即将累加结果写回GM，而是在UB中批量累加（最多10次），当遇到不同索引或达到阈值时才执行原子写回。这种策略将原子操作次数减少约90%，显著降低了GM竞争。双缓冲(addResQue_[2])实现了读写流水线并行：当队列0执行原子写回时，队列1可以继续累加新的梯度，二者通过switchId标志切换。LIMIT_COUNT_NUM (10)的选择是一个经验值，平衡了UB占用和原子操作频率。

**Expert implementation:**
```cpp
struct AddParam {
    uint64_t mask;
    uint64_t repeatTime;
    int64_t lastIndices;
    bool switchId;  // 双缓冲切换标志
};

__aicore__ inline void ComputeAndCopyOut(const uint64_t progress, const uint64_t dimJ)
{
    bool isNeedSwitch = CheckIsNeedSwitchAddQue(currentId);
    bool isLimite = addCount_[addParam_.switchId] == LIMIT_COUNT_NUM;
    
    AtomicAddInUb(gradLocal);
    
    if (isLimite || isLastRow) {
        CopyOut(addParam_.switchId, currentId, dimJ);
    }
    if (isNeedSwitch) {
        CopyOut(!addParam_.switchId, addParam_.lastIndices, dimJ);
    }
}
```

**vs. baseline:**
```cpp
// Core partitioning - use single core to avoid write conflicts
const uint32_t nCores = 1;
uint32_t rowsPerCore = batchSize;
```

Benefit: 多核并行带来接近线性的加速比（理论8核8x），批量原子操作减少90%的GM竞争
Trade-off: 需要额外的UB空间维护双缓冲（约2 * embedding_dim * sizeof(float)）

---

## Variant Q: 双缓冲与流水线并行
Source: fake_quant_affine_cachemask

专家实现使用了BUFFER_NUM = 2的双缓冲机制，配合PipeBarrier<PIPE_ALL>()实现了CopyIn-Compute-CopyOut的三级流水线并行。这种设计使得数据传输和计算可以重叠进行，显著提高了计算单元的利用率。在Process()函数中，每个tile的处理流程为：1) CommonCopyIn - 从GM读取数据到UB；2) PipeBarrier - 确保数据就绪；3) Compute - 执行量化计算；4) PipeBarrier - 确保计算完成；5) CommonCopyOut - 将结果写回GM。双缓冲的设计允许在计算当前tile的同时，预取下一个tile的数据，从而隐藏数据传输延迟。

**Expert implementation:**
```cpp
// 专家实现双缓冲流水线
constexpr int32_t BUFFER_NUM = 2;
pipe.InitBuffer(outQueueOut, BUFFER_NUM, this->tileLength * sizeof(yType));
pipe.InitBuffer(inQueueData, BUFFER_NUM, this->tileLength * sizeof(yType));

__aicore__ inline void Process() {
    for (uint32_t i = 0; i < this->circleNum; i++) {
        for (uint32_t j = 0; j < this->tileNum; j++) {
            this->CommonCopyIn(inQueueData, xGm, calcOffset, this->tileLength);
            PipeBarrier<PIPE_ALL>();
            Compute(this->tileLength);
            PipeBarrier<PIPE_ALL>();
            this->CommonCopyOut(outQueueOut, outQueueMask, yGm, maskGm, calcOffset, this->tileLength);
            PipeBarrier<PIPE_ALL>();
        }
    }
}
```

**vs. baseline:**
```cpp
// baseline无流水线
__aicore__ inline void Process() {
    for (uint32_t i = 0; i < this->innerLoops; i++) {
        CopyIn(i);
        Compute(i);
        CopyOut(i);
    }
}
```

Benefit: 通过双缓冲和流水线并行，隐藏数据传输延迟，提高计算单元利用率，整体性能提升显著
Trade-off: 代码复杂度增加，需要仔细管理PipeBarrier的位置；UB内存需求增加（双缓冲需要2倍buffer）

---

## Variant R: 模板继承架构设计
Source: foreach_add_scalar_list

专家实现采用三层模板继承结构：KernelForeachBase处理基础内存管理，KernelForeachUnary处理单输入流水线和双缓冲，ForeachOneScalarListBinary处理scalar list特化逻辑。这种架构实现了代码的高度复用，同系列的foreach算子可以复用同一套基类。

**Expert implementation:**
```cpp
template <typename T, typename Predicate, int32_t bufferNum, ...>
class KernelForeachUnary : public KernelForeachBase<T> {
    __aicore__ inline void Process() {
        for (uint16_t i = Base::tensorStart; i <= Base::tensorEnd; i++) {
            ProcessPlusInLoop(i, cursorStart);
            SingleTensorProcess(dataCount, float32Tensor);
        }
    }
};
```

**vs. baseline:**
```cpp
class ForeachAddScalarListCustomKernel {
    __aicore__ inline void Process() {
        for (uint32_t i = 0; i < inner_loops; i++) {
            CopyIn(tile_start);
            Compute();
            CopyOut(tile_start);
        }
    }
};
```

Benefit: 代码复用率高，新算子开发成本低，维护方便
Trade-off: 模板代码复杂，编译时间增加

---

## Variant S: 双缓冲与数据预取
Source: inplace_add_rms_norm

专家实现广泛使用了双缓冲（DOUBLE_BUFFER_NUM）机制来隐藏内存访问延迟。双缓冲允许在计算当前数据块的同时，预取下一个数据块到另一个缓冲区中。这种技术显著减少了计算单元的空闲等待时间，提高了整体吞吐量。在Arch35架构的RegBase实现中，双缓冲被用于输入队列（inQueueX1、inQueueX2）和输出队列（outQueueY、outQueueX、outQueueRstd）。

**Expert implementation:**
```cpp
// 专家实现 - 双缓冲
constexpr int32_t DOUBLE_BUFFER_NUM = 2;
pPipe->InitBuffer(inQueueX1, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
pPipe->InitBuffer(inQueueX2, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
pPipe->InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
pPipe->InitBuffer(outQueueX, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
```

**vs. baseline:**
```cpp
// baseline - 单缓冲
pipe.InitBuffer(inQueueX, 1, this->cols * sizeof(float));
pipe.InitBuffer(inQueueY, 1, this->cols * sizeof(float));
pipe.InitBuffer(outQueue, 1, this->cols * sizeof(float));
```

Benefit: 隐藏内存访问延迟，提高计算单元利用率
Trade-off: 增加UB内存占用

---

## Variant T: 双缓冲与流水线优化
Source: layer_norm_v3

专家实现广泛使用双缓冲（Double Buffer）技术隐藏内存延迟。TQue<QuePosition::VECIN, DOUBLE_BUFFER>实现输入队列双缓冲，TQue<QuePosition::VECOUT, DOUBLE_BUFFER>实现输出队列双缓冲。Ping-Pong机制允许DMA搬运与Vector计算并行。同时，使用LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>()显式控制内存屏障，确保数据一致性。

**Expert implementation:**
```cpp
// 双缓冲实现
pipe_->InitBuffer(inQueueX, DOUBLE_BUFFER, td_->tileLength * sizeof(T));
pipe_->InitBuffer(outQueueY, DOUBLE_BUFFER, td_->tileLength * sizeof(T));
pipe_->InitBuffer(outQueueMean, DOUBLE_BUFFER, AGGREGATION_COUNT * sizeof(float));

// 内存屏障
LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
```

**vs. baseline:**
```cpp
// 单缓冲实现
pipe.InitBuffer(inQueueX, 1, this->tileLength * sizeof(float));
pipe.InitBuffer(outQueueY, 1, this->tileLength * sizeof(float));
```

Benefit: 隐藏内存延迟，计算与数据搬运并行，吞吐量提升30-50%
Trade-off: UB空间占用增加一倍

---

## Variant U: 双缓冲与流水线优化
Source: layer_norm_v4

专家实现广泛使用双缓冲技术隐藏内存访问延迟。在pipe中初始化双缓冲队列，计算当前数据块时预取下一个数据块，实现计算与数据传输的流水线重叠。通过CopyInX和CaculateWithHighLevelApi的流水线编排，确保计算指令与数据传输指令交错执行。baseline使用单缓冲，没有考虑流水线优化。

**Expert implementation:**
```cpp
// 双缓冲
static constexpr uint32_t DOUBLE_BUFFER = 2;
pipe.InitBuffer(xQueue, DOUBLE_BUFFER, xBufferSize * sizeof(T));
pipe.InitBuffer(yQueue, DOUBLE_BUFFER, xBufferSize * sizeof(T));
pipe.InitBuffer(batchMeanQueue, DOUBLE_BUFFER, aFactorAlignF32 * sizeof(float));
```

**vs. baseline:**
```cpp
// 单缓冲
pipe.InitBuffer(inQueueX, 1, this->cols * sizeof(float));
pipe.InitBuffer(outQueue, 1, this->cols * sizeof(float));
```

Benefit: 隐藏内存延迟，提升计算单元利用率，通常可获得1.5-2倍性能提升
Trade-off: 增加UB内存占用，需要仔细设计流水线编排避免数据竞争

---

## Variant V: 双缓冲流水线优化
Source: modulate

专家实现采用双缓冲（DOUBLE_BUFFER=2）机制隐藏数据传输延迟。Ping-Pong机制使两个buffer交替使用，一个用于数据传输时另一个用于计算。Queue深度配置TQue<QuePosition::VECIN,DOUBLE_BUFFER>表示输入队列深度为2。专家实现的优化细节是X/Y使用双缓冲（输入和输出需要流水线并行），Scale/Shift使用单缓冲（只需加载一次）。

**Expert implementation:**
```cpp
// 专家实现: 优化的双缓冲配置
constexpr int64_t DOUBLE_BUFFER = 2;
this->pipe->InitBuffer(this->QueueX, DOUBLE_BUFFER, this->ubLength * sizeof(T));
this->pipe->InitBuffer(this->QueueY, DOUBLE_BUFFER, this->ubLength * sizeof(T));
if(this->parameterStatus != NO_SCALE) {
    this->pipe->InitBuffer(this->QueueScale, 1, this->ubLength * sizeof(T));
}
if(this->parameterStatus != NO_SHIFT) {
    this->pipe->InitBuffer(this->QueueShift, 1, this->ubLength * sizeof(T));
}
TQue<QuePosition::VECIN, DOUBLE_BUFFER> QueueX;
TQue<QuePosition::VECOUT, DOUBLE_BUFFER> QueueY;
TQue<QuePosition::VECIN, 1> QueueScale;
TQue<QuePosition::VECIN, 1> QueueShift;
```

**vs. baseline:**
```cpp
// baseline: 基础双缓冲
constexpr int64_t DOUBLE_BUFFER = 2;
pipe.InitBuffer(inQueueX, DOUBLE_BUFFER, UB_BUFFER_SIZE * sizeof(float));
pipe.InitBuffer(outQueueY, DOUBLE_BUFFER, UB_BUFFER_SIZE * sizeof(float));
```

Benefit: 数据传输和计算重叠场景下性能提升30-50%
Trade-off: 需要额外的UB空间（2x buffer大小）

---

## Variant W: 双缓冲与Ping-Pong机制
Source: multi_scale_deformable_attn_function

高性能模板KernelMultiScaleDeformableAttnOpt采用了双缓冲（Ping-Pong）机制来隐藏数据传输延迟。具体来说，valueQue_和outputQue_都分配了两倍大小的缓冲区。通过ping变量在两个缓冲区之间切换，使得计算和数据传输可以并行进行。同时，使用事件机制（HardEvent::V_MTE2、HardEvent::MTE2_V、HardEvent::V_MTE3）精确控制同步点，确保数据一致性。

**Expert implementation:**
```cpp
// UB缓冲区初始化 - 双缓冲
pipe_->InitBuffer(valueQue_, TWO * alignedCornerEmbedDims_ * B32_BYTE_SIZE);
pipe_->InitBuffer(outputQue_, TWO * alignedHeadEmbedDims_ * B32_BYTE_SIZE);

// Process函数中的Ping-Pong切换
uint8_t ping = 0;
for (uint32_t batch = 0; batch < batchSize_; ++batch) {
    for (uint32_t query = startOffset_; query < endOffset_; ++query) {
        WaitFlag<HardEvent::MTE3_V>(ping);
        Duplicate<float, false>(output[ping * alignedHeadEmbedDims_], 0.f, ...);
        // ... 计算 ...
        DataCopy(outputGm_[baseDstOffset_], output[ping * alignedHeadEmbedDims_], ...);
        SetFlag<HardEvent::MTE3_V>(ping);
        ping = 1 - ping;  // 切换缓冲区
    }
}
```

**vs. baseline:**
```cpp
// baseline实现不存在，无法提供对比代码
```

Benefit: 隐藏数据传输延迟，提高整体吞吐量
Trade-off: 增加了UB内存使用量，需要管理双缓冲的同步

---

## Variant X: 双缓冲(Double Buffer)机制
Source: rms_norm_grad

专家实现大量使用了双缓冲技术来隐藏内存访问延迟。在InitInputQue中可以看到，输入队列inQueDY_、inQueX_、inQueRstd_都使用了BUFFER_NUM_DB（值为2）作为buffer数量。这意味着在计算当前数据的同时，可以异步加载下一批数据。通过PIPE_V和PIPE_MTE2之间的流水，最大化计算和内存访问的并行度。baseline实现使用单缓冲，计算和内存访问串行执行，硬件利用率较低。

**Expert implementation:**
```cpp
bufferNum_ = BUFFER_NUM_DB;  // = 2
pipe.InitBuffer(inQueDY_, bufferNum_, bufferLenSize_);
pipe.InitBuffer(inQueX_, bufferNum_, bufferLenSize_);
pipe.InitBuffer(outQueDX_, bufferNum_, bufferLenSize_);
```

**vs. baseline:**
```cpp
pipe.InitBuffer(inQueueX, 1, cols * sizeof(float));
pipe.InitBuffer(inQueueGradOut, 1, cols * sizeof(float));
pipe.InitBuffer(outQueueGradX, 1, cols * sizeof(float));
```

Benefit: 在计算密集场景可提升30-50%性能；有效隐藏Global Memory访问延迟
Trade-off: UB内存使用量增加一倍；代码复杂度增加

---

## Variant Y: 流水线并行与双缓冲
Source: scaled_masked_softmax_grad_v2

在Large HeadDim场景中，专家实现采用精细的流水线并行控制，通过SetFlag/WaitFlag机制实现指令级并行。数据搬运与计算重叠，当前迭代的计算与下一次迭代的数据搬运并行，使用多个eventId（MTE3_MTE2, MTE2_V, MTE3_V, V_MTE3等）管理不同阶段依赖。

**Expert implementation:**
```cpp
int32_t eventIdMTE32MTE2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
int32_t eventIdMTE32V = GetTPipePtr()->FetchEventID(HardEvent::MTE3_V);
int32_t eventIdMTE22VA = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
int32_t eventIdV2MTE3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);

for (uint64_t loop = 0; loop < this->loopTimes; ++loop) {
    ScaledMaskedSoftmaxGradV2Base<T>::CopyIn(yGradLocal, yLocal, offset);
    SetFlag<HardEvent::MTE2_V>(eventIdMTE22VA);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE22VA);
    if (loop != 0) {
        WaitFlag<HardEvent::MTE3_V>(eventIdMTE32V);
    }
    ComputeSoftmaxGradBit16(...);
    SetFlag<HardEvent::V_MTE3>(eventIdV2MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdV2MTE3);
    ScaledMaskedSoftmaxGradV2Base<T>::CopyOut(out, offset);
    if (loop != this->loopTimes - 1) {
        SetFlag<HardEvent::MTE3_V>(eventIdMTE32V);
    }
}
```

**vs. baseline:**
```cpp
for (uint32_t i = 0; i < this->innerLoops; i++) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
}
```

Benefit: 提高指令级并行度，隐藏内存访问延迟，提升整体吞吐量
Trade-off: 代码复杂度大幅增加，需要深入理解硬件流水线；调试困难，时序问题难以定位

---

## Variant Z: 精细化的UB内存Tiling
Source: scatter_elements_v2

专家实现在Host端进行详细的UB内存容量规划，考虑数据对齐、双缓冲、中间缓存等多方面因素。通过计算inputAlign、indicesAlign、updatesAlign等参数，确保UB使用效率最大化。针对indices和input的不同大小关系，设计了三种tiling策略：indices优先、input优先、两者均分。同时考虑数据类型大小进行32B对齐，充分利用Vector Unit的并行能力。

**Expert implementation:**
```cpp
// Expert: 精细化UB规划
if (indicesSum < max_ub / HALF_UB) {
    inputEach = (max_ub - indicesSum) / inputSize;
    inputLoop = (inputOnePiece - 1) / inputEach + 1;
} else if (inputSum < max_ub / HALF_UB) {
    indicesEach = (max_ub - inputSum) / (inputSize + indicesSize);
    indicesLoop = (indicesOneTime - 1) / indicesEach + 1;
}
inputAlign = ((inputEach - 1) / inputDataAlign + 1) * inputDataAlign;
```

**vs. baseline:**
```cpp
// Baseline: 固定buffer，无tiling
uint32_t bufferSize = 256;
pipe.InitBuffer(inQueueVar, BUFFER_NUM, bufferSize * sizeof(T));
pipe.InitBuffer(inQueueIndices, BUFFER_NUM, bufferSize * sizeof(U));
pipe.InitBuffer(inQueueUpdates, BUFFER_NUM, bufferSize * sizeof(T));
```

Benefit: UB利用率最大化，减少GM访问次数，充分利用Vector Unit并行能力
Trade-off: Host端tiling计算复杂，需要精确计算内存布局

---

## Variant 27: Split分块与双缓冲策略
Source: trans_quant_param_v2

专家实现采用了经典的Split分块策略来处理大规模数据。SPLIT_SIZE = 65536字节（64KB）的块大小是根据UB（Unified Buffer）容量精心选择的。算法将数据划分为多个循环处理，每个循环处理一个块，并通过PipeBarrier和SetFlag/WaitFlag机制确保计算和内存传输的流水线并行。双缓冲技术通过同时分配offsetUb_（用于中间结果）和resUb_（用于最终结果）实现，允许在计算当前块的同时预取下一个块的数据。这种策略显著提高了内存带宽利用率。

**Expert implementation:**
```cpp
constexpr int32_t SPLIT_SIZE = 65536;

uint32_t eachLength = SPLIT_SIZE / sizeof(uint64_t);
uint32_t loops = Max<uint32_t>(scaleLength_, offsetLength_) / eachLength;

pipe_->InitBuffer(offsetUb_, 2 * alignedLength * sizeof(float));
pipe_->InitBuffer(resUb_, SPLIT_SIZE);

for (uint32_t loopidx = 0; loopidx < loops; ++loopidx) {
    // 计算
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
    // 拷贝
    DataCopy(yGm_[eachLength * loopidx], resTensor, ub2GmParams);
}
```

**vs. baseline:**
```cpp
scale_ub = tl.alloc_ub(tile_size, dtype=tl.float32)
zero_point_ub = tl.alloc_ub(tile_size, dtype=tl.int32)
```

Benefit: 提高内存带宽利用率，实现计算和传输流水线并行，支持大规模数据处理
Trade-off: 需要更多的UB内存分配，增加了同步开销

---

## Variant 28: Flash Attention BMM双缓冲与L1 IBShare
Source: flash_attention_score, fused_floyd_attention

Flash Attention的两次连续矩阵乘法(BMM1: Q@K^T, BMM2: Softmax@V)通过REGIST_MATMUL_OBJ宏注册到同一Pipeline，启用L1 Input Buffer Share模式。BMM1和BMM2的Tiling数据分别配置，但共享L1中的输入缓冲区，避免Q矩阵在BMM1使用后被驱逐、BMM2需要重新加载。Pipeline内部自动管理双缓冲：当前块的Softmax计算与下一块的K/V预取可以重叠执行。

**Expert implementation:**
```cpp
// 双MatMul注册到同一Pipeline，启用L1 IBShare
constexpr MatmulConfig CFG_IBSHARE_EXCEED = GetIBShareNormConfig(true);

GET_TILING_DATA_MEMBER(FlashAttentionScoreGeneralTilingData, bmm1TilingData, bmm1TilingDataVar, tiling);
GET_TILING_DATA_MEMBER(FlashAttentionScoreGeneralTilingData, bmm2TilingData, bmm2TilingDataVar, tiling);
REGIST_MATMUL_OBJ(&tPipe, GetSysWorkSpacePtr(), op.bmm1, bmm1tiling, op.bmm2, bmm2tiling);

// S2维度分块迭代：BMM1 → Softmax(UB内) → BMM2，Q的L1数据跨BMM复用
// BMM1: Q[S1,D] @ K[S2_block,D]^T -> scores[S1,S2_block]
// Softmax: online softmax with running max/sum
// BMM2: attn[S1,S2_block] @ V[S2_block,D] -> partial_output[S1,D]
```

**vs. baseline:**
```cpp
// 两次独立MatMul，L1数据不共享
matmul1.Init(tPipe, bmm1Tiling);
matmul1.IterateAll(scores);         // Q的L1数据被释放
matmul2.Init(tPipe, bmm2Tiling);    // 需要重新从GM加载
matmul2.IterateAll(output);
```

Benefit: L1 IBShare减少BMM2的GM访问；在线Softmax避免S1×S2完整矩阵存储；计算与预取重叠
Trade-off: IBShare限制L1使用灵活性；两个BMM的tiling需要协调

---

## Variant 29: Conv2D L1 Ping-Pong与L0 Ping-Pong
Source: conv2d_v2

Conv2D通过编译期模板参数控制L1和L0两级缓存的双缓冲策略。L1PingPong控制Feature Map数据在L1中的双缓冲（计算当前块时预取下一块），L0PingPong控制Weight数据在L0中的双缓冲。WeightUbTrans模式在使用L1双缓冲的同时，先将Weight经UB格式转换后再写入L1，避免计算阶段的格式转换开销。多个模板参数的组合在编译期确定最优流水线配置。

**Expert implementation:**
```cpp
// 编译期模板参数控制两级缓存双缓冲
template<int8_t FmapTiling, int8_t WeightTiling,
         int8_t L1PingPong,    // L1层Feature Map双缓冲
         int8_t L0PingPong,    // L0层Weight双缓冲
         int8_t WeightUbTrans, // Weight经UB转换后写入L1
         int8_t FmapCopyMode, int8_t InnerBatch, ...>
struct Conv2DV1Param : public Conv2dParam {};

// 数据流: GM -> [UB转换] -> L1(Ping/Pong) -> L0(Ping/Pong) -> Cube计算
// L1PingPong: Fmap块A在Cube计算时，DMA预取Fmap块B到L1另一半
// L0PingPong: Weight块A在Cube计算时，Weight块B从L1预取到L0另一半
```

**vs. baseline:**
```cpp
// 无L1/L0双缓冲，串行加载和计算
for (int tile = 0; tile < numTiles; tile++) {
    DataCopy(l1Buffer, fmapGm[offset], tileSize);  // 等待加载完成
    conv.Compute(l1Buffer, weightL1, output);       // 计算
}
```

Benefit: L1和L0两级流水线并行，最大化隐藏数据加载延迟；编译期模板选择零运行时开销
Trade-off: L1/L0双缓冲各占用双倍缓存空间；模板参数组合较多
