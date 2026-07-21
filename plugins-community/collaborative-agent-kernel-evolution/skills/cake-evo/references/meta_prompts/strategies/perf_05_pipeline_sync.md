# P5: Pipeline Synchronization (流水线同步控制)
## Overview
专家实现大量使用Ascend C向量化指令（Muls, Mul, Add, ReduceSum, Cast等）替代标量循环，显著提升计算吞吐量。同时通过PipeBarrier和SetFlag/WaitFlag机制实现精确的流水线同步——在计算和数据传输之间插入同步点，确保数据一致性同时最大化流水线并行。例如，在加载gamma/beta权重时使用SetFlag<HardEvent::V_MTE2>标记计算完成，然后启动数据传输，再使用WaitFlag<HardEvent::MTE2_V>等待数据就绪，实现计算与数据传输的流水线重叠。baseline实现完全使用标量循环（GetValue/SetValue）和DataCopyPad，未使用向量化指令，计算效率低下。

## When to Use
- All double-buffered kernels
- 保证数据依赖正确性，避免数据竞争和脏读
- 精确控制异步执行单元之间的依赖关系，最大化各执行单元的并行度
- 最大化指令级并行，提高Vector单元利用率

## Trade-off
- 同步指令会引入一定的流水线气泡
- 代码复杂度显著增加，需要深入理解硬件架构
- 增加同步开销，需要精细设计Barrier位置

**Source operators**: adaptive_avg_pool3d, add_rms_norm_cast, add_rms_norm_dynamic_quant, apply_adam_w_v2, dequant_bias, embedding_dense_grad_v2, foreach_abs, foreach_add_list, foreach_add_scalar_list, gather_elements_v2, gemma_rms_norm, inplace_add_rms_norm, multi_scale_deformable_attention_grad, norm_common, rms_norm_grad, rms_norm_quant, scatter_elements_v2

---

## Variant A: 显式流水线同步
Source: adaptive_avg_pool3d

使用显式同步指令管理流水线：SToVSync()确保Scalar到Vector同步，MTE3ToVSync()确保Global Memory加载到Vector同步。合理使用同步指令保证复杂流水线并行场景下数据依赖正确性。

**Expert implementation:**
```cpp
// 显式同步
__aicore__ inline void SToVSync() {
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
}

// 使用场景
GetIndexFromBuffer(indexBuf, bufIdx, bufIdx, index);
SToVSync();  // 确保索引已准备好
```

**vs. baseline:**
```cpp
// baseline无显式同步
```

Benefit: 保证数据依赖正确性，避免数据竞争和脏读
Trade-off: 同步指令会引入一定的流水线气泡

---

## Variant B: 精细化事件同步与流水线优化
Source: add_rms_norm_cast

专家实现大量使用显式的事件同步机制（SetFlag/WaitFlag）来控制不同流水线阶段（MTE2, MTE3, V, S）的执行顺序。这比简单的PipeBarrier提供了更精细的控制，允许最大化并行度。同步策略包括：MTE2_V数据从GM搬入UB完成后触发VPU可以开始计算；V_MTE3向量计算完成后触发结果可以搬出到GM；MTE3_V数据搬出完成后触发VPU可以复用缓冲区；V_S向量计算完成后Scalar单元可以读取结果。

**Expert implementation:**
```cpp
// 精细化事件同步
event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
SetFlag<HardEvent::MTE2_V>(eventMte2V);
// ... 计算 ...
WaitFlag<HardEvent::MTE2_V>(eventMte2V);
Mul(xFp32, xFp32, xFp32, numCol);
PipeBarrier<PIPE_V>();
// ... 更多计算 ...
event_t eventVMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
SetFlag<HardEvent::V_MTE3>(eventVMte3);
WaitFlag<HardEvent::V_MTE3>(eventVMte3);
DataCopyCustom<T>(xGm[gm_bias], x1Local_in, numCol);
```

**vs. baseline:**
```cpp
// 简单顺序执行，无显式流水线优化
AscendC::Cast(addedLocal, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
AscendC::Cast(tempLocal, residualLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
AscendC::Add(addedLocal, addedLocal, tempLocal, this->tileLength);
// 没有显式事件同步
```

Benefit: 精确控制异步执行单元之间的依赖关系，最大化各执行单元的并行度
Trade-off: 代码复杂度显著增加，需要深入理解硬件架构

---

## Variant C: 精细的Pipe Barrier流水线控制
Source: add_rms_norm_dynamic_quant

专家实现在Kernel端大量使用PipeBarrier<PIPE_V>()和SetFlag/WaitFlag进行流水线同步。通过显式控制Vector单元与Scalar单元之间的同步，确保数据依赖的正确性，提高指令级并行，优化内存访问。相比baseline实现中较少的同步点，专家实现通过更精细的控制实现了更高的指令吞吐。

**Expert implementation:**
```cpp
Cast(xLocalFp32, x1Local, RoundMode::CAST_NONE, elementCount);
Cast(yLocalFp32, x2Local, RoundMode::CAST_NONE, elementCount);
PipeBarrier<PIPE_V>();
Add(xLocalFp32, xLocalFp32, yLocalFp32, elementCount);
PipeBarrier<PIPE_V>();
event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
SetFlag<HardEvent::S_V>(eventSV);
WaitFlag<HardEvent::S_V>(eventSV);
```

**vs. baseline:**
```cpp
AscendC::Cast(addOutLocal, x1Local, AscendC::RoundMode::CAST_RINT, tileLength);
AscendC::Cast(tempLocal, x2Local, AscendC::RoundMode::CAST_RINT, tileLength);
AscendC::Add(addOutLocal, addOutLocal, tempLocal, tileLength);
```

Benefit: 最大化指令级并行，提高Vector单元利用率
Trade-off: 增加同步开销，需要精细设计Barrier位置

---

## Variant D: 流水线屏障(PipeBarrier)的精细化控制
Source: apply_adam_w_v2

专家实现通过在计算流程中插入大量的`PipeBarrier<PIPE_V>()`屏障，确保Vector单元的数据依赖性得到正确处理。在AdamW算法的每个计算步骤之间都插入了屏障，如Weight Decay计算后、Momentum更新后、Variance更新后等。这种设计虽然增加了代码复杂度，但确保了数据一致性，避免了流水线冒险(Hazard)。

**Expert implementation:**
```cpp
Muls(dataOutLocal[varOffset_], dataLocal[varOffset_], realWeightDecay_, dataCount);
PipeBarrier<PIPE_V>();
Sub(dataOutLocal[maxGradOutOffset_], dataLocal[gradOffset_], dataLocal[expAvgOffset_], dataCount);
PipeBarrier<PIPE_V>();
Muls(dataLocal[varOffset_], dataOutLocal[maxGradOutOffset_], oneSubBeta1_, dataCount);
PipeBarrier<PIPE_V>();
Add(dataOutLocal[expAvgOffset_], dataLocal[varOffset_], dataLocal[expAvgOffset_], dataCount);
```

**vs. baseline:**
```cpp
// Compute without explicit barriers
Muls(outLocal[0], var, decayFactor, count);
Muls(m, m, beta1_, count);
Muls(var, grad, (1.0f - beta1_), count);
Add(m, m, var, count);
```

Benefit: 确保数据一致性；避免流水线冒险；硬件安全
Trade-off: 代码量增加；可能引入微小延迟

---

## Variant E: 数据类型转换的延迟隐藏
Source: dequant_bias

专家实现在weight_scale和bias的数据加载时采用双阶段转换策略：先将数据从GM拷贝到临时buffer，使用Cast指令转换为float32，通过event_t同步确保转换完成后再使用。这种设计实现了类型无关性、精度保证、延迟隐藏。

**Expert implementation:**
```cpp
// 专家实现：延迟隐藏的类型转换
if constexpr(!IsSameType<WSTYPE, float>::value) {
    LocalTensor<WSTYPE> tmpBuf = wsBuf_.GetWithOffset<WSTYPE>(wsTypeNum, wsBufferSize_ / 2);
    DataCopyPad(tmpBuf, weightScaleGM_, wsParams, padParams);
    
    // 事件同步
    event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2V);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
    
    // 类型转换
    Cast(wsLocal_, tmpBuf, RoundMode::CAST_NONE, N_);
    PipeBarrier<PIPE_V>();
}
```

**vs. baseline:**
```cpp
// baseline：直接在Init中转换
scaleVal = scaleGm.GetValue(0);
int8_t zeroPointInt8 = zeroPointGm.GetValue(0);
zeroPointVal = static_cast<float>(zeroPointInt8);
```

Benefit: 支持多类型输入，精度保证，延迟隐藏
Trade-off: 需要额外的临时buffer空间

---

## Variant F: 显式流水线同步
Source: embedding_dense_grad_v2

专家实现使用显式的PIPE_*系列函数控制指令流水线，而不是依赖隐式同步。这是因为：数据依赖（Cast操作依赖DataCopy完成，必须等待MTE2事件）、乱序执行（NPU是乱序执行架构，需要显式屏障保证正确性）、性能调优（最小化同步点可以暴露更多ILP）。常用的同步原语包括PIPE_MTE2_V（MTE2到Vector）、PIPE_V_MTE3（Vector到MTE3）、PIPE_MTE3_S（MTE3到Scalar）。

**Expert implementation:**
```cpp
__aicore__ inline void PIPE_MTE2_V()
{
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
}

// Usage
DataCopyPad(gradLocalCasted[gradLocalCastedOffset], gradGm_[gradAddrOffset], ...);
PIPE_MTE2_V();
Cast(gradLocal, gradLocalCasted[gradLocalCastedOffset], ...);
```

**vs. baseline:**
```cpp
// Implicit synchronization
AscendC::DataCopyPad(...);
AscendC::Add(...);
```

Benefit: 消除不必要的同步等待，提升流水线利用率15-30%
Trade-off: 代码复杂度增加，需要仔细设计同步点

---

## Variant G: 流水线屏障同步
Source: foreach_abs

专家实现大量使用PipeBarrier<PIPE_V>()来确保向量指令的按序执行和数据一致性。这对于BF16的Cast-Compute-Cast流程尤为重要，必须确保前一个Cast完成才能开始计算，计算完成才能开始后一个Cast。Pipeline Barrier的作用是强制同步不同流水阶段的指令，避免数据竞争和乱序执行导致的错误。

**Expert implementation:**
```cpp
PipeBarrier<PIPE_V>();
Cast(float32Tensor, x1Local[...], ...);
PipeBarrier<PIPE_V>();
op(float32Tensor[offset], float32Tensor, dataCount);
PipeBarrier<PIPE_V>();
```

**vs. baseline:**
```cpp
// 直接使用Abs指令，无屏障
AscendC::Abs(outputLocal, inputLocal, currentTileSize);
```

Benefit: 保证计算正确性和数据一致性，避免乱序执行错误
Trade-off: 增加同步开销，可能影响性能

---

## Variant H: Pipeline Barrier精确控制
Source: foreach_add_list

专家实现使用PipeBarrier<PIPE_V>()进行向量管道同步，确保数据依赖的正确性。在Axpy指令后添加Barrier确保计算完成后再进行数据拷贝。这种细粒度同步比全局同步更高效，只阻塞向量管道而不影响其他管道（如MTE）。

**Expert implementation:**
```cpp
// 专家实现: 精确Pipeline控制
Axpy<T, T>(srcLocal1, srcLocal2, scalarVal, uValue);
if (dstLocal.GetPhyAddr() != srcLocal1.GetPhyAddr()) {
    PipeBarrier<PIPE_V>();  // 确保Axpy完成
    DataCopy(dstLocal, srcLocal1, uValue);
}
```

**vs. baseline:**
```cpp
// baseline: 无显式同步
AscendC::Add(outputLocal, inputLocal, otherLocal, this->tileSize);
inputQueue.FreeTensor(inputLocal);
otherQueue.FreeTensor(otherLocal);
outputQueue.EnQue(outputLocal);
```

Benefit: 确保数据依赖正确性，避免竞态条件，同时最小化同步开销
Trade-off: 需要开发者精确理解数据依赖关系

---

## Variant I: PipeBarrier流水线同步
Source: foreach_add_scalar_list

专家实现在计算前后使用PipeBarrier<PIPE_V>()确保vector指令流水线的正确性，防止指令重排序导致的计算错误。baseline实现缺少这种同步机制，可能在某些边界情况下出现竞态条件。

**Expert implementation:**
```cpp
PipeBarrier<PIPE_V>();
op(outLocal, dataLocal, scalarVal, dataCount);
PipeBarrier<PIPE_V>();
```

Benefit: 确保指令执行顺序正确，避免数据竞争
Trade-off: 轻微的同步开销

---

## Variant J: 流水线同步优化
Source: gather_elements_v2

专家实现中大量使用PipeBarrier和同步原语（SetFlag/WaitFlag）来优化指令流水线。关键优化点包括：MTE3到MTE2同步确保数据从Global Memory搬运到UB完成后才进行计算、V到MTE2同步确保向量计算完成后才进行数据搬出、PIPE_V屏障确保向量指令之间的数据依赖正确性。这些同步点精心布置在数据生产者和消费者之间，既保证了正确性，又最大程度允许指令流水线重叠执行。

**Expert implementation:**
```cpp
__aicore__ inline void MTE3ToMTE2Sync() {
    event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}

__aicore__ inline void GatherInUb(...) {
    Muls(idxLocal, idxLocal, static_cast<T_IDX>(sizeof(T_X)), computeNum);
    PipeBarrier<PIPE_V>();  // 确保Muls完成后再执行Gather
    Gather(yLocal, xLocal, gatherOffsetLocal, (uint32_t)0, computeNum);
}
```

Benefit: 保证正确性的同时最大化指令流水线重叠
Trade-off: 增加了代码复杂度和同步开销

---

## Variant K: 流水线同步优化
Source: gemma_rms_norm

专家实现通过显式的 PipeBarrier 和 SetFlag/WaitFlag 指令精确控制 Vector 和 Scalar 单元的流水线同步。这种细粒度同步允许计算和内存访问最大化重叠。例如，在计算 rstd 时，先启动 Vector 计算，然后设置标志等待 Scalar 单元读取结果，期间 Vector 单元可以继续执行其他任务。baseline实现缺少这种显式同步，依赖自动同步，无法充分利用流水线并行。

**Expert implementation:**
```cpp
// 专家实现 - 显式流水线同步
event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));

Mul(sqx, xLocal, xLocal, num_col);
PipeBarrier<PIPE_V>();

float reduceOut = ReduceSumHalfInterval(sqx, num_col);
SetFlag<HardEvent::V_S>(eventVS);
WaitFlag<HardEvent::V_S>(eventVS);
float rstdValue = 1 / sqrt(reduceOut * avg_factor + epsilon);
SetFlag<HardEvent::S_V>(eventSV);
WaitFlag<HardEvent::S_V>(eventSV);
```

**vs. baseline:**
```cpp
// baseline - 自动同步
AscendC::Mul(tempLocal, inputLocal, inputLocal, this->tileLength);
AscendC::ReduceSum(sharedLocal, tempLocal, sharedLocal, this->tileLength);
float tileSqSum = sharedLocal.GetValue(0);
```

Benefit: 最大化 Vector 和 Scalar 单元并行度，减少流水线气泡，隐藏内存延迟
Trade-off: 编程复杂度增加，需要深入理解硬件流水线

---

## Variant L: 流水线屏障与事件同步
Source: inplace_add_rms_norm

专家实现中精细地使用了PipeBarrier和事件同步机制来确保数据一致性并最大化指令级并行。PipeBarrier<PIPE_V>()用于在向量计算之间建立同步点，确保前一个计算完成后再开始下一个计算。事件同步（SetFlag/WaitFlag）用于跨流水线阶段的同步，如V到S（向量到标量）和S到V（标量到向量）的同步。这种精细的同步控制允许编译器和硬件进行更好的指令调度，减少流水线停顿。

**Expert implementation:**
```cpp
// 专家实现 - 精细同步
Mul(sqx, xLocal, xLocal, numCol);
PipeBarrier<PIPE_V>();

Muls(sqx, sqx, avgFactor, numCol);
PipeBarrier<PIPE_V>();

ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);
PipeBarrier<PIPE_V>();

// 事件同步
event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(event_v_s);
WaitFlag<HardEvent::V_S>(event_v_s);
float rstdValue = sqx.GetValue(0);
event_t event_s_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
SetFlag<HardEvent::S_V>(event_s_v);
WaitFlag<HardEvent::S_V>(event_s_v);
```

**vs. baseline:**
```cpp
// baseline - 无显式同步
AscendC::Add(sumLocal, xLocal, yLocal, this->cols);
AscendC::Mul(squareLocal, sumLocal, sumLocal, this->cols);
AscendC::ReduceSum(sharedLocal, squareLocal, sharedLocal, this->cols);
```

Benefit: 确保数据一致性，最大化指令级并行
Trade-off: 增加代码复杂度，需要深入理解流水线架构

---

## Variant M: 8事件ID驱动的异步流水线
Source: multi_scale_deformable_attention_grad

专家实现使用了8个独立的事件ID来管理不同计算单元之间的同步，这是实现高吞吐量的关键：eventIdMte2ToV用于MTE2(数据加载)到Vector的同步；eventIdMte3ToV用于MTE3(数据存储)到Vector的同步；eventIdVToMte2和eventIdVToMte3用于Vector到内存单元的同步；此外还有专门用于weight、x、y梯度存储的事件。这种精细的事件管理允许计算和内存传输overlap，例如在等待MTE2加载数据的同时，Vector单元可以处理上一批数据。

**Expert implementation:**
```cpp
eventIdMte2ToV = pipe->AllocEventID<HardEvent::MTE2_V>();
eventIdVToMte3 = pipe->AllocEventID<HardEvent::V_MTE3>();
eventIdVToMteWeight = pipe->AllocEventID<HardEvent::V_MTE3>();
// ... 使用
SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
Muls(wv1Local, zerosLocal[...], w1, embedDims);
SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
```

**vs. baseline:**
```cpp
// baseline无显式事件同步
CopyInGradOut(b, q, h);
ComputeGradAttnWeight();
CopyOutGradAttnWeight(weightOffset);
```

Benefit: 实现计算和内存传输overlap，隐藏内存延迟，预期性能提升30-50%
Trade-off: 编程复杂度高，需要仔细管理事件依赖关系

---

## Variant N: 向量化计算与流水线优化
Source: norm_common

专家实现大量使用Ascend C向量化指令（Muls, Mul, Add, ReduceSum, Cast等）替代标量循环，显著提升计算吞吐量。同时通过PipeBarrier和SetFlag/WaitFlag机制实现精确的流水线同步——在计算和数据传输之间插入同步点，确保数据一致性同时最大化流水线并行。例如，在加载gamma/beta权重时使用SetFlag<HardEvent::V_MTE2>标记计算完成，然后启动数据传输，再使用WaitFlag<HardEvent::MTE2_V>等待数据就绪，实现计算与数据传输的流水线重叠。baseline实现完全使用标量循环（GetValue/SetValue）和DataCopyPad，未使用向量化指令，计算效率低下。

**Expert implementation:**
```cpp
// 专家实现：向量化计算与流水线优化
__aicore__ inline void ProcessBasicBlock(uint32_t nRow, ...)
{
    Muls(yLocal, xLocal, coefficient, tileLength);
    PipeBarrier<PIPE_V>();
    
    for (uint32_t rowIdx = 0; rowIdx < nRow; ++rowIdx) {
        ReduceSum(yLocal[currentRowOffset], yLocal[currentRowOffset], yLocal[currentRowOffset], rowSize);
        acc_val = GetAccVal();
        value = *reinterpret_cast<float*>(&acc_val);
        meanLocal.SetValue(rowIdx, static_cast<float>(value));
    }
    
    // 流水线：异步加载gamma权重
    if (!nullptrGamma && rowIdx == 0) {
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        DataCopyPad(xLocal.ReinterpretCast<Tweight>()[...], gammaGm, dataCopyParams, padParams);
    }
    
    PipeBarrier<PIPE_V>();
    Mul(xLocal, yLocal, yLocal, tileLength);
    PipeBarrier<PIPE_V>();
    Muls(xLocal, xLocal, coefficient, tileLength);
}
```

**vs. baseline:**
```cpp
// baseline：标量循环实现
__aicore__ inline float ComputeSum()
{
    LocalTensor<float> x_local = inQueue_x.DeQue<float>();
    float sum = 0.0f;
    for (uint32_t i = 0; i < tile_length; i++) {
        sum += x_local.GetValue(i);
    }
    inQueue_x.FreeTensor(x_local);
    return sum;
}

__aicore__ inline void ComputeNormalizeScaleShift(float mean, float std)
{
    for (uint32_t i = 0; i < tile_length; i++) {
        float normalized = (x_local.GetValue(i) - mean) / std;
        float result = normalized * weight_local.GetValue(i) + bias_local.GetValue(i);
        y_local.SetValue(i, result);
    }
}
```

Benefit: 向量化计算可提升性能3-10倍，流水线优化隐藏数据传输延迟，综合性能提升可达5-20倍
Trade-off: 需要深入理解AI Core架构和指令流水线，代码可读性降低

---

## Variant O: AtomicAdd跨核累加
Source: rms_norm_grad

专家实现使用SetAtomicAdd进行跨核的原子累加操作，允许多个核心同时向同一个Global Memory地址写入数据，无需额外的同步和reduce步骤。这大大简化了dgamma的跨核累加逻辑，提升了性能。baseline实现没有使用原子操作，可能需要额外的处理来完成跨核累加。

**Expert implementation:**
```cpp
__aicore__ inline void CopyDgammaOut() {
    LocalTensor<float> dgammaOut = outQueDgamma_.DeQue<float>();
    SetAtomicAdd<float>();
    DataCopy(dgammaGm_, dgammaOut, ROUND_UP(colVal_, ALIGN_32));
    SetAtomicNone();
}
```

**vs. baseline:**
```cpp
// 未使用AtomicAdd
```

Benefit: 高效的跨核累加；无需额外的workspace和reduce kernel
Trade-off: 结果非确定性；需要硬件支持原子操作

---

## Variant P: 精细同步控制
Source: rms_norm_grad

专家实现使用了多种同步机制确保数据一致性：PipeBarrier<PIPE_V>()用于向量计算流水线同步；SetFlag/WaitFlag用于事件驱动的精确同步；SyncAll用于跨核同步。通过精细控制同步点，避免不必要的等待，提升流水线效率。baseline实现缺乏显式同步控制，可能导致流水线效率低下或数据竞争。

**Expert implementation:**
```cpp
event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(eventVS);
WaitFlag<HardEvent::V_S>(eventVS);
float rstd_value = rstdLocal.GetValue(0);
event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
SetFlag<HardEvent::S_V>(eventSV);
WaitFlag<HardEvent::S_V>(eventSV);
```

**vs. baseline:**
```cpp
// 简单处理，无显式同步
```

Benefit: 最大化流水线并行度；精确控制数据依赖；提升硬件利用率
Trade-off: 代码复杂度增加；需要深入理解硬件流水线

---

## Variant Q: 流水线显式同步
Source: rms_norm_quant

专家实现使用SetFlag/WaitFlag显式控制MTE2（内存传输引擎）到Vector（向量计算单元）的同步点。通过EVENT_ID区分不同同步点，实现细粒度的流水线控制。这种显式同步相比自动同步可减少等待时间，提升指令级并行度。

**Expert implementation:**
```cpp
DataCopyCustom<T>(fp16_g, gm_g_[sliceOffset], numel);
AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
Cast(fp32_g[OFFSET_GAMMA * slice_size_], fp16_g, AscendC::RoundMode::CAST_NONE, numel);
AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
```

**vs. baseline:**
```cpp
AscendC::DataCopy(xLocal, xGm[offset], count);
inQueueX.EnQue(xLocal);
```

Benefit: 隐藏内存访问延迟，提升指令级并行度
Trade-off: 代码复杂度增加，需要仔细设计同步点

---

## Variant R: 流水线优化与内存访问隐藏
Source: scatter_elements_v2

专家实现通过PIPE_*宏显式控制MTE（Memory Transfer Engine）和Vector Unit之间的同步，实现指令级并行。在Small Mode中，采用MTE3→MTE2→V→MTE3的流水线，确保数据搬运和计算重叠。使用PipeBarrier进行精确的流水级间同步，避免数据竞争。对于类型转换操作，插入PIPE_MTE2_V同步点确保数据准备好后再进行Vector运算。

**Expert implementation:**
```cpp
// Expert: 显式流水线控制
PIPE_MTE3_MTE2();
CopyInIndex(indicesIndex);
DataCopyPad(updatesLocal, updatesGm[updatesIndex], updatesExtParams, tPadParams);
DataCopyPad(inputLocal, inputGm[inputIndex], inputExtParams, tPadParams);
if constexpr (IS_CAST_FLOAT) {
    PIPE_MTE2_V();
    Cast(inputTemp, inputLocal, RoundMode::CAST_NONE, inputAlign);
}
PipeBarrier<PIPE_ALL>();
```

**vs. baseline:**
```cpp
// Baseline: 无流水线控制，同步执行
DataCopy(varLocal, inputGm[varOffset], inputOneTime);
DataCopy(indicesLocal, indicesGm[indicesOffset], indicesOneTime);
DataCopy(updatesLocal, updatesGm[updatesOffset], updatesOneTime);
// 处理scatter...
DataCopy(inputGm[varOffset], varLocal, inputOneTime);
```

Benefit: 指令级并行，隐藏内存访问延迟，最大化计算单元利用率
Trade-off: 需要精确控制同步点，调试复杂
