# P10: Vectorized Data Copy & Instructions (向量化数据与指令)
## Overview
专家实现在mask生成逻辑中采用了Compare+Select的组合指令，而非简单的CompareScalar。这种策略的优势在于：1) Compare指令产生bitmask结果，可以进行向量化操作；2) Select指令根据bitmask进行条件选择，实现了向量化条件判断；3) 通过两次Compare+Select组合，可以高效地实现区间判断（quant_min <= x <= quant_max）。相比之下，baseline实现使用了CompareScalar+手动循环AND操作的方式，虽然逻辑清晰，但无法充分利用向量化引擎。专家实现还通过Mul指令（Mul(curHf16Temp, selectTemp, curHf16Temp, calCount)）实现了逻辑AND操作，利用FP16的数值计算代替位运算，这也是一种向量化友好的优化策略。

## When to Use
- CopyIn/CopyOut optimization
- 向量化转置可将内存访问模式优化为连续访问，减少bank conflict，提升Vector指令效率4-8倍
- 代码复用性高，参数配置集中管理
- 利用Vector单元并行能力；保持代码一致性

## Trade-off
- 需要额外的UB buffer存储转置数据
- 增加一层函数调用开销(通常可内联)
- 对于单次运算略显复杂

**Source operators**: adaptive_max_pool3d_grad, add_rms_norm_dynamic_quant, apply_adam_w_v2, deep_norm, dynamic_block_quant, fake_quant_affine_cachemask, foreach_add_list, foreach_add_scalar_list, foreach_addcdiv_list, gather_elements_v2, grouped_dynamic_mx_quant, inplace_add_rms_norm, layer_norm_v4, linear_index, max_pool_grad_with_argmax_common, max_pool_with_argmax_v3, rms_norm_quant, scaled_masked_softmax_grad_v2, sparse_to_dense

---

## Variant A: Transpose向量化优化
Source: adaptive_max_pool3d_grad

专家实现大量使用TransDataTo5HD指令进行矩阵转置，这是Ascend C提供的高性能向量化指令。转置的目的是将数据从[N, C, D, H, W]布局转换为适合Vector指令处理的布局。TransposeBase16M8用于[row, col] -> [col, row]，row对齐16，col对齐8；TransposeBase8M16用于row对齐8，col对齐16；TransposeBase16M16用于row/col都对齐16。转置后可以使用连续的Vector指令处理数据，提高指令级并行度。这是内存访问模式优化的核心技巧。

**Expert implementation:**
```cpp
template <typename T>
__aicore__ inline void TransposeBase16M8(LocalTensor<T>& dstUb, LocalTensor<T>& srcUb, 
                                         uint64_t rowNum, uint64_t colNum) {
    uint64_t srcAddrList[TRANS_ADDR_LEN];
    uint64_t dstAddrList[TRANS_ADDR_LEN];
    for (uint64_t r = 0; r < rowNum / TRANS_ADDR_LEN; r++) {
        for (uint64_t i = 0; i < TRANS_ADDR_LEN; i++) {
            srcAddrList[i] = (uint64_t)(srcUb[r * TRANS_ADDR_LEN * colNum + i * colNum].GetPhyAddr());
            dstAddrList[i] = (uint64_t)(dstUb[r * TRANS_ADDR_LEN + i / 2 * rowNum + i % 2 * BLOCK_NUM_32].GetPhyAddr());
        }
        struct TransDataTo5HDParams transDataParams;
        transDataParams.repeatTimes = colNum / BLOCK_NUM_32;
        TransDataTo5HD<float>(dstAddrList, srcAddrList, transDataParams);
    }
}

if constexpr (is_same<TGrad, float>::value) {
    TransposeBase16M8(gradTranUb, gradUb, params_.singleCoreNc, block_.dohowoAlign8);
} else {
    TransposeBase16M16(gradTranUb, gradUb, params_.singleCoreNc, block_.dohowoAlign16);
}
```

Benefit: 向量化转置可将内存访问模式优化为连续访问，减少bank conflict，提升Vector指令效率4-8倍
Trade-off: 需要额外的UB buffer存储转置数据

---

## Variant B: 内存访问封装DataCopyEx
Source: add_rms_norm_dynamic_quant

**Expert implementation:**
```cpp
template <typename T, template <typename U> typename R, template <typename U> typename S>
__aicore__ inline void DataCopyEx(const R<T>& dst, const S<T>& src, const uint32_t len, const uint32_t count = 1, const bool ubAligned = false) {
    DataCopyExtParams copyParams;
    copyParams.blockCount = count;
    copyParams.blockLen = len * sizeof(T);
    if constexpr (is_same<R<T>, AscendC::LocalTensor<T>>::value) {
        copyParams.srcStride = 0;
        copyParams.dstStride = (ubAligned) ? 1 : 0;
        DataCopyPad(dst, src, copyParams, {});
    }
}
DataCopyEx(x1x2LocalIn[0], this->x2Gm[gmOffset], this->numLastDim, rowCount, this->ubAligned);
```

**vs. baseline:**
```cpp
AscendC::DataCopyPad(x1Local, x1Gm[offset], {1, static_cast<uint16_t>(tileLength * sizeof(half)), 0, 0, 0}, {false, 0, 0, 0});
```

Benefit: 代码复用性高，参数配置集中管理
Trade-off: 增加一层函数调用开销(通常可内联)

---

## Variant C: ScalarPow的向量化实现
Source: apply_adam_w_v2

针对Adam算法中需要的幂运算（计算bias correction），专家实现没有使用标量ALU，而是通过Vector单元的`Power`指令实现。`ScalarPow`函数将标量`x`广播到Vector寄存器（通过`Duplicate`指令填充32字节/8个FP32），然后使用`Power`指令进行向量化计算。

**Expert implementation:**
```cpp
LocalTensor<float> baseLocal = powTempBuf1_.Get<float>();
LocalTensor<float> outLocal = powTempBuf2_.Get<float>();
PipeBarrier<PIPE_V>();
Duplicate(baseLocal, x, BLOCK_SIZE_FOR_FLOAT32);
PipeBarrier<PIPE_V>();
Power<float, false>(outLocal, baseLocal, y, BLOCK_SIZE_FOR_FLOAT32);
event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
SetFlag<HardEvent::V_S>(eventIdVToS);
WaitFlag<HardEvent::V_S>(eventIdVToS);
float result = outLocal.GetValue(0);
```

**vs. baseline:**
```cpp
// Scalar pow (CPU way)
float biasCorrection1 = 1.0f - powf(beta1_, step);
```

Benefit: 利用Vector单元并行能力；保持代码一致性
Trade-off: 对于单次运算略显复杂

---

## Variant D: 向量化 Reduce 优化
Source: deep_norm

专家实现针对短维度场景设计了专门的 ReduceSumShort 函数，充分利用 Ascend C 的向量化指令并行能力。该函数将输入数据按 32B 对齐分组，通过 BlockReduceSum 指令实现高效的并行归约。相比 baseline 中简单的循环累加，这种向量化实现可以充分利用 SIMD 单元的并行性，在小维度场景下获得显著的性能提升。

**Expert implementation:**
```cpp
// 专家实现 - 向量化 ReduceSumShort
Duplicate<float>(tmp_local, ZERO, repeat * elementNum);
PipeBarrier<PIPE_V>();
for (index = 0; index + elementNum <= num_last_dim; index += elementNum) {
    Add(tmp_local, tmp_local, src_local[index], elementNum, repeat, {1, 1, 1, 1, 1, repStride});
    PipeBarrier<PIPE_V>();
}
// BlockReduceSum 向量化归约
if (repeatTimes != 0) {
    BlockReduceSum<float>(dst_local, tmp_local, repeatTimes, maxRepeat, 1, 1, elementNum);
}
```

**vs. baseline:**
```cpp
// baseline - 简单 ReduceSum
AscendC::ReduceSum(sharedTensor, inputTensor, sharedTensor, actualTileLen);
float tileSum = sharedTensor.GetValue(0);
rowSum += tileSum;
```

Benefit: 充分利用 SIMD 并行性，小维度场景性能提升显著
Trade-off: 代码复杂度增加，需要处理对齐和尾部数据

---

## Variant E: 纯向量化计算
Source: dynamic_block_quant

专家实现完全避免了标量循环，所有计算都使用向量指令。关键设计包括：1) 使用Duplicate预计算常量tensor（如1.0/127），避免在循环中重复计算；2) 使用参数化的向量操作（Div/Mul）通过BinaryRepeatParams指定repeat次数和mask；3) 使用硬件Cast指令进行类型转换和舍入。特别值得注意的是，专家实现使用quantMask、quantMask2、quantMask3等预计算tensor替代标量常量，使得整个量化流程可以通过向量指令完成。这种设计充分利用了向量单元的并行能力，理论上可以达到接近峰值算力。

**Expert implementation:**
```cpp
Duplicate(quantMask, (float)(1.0), NUM_EIGHT);
Duplicate(quantMask2, (float)(1.0) / NUM_INT_8_MAX, NUM_EIGHT);
Div(scaleLocalTmp, quantMask, scaleLocal, NUM_SIX_FOUR, repeatTimes, {1, 0, 1, NUM_EIGHT, 0, NUM_EIGHT});
Cast(xLocalTmpInt8, xLocalTmpHalf, RoundMode::CAST_NONE, calcNum);
```

**vs. baseline:**
```cpp
for (uint32_t i = 0; i < blockSize; i++) {
    float val = quantizedFloat.GetValue(i);
    int rounded = static_cast<int>(val + 0.5f);
    quantizedLocal.SetValue(i, static_cast<int8_t>(clamped));
}
```

Benefit: 充分利用向量单元并行性，接近峰值性能
Trade-off: 需要预计算常量tensor，增加UB使用

---

## Variant F: 指令选择优化 (Select vs CompareScalar)
Source: fake_quant_affine_cachemask

专家实现在mask生成逻辑中采用了Compare+Select的组合指令，而非简单的CompareScalar。这种策略的优势在于：1) Compare指令产生bitmask结果，可以进行向量化操作；2) Select指令根据bitmask进行条件选择，实现了向量化条件判断；3) 通过两次Compare+Select组合，可以高效地实现区间判断（quant_min <= x <= quant_max）。相比之下，baseline实现使用了CompareScalar+手动循环AND操作的方式，虽然逻辑清晰，但无法充分利用向量化引擎。专家实现还通过Mul指令（Mul(curHf16Temp, selectTemp, curHf16Temp, calCount)）实现了逻辑AND操作，利用FP16的数值计算代替位运算，这也是一种向量化友好的优化策略。

**Expert implementation:**
```cpp
// 专家实现 - Compare + Select向量化组合
Compare(maskTemp, curHf16Temp, quantMinTensor, CMPMODE::GE, calCount);
BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
Select(selectTemp, maskTemp, oneTensor, zeroTensor, 
       SELMODE::VSEL_TENSOR_TENSOR_MODE, this->mask, repeatTimes, repeatParams);
Compare(maskTemp, curHf16Temp, quantMaxTensor, CMPMODE::LE, calCount);
Select(curHf16Temp, maskTemp, oneTensor, zeroTensor, 
       SELMODE::VSEL_TENSOR_TENSOR_MODE, this->mask, repeatTimes, repeatParams);
Mul(curHf16Temp, selectTemp, curHf16Temp, calCount);  // 逻辑AND
```

**vs. baseline:**
```cpp
// baseline - CompareScalar + 手动循环
AscendC::CompareScalar(maskGeLocal, tempLocal, quantMinF, AscendC::CMPMODE::GE, this->tileSize);
AscendC::CompareScalar(maskLeLocal, tempLocal, quantMaxF, AscendC::CMPMODE::LE, this->tileSize);
for (uint32_t j = 0; j < maskSize; j++) {
    maskLocal.SetValue(j, maskGeLocal.GetValue(j) & maskLeLocal.GetValue(j));
}
```

Benefit: 充分利用向量化引擎，提高mask生成效率；避免标量循环，减少指令开销
Trade-off: 指令复杂度增加，需要理解Compare+Select的组合用法；FP16 Mul代替位运算可能有精度损失

---

## Variant G: Axpy指令优化
Source: foreach_add_list

专家实现针对浮点类型使用Axpy指令替代独立的Muls+Add组合。Axpy指令可以同时完成y = x + alpha * y的操作，减少指令数量和内存访问。在AddListFloatAdapter中，直接调用Axpy完成x1 + alpha * x2的计算，避免了单独乘法再加法的开销。

**Expert implementation:**
```cpp
// 专家实现: Axpy单指令
template <typename T>
__aicore__ void AddListFloatAdapter(...) {
    Axpy<T, T>(srcLocal1, srcLocal2, scalarVal, uValue);
    if (dstLocal.GetPhyAddr() != srcLocal1.GetPhyAddr()) {
        PipeBarrier<PIPE_V>();
        DataCopy(dstLocal, srcLocal1, uValue);
    }
}
```

**vs. baseline:**
```cpp
// baseline: Muls + Add组合
if (this->alpha != 1.0f) {
    AscendC::Muls(otherLocal, otherLocal, this->alpha, this->tileSize);
}
AscendC::Add(outputLocal, inputLocal, otherLocal, this->tileSize);
```

Benefit: 减少指令数量，降低内存访问，提升计算效率约10-20%
Trade-off: 仅适用于浮点类型，INT32类型仍需使用Muls+Add

---

## Variant H: Adds专用指令优化
Source: foreach_add_scalar_list

专家实现直接使用Adds指令完成tensor加scalar的操作，而baseline实现使用了Duplicate填充scalar到tensor，然后再使用Add进行相加。Adds是专门为tensor加scalar设计的指令，在硬件层面直接支持scalar操作数，避免了额外的内存操作和中间结果存储。

**Expert implementation:**
```cpp
template <typename T>
__aicore__ void AddsAdapter(...) {
    Adds(dstLocal, srcLocal, scalarValue, static_cast<uint32_t>(uValue));
}
op(outLocal, dataLocal, scalarVal, dataCount);
```

**vs. baseline:**
```cpp
AscendC::Duplicate(out_ub, scalar_value, tile_size);
AscendC::Add(out_ub, x_ub, out_ub, tile_size);
```

Benefit: 减少约50%的vector指令数，降低内存带宽需求15-20%
Trade-off: 需要了解Ascend C的专用指令集

---

## Variant I: 寄存器级APT优化
Source: foreach_add_scalar_list

专家实现提供了APT（Advanced Pipeline Tuning）版本，使用寄存器级别的操作（RegTensor, MaskReg）来进一步优化性能。APT版本直接使用__VEC_SCOPE__和寄存器操作，绕过部分自动流水线调度，由开发者手动控制指令流。这种方式允许更精细的指令调度，可以减少流水线气泡。

**Expert implementation:**
```cpp
__VEC_SCOPE__ {
    MaskReg maskReg;
    RegTensor<float> inRegToFloat;
    RegTensor<float> scaleValReg;
    RegTensor<float> outReg;
    for (uint16_t i = 0; i < repeatTimes; i++) {
        maskReg = UpdateMask<float>(sreg);
        ops::LoadOneTensorForDtypeT<T>(inUbAddr, inRegToFloat, maskReg, i * dataCountPerLoop);
        Adds(outReg, inRegToFloat, scaleVal, maskReg);
        ops::StoreOneTensorForDtypeT<T>(outUbAddr, outReg, maskReg, i * dataCountPerLoop);
    }
}
```

Benefit: 在计算密集型场景下可进一步提升10-20%性能
Trade-off: 代码复杂度显著提高，需要对硬件指令集有深入理解

---

## Variant J: Axpy指令融合优化
Source: foreach_addcdiv_list

专家实现在AddcDivListFloatAdapter中使用了Axpy指令代替独立的Mul和Add。Axpy（A*X Plus Y）是计算y = y + alpha * x的融合指令，在硬件级别将乘法和加法融合为一条指令，减少了指令发射开销和中间结果存储。此外，专家实现使用PipeBarrier<PIPE_V>()确保Vector流水线的同步，避免数据竞争。baseline实现使用了独立的Muls和Add指令，没有利用融合指令优化。

**Expert implementation:**
```cpp
// 专家实现: Axpy融合指令
Div(tensor2Local, tensor2Local, tensor3Local, uValue);
PipeBarrier<PIPE_V>();
Axpy<T, T>(tensor1Local, tensor2Local, scalarVal, uValue);  // y = y + alpha * x
```

**vs. baseline:**
```cpp
// baseline: 独立指令
AscendC::Div(divLocal, tensor1Local, tensor2Local, current_tile_size);
AscendC::Muls(divLocal, divLocal, this->value, current_tile_size);
AscendC::Add(outLocal, inputLocal, divLocal, current_tile_size);
```

Benefit: 减少指令数，降低指令发射开销，减少中间结果存储
Trade-off: 需要理解和正确使用融合指令

---

## Variant K: Transpose优化的内存访问模式
Source: gather_elements_v2

在非最后一维gather场景下，直接按gather维度访问会导致跨步访问（strided access），内存访问效率极低。专家实现通过Transpose-Gather-ReTranspose三步法解决这个问题：将输入tensor转置使gather维度变为连续，在转置后的tensor上进行高效的连续内存gather操作，再将结果转置回原始形状。这种优化利用了Ascend芯片的TransDataTo5HD指令。

**Expert implementation:**
```cpp
__aicore__ inline void Process() {
    for (size_t preDimId = 0; preDimId < curGroupPreDim; preDimId++) {
        for (size_t postDimPartId = 0; postDimPartId < postDimPartNum; postDimPartId++) {
            TransposeProcess(xBaseOffset, idxBaseOffset, carryNumAlign_, ...);
            this->MTE3ToMTE2Sync();
            GatherProcess(carryNumAlign_);
            this->MTE3ToMTE2Sync();
            ReTransposeProcess(idxBaseOffset, carryNumAlign_);
        }
    }
}

template <typename T>
__aicore__ inline void TransposeByte4(LocalTensor<T>& dstLocal, LocalTensor<T>& srcLocal, ...) {
    uint64_t srcList[TRANS_LEN];
    uint64_t dstList[TRANS_LEN];
    TransDataTo5HDParams transDataParams;
    TransDataTo5HD<T>(dstList, srcList, transDataParams);
}
```

Benefit: 将跨步访问转为连续访问，显著提升内存带宽利用率
Trade-off: 增加了两次transpose开销，需要额外workspace

---

## Variant L: 向量化Gather指令优化
Source: gather_elements_v2

在Transpose和LastDim模式下，专家实现充分利用了Ascend芯片的向量化Gather指令，而非baseline中的逐元素访问。具体包括索引预处理（将索引值乘以数据类型大小转换为字节偏移）、向量Gather单条指令完成多个元素的gather、负索引的向量化处理。相比baseline中的循环GetValue/SetValue，向量化gather可以利用SIMD并行性。

**Expert implementation:**
```cpp
__aicore__ inline void GatherInUb(const uint64_t& xGatherDimSlice, const uint64_t& computeNum) {
    LocalTensor<T_X> xLocal = dataInQue_.DeQue<T_X>();
    LocalTensor<T_IDX> idxLocal = xLocal[CeilAlign(xGatherDimSlice, this->xAlign_)].template ReinterpretCast<T_IDX>();
    LocalTensor<T_X> yLocal = dataOutQue_.AllocTensor<T_X>();
    
    Muls(idxLocal, idxLocal, static_cast<T_IDX>(sizeof(T_X)), computeNum);
    PipeBarrier<PIPE_V>();
    LocalTensor<uint32_t> gatherOffsetLocal = idxLocal.template ReinterpretCast<uint32_t>();
    Gather(yLocal, xLocal, gatherOffsetLocal, (uint32_t)0, computeNum);
    
    dataInQue_.EnQue<T_X>(xLocal);
    dataOutQue_.EnQue<T_X>(yLocal);
}
```

**vs. baseline:**
```cpp
for (uint32_t j = 0; j < idxGatherDim; j++) {
    int32_t gatherIdx = idxLocal.GetValue(idxRowOffset + j);
    float value = xLocal.GetValue(xRowOffset + gatherIdx);
    yLocal.SetValue(yRowOffset + j, value);
}
```

Benefit: 利用SIMD并行性，显著提升吞吐量
Trade-off: 需要额外的索引预处理和同步

---

## Variant M: 寄存器级向量化优化
Source: grouped_dynamic_mx_quant

专家实现大量使用AscendC::MicroAPI进行寄存器级操作，这是性能优化的核心手段。RegTensor直接操作绕过LocalTensor的抽象层，减少数据搬运开销。通过vfLen（Vector寄存器长度）确定每次处理的元素数量，最大化Vector单元吞吐量。Interleave指令的使用实现了数据的奇偶交错存储，优化了内存访问模式。在Compute函数中，使用了超过20个RegTensor进行并行计算。

**Expert implementation:**
```cpp
__VEC_SCOPE__
{
    AscendC::MicroAPI::RegTensor<T> xRegTensor;
    AscendC::MicroAPI::RegTensor<uint16_t> expMaxRegTensor;
    AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<uint16_t>(pnum);
    
    DataCopy(xRegTensor, xAddr + i * vfLen);
    AscendC::MicroAPI::And(expMaxRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor,
        maxEleRegTensor, p0);
    
    for (uint16_t j = 1; j < static_cast<uint16_t>(blockCount); j++) {
        AscendC::MicroAPI::Max(expMaxRegTensor, expMaxRegTensor, expRegTensor, p0);
    }
}
```

**vs. baseline:**
```cpp
LocalTensor<T> xLocal = this->calcBuf.template Get<T>();
for (uint16_t j = 0; j < blockCount; j++) {
    DataCopy(xLocal[j * vfLen], xAddr + j * dataLen + i * vfLen, vfLen);
}
```

Benefit: 最大化Vector单元吞吐量；减少内存访问次数；精细控制指令调度
Trade-off: 代码可读性降低；需要深入理解硬件寄存器架构

---

## Variant N: 寄存器级向量化优化（Arch35）
Source: inplace_add_rms_norm

在Arch35架构（ASCEND910_95）上，专家实现使用了RegTensor和微指令API进行寄存器级向量化优化。这种优化技术将数据直接加载到向量寄存器中进行计算，避免了频繁的UB内存访问。实现中的__VEC_SCOPE__宏定义了向量计算的作用域，在此作用域内使用RegTensor进行高效的向量化运算。关键优化包括：使用CreateMask进行灵活的掩码控制、通过LoadTensorForDtypeTIn和StoreTensorForDtypeTOut实现高效的数据加载/存储、以及循环折叠（loop folding）技术来减少循环开销。

**Expert implementation:**
```cpp
// 专家实现 - Arch35寄存器级优化
__VEC_SCOPE__
{
    RegTensor<float> x1Reg;
    RegTensor<float> x2Reg;
    RegTensor<float> xSum1Reg;
    
    MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
    for (uint16_t i = 0; i < loopRows; ++i) {
        LoadTensorForDtypeTIn<T>(x1InUb, x1Reg, pregFull, offset);
        LoadTensorForDtypeTIn<T>(x2InUb, x2Reg, pregFull, offset);
        Add(xSum1Reg, x1Reg, x2Reg, pregFull);
        StoreTensorForDtypeTOut<T>(xOutInUb, xSum1Reg, pregFull, offset);
    }
}
```

**vs. baseline:**
```cpp
// baseline - 标准UB计算
AscendC::Add(sumLocal, xLocal, yLocal, this->cols);
AscendC::Mul(squareLocal, sumLocal, sumLocal, this->cols);
AscendC::ReduceSum(sharedLocal, squareLocal, sharedLocal, this->cols);
```

Benefit: 在Arch35上可获得显著的指令吞吐提升
Trade-off: 仅适用于特定架构，代码可移植性降低

---

## Variant O: Transpose策略的内存访问优化
Source: layer_norm_v4

Transpose策略针对中等row size（≤64）优化。Layer Norm需要对每行reduce，原始布局中行数据不连续（跨步访问）。通过转置将reduce维度变为连续访问，显著提升带宽利用率。实现涉及TransDataTo5HDImpl转置、Reshape重排、二分归约等复杂内存布局转换。

**Expert implementation:**
```cpp
template <typename T_TRANS>
__aicore__ inline void DoTranspose(LocalTensor<T_TRANS>& dstTensor, LocalTensor<T_TRANS>& srcTensor) {
    __ubuf__ T_TRANS* srcLocalList[TRANSPOSE_C0_SIZE];
    __ubuf__ T_TRANS* dstLocalList[TRANSPOSE_C0_SIZE];
    for (uint32_t i = 0; i < TRANSPOSE_C0_SIZE; i++) {
        srcLocalList[i] = srcAddr + rFormerAxisAlign * i;
        dstLocalList[i] = dstAddr + B32_BLOCK_ALIGN_NUM * i;
    }
    struct TransDataTo5HDParams transDataParams;
    transDataParams.repeatTimes = rFormerAxisAlign / B32_BLOCK_ALIGN_NUM;
    TransDataTo5HDImpl(dstLocalList, srcLocalList, transDataParams);
}
```

**vs. baseline:**
```cpp
// 直接跨步访问，无转置优化
AscendC::DataCopy(inputLocal, inputGm[rowIdx * this->cols], this->cols);
```

Benefit: row size较小时，转置开销远小于跨步访问开销，显著提升内存带宽利用率
Trade-off: 实现复杂，需要处理转置后的复杂索引计算和边界情况

---

## Variant P: RegBase策略的寄存器级优化
Source: layer_norm_v4

RegBase策略直接使用寄存器操作而非高层API，完全控制指令流水线。通过显式PipeBarrier和LocalMemBar控制指令发射顺序，使用RegTensor显式管理向量寄存器，通过MaskReg实现条件执行避免分支预测失败，设计高效的二叉树归约算法。baseline完全依赖高层API，失去底层优化机会。

**Expert implementation:**
```cpp
__VEC_SCOPE__ {
    RegTensor<float> x, mean_sum, mean;
    MaskReg pregLoop = UpdateMask<float>(sreg0);
    LoadOneTensorForDtypeT(xInUb, x, pregLoop, (k * xyUbOffset));
    Muls(mean_sum, x, n, pregLoop);
    ReduceSum(mean, mean_sum, pregLoop);
    Muls(mean, mean, nCorrectionFactor, pregMerge);
}
LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
```

**vs. baseline:**
```cpp
// 使用高层API
AscendC::ReduceSum(sharedLocal, inputLocal, sharedLocal, this->cols);
float rowSum = sharedLocal.GetValue(0);
```

Benefit: 最大化指令级并行，精确控制寄存器分配，避免分支预测失败，达到理论峰值性能
Trade-off: 代码极其复杂，需要深入理解Ascend芯片微架构，维护成本高

---

## Variant Q: 负数索引向量化处理
Source: linear_index

专家实现采用了一种巧妙的向量化方法处理负数索引转正数的问题。传统做法是使用标量条件判断（if idx < 0 then idx += target），这在向量化架构上效率极低。专家实现利用算术右移指令(ShiftRight)的特性：对int32进行31位算术右移，负数结果为-1(0xFFFFFFFF)，正数结果为0。然后乘以target边界值，再与原索引相减，实现了无分支的负数转正数转换。这种方法完全避免了分支预测失败，并且可以充分利用Vector单元的并行计算能力。

**Expert implementation:**
```cpp
constexpr int INT32_OFFSET = 31;

// 左移31位，负数索引结果为-1，正数索引结果为0
ShiftRight(indicesTemp, indices32Local, INT32_OFFSET, static_cast<int>(indicesAlign));
PipeBarrier<PIPE_V>();
// 乘以边界值
Muls(indicesTemp, indicesTemp, static_cast<int>(target), static_cast<int>(indicesAlign));
PipeBarrier<PIPE_V>();
// 负数索引加上边界值
Sub(indices32Local, indices32Local, indicesTemp, static_cast<int>(indicesAlign));
```

**vs. baseline:**
```cpp
for (uint32_t i = 0; i < tileSize; i++) {
    int32_t idx_val = indicesLocal.GetValue(i);
    if (idx_val < 0) {
        idx_val = idx_val + static_cast<int32_t>(target);
    }
}
```

Benefit: 向量化处理比标量循环快8-16倍（取决于vector长度），完全消除分支预测失败
Trade-off: 需要对齐到vector长度，可能浪费少量计算资源

---

## Variant R: MicroAPI高性能指令
Source: max_pool_grad_with_argmax_common

专家实现大量使用了Ascend C的MicroAPI来替代传统的向量指令，获得显著的性能提升。关键优化包括：DataCopyGather/Scatter使用gather/scatter指令实现非连续内存访问的高效加载和存储；Arange指令高效生成连续整数序列；Pack/Unpack在16位和32位数据类型之间转换；向量化Mask操作通过CreateMask、UpdateMask、MaskAnd等操作实现精细的条件控制，避免分支跳转。这些MicroAPI指令通常对应底层的SIMD硬件指令，可以充分利用AI Core的并行计算能力。

**Expert implementation:**
```cpp
// 专家实现：向量化Gather/Scatter
template <typename T>
__aicore__ inline void GradientAcc(__local_mem__ computeType* yAddr, 
                                   MicroAPI::RegTensor<computeType>& gradReg,
                                   MicroAPI::RegTensor<T>& argmaxReg, 
                                   MicroAPI::MaskReg& pregArgmax)
{
    AscendC::MicroAPI::RegTensor<computeType> scatterAccResReg;
    // 从yAddr中gather当前值（向量化）
    AscendC::MicroAPI::DataCopyGather(scatterAccResReg, yAddr, 
        (AscendC::MicroAPI::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
    // 累加梯度（向量化）
    AscendC::MicroAPI::Add(scatterAccResReg, scatterAccResReg, gradReg, pregArgmax);
    // scatter回yAddr（向量化）
    AscendC::MicroAPI::DataCopyScatter(yAddr, scatterAccResReg, 
        (AscendC::MicroAPI::RegTensor<uint32_t>&)argmaxReg, pregArgmax);
}
```

**vs. baseline:**
```cpp
// baseline实现：标量操作
__aicore__ inline int64_t GetArgmaxValue(uint32_t offset)
{
    AscendC::LocalTensor<int64_t> argmaxLocal = argmaxQueue.DeQue<int64_t>();
    int64_t value = argmaxLocal.GetValue(offset);  // 标量读取
    argmaxQueue.EnQue(argmaxLocal);
    return value;
}

// 标量累加
float grad_val = gradOutLocal.GetValue(c_offset);
float current_val = gradInLocal.GetValue(0);
float new_val = current_val + grad_val;
tempLocal.SetValue(0, new_val);
```

Benefit: 纯向量化操作充分利用AI Core并行能力；Gather/Scatter高效处理非连续内存访问
Trade-off: MicroAPI学习曲线较陡；需要理解底层硬件特性

---

## Variant S: Gather/Scatter 向量化内存访问
Source: max_pool_with_argmax_v3

专家实现使用 MicroAPI::DataCopyGather/DataCopyScatter 指令，在单个向量周期内从非连续内存位置加载/存储数据，完美匹配max pool的滑动窗口访问模式。MaxPoolWithArgMaxV3GatherImpl 函数一次性加载kernel窗口内所有数据，使用向量化Max和Compare指令并行计算。baseline使用标量循环GetValue/SetValue逐元素访问。

**Expert implementation:**
```cpp
// 专家实现 - Gather/Scatter 向量化
__aicore__ inline void MaxPoolWithArgMaxV3GatherImpl(...) {
    GenGatterIndex4D<int32_t>(gatterStartIdx, rate4D, num3D, rate3D, num2D, rate2D, num1D);
    
    for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
        for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
            MicroAPI::DataCopyGather(vd1, xAddr, gatterIndexReg, computeT1);
            AscendC::MicroAPI::Compare<T1, CMPMODE::GT>(gtMask, vd1, vd0, computeT1);
            MicroAPI::Max(vd0, vd1, vd0, computeT1);
        }
    }
    
    AscendC::MicroAPI::DataCopyScatter(argmaxAddr, argmaxRes, scatterIndexReg, computeT2);
    AscendC::MicroAPI::DataCopyScatter(maxValueAddr, vd0, scatterIdxU16Reg, computeT1);
}
```

**vs. baseline:**
```cpp
// baseline - 标量循环
for (uint32_t i = 0; i < count; i++) {
    float inputVal = inputLocal.GetValue(i);
    float maxVal = maxLocal.GetValue(i);
    if (inputVal > maxVal) {
        maxLocal.SetValue(i, inputVal);
        idxLocal.SetValue(i, static_cast<int32_t>(perChannelIdx));
    }
}
```

Benefit: 充分利用SIMD并行性，性能提升5-20倍
Trade-off: 需要精心设计的索引生成逻辑，代码复杂度提高

---

## Variant T: 多维索引生成（ArithProgression）
Source: max_pool_with_argmax_v3

专家实现使用 MicroAPI::Arange 和算术运算生成多维索引，避免显式循环嵌套。GenGatterIndex4D 将一维线性索引转换为4D空间坐标，然后计算Gather偏移量。这种方法完全向量化，减少分支，且通过模板参数支持2D/3D/4D不同维度。

**Expert implementation:**
```cpp
// 专家实现 - 向量化4D索引生成
template <typename T>
__aicore__ inline void GenGatterIndex4D(MicroAPI::RegTensor<T>& indexReg, 
    T rate4D, T num3D, T rate3D, T num2D, T rate2D, T num1D, T rate1D = 1) {
    AscendC::MicroAPI::Arange(indexReg, 0);
    AscendC::MicroAPI::Div(segmentScalarReg3, indexReg, constReg(num3D), preg);
    AscendC::MicroAPI::Muls(tmpReg, segmentScalarReg3, num3D, preg);
    AscendC::MicroAPI::Sub(indexReg, indexReg, tmpReg, preg);
    // ... 继续3D/2D/1D分解
    AscendC::MicroAPI::Add(indexReg, indexReg, segmentScalarReg, preg);
}
```

**vs. baseline:**
```cpp
// baseline - 标量计算
uint32_t perChannelIdx = ih * width + iw;
```

Benefit: 完全向量化，无控制流开销，索引生成与计算重叠
Trade-off: 需要理解复杂的索引算术，调试难度增加

---

## Variant U: Gamma预加载复用
Source: rms_norm_quant

在FastCompute模式下，专家实现将gamma在循环外预加载到UB中，避免每行重复从GM读取。这种预加载策略对于RMSNorm的计算模式非常有效——gamma在行间共享，而x每行独立。通过CopyInGama函数一次性加载gamma到calc_buf_，后续所有行的计算都复用这份数据。

**Expert implementation:**
```cpp
AscendC::LocalTensor<T> fp16_g = fp32_xy_buf_.Get<T>(num_col_align_f32);
DataCopyCustom<T>(fp16_g, gm_g_, num_col_);
AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
Cast(fp32_g[OFFSET_GAMMA * num_col_align_f32], fp16_g, AscendC::RoundMode::CAST_NONE, num_col_);

while (pid < row_work_) {
    CopyIn(offset, num_col_);
    Compute();  // 使用预加载的gamma
    CopyOut(offset, num_col_);
    ++pid;
}
```

**vs. baseline:**
```cpp
AscendC::LocalTensor<float> wLocal = inQueueW.AllocTensor<float>();
AscendC::DataCopy(wLocal, weightGm[colStart], count);
inQueueW.EnQue(wLocal);
```

Benefit: GM带宽压力显著降低，内存访问和计算更好流水线化
Trade-off: 需要额外的UB空间存储预加载的gamma

---

## Variant V: Mask数据搬运模式优化
Source: scaled_masked_softmax_grad_v2

专家实现针对mask的不同广播模式（BNSD/1NSD/B1SD/11SD）实现了专门的CopyIn策略。通过CopyInMaskBlock和CopyInMaskLines的组合，最小化冗余数据搬运。特别是在广播场景下，避免从GM重复读取相同数据，而是从已读取的local数据复制。

**Expert implementation:**
```cpp
__aicore__ inline void CopyInMask1NSD(
    LocalTensor<bool>& maskLocal, uint64_t& lineCnt, const LineInfo& start, const LineInfo& end) {
    if (start.currentBatch == end.currentBatch) {
        CopyInMaskLines(maskLocal, lineCnt, lineNum, start.currentLineInMask);
    } else {
        CopyInMaskLines(maskLocal, lineCnt, linePerBatch - start.currentLineInBatch, start.currentLineInMask);
        CopyInMaskBlock(maskLocal, lineCnt, linePerBatch, end.currentBatch - start.currentBatch - 1, 0);
        if (end.currentLineInBatch != 0) {
            CopyInMaskLines(maskLocal, lineCnt, end.currentLineInBatch, 0);
        }
    }
}
```

**vs. baseline:**
```cpp
// baseline不支持mask
```

Benefit: 减少GM访问次数，节省内存带宽，提高mask处理效率
Trade-off: 代码复杂度增加，需要维护4种mask模式的专门处理逻辑

---

## Variant W: SIMT向量化并行计算
Source: sparse_to_dense

专家实现采用SIMT (Single Instruction Multiple Thread)架构处理稀疏索引到密集索引的映射计算。核心思想是将稀疏索引的写入操作并行化到1024个线程：1) 线程分配：使用USED_THREAD = 1024个线程，每个线程处理一部分稀疏索引；2) 索引计算：每个线程通过Simt::GetThreadIdx()获取线程ID，按照步长Simt::GetThreadNum()遍历分配到的索引；3) 多维索引映射：通过循环累加计算多维索引对应的线性偏移，公式为outputIdx += indices[idx * numDims + i] * strides。这种SIMT设计相比baseline的串行处理（逐个索引顺序处理）可以实现接近1024倍的并行加速，尤其是在稀疏索引数量较大的场景下效果显著。

**Expert implementation:**
```cpp
AscendC::Simt::VF_CALL<SparseToDenseSimtCompute>(
    AscendC::Simt::Dim3(USED_THREAD),  // 1024 threads
    numValues, numDims, sparseUsedCoreNum, sparseBlockOffset, ...);

__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) 
inline void SparseToDenseSimtCompute(...) {
    for (COMP_T idx = sparseBlockOffset + Simt::GetThreadIdx(); 
         idx < sparseBlockOffset + sparseDataNum;
         idx += Simt::GetThreadNum()) {
        // Compute output index and write
    }
}
```

**vs. baseline:**
```cpp
for (uint32_t idx = startIdx; idx < endIdx; idx++) {
    ProcessIndex(idx);  // Serial processing
}
```

Benefit: 1024线程并行，理论上可达1024倍加速
Trade-off: 需要SIMT支持，增加了代码复杂度
