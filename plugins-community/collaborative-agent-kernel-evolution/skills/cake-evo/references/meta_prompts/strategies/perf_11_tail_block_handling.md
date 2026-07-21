# P11: Tail Block Handling (尾块与边界处理)
## Overview
专家实现精心设计了尾块处理机制以解决数据量不能被核心数整除的问题：1) 双轨制计算：同时计算正常核（normCoreHandleDefaultValues/normCoreHandleSparses）和尾核（tailCoreHandleDefaultValues/tailCoreHandleSparses）的处理数据量；2) 循环展开：正常循环使用defaultUbFactor批量处理，尾循环单独处理剩余数据；3) Loop/Tail分离：Host端计算normBlockLoop/tailBlockLoop和normBlockTailLoopSize/tailBlockTailLoopSize，Kernel端根据blockIdx_判断自己是正常核还是尾核。这种设计确保了在所有场景下都能实现近完美的负载均衡，避免了因数据量不是核心数倍而导致的性能损失。

## When to Use
- Pooling/gather with uneven splits
- 正确处理非对齐数据，避免数据竞争导致的错误结果
- 避免atomic操作开销，正确处理尾块数据布局
- 向量化处理可一次处理64个元素，相比标量处理提升4-8倍性能

## Trade-off
- AtomicAdd操作有一定性能开销
- 需要使用额外的tmpPattern buffer
- 需要预计算和存储kernel索引，增加UB使用

**Source operators**: adaptive_avg_pool3d, adaptive_max_pool3d_grad, batch_norm_v3, dynamic_block_quant, foreach_add_list, foreach_add_scalar_list, max_pool_grad_with_argmax_common, rms_norm_grad, sparse_to_dense

---

## Variant A: 余数处理与边界对齐
Source: foreach_add_list, foreach_add_scalar_list

专家实现通过isRemainder标志位处理非对齐的余数数据。当张量大小不是maxDataCount的整数倍时，最后一个tile使用DataCopyPadExtParams进行带填充的数据拷贝，避免访问越界。主循环使用DataCopy高效处理完整tile，余数部分使用DataCopyPad安全处理。

**Expert implementation:**
```cpp
// 专家实现: 余数处理
for (uint32_t i = 0; i < copyTimes; i++) {
    bool isRemainder = (i == copyTimes - 1 && copyTimesRemainder > 0);
    uint32_t tempDataCount = isRemainder ? copyTimesRemainder : Base::maxDataCount;
    if (isRemainder) {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(dataCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(dataLocal, inTensorsGM[...], copyParams, padParams);
    } else {
        DataCopy(dataLocal, inTensorsGM[...], dataCount);
    }
}
```

**vs. baseline:**
```cpp
// baseline: 固定tile处理
for (uint32_t i = 0; i < this->innerLoops; i++) {
    CopyIn(i);
    Compute(i);
    CopyOut(i);
}
```

Benefit: 保证数据安全性，避免越界访问，同时主循环保持高效; 确保所有边界情况下的计算正确性
Trade-off: 代码复杂度增加，需要处理分支逻辑; 余数处理增加了少量代码复杂度

---

## Variant B: 非对齐数据的原子操作处理
Source: adaptive_avg_pool3d

针对数据非32B对齐的情况，使用AtomicAdd避免数据竞争。在尾块处理时，如果剩余数据长度不足一个block，使用Duplicate清零超出部分，然后设置AtomicAdd模式执行DataCopy，最后恢复AtomicNone模式。

**Expert implementation:**
```cpp
// 原子操作处理非对齐
uint64_t mask0 = (1ul << numPerBlock) - (1ul << validDataLen);
uint64_t mask[2] = {mask0, 0};
Duplicate<T>(outputLocal, 0, mask, 1, 1, 1);
SetAtomicAdd<T>();
DataCopy(outputGlobal[offset], outputLocal, cTailAlign);
SetAtomicNone();
```

**vs. baseline:**
```cpp
// baseline无尾块特殊处理
AscendC::DataCopyPad(outputGm[out_offset], outputLocal, 
                     {1, static_cast<uint16_t>(C * sizeof(float)), 0, 0, 0});
```

Benefit: 正确处理非对齐数据，避免数据竞争导致的错误结果
Trade-off: AtomicAdd操作有一定性能开销

---

## Variant C: GatherMask指令处理尾块
Source: adaptive_avg_pool3d

对于最后一个输出点的尾块，使用GatherMask指令进行数据重排。当数据长度不是32B整数倍时，将分散的有效数据gather到连续buffer中再写入Global Memory。使用pattern buffer定义gather模式，float类型用uint32_t pattern，half/bfloat16_t用uint16_t pattern。

**Expert implementation:**
```cpp
// GatherMask处理尾块
if constexpr (std::is_same_v<T, float>) {
    LocalTensor<uint32_t> bufPattern = tmpPattern.Get<uint32_t>();
    int32_t preLeftShift = numPerBlock + lastLeftShift;
    bufPattern.SetValue(0, (1u << preLeftShift) - (1u << lastLeftShift));
    GatherMask(outputLocal[gatherOffset], outputLocal[gatherOffset], 
               bufPattern, true, mask, {1, 1, 8, 8}, rsvdCnt);
}
```

**vs. baseline:**
```cpp
// baseline无尾块特殊处理
```

Benefit: 避免atomic操作开销，正确处理尾块数据布局
Trade-off: 需要使用额外的tmpPattern buffer

---

## Variant D: Kernel索引预生成与Mask向量化
Source: adaptive_max_pool3d_grad

专家实现在Normal策略中使用了预生成kernel索引+mask选择的技术。GenkernelIndex预先生成kernel内所有位置的索引（flattened index），使用Duplicate+Adds指令生成等差数列，利用向量化指令批量生成kd*kh*kw个索引。然后使用Compare+Select指令批量生成mask和选择梯度，Compare生成uint8 mask（标识哪些kernel位置匹配argmax），Select根据mask选择对应的梯度值。相比baseline的逐个元素处理，这种向量化方法可以一次处理64个元素（VL=64），大幅提升计算效率。

**Expert implementation:**
```cpp
__aicore__ inline void GenkernelIndex(LocalTensor<float>& dstLocal) {
    float firstValue = 0;
    Duplicate(dstLocal, firstValue, params_.singleCoreNc);
    PipeBarrier<PIPE_V>();
    for (uint64_t wIdx = 1; wIdx < kW; wIdx++) {
        Adds(dstLocal[wIdx * params_.singleCoreNc], dstLocal, 1.f * wIdx, params_.singleCoreNc);
    }
    PipeBarrier<PIPE_V>();
    for (uint64_t hIdx = 1; hIdx < kH; hIdx++) {
        Adds(dstLocal[hIdx * kW * params_.singleCoreNc], dstLocal, 1.f * (hIdx * kW), kW * params_.singleCoreNc);
    }
}

LocalTensor<uint8_t> maskUb = maskBuf.Get<uint8_t>();
Compare(maskUb, kernelIdx, indicesFloat[params_.singleCoreNc * woCntIndex], 
        CMPMODE::EQ, mask, params_.maxKdhwLen, repeatParams);
PipeBarrier<PIPE_V>();
SelectGrad(gradSelUb, maskUb, gradTranUb, woCntIndex);
```

**vs. baseline:**
```cpp
int64_t kd = idx / (kernel_h * kernel_w);
int64_t tmp_idx = idx % (kernel_h * kernel_w);
int64_t kh = tmp_idx / kernel_w;
int64_t kw = tmp_idx % kernel_w;
```

Benefit: 向量化处理可一次处理64个元素，相比标量处理提升4-8倍性能
Trade-off: 需要预计算和存储kernel索引，增加UB使用

---

## Variant E: Adaptive Pooling重叠检测与处理
Source: adaptive_max_pool3d_grad

Adaptive Max Pooling的一个特点是kernel可能重叠（当输入尺寸不能被输出尺寸整除时）。专家实现专门处理了这种情况：通过diDim % doDim != 0等判断是否存在重叠，传递isOverLap标志到kernel选择不同的计算路径，在overlap场景使用atomic add保证正确性，FP16/BF16场景使用FP32 workspace累加。baseline实现没有检测重叠情况，而是简单地逐个处理元素，这在大数据量场景下会导致性能问题和潜在的竞态条件。

**Expert implementation:**
```cpp
bool isDOverlap = (maxPoolGradParams.diDim % maxPoolGradParams.doDim) != 0UL;
bool isHOverlap = (maxPoolGradParams.hiDim % maxPoolGradParams.hoDim) != 0UL;
bool isWOverlap = (maxPoolGradParams.wiDim % maxPoolGradParams.woDim) != 0UL;
maxPoolGradParams.isOverLap = (isDOverlap || isHOverlap || isWOverlap);

if constexpr (IsOverlap) {
    SetAtomicAdd<T>();
}
DataCopyPad(outGm[block_.offsetY], yUb[...], copyParamsY);
if constexpr (IsOverlap) {
    SetAtomicNone();
}
```

**vs. baseline:**
```cpp
uint32_t d_start = d_out * depthIn / depthOut;
uint32_t d_end = (d_out + 1) * depthIn / depthOut;
// 无重叠检测
```

Benefit: 保证重叠场景下的计算正确性，避免竞态条件
Trade-off: atomic add带来一定性能开销

---

## Variant F: 多维度Tiling与并行策略
Source: batch_norm_v3

专家实现支持Channel维度（A）和Spatial维度（R0/R1）的多级并行切分，根据数据Shape特征动态选择最优切分策略：1）Channel维度并行（A维度）：将不同Channel分配给不同AI Core并行计算，通过blockFactor控制每核处理的Channel数；2）R0维度切分：当Spatial维度较大时，在UB级别对R0进行循环切分，通过r0UbFactor控制每次计算的Spatial元素数；3）R1维度补充切分：当R0维度较小但R1维度较大时，采用R1补充切分策略，将多个N（batch）合并处理，提高计算并行度；4）尾块优化：对每个维度的切分都考虑了尾块（Tail）处理，通过tailCoreBlockFactor、aUbTail、r0UbTail等参数精确控制边界处理。

**Expert implementation:**
```cpp
// Channel维度并行切分
int64_t blockFactor = Ops::Base::CeilDiv(commonParams.patternA, static_cast<int64_t>(commonParams.coreNum));
usedCoreNum = Ops::Base::CeilDiv(commonParams.patternA, blockFactor);
td_.set_blockFactor(blockFactor);
td_.set_tailCoreBlockFactor(commonParams.patternA - (usedCoreNum - 1) * blockFactor);

// R0维度UB切分
td_.set_r0UbLoop(Ops::Base::CeilDiv(commonParams.patternR0, r0UbFactor));
td_.set_r0UbTail(commonParams.patternR0 - (td_.get_r0UbLoop() - 1) * r0UbFactor);

// R1补充切分策略
if ((commonParams.patternR0Align <= (r0UbFactor / TWO_NUM)) && commonParams.patternR1 > 1) {
    int64_t procNR0 = Ops::Base::FloorDiv(r0UbFactor, commonParams.patternR0Align);
    int64_t nR0Loop = Ops::Base::CeilDiv(commonParams.patternR1, procNR0);
    td_.set_procNR0(procNR0);
    td_.set_nR0Loop(nR0Loop);
}
```

**vs. baseline:**
```cpp
# baseline简单Channel并行
n_cores = 32
channels_per_core = c // n_cores
tile_size = min(max_ub_elements // 2, elements_per_channel)
```

Benefit: 多核利用率最大化；UB内存利用率最优；支持任意Shape配置
Trade-off: Tiling逻辑复杂；需要根据硬件特性调优参数

---

## Variant G: 内存访问对齐优化
Source: dynamic_block_quant

专家实现针对昇腾硬件的内存访问特性进行了精细优化。硬件要求Global Memory访问必须32字节对齐，因此专家实现设计了多层次的padding策略：1) 计算colPadExtNum（需要pad的数据量）、colPadToBlockNum（pad到block边界的数据量）、colClearExtNum（需要清零的数据量）；2) 根据对齐情况选择不同的DataCopy变体：直接DataCopy（无需pad）、DataCopyParams（指定gap）、DataCopyPadExt（显式padding）；3) 对输出使用DataCopyPad处理非对齐的尾块。这种设计确保了所有内存访问都符合硬件对齐要求，避免了额外的数据搬运开销和潜在的错误。

**Expert implementation:**
```cpp
if (colPadExtNum == 0) {
    DataCopy(xLocal, xGM[rowStart * colNum], rowCount * colNum);
} else if (colPadToBlockNum == 0) {
    DataCopyParams copyParams{static_cast<uint16_t>(rowCount), 
        static_cast<uint16_t>(colNum * sizeof(T) / SINGLE_DATA_BLOCK_SIZE), 0,
        static_cast<uint16_t>(colPadExtNum * sizeof(T) / SINGLE_DATA_BLOCK_SIZE)};
    DataCopy(xLocal, xGM[rowStart * colNum], copyParams);
} else {
    DataCopyExtParams copyExtParams{static_cast<uint16_t>(rowCount), 
        static_cast<uint16_t>(colNum * sizeof(T)), 0,
        static_cast<uint16_t>(colPadExtNum * sizeof(T) / SINGLE_DATA_BLOCK_SIZE), 0};
    DataCopyPadExtParams<T> padParams{static_cast<uint16_t>(1), 0, 
        static_cast<uint8_t>(colPadToBlockNum), 0};
    DataCopyPad(xLocal, xGM[rowStart * colNum], copyExtParams, padParams);
}
```

**vs. baseline:**
```cpp
AscendC::DataCopy(inputLocal, inputGm[blockStart], blockSize);
```

Benefit: 符合硬件对齐要求，避免性能损失和数据错误
Trade-off: 增加代码复杂度，需要计算各种padding参数

---

## Variant H: Argmax索引转换优化
Source: max_pool_grad_with_argmax_common

专家实现针对argmax索引的格式转换进行了深度优化。原始argmax索引通常是flattened格式（HWC），需要转换为分层的H、W、C索引才能进行梯度散布。专家实现通过IndexConvNhwc函数使用纯向量化操作完成这个转换，避免了标量运算。关键技巧包括：使用Div指令计算H索引，使用Mul+Sub计算W索引，使用Arange生成C索引，最后通过Muls和Add组合成最终的扁平索引。这种向量化转换比逐元素循环快数倍。

**Expert implementation:**
```cpp
// 专家实现：向量化索引转换
template <typename T, const uint32_t IS_MUL_C = 0>
__aicore__ inline void IndexConvNhwc(MicroAPI::RegTensor<T>& argmaxReg, 
                                     MicroAPI::RegTensor<int32_t>& hIndexReg,
                                     MicroAPI::RegTensor<int32_t>& wIndexReg, 
                                     ...)
{
    AscendC::MicroAPI::RegTensor<T> hTmpIndexReg;
    AscendC::MicroAPI::RegTensor<T> wTmpIndexReg;
    AscendC::MicroAPI::MaskReg allMask = 
        AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();

    // H = argmax / W（向量化除法）
    AscendC::MicroAPI::Div(hTmpIndexReg, argmaxReg, wOutputConstReg, allMask);
    AscendC::MicroAPI::Adds(hIndexReg, hTmpIndexReg, T(-curHIndex), allMask);

    // W = argmax - H * W（向量化乘减）
    AscendC::MicroAPI::Mul(wTmpIndexReg, hTmpIndexReg, wOutputConstReg, allMask);
    AscendC::MicroAPI::Sub(wTmpIndexReg, argmaxReg, wTmpIndexReg, allMask);
    AscendC::MicroAPI::Adds(wIndexReg, wTmpIndexReg, T(-curWIndex), allMask);

    // 组合成最终索引: ((H * W) + W) * C + C + N
    AscendC::MicroAPI::Muls((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, 
                            hIndexReg, wOutputActual, allMaskU32);
    AscendC::MicroAPI::Add((AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg,
                           (AscendC::MicroAPI::RegTensor<int32_t>&)argmaxReg, 
                           wIndexReg, allMaskU32);
    // ... 继续计算C和N偏移
}
```

**vs. baseline:**
```cpp
// baseline实现：标量索引转换
int64_t idx = GetArgmaxValue(c_offset);
// 将flattened索引转换为(h, w)坐标
uint32_t h_idx = idx / W_in;
uint32_t w_idx = idx % W_in;
```

Benefit: 纯向量化索引转换，性能比标量循环提升数倍
Trade-off: 需要理解索引计算逻辑，代码可读性降低

---

## Variant I: DataCopyPad对齐处理
Source: rms_norm_grad

专家实现针对不同数据类型和场景使用不同的数据拷贝策略：使用DataCopyPad支持padding和对齐；通过DataCopyExtParams精确控制拷贝参数；对于非对齐的尾块使用逐元素处理。这种精细的内存访问控制确保了数据的正确性和访问效率。baseline实现使用简单的DataCopy，可能无法正确处理非对齐数据。

**Expert implementation:**
```cpp
DataCopyExtParams dataCopyParams{(uint16_t)calcLen, (uint32_t)(colVal_ * sizeof(T_DY)), 0, 0, 0};
DataCopyPadExtParams<T_DY> padParams{true, 0, 0, 0};
DataCopyPad(xLocal[calcLen * colValAlign_], xGm_[rowIdx * colVal_], dataCopyParams, padParams);
```

**vs. baseline:**
```cpp
AscendC::DataCopy(xLocal, xGm[rowIdx * cols], cols);
```

Benefit: 支持非对齐数据传输；精确的padding控制；提升内存访问效率
Trade-off: API复杂度增加；需要更多参数配置

---

## Variant J: 尾块处理与负载均衡
Source: sparse_to_dense

专家实现精心设计了尾块处理机制以解决数据量不能被核心数整除的问题：1) 双轨制计算：同时计算正常核（normCoreHandleDefaultValues/normCoreHandleSparses）和尾核（tailCoreHandleDefaultValues/tailCoreHandleSparses）的处理数据量；2) 循环展开：正常循环使用defaultUbFactor批量处理，尾循环单独处理剩余数据；3) Loop/Tail分离：Host端计算normBlockLoop/tailBlockLoop和normBlockTailLoopSize/tailBlockTailLoopSize，Kernel端根据blockIdx_判断自己是正常核还是尾核。这种设计确保了在所有场景下都能实现近完美的负载均衡，避免了因数据量不是核心数倍而导致的性能损失。

**Expert implementation:**
```cpp
normBlockLoop_ = Ops::Base::CeilDiv(normCoreHandleDefaultValues_, defaultUbFactor_);
normBlockTailLoopSize_ = normCoreHandleDefaultValues_ - defaultUbFactor_ * (normBlockLoop_ - 1);
tailBlockLoop_ = Ops::Base::CeilDiv(tailCoreHandleDefaultValues, defaultUbFactor_);
tailBlockTailLoopSize_ = tailCoreHandleDefaultValues - defaultUbFactor_ * (tailBlockLoop_ - 1);

// Kernel:
loop_ = tilingData_.normBlockLoop;
tailLoopSize_ = tilingData_.normBlockTailLoopSize;
if (blockIdx_ == tilingData_.defaultValueUsedCoreNum - 1) {
    loop_ = tilingData_.tailBlockLoop;
    tailLoopSize_ = tilingData_.tailBlockTailLoopSize;
}
```

**vs. baseline:**
```cpp
uint32_t startIdx = coreId * this->indicesPerCore;
uint32_t endIdx = startIdx + this->indicesPerCore;
if (endIdx > this->numIndices) {
    endIdx = this->numIndices;  // Simple truncation
}
```

Benefit: 近完美的多核负载均衡，避免尾核空闲
Trade-off: 增加了Tiling数据结构和计算逻辑的复杂度
