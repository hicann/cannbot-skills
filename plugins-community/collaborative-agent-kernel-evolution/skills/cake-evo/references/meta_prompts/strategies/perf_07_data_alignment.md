# P7: 32B Alignment + DataCopyPad (数据对齐与填充)
## Overview
专家实现充分利用了Ascend C的向量化指令（如Adds、Cast、Duplicate等）进行高效计算。在foreach_add_scalar.cpp中，通过AddsAdapter函数包装Adds指令，实现了类型安全的高性能加法操作。对于BF16和FP16类型，专家实现采用FP32中间计算（通过Cast转换），然后使用RoundMode::CAST_RINT进行四舍五入，在保证精度的同时充分利用向量化指令的吞吐量。此外，专家实现使用了ListTensorDesc来描述tensor列表的内存布局，支持非连续存储的tensor（通过AutoContiguous标记在Host端处理）。baseline实现虽然使用了Add和Duplicate指令，但没有考虑向量化对齐和数据类型转换优化。

## When to Use
- Non-aligned input shapes
- 精细的传输控制减少DMA启动次数，提升内存带宽利用率20-40%
- 避免非对齐访问带来的性能损失，最大化向量指令的吞吐量
- 支持非对齐数据访问，减少数据预处理开销

## Trade-off
- 需要复杂的参数计算
- 可能需要额外的内存开销用于对齐填充
- 需要额外计算stride参数

**Source operators**: adaptive_max_pool3d_grad, add_rms_norm_cast, ascend_quant_v2, batch_norm_v3, clipped_swiglu, deep_norm, dequant_bias, dynamic_mx_quant, dynamic_quant_update_scatter_v2, fake_quant_affine_cachemask, foreach_add_list, foreach_add_scalar, foreach_addcdiv_list, gemma_rms_norm, grouped_dynamic_mx_quant, inplace_add_rms_norm, layer_norm_v4, linear_index, max_pool_grad_with_argmax_common, modulate, multi_scale_deformable_attn_function, norm_common, rms_norm_quant, scaled_masked_softmax_grad_v2, scaled_masked_softmax_v2, sparse_to_dense, trans_quant_param_v2

---

## Variant A: 32字节对齐的数据拷贝优化
Source: foreach_add_list, foreach_addcdiv_list

专家实现中通过ADDCDIV_LIST_BYTE_PER_BLOCK = 32常量实现了32字节对齐的数据拷贝优化。当数据大小是32字节的整数倍时，使用DataCopy进行高效拷贝；否则，计算对齐后的数据量进行拷贝。32字节是昇腾芯片的内存访问对齐粒度，对齐访问可以最大化内存带宽利用率。

**Expert implementation:**
```cpp
// 专家实现: 32字节对齐优化
constexpr uint8_t ADDCDIV_LIST_BYTE_PER_BLOCK = 32;
if (uValue * sizeof(T) % ADDCDIV_LIST_BYTE_PER_BLOCK == 0) {
    DataCopy(dstLocal, tensor1Local, uValue);
} else {
    int32_t dataCountInBlock = ADDCDIV_LIST_BYTE_PER_BLOCK / sizeof(T);
    DataCopy(dstLocal, tensor1Local, (uValue + dataCountInBlock - 1) / dataCountInBlock * dataCountInBlock);
}
```

**vs. baseline:**
```cpp
// baseline: 无对齐优化
AscendC::DataCopy(outputGm[tile_offset], outLocal, current_tile_size);
```

Benefit: 提高内存访问效率，减少非对齐访问开销约5-10%; 最大化内存带宽利用率，提升数据拷贝效率
Trade-off: 代码复杂度增加; 需要处理对齐计算逻辑

---

## Variant B: DataCopyExt优化内存传输
Source: adaptive_max_pool3d_grad

专家实现使用DataCopyExtParams结构进行精细的内存传输控制。blockCount/blockLen支持多维数据块的批量传输，srcStride/dstStride处理非连续内存布局，DataCopyPadExtParams支持padding和自动对齐。相比baseline的DataCopyPad基本用法，专家实现通过精心计算的stride参数，可以一次性传输整个NC块的数据，减少了DMA启动次数，提高了内存带宽利用率。针对非对齐数据，还实现了拆分为32B对齐的小块+tail处理的策略。

**Expert implementation:**
```cpp
DataCopyExtParams copyParamsGrad;
copyParamsGrad.blockCount = core_.ncShape;
copyParamsGrad.blockLen = block_.dohowoShape * sizeof(TGrad);
copyParamsGrad.srcStride = (params_.doDim * params_.hoDim * params_.woDim - block_.dohowoShape) * sizeof(TGrad);
copyParamsGrad.dstStride = 0;
DataCopyPadExtParams<TGrad> padGrad{false, 0, 0, 0};
DataCopyPad(gradUb, gradGm[block_.offsetGrad], copyParamsGrad, padGrad);

if (baseblockLen > REPEAT_BASE_BLOCK_LEN && baseblockLen % REPEAT_BASE_BLOCK_LEN == 0) {
    // 直接使用大块传输
} else {
    uint16_t baseBlockCount = totalBlockLen / REPEAT_BASE_BLOCK_LEN;
    // 拆分为32B对齐的小块+tail处理
}
```

**vs. baseline:**
```cpp
AscendC::DataCopyPad(gradOutLocal, gradOutputGm[grad_out_base + c0], 
                    {1, static_cast<uint16_t>(actualTileC * sizeof(float)), 0, 0, 0}, 
                    {false, 0, 0, 0});
```

Benefit: 精细的传输控制减少DMA启动次数，提升内存带宽利用率20-40%
Trade-off: 需要复杂的参数计算

---

## Variant C: 数据对齐与向量化访问
Source: add_rms_norm_cast

专家实现充分考虑了Ascend C的向量计算单元特性，对数据进行对齐处理。使用numColAlign确保列数是ONE_BLK_SIZE / sizeof(T)的整数倍，从而最大化向量指令效率。对齐策略包括：使用AlignUp函数对列数进行上对齐；在SplitD和MultiN模式中，根据block size对齐数据；使用DataCopyCustom替代DataCopyPad，提供更高的搬移效率。

**Expert implementation:**
```cpp
// 显式对齐处理
uint32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
this->numColAlign = AlignUp(numCol, numPerBlock);
// 在计算中使用对齐后的尺寸
for (uint32_t i_i = 0; i_i < calc_row_num; i_i++) {
    ReduceSumCustom(rstdLocal[i_i * NUM_PER_BLK_FP32], sqx[i_i * numColAlign], reduce_buf_local, numCol);
}
```

Benefit: 避免非对齐访问带来的性能损失，最大化向量指令的吞吐量
Trade-off: 可能需要额外的内存开销用于对齐填充

---

## Variant D: DataCopyPad和对齐优化
Source: ascend_quant_v2

使用DataCopyPad代替DataCopy，支持自动填充和非对齐访问，避免额外的数据整理开销。GetXInCopyParams和GetOutCopyParams函数根据tilingData动态计算srcStride和dstStride，实现高效的strided memory access。

**Expert implementation:**
```cpp
// DataCopyPad支持填充和对齐
DataCopyExtParams copyParams;
DataCopyPadExtParams<T> padParams = {false, 0, 0, 0};
GetXInCopyParams(tilingData_, xN, xLen, tilingData_.dim1, copyParams);
DataCopyPad(xLocal, xGm_[xInOffset], copyParams, padParams);

// 动态stride计算
if (runTilingData.baseLen > xLen) {
    copyParams.dstStride = (runTilingData.baseLen - xLen) * sizeof(T) / BLOCK_SIZE;
}
if (lastDimLen > xLen) {
    copyParams.srcStride = (lastDimLen - xLen) * sizeof(T);
}
```

**vs. baseline:**
```cpp
// 简单DataCopy
AscendC::DataCopy(inputLocal, inputGm[offset], this->tileSize);
```

Benefit: 支持非对齐数据访问，减少数据预处理开销
Trade-off: 需要额外计算stride参数

---

## Variant E: 数据对齐与原地计算优化
Source: batch_norm_v3

专家实现采用了严格的数据对齐策略。对于FP16/BF16数据，要求16字节对齐；对于FP32数据，要求8字节对齐。这种对齐不仅满足了硬件的访问粒度要求，还使得原地Cast成为可能——即可以在同一个UB缓冲区中先存放低精度数据，然后直接将其reinterpret cast为FP32进行计算，无需额外的缓冲区。

**Expert implementation:**
```cpp
// Host端对齐计算
commonParams.patternR0Align = (commonParams.xDtype == ge::DT_FLOAT) ?
    Ops::Base::CeilAlign(commonParams.patternR0, B32_BLOCK_ALIGN_NUM) :
    Ops::Base::CeilAlign(commonParams.patternR0, B16_BLOCK_ALIGN_NUM);
// UB分块对齐
aUbFactor = Ops::Base::CeilAlign(td_.get_blockFactor(), B16_BLOCK_ALIGN_NUM);
```

**vs. baseline:**
```cpp
# baseline无对齐处理
base_offset = elements_processed * c + channel_idx
```

Benefit: 原地Cast节省50%的UB空间；对齐访问提升Vector Unit效率约20%
Trade-off: 需要处理Padding数据；对齐后的数据量可能略大于原始数据

---

## Variant F: SkipPad高效均值减法
Source: batch_norm_v3

在BatchNorm中，减均值是一个关键操作。由于数据对齐需求，UB中存储的数据可能包含Padding（补0）区域。直接对整个对齐后的数据进行减法会影响后续方差计算（因为Padding区域的0会被错误地计入）。专家实现了SkipPadSubMean函数，高效跳过Padding区域进行均值减法：1）向量化跳过策略：利用Vector指令的repeat和stride参数，在单个指令中跳过Padding区域；2）多级循环优化：对于大数据量，采用外层repeat循环+内层element循环的分层策略，避免超出硬件repeat限制（255）；3）条件分支优化：根据数据特征（forLoopNum vs patternR1）选择最优循环策略。

**Expert implementation:**
```cpp
// SkipPad高效均值减法
if ((r0ForLoopNum < lineNum) && (patternR0Align < (UINT8_MAX_NUM * B32_BLOCK_ALIGN_NUM))) {
    uint8_t repStride = patternR0Align / B32_BLOCK_ALIGN_NUM;
    for (int64_t i = 0; i < r0ForLoopNum; i++) {
        Adds(calcTensor[i * ELEM_PER_REP_FP32], calcTensor[i * ELEM_PER_REP_FP32], 
             -finalMean, ELEM_PER_REP_FP32, lineNum, {1, 1, repStride, repStride});
    }
    // 处理剩余元素
    if (r0ForRemainNum > 0) {
        int64_t repeatForLoopNum = lineNum / UINT8_MAX_NUM;
        for (int64_t i = 0; i < repeatForLoopNum; i++) {
            Adds(calcTensor[...], calcTensor[...], -finalMean, r0ForRemainNum, UINT8_MAX_NUM, {1, 1, repStride, repStride});
        }
    }
}
```

**vs. baseline:**
```cpp
# baseline无Padding处理
tl.vsub_scalar(temp_ub, data_ub, mean_val)
```

Benefit: 向量化跳过Padding比逐元素处理快5-10倍；避免Padding数据污染方差计算结果
Trade-off: 代码复杂度增加；需要处理多种边界情况

---

## Variant G: 32字节对齐与向量化
Source: clipped_swiglu

专家实现处处考虑32字节对齐（BLOCK_SIZE=32），这是AscendC向量指令的最优访问粒度。通过AlignBytes函数确保所有buffer大小和数据搬运操作都满足32字节对齐要求。对齐的重要性：1)向量指令效率：未对齐访问可能导致性能下降；2)内存访问效率：32字节对齐匹配硬件缓存行大小；3)UB利用率：对齐后UB空间利用更规整。

**Expert implementation:**
```cpp
constexpr static int64_t BLOCK_SIZE = 32;

__aicore__ inline int64_t AlignBytes(int64_t number) {
    return (number + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
}

xQueSpace_ = SWI_FACTOR * DTYPE_FACTOR * AlignBytes(ubMaxPair_ * static_cast<int64_t>(sizeof(bfloat16_t)));
```

Benefit: 确保向量指令最优执行，提升内存访问效率
Trade-off: 少量内存空间用于对齐填充

---

## Variant H: 内存对齐与数据搬运优化
Source: deep_norm

专家实现对内存访问进行了精细的对齐控制。通过 ROUND_UP 宏将数据大小向上对齐到 blockNumEl 的倍数，确保向量指令的高效执行。对于 AICore >= 220 的平台，使用 DataCopyPad 替代普通 DataCopy，支持自动 Padding 功能，可以处理非对齐的数据尾部而无需额外的循环处理。

**Expert implementation:**
```cpp
// 专家实现 - DataCopyPad 对齐优化
__aicore__ inline uint32_t ROUND_UP(uint32_t x) {
    return (x + blockNumEl - 1) / blockNumEl * blockNumEl;
}

#if __CCE_AICORE__ >= 220
DataCopyPadParams temp;
temp.isPad = true;
temp.paddingValue = 0;
temp.rightPadding = ROUND_UP(length) - length;
DataCopyPad(x_local, x_gm[offset], copyInput, temp);
#else
DataCopy(x_local, x_gm[offset], ROUND_UP(size));
#endif
```

**vs. baseline:**
```cpp
// baseline - 基础 DataCopy
AscendC::DataCopyPad(inputLocal, inputGm[offset], 
                     {1, static_cast<uint16_t>(actualTileLen * sizeof(float)), 0, 0}, 
                     {false, 0, 0, 0});
```

Benefit: 提高向量指令效率，减少条件分支
Trade-off: 需要处理不同硬件平台的兼容性

---

## Variant I: 内存对齐与访问模式优化
Source: dequant_bias

专家实现采用64字节对齐策略（nAlign_），这是Ascend NPU内存访问的最优粒度。关键优化包括对齐计算、条件stride设置、DataCopyPad使用。这种对齐策略确保Global Memory访问合并，最大化内存带宽利用率。

**Expert implementation:**
```cpp
// 专家实现：内存对齐优化
nAlign_ = (N * sizeof(float) + 63) / 64 * 64 / sizeof(float);  // 64字节对齐

// 条件stride设置
if ((nAlign_ - N_) * sizeof(XTYPE) >= BLOCK_SIZE) {
    oneBlockMore_ = true;
}

// DataCopy使用对齐参数
DataCopyExtParams extParams;
extParams.blockCount = inRows;
extParams.blockLen = N_ * sizeof(XTYPE);
extParams.srcStride = 0;
extParams.dstStride = oneBlockMore_ ? 1 : 0;
```

**vs. baseline:**
```cpp
// baseline：未考虑对齐
AscendC::DataCopyPad(inputLocal, inputGm[tileStart], {1, static_cast<uint16_t>(tileSize * sizeof(int8_t)), 0, 0, 0}, {false, 0, 0, 0});
```

Benefit: 最大化内存带宽利用率，避免bank conflict
Trade-off: 需要额外的对齐计算和条件判断

---

## Variant J: MicroAPI 寄存器级优化
Source: dynamic_mx_quant

baseline 使用高层 AscendC API (Abs, ReduceMax, Muls, ClampMax 等)，这些 API 虽然易用但性能不是最优。专家实现完全基于 MicroAPI 进行寄存器级编程，直接操作 Vector Unit 寄存器。优势包括：零开销抽象（MicroAPI 直接映射到硬件指令，无函数调用开销）、寄存器复用（通过 RegTensor 显式管理寄存器生命周期，最大化寄存器利用率）、掩码向量操作（使用 MaskReg 实现条件执行，避免分支预测失败）、非对齐数据访问（DataCopyUnAlign 系列指令高效处理非对齐内存访问）、Packed 数据操作（FP4/FP8 使用 packed 格式，需要专门的 pack/unpack 指令）。

**Expert implementation:**
```cpp
// MicroAPI 寄存器级优化 (性能最优)
__VEC_SCOPE__ {
    AscendC::MicroAPI::RegTensor<calcType> xRegTensor;
    AscendC::MicroAPI::RegTensor<calcTypeInt> expRegTensor;
    AscendC::MicroAPI::RegTensor<calcTypeInt> expMaxRegTensor;
    
    // 使用掩码进行向量化比较
    uint32_t pnum = vfLen;
    AscendC::MicroAPI::MaskReg p0 = AscendC::MicroAPI::UpdateMask<calcTypeInt>(pnum);
    
    // 直接提取指数位进行 max 计算
    this->template LoadData<calcType>(xAddr, i * vfLen, xRegTensor, p0);
    AscendC::MicroAPI::And(
        expMaxRegTensor, (AscendC::MicroAPI::RegTensor<calcTypeInt>&)xRegTensor, maxEleRegTensor, p0);
}
```

**vs. baseline:**
```cpp
// 高层 API (性能次优)
AscendC::Abs(absLocal, blockLocal, blockSize);
AscendC::ReduceMax(sharedLocal, absLocal, sharedLocal, blockSize);
float blockMax = sharedLocal.GetValue(0);
```

Benefit: 相比高层 API 可提升 20-50% 性能，特别是在数据量大的情况下
Trade-off: 代码可读性下降，需要对硬件指令集有深入理解；调试难度增加

---

## Variant K: 向量化内存访问与对齐(Vectorized Memory Access)
Source: dynamic_quant_update_scatter_v2

专家实现通过精心的内存对齐策略优化向量化访问效率。对于输入数据，计算sizeHalfLen = (tilingData_.rowLen + FIFTEEN) / SIXTEEN * SIXTEEN确保每行数据16字节对齐；对于输出(INT4)，计算outAlignLen = (tilingData_.rowLen + SIXTY_THREE) / SIXTY_FOUR * SIXTY_FOUR确保64字节对齐。当数据自然不对齐时，使用DataCopyPad进行零填充对齐，避免非对齐访问的性能损失。这种对齐策略与Ascend C的Vector指令宽度(128-bit/256-bit)相匹配。

**Expert implementation:**
```cpp
// 对齐计算
sizeHalfLen = (tilingData_.rowLen + FIFTEEN) / SIXTEEN * SIXTEEN;  // 16字节对齐
rightPadding = sizeHalfLen - tilingData_.rowLen;
if (rightPadding > 0) {
    isPad = true;
}
outAlignLen = (tilingData_.rowLen + SIXTY_THREE) / SIXTY_FOUR * SIXTY_FOUR;  // 64字节对齐

// 带填充的DataCopy
if (isPad) {
    DataCopyParams copyParams{(uint16_t)multiRow, (uint16_t)(tilingData_.rowLen * sizeof(xDtype)), 0, 0};
    DataCopyPadParams padParams{true, 0, rightPadding, 0};
    DataCopyPad(inLocal, inGm[loopNum * lenGMMultiRow], copyParams, padParams);
} else {
    DataCopy(inLocal, inGm[loopNum * lenGMMultiRow], multiRow * tilingData_.rowLen);
}
```

**vs. baseline:**
```cpp
// baseline可能直接按原始长度处理
DataCopy(inLocal, inGm, rowLen);
```

Benefit: 充分发挥Vector指令吞吐能力，避免非对齐访问的性能损失
Trade-off: 需要额外的padding内存，可能轻微增加UB占用

---

## Variant L: 向量化指令与数据对齐
Source: fake_quant_affine_cachemask

专家实现充分利用了昇腾向量化引擎的特性，通过dataPerRepeat（每repeat处理的数据量）进行向量化计算。在Tiling阶段，tileLength被计算为dataPerRepeat的整数倍，确保了向量化指令的高效执行。在kernel实现中，使用了BinaryRepeatParams结构来配置向量化操作的重复参数，包括repeatTimes（重复次数）和mask（每repeat处理的数据掩码）。此外，专家实现还通过totalLengthAligned确保数据在内存中的对齐（32字节边界），这对于向量化内存访问的性能至关重要。

**Expert implementation:**
```cpp
// 专家实现向量化优化
constexpr uint32_t BYTE_BLOCK = 32;
constexpr uint32_t BYTE_REPEAT = 256;

// 数据对齐计算
alignNum = BYTE_BLOCK / SIZE_OF_FP32;
if (dType == ge::DT_FLOAT16) {
    alignNum = BYTE_BLOCK / SIZE_OF_FP16;
}
totalLengthAligned = ((calcLength + alignNum - 1) / alignNum) * alignNum;

// 向量化计算配置
repeatTimes = (this->tileLength + this->mask - 1) / this->mask;
BinaryRepeatParams repeatParams = {1, 1, 1, 8, 8, 8};
Select(selectTemp, maskTemp, oneTensor, zeroTensor, 
       SELMODE::VSEL_TENSOR_TENSOR_MODE, this->mask, repeatTimes, repeatParams);
```

**vs. baseline:**
```cpp
// baseline简单固定tileSize
uint32_t tileSize = elementsPerCore < 2048 ? elementsPerCore : 2048;
```

Benefit: 充分利用向量化引擎，提高指令吞吐率；数据对齐减少内存访问开销
Trade-off: 需要对齐填充，可能浪费少量内存；tileSize必须是dataPerRepeat的整数倍，灵活性降低

---

## Variant M: 向量化指令与数据对齐
Source: foreach_add_scalar

专家实现充分利用了Ascend C的向量化指令（如Adds、Cast、Duplicate等）进行高效计算。在foreach_add_scalar.cpp中，通过AddsAdapter函数包装Adds指令，实现了类型安全的高性能加法操作。对于BF16和FP16类型，专家实现采用FP32中间计算（通过Cast转换），然后使用RoundMode::CAST_RINT进行四舍五入，在保证精度的同时充分利用向量化指令的吞吐量。此外，专家实现使用了ListTensorDesc来描述tensor列表的内存布局，支持非连续存储的tensor（通过AutoContiguous标记在Host端处理）。baseline实现虽然使用了Add和Duplicate指令，但没有考虑向量化对齐和数据类型转换优化。

**Expert implementation:**
```cpp
template <typename T>
__aicore__ void AddsAdapter(
    const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, const T& scalarValue, const int32_t& uValue) {
    Adds(dstLocal, srcLocal, scalarValue, static_cast<uint32_t>(uValue));
}

// BF16精度保护计算
if constexpr (IsSameType<T, bfloat16_t>::value || IsSameType<T, half>::value) {
    Cast(this->castLocal, inLocal, RoundMode::CAST_NONE, dataCount);
    Adds(this->castLocal, this->castLocal, float(this->scalarVal), dataCount);
    Cast(outLocal, this->castLocal, RoundMode::CAST_RINT, dataCount);
}
```

**vs. baseline:**
```cpp
float scaledScalar = this->scalar * this->alpha;
AscendC::Duplicate(scalarLocal, scaledScalar, this->tileSize);
AscendC::Add(outLocal, xLocal, scalarLocal, this->tileSize);
```

Benefit: 充分利用向量化指令吞吐量，保证计算精度，支持灵活的内存布局
Trade-off: 需要处理不同数据类型的转换逻辑，增加代码复杂度

---

## Variant N: 优化 DataCopy 策略
Source: gemma_rms_norm

专家实现封装了 DataCopyCustom 函数，针对不同 NPU 架构（220/310/910 等）使用不同的数据搬运策略。对于非对齐的数据访问，实现了专门的 padding 处理逻辑，避免非法内存访问。baseline实现使用标准的 DataCopyPad，缺少针对特定架构的优化。

**Expert implementation:**
```cpp
// 专家实现 - 条件编译优化 DataCopy
template <typename T, typename U, typename R>
__aicore__ inline void DataCopyCustom(const U& dstTensor, const R& srcTensor, const uint32_t count) {
#if (defined(__CCE_AICORE__) && __CCE_AICORE__ == 220) || (defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
    DataCopyParams copyParams;
    copyParams.blockLen = count * sizeof(T);
    DataCopyPad(dstTensor, srcTensor, copyParams, padParams);
#else
    // 其他架构的标准实现
    if (count % numPerBlock == 0) {
        DataCopy(dstTensor, srcTensor, count);
    } else {
        // 非对齐处理...
    }
#endif
}
```

**vs. baseline:**
```cpp
// baseline - 标准 DataCopyPad
AscendC::DataCopyPad(inputLocal, inputGm[offset], 
    {1, static_cast<uint16_t>(actualTileLen * sizeof(float)), 0, 0}, 
    {false, 0, 0, 0});
```

Benefit: 针对不同 NPU 架构的最优数据搬运，减少 MTE 开销
Trade-off: 需要维护多套条件编译代码

---

## Variant O: 精细的内存访问模式控制
Source: grouped_dynamic_mx_quant

专家实现对Global Memory的访问模式进行了精细控制，以最大化内存带宽利用率。DataCopyExtParams结构化参数明确定义blockCount、srcStride、dstStride等参数，优化跨步访问。DataCopyPadExtParams处理不对齐的数据访问，避免越界访问。mxscale的E8M0_2格式要求两个scale值交织存储，通过Interleave指令高效实现。

**Expert implementation:**
```cpp
DataCopyExtParams inCopyParams_ = {
    static_cast<uint16_t>(blockCount),
    static_cast<uint32_t>(dataLen * sizeof(T)),
    static_cast<uint32_t>((postAxisSize_ - dataLen) * sizeof(T)),
    static_cast<uint32_t>((dataLen +31)/32*2-(dataLen+15)/16),
    static_cast<uint32_t>(0)
};

DataCopyExtParams scaleCopyOutParams = {
    static_cast<uint16_t>((blockCount + 63) / 64), 
    static_cast<uint32_t>(dataLen * 2 * sizeof(uint8_t)),
    static_cast<uint32_t>(2 * ((dataLen + 31) / 32 * 32 - dataLen) / 32),
    static_cast<uint32_t>(2 * (postAxisSize_ - dataLen) * sizeof(uint8_t)), 
    static_cast<uint32_t>(0)
};
```

**vs. baseline:**
```cpp
DataCopyPad(x, xGm_[offset], inCopyParams_, padParams_);
```

Benefit: 最大化内存带宽利用率；支持复杂数据布局；安全的边界处理
Trade-off: 参数计算复杂；需要对内存布局有深入理解

---

## Variant P: 自定义DataCopy与对齐优化
Source: inplace_add_rms_norm

专家实现中自定义了DataCopy函数来处理非对齐的数据传输。标准的DataCopy要求数据按块对齐（32字节），但在实际应用中输入数据可能不满足此要求。实现中的DataCopyCustom函数通过检测数据对齐情况，选择合适的传输策略：对于对齐的数据使用标准DataCopy；对于非对齐的数据使用DataCopyPad或分段传输策略。此外，实现中还使用了DataCopyExtParams和DataCopyPadExtParams来精细控制数据传输的参数。

**Expert implementation:**
```cpp
// 专家实现 - 自定义DataCopy
template <typename T, typename U, typename R>
__aicore__ inline void DataCopyCustom(const U& dstTensor, const R& srcTensor, const uint32_t count)
{
    int32_t numPerBlock = ONE_BLK_SIZE / sizeof(T);
    if (count % numPerBlock == 0) {
        DataCopy(dstTensor, srcTensor, count);
    } else {
        int32_t num = count / numPerBlock * numPerBlock;
        DataCopy(dstTensor, srcTensor, num);
        // 处理尾部数据
        for (int32_t i = 0; i < numPerBlock; i++) {
            T tensorValue = srcTensor.GetValue(count - numPerBlock + i);
            srcTensor.SetValue(i, tensorValue);
        }
        DataCopy(dstTensor[count - numPerBlock], srcTensor, numPerBlock);
    }
}

// DataCopyExtParams使用
DataCopyExtParams copyParams{
    static_cast<uint16_t>(curRows),
    static_cast<uint32_t>(numCol * sizeof(T)),
    static_cast<uint32_t>(srcStride),
    static_cast<uint32_t>(0),
    0
};
DataCopyPad(yGm[offset], yLocal, copyParams);
```

**vs. baseline:**
```cpp
// baseline - 标准DataCopy
AscendC::DataCopy(xLocal, xGm[rowIdx * this->cols], this->cols);
AscendC::DataCopy(yLocal, yGm[rowIdx * this->cols], this->cols);
```

Benefit: 处理非对齐数据，提高灵活性和性能
Trade-off: 代码更复杂，需要处理多种边界情况

---

## Variant Q: 内存对齐与Padding策略
Source: layer_norm_v4

专家实现重视内存对齐，这是Ascend芯片高效访问的关键。Row对齐确保每行数据对齐到16字节（FP32）或32字节（FP16）边界，DataCopyPad使用带padding的数据拷贝自动处理不对齐边界，UB空间分配按32字节对齐。baseline完全没有考虑对齐问题，直接使用原始size分配buffer。

**Expert implementation:**
```cpp
uint64_t alignment = BLOCK_SIZE / DTYPE_SIZE_MAP.at(commonParams.tensorDtype);
commonParams.rowAlign = (commonParams.rowSize + alignment - 1) / alignment * alignment;

DataCopyPadExtParams<T> dataCopyPadExtParams;
dataCopyPadExtParams.isPad = (this->r != this->rAlign);
dataCopyPadExtParams.rightPadding = (this->rAlign - this->r);
DataCopyPad(xInUb, xGm[offset], copyInParams, dataCopyPadExtParams);
```

Benefit: DMA传输和向量指令以最大带宽执行，避免非对齐访问的性能损失和未定义行为
Trade-off: 增加内存占用（需要分配对齐后的大小），需要处理padding数据的正确性

---

## Variant R: DataCopyPad批量传输
Source: linear_index

专家实现使用DataCopyPad进行数据从GM到UB的搬运，并配置了适当的填充参数。DataCopyPad支持非对齐的数据传输，并可以自动处理数据填充，这对于处理尾部数据（tail）特别有用。通过设置indicesAlign参数（对齐到32字节），确保后续向量化计算可以正确执行。此外，专家实现还使用了DataCopyExtParams来配置传输参数，包括stride等，这在处理多维数据时尤为重要。

**Expert implementation:**
```cpp
DataCopyExtParams indicesExtParams = {(uint16_t)1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
DataCopyPadExtParams<T> tPadParams = {false, 0, 0, static_cast<T>(0)};

if constexpr (IS_CAST_INT) {
    DataCopyPadGm2UBImpl(
        (__ubuf__ uint32_t*)indicesLocal.GetPhyAddr(),
        (__gm__ uint32_t*)indicesGm[indicesOffset].GetPhyAddr(), 
        indicesExtParams, padParams);
} else {
    DataCopyPad(indices32Local, indicesGm[indicesOffset], indicesExtParams, tPadParams);
}
```

**vs. baseline:**
```cpp
if (indicesDtype == 1) {
    for (uint32_t i = 0; i < tileSize; i++) {
        int64_t val = indicesGmInt64.GetValue(offset + i);
        indicesLocal.SetValue(i, static_cast<int32_t>(val));
    }
} else {
    for (uint32_t i = 0; i < tileSize; i++) {
        int32_t val = indicesGmInt32.GetValue(offset + i);
        indicesLocal.SetValue(i, val);
    }
}
```

Benefit: DataCopyPad批量传输比标量GetValue快数十倍，硬件DMA引擎 offload 数据搬运
Trade-off: 需要对齐处理，可能产生少量填充开销

---

## Variant S: 内存布局优化与LoopMode
Source: max_pool_grad_with_argmax_common

专家实现通过SetLoopModePara和ResetLoopModePara实现了高效的多维数据搬运。当处理NHWC布局的多维数据时，直接使用DataCopy会导致大量的stride计算和边界检查开销。专家实现使用LoopMode来优化这种场景，通过预定义loop1和loop2的stride参数，让硬件自动处理多维数据的内存访问模式。此外，专家实现还使用了DataCopyPadExtParams来处理数据对齐问题，确保向量化操作的高效执行。

**Expert implementation:**
```cpp
// 专家实现：LoopMode优化多维数据搬运
LoopModeParams loopModeParamsT1;
loopModeParamsT1.loop1Size = hArgmaxActual_;
loopModeParamsT1.loop2Size = nOutputActual_;
loopModeParamsT1.loop1SrcStride = wArgmax_ * cOutput_ * sizeof(T1);
loopModeParamsT1.loop2SrcStride = argmaxPlaneSize_ * cOutput_ * sizeof(T1);
loopModeParamsT1.loop1DstStride = wArgmaxActual_ * cOutputAligned_ * sizeof(T1);
loopModeParamsT1.loop2DstStride = hArgmaxActual_ * wArgmaxActual_ * cOutputAligned_ * sizeof(T1);

SetLoopModePara(loopModeParamsT1, DataCopyMVType::OUT_TO_UB);
DataCopyPad(gradLocal, gradGm_[argmaxGmOffset], copyOutParamT1, paramsT1);
ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
```

**vs. baseline:**
```cpp
// baseline实现：基础DataCopyPad
AscendC::DataCopyPad(argmaxLocal, argmaxGm[base], 
                    {1, static_cast<uint16_t>(tile_c * sizeof(int64_t)), 0, 0, 0}, 
                    {false, 0, 0, 0});
```

Benefit: LoopMode减少地址计算开销，提升多维数据搬运效率；硬件自动处理stride
Trade-off: 需要正确配置loop参数，调试难度增加

---

## Variant T: 动态UB长度计算
Source: modulate

专家实现针对不同数据类型采用动态UB长度计算策略。不同数据类型占用不同字节数（FLOAT=4B, FLOAT16/BFLOAT16=2B），直接影响UB能容纳的元素数量。专家实现在Host端根据dataTypeSize动态计算ubLength，通过'/ ALIGNED_SIZE * ALIGNED_SIZE'确保内存对齐，满足硬件约束。ubTensorNum根据参数状态动态调整（SCALE_AND_SHIFT时使用6个buffer，否则使用5个）。相比固定UB大小的实现，动态计算可使UB利用率提升20-50%，特别是对于小数据类型能处理更多元素。

**Expert implementation:**
```cpp
// 专家实现: 动态计算UB长度
switch (dataType) {
    case ge::DT_FLOAT:
        this->dataTypeSize = 4;
        break;
    case ge::DT_FLOAT16:
        this->dataTypeSize = 2;
        break;
    case ge::DT_BF16:
        this->dataTypeSize = 2;
        break;
}
int64_t ubTensorNum = (this->tilingData.parameterStatus == SCALE_AND_SHIFT) ? UB_TENSOR_NUM_ALL : UB_TENSOR_NUM;
this->tilingData.ubLength =
    static_cast<int64_t>(ubSizePlatForm) / ubTensorNum / this->dataTypeSize / ALIGNED_SIZE * ALIGNED_SIZE;
```

**vs. baseline:**
```cpp
// baseline: 固定UB buffer大小
constexpr int64_t UB_BUFFER_SIZE = 16384;
pipe.InitBuffer(inQueueX, DOUBLE_BUFFER, UB_BUFFER_SIZE * sizeof(float));
```

Benefit: UB利用率提升20-50%，特别是小数据类型场景
Trade-off: Host端增加数据类型判断逻辑，但开销极小

---

## Variant U: DataCopyPad非对齐数据传输
Source: modulate

专家实现使用DataCopyPad API处理非32B对齐的数据传输。AscendC的DataCopy要求数据传输长度为32B对齐，实际场景中D维度可能不是32B的整数倍。专家实现配置CopyParams为{1,opCopyLength*sizeof(T),0,0,0}和PadParams为{true,0,0,0}表示使用padding但不填充特定值。支持任意shape，无需用户手动对齐，提升易用性，性能损失极小。

**Expert implementation:**
```cpp
// 专家实现: DataCopyPad
template <typename T>
__aicore__ inline void CopyCommom(int64_t offset, int64_t opCopyLength, 
                                   LocalTensor<T> &localData, GlobalTensor<T> &gmData) {
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(opCopyLength * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        DataCopyPad(localData[this->ubLength], gmData[offset], copyParams, padParams);
    } else {
        DataCopyPad(localData, gmData[offset], copyParams, padParams);
    }
}
```

**vs. baseline:**
```cpp
// baseline: 标准DataCopy
AscendC::DataCopy(scaleLocal, scaleGm[offset], inputD);
AscendC::DataCopy(xLocal, xGm[offset], tileSize * inputD);
AscendC::DataCopy(yGm[offset], yLocal, tileSize * inputD);
```

Benefit: 支持任意shape，无需手动对齐，提升易用性
Trade-off: 相比DataCopy有轻微性能损失，但在非对齐场景下是必需的

---

## Variant V: 内存对齐访问
Source: modulate

专家实现使用alignedD确保内存对齐访问，满足硬件对齐要求。AscendC Vector指令通常要求32B对齐，计算公式为alignedD=(opCopyLength+alignedLength-1)&~(alignedLength-1)。对齐策略包括计算对齐长度（根据数据类型计算需要对齐的元素数）、分配对齐buffer（UB buffer按alignedD分配）、访问对齐地址（使用jL*alignedD作为行内偏移）。

**Expert implementation:**
```cpp
// 专家实现: 使用alignedD
int64_t alignedLength = ALIGNED_SIZE / sizeof(T);
this->alignedD = (opCopyLength + alignedLength - 1) & ~(alignedLength - 1);
// 使用alignedD访问
for (int64_t jL = 0; jL < handleL; ++jL) {
    Mul(xLocal[jL * this->alignedD], xLocal[jL * this->alignedD], scaleLocal, opCopyLength);
    Add(yLocal[jL * this->alignedD], yLocal[jL * this->alignedD], shiftLocal, opCopyLength);
}
```

**vs. baseline:**
```cpp
// baseline: 直接访问
int64_t tileSize = UB_BUFFER_SIZE / inputD;
AscendC::DataCopy(xLocal, xGm[offset], tileSize * inputD);
// 使用inputD直接访问
AscendC::Muls(yLocal[offset], xLocal[offset], 1.0f, inputD);
```

Benefit: 避免未对齐访问导致的硬件异常或性能下降
Trade-off: 可能需要分配稍大的UB buffer

---

## Variant W: 数据对齐与向量化优化
Source: multi_scale_deformable_attn_function

专家实现中广泛使用了数据对齐策略来确保向量指令的高效执行。AlignUp函数用于将数据大小对齐到32字节（B32）边界，这是Ascend C向量指令的基本单位。关键对齐参数包括：alignedNumPoints_、alignedOneHeadNum_、alignedOneQueryNum_、alignedEmbedDims_等。这些对齐确保了DataCopy、Mul、Add等向量指令可以以最大吞吐量执行，避免了非对齐访问的性能损失。

**Expert implementation:**
```cpp
// 对齐计算
alignedNumPoints_ = AlignUp(num_points, B32_DATA_NUM_PER_BLOCK);
alignedOneHeadNum_ = numLevels_ * alignedNumPoints_;
alignedOneQueryNum_ = AlignUp(numHeads_ * alignedOneHeadNum_, B32_DATA_NUM_PER_REPEAT);
alignedEmbedDims_ = AlignUp(embedDims_, B32_DATA_NUM_PER_BLOCK);

// 向量化参数计算
embedBlk_ = alignedEmbedDims_ / B32_DATA_NUM_PER_BLOCK;
outBlk_ = numHeads_ * embedBlk_;
rptTimes_ = alignedOneQueryNum_ / B32_DATA_NUM_PER_REPEAT;

// 数据广播
Brcb(shapeBrc[k * alignedOneQueryNum_], shapeBrc[k * alignedOneQueryNum_], 1, {1, 8});
```

**vs. baseline:**
```cpp
// baseline实现不存在，无法提供对比代码
```

Benefit: 最大化向量指令吞吐量，避免非对齐访问的性能损失
Trade-off: 增加了内存使用量（由于对齐填充），需要计算对齐参数

---

## Variant X: 内存对齐与数据布局优化
Source: norm_common

专家实现通过rowAlign参数确保每行数据按32字节（或16个float）对齐，满足AI Core的内存访问对齐要求。对齐后的布局使向量化指令可以高效工作，避免非对齐访问的性能损失。同时，SingleRead策略通过一次读取多行数据（nRow行）到UB，提高了数据局部性和复用率。baseline实现未进行内存对齐处理，直接使用原始shape，可能导致非对齐访问性能下降。

**Expert implementation:**
```cpp
// 专家实现：内存对齐计算
ge::graphStatus GetCommonShapeAttrsInfo(...)
{
    commonParams.coefficient = static_cast<float>(1.0) / static_cast<float>(commonParams.rowSize);
    uint64_t alignment = 16;  // 16个元素对齐
    if (DTYPE_SIZE_MAP.find(commonParams.tensorDtype) != DTYPE_SIZE_MAP.end()) {
        alignment = BLOCK_SIZE / DTYPE_SIZE_MAP.at(commonParams.tensorDtype);
    }
    commonParams.rowAlign = (commonParams.rowSize + alignment - 1) / alignment * alignment;
}

// 使用rowAlign进行数据访问
for (uint32_t rowIdx = 0; rowIdx < nRow; ++rowIdx) {
    uint32_t currentRowOffset = rowIdx * rowAlign;  // 使用对齐后的stride
    ReduceSum(yLocal[currentRowOffset], yLocal[currentRowOffset], yLocal[currentRowOffset], rowSize);
}
```

**vs. baseline:**
```cpp
// baseline：无内存对齐处理
uint32_t offset = row_idx * norm_size + col_start;
CopyInInput(offset);
```

Benefit: 满足AI Core对齐要求，避免非对齐访问的性能损失，向量化指令效率提升20-50%
Trade-off: 需要额外存储空间进行padding，内存占用略有增加

---

## Variant Y: 内存对齐优化
Source: rms_norm_quant

专家实现对内存访问进行了精细的对齐控制，定义了REPEAT_TIME_256（256B对齐，用于INT8）、REPEAT_TIME_16（16B对齐，用于FP16）、REPEAT_TIME_64（64B对齐，用于FP32）等常量。根据数据类型和UB大小动态计算对齐后的列数，确保向量化指令能够以最优带宽执行。

**Expert implementation:**
```cpp
static constexpr uint32_t REPEAT_TIME_256 = 256;
static constexpr uint32_t REPEAT_TIME_16 = 16;
static constexpr uint32_t REPEAT_TIME_64 = 64;

if (num_col_ <= slice_size_) {
    num_col_align_int8 = (num_col_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
    num_col_align_int4 = (num_col_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
    num_col_align_f16 = (num_col_ + REPEAT_TIME_16 - 1) / REPEAT_TIME_16 * REPEAT_TIME_16;
    num_col_align_f32 = (num_col_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
}
```

**vs. baseline:**
```cpp
uint32_t tileCols = 256;
```

Benefit: 向量化指令效率最大化，内存带宽利用率提升
Trade-off: 需要额外的padding空间

---

## Variant Z: 数据对齐与Padding策略
Source: scaled_masked_softmax_grad_v2

专家实现通过paddedHeadDim确保数据64字节对齐，这是Ascend C向量化计算的关键。对齐计算使用CeilDiv(headDim, ALIGNED_NUM) * ALIGNED_NUM，条件DataCopyPad仅当headDim % ALIGNED_NUM != 0时使用带padding的拷贝，Duplicate填充在计算前用MASK_VALUE填充padding区域。

**Expert implementation:**
```cpp
paddingNum = (paddedHeadDim_ - headDim_) % (BLK_LEN / dataSize);
dupNum = paddedHeadDim_ - headDim_ - paddingNum;
copyPadExtParams = {true, 0, static_cast<uint8_t>(paddingNum), MASK_VALUE};

if (headDim_ % ALIGNED_NUM == 0) {
    DataCopy(yGradLocal, yGradGm[offset], moveNum);
} else {
    DataCopyPad(yGradLocal, yGradGm[offset], copyExtParams, copyPadExtParams);
}

if (this->dupNum != 0) {
    for (uint64_t i = 0; i < this->lineNum; ++i) {
        Duplicate(yGradLocal[i * this->paddedHeadDim_ + this->alignedHeadDim], MASK_VALUE, this->dupNum);
    }
    PipeBarrier<PIPE_V>();
}
```

**vs. baseline:**
```cpp
AscendC::DataCopyPad(gradOutputLocal, gradOutputGm[loopIdx * this->tileSize], 
                     {1, static_cast<uint16_t>(this->tileSize * sizeof(float)), 0, 0}, 
                     {false, 0, 0, 0});
```

Benefit: 保证向量化计算效率，避免未对齐访问的性能损失
Trade-off: 增加padding计算和Duplicate开销，增加代码复杂度

---

## Variant 27: 32字节对齐与DataCopyPad
Source: scaled_masked_softmax_v2

专家实现严格执行32字节对齐策略，这是昇腾芯片内存访问的基本要求。对于输入x（FP32/FP16/BF16），计算每行需要补充的数据个数xPaddingNum，使得width + xPaddingNum能被32字节对齐；对于mask（bool，1字节），计算maskPaddingNum使得width + maskPaddingNum能被32对齐。在Kernel端，使用DataCopyPad替代普通DataCopy，通过DataCopyPadExtParams指定padding值和padding数量，实现从GM到UB的自动对齐填充。

**Expert implementation:**
```cpp
// 专家实现对齐处理
void SetPaddingInfo() {
    uint64_t alignedXBlock = AlignedBytes / xDtypeSize;  // 32 / 4 = 8
    uint64_t xLeft = width % alignedXBlock;
    uint64_t xPaddingNum = xLeft > 0 ? alignedXBlock - xLeft : 0u;
    tiling.set_padLineNum(width + xPaddingNum);
    
    uint64_t alignedMaskBlock = AlignedBytes / BOOL_SIZE;  // 32 / 1 = 32
    uint64_t maskLeft = width % alignedMaskBlock;
    uint64_t maskPaddingNum = maskLeft > 0 ? alignedMaskBlock - maskLeft : 0u;
    tiling.set_alignedMaskWidth(width + maskPaddingNum);
}

// Kernel端
DataCopyExtParams params = {static_cast<uint16_t>(linePerIter), static_cast<uint32_t>(tilingData.width * sizeof(T)), 0, 0, 0};
DataCopyPadExtParams<T> extParams = {true, 0, static_cast<uint8_t>(tilingData.paddingNum), 0.0};
DataCopyPad(xTensor, gmX[idx * gmOffsetPerIdx], params, extParams);
```

**vs. baseline:**
```cpp
// baseline 无对齐处理
AscendC::DataCopy(xLocal, xGm[xOffset], tileLength);
AscendC::DataCopy(maskLocal, maskGm[maskOffset], tileLength);
```

Benefit: 向量指令执行效率最大化，避免非对齐访问的性能损失或硬件异常，性能提升5-15%
Trade-off: 需要额外的padding计算和存储，内存占用略增

---

## Variant 28: UB优化与批量数据搬运
Source: sparse_to_dense

专家实现在Default Value填充阶段充分利用UB (Unified Buffer)进行批量操作：1) Duplicate指令：使用Duplicate(outputLocal, defaultValueScalar, defaultUbFactor)将默认值快速复制到UB中，这是一条高度优化的向量指令；2) 批量DataCopy：通过DataCopyPad一次性将UB中的数据批量搬运到Global Memory，而不是逐个元素写入；3) UB大小感知：Host端根据UB大小计算defaultUbFactor，确保每次搬运的数据量最大化同时不超出UB容量。这种批量操作相比baseline的逐个元素处理，可以显著减少指令发射开销和内存访问延迟。

**Expert implementation:**
```cpp
LocalTensor<Y_T> outputLocal = outputQue_.AllocTensor<Y_T>();
Duplicate(outputLocal, defaultValueScalar, defaultUbFactor);
outputQue_.EnQue<Y_T>(outputLocal);
outputLocal = outputQue_.DeQue<Y_T>();
DataCopyExtParams outputCopyParamsNorm{1, 
    static_cast<uint32_t>(defaultUbFactor * sizeof(Y_T)), 0, 0, 0};
DataCopyPad(y_[outputOffset], outputLocal, outputCopyParamsNorm);
```

**vs. baseline:**
```cpp
AscendC::LocalTensor<float> valLocal = valuesQueue.AllocTensor<float>();
AscendC::DataCopyPad(valLocal, valuesGm[idx], 
    {1, static_cast<uint16_t>(sizeof(float)), 0, 0});
outLocal.SetValue(0, valLocal.GetValue(0));
```

Benefit: 减少指令发射开销，提高内存带宽利用率
Trade-off: 需要更复杂的UB管理，增加了代码复杂度

---

## Variant 29: SafeDataCopy - 非对齐内存访问处理
Source: trans_quant_param_v2

SafeDataCopy模板函数是专家实现中内存访问优化的核心。NPU的DMA传输通常要求32字节对齐，但实际的tensor shape往往无法满足这一要求。专家实现通过三种策略处理这种情况：1)对于支持DataCopyPad的新平台，直接使用硬件提供的填充功能；2)对于已对齐的数据，使用标准DataCopy；3)对于未对齐的数据，采用地址回退策略（rollback strategy）——将最后一个不完整块的数据复制到前一个完整块中，确保DMA传输的完整性。这种策略避免了内存踩踏，同时最大限度地利用了硬件能力。特别值得一提的是，该模板还支持forAtomicAdd模式，在原子加场景下将回退部分置零以避免重复累加。

**Expert implementation:**
```cpp
template <bool forAtomicAdd = false, typename T>
__aicore__ inline void SafeDataCopy(...) {
    if constexpr (IsDataCopyPadSupport() && sizeof(T) < 8) {
        DataCopyPad(dstGlobal, srcLocal, copyParams);
    } else if (likely(!(calCount % numElemsPerBlock))) {
        DataCopy(dstGlobal, srcLocal, copyParams);
    } else {
        const int numAlignedBlocks = calCount / numElemsPerBlock * numElemsPerBlock;
        DataCopy(dstGlobal, srcLocal, numAlignedBlocks);
        const size_t rollbackDstIdx = numAlignedBlocks - numElemsPerBlock;
        for (int i = 0; i < numElemsPerBlock; ++i) {
            srcLocal.SetValue((rollbackDstIdx + i), srcLocal.GetValue(rollbackSrcIdx + i));
        }
        DataCopy(dstGlobal[calCount - numElemsPerBlock], srcLocal[rollbackDstIdx], numElemsPerBlock);
    }
}
```

**vs. baseline:**
```cpp
with tl.copyin():
    tl.load(scale_ptr + offsets, scale_ub)
with tl.copyout():
    tl.store(output_scale_ptr + offsets, output_scale_ub)
```

Benefit: 安全处理非对齐内存访问，避免内存踩踏，支持多种硬件平台
Trade-off: 增加了代码复杂度，非对齐访问有一定的性能开销
