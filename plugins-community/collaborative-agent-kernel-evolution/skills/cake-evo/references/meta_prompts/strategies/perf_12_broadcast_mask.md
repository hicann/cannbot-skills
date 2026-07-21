# P12: Broadcast & Mask Operations (广播与掩码操作)
## Overview
专家实现使用SelectWithBytesMask高阶API实现mask应用，替代手动if-else或Add操作。该API的语义是：当mask对应位置为true时，dst取value；否则取src。这种设计正好符合scaled masked softmax的需求：mask为true的位置应该被替换为MASK_VAL（-10000.0），这是一个极小的值，在softmax中会变成接近0的概率。相比baseline中使用Add操作（要求mask是float类型且值为0/-inf），这种bool mask方式更加内存高效（1字节 vs 4字节）且语义清晰。

## When to Use
- Operators with mask inputs, broadcasting dimensions, or conditional selection
- 统一的向量计算流程，更高的执行效率，更好的精度控制
- 支持更通用的广播模式，编译期确定分支，零运行时开销，针对特定模式深度优化
- 精确mask应用，避免乘法mask的精度损失

## Trade-off
- 需要额外的广播操作，但通常可以被内存访问隐藏
- Host端需要额外的广播模式识别逻辑，增加代码复杂度
- 需要额外buffer，增加计算步骤

**Source operators**: apply_adagrad_d, masked_scatter_with_position, scaled_masked_softmax_grad_v2, scaled_masked_softmax_v2

---

## Variant A: 标量数据的广播优化
Source: apply_adagrad_d

对于学习率lr这种标量输入，专家实现使用Vec::Duplicate<U>将其广播为向量，而不是像baseline那样在标量层面处理。这样可以统一向量计算流程，避免标量-向量混合操作；允许在更高精度下进行广播（Duplicate在FP32上进行）；更好地利用向量单元的全带宽。baseline提取标量后使用Muls进行标量-向量乘法，效率较低。

**Expert implementation:**
```cpp
// 专家实现 - 标量广播为向量
using OpCopyInLr = Bind<Vec::Duplicate<U>, Placeholder::In2<U, Placeholder::ScalarAttr<true>>>;
using OpLrCast = Bind<Vec::Cast<T, U, 0>, OpCopyInLr>;
using OpLrMulGrad = Bind<Vec::Mul<T>, OpGradCast, OpLrCast>;  // 向量-向量乘法
```

**vs. baseline:**
```cpp
// baseline - 标量直接使用
lrScalar = lrGm.GetValue(0);
AscendC::Muls(lrMulGradLocal, gradLocal, this->lrScalar, this->tileSize);
```

Benefit: 统一的向量计算流程，更高的执行效率，更好的精度控制
Trade-off: 需要额外的广播操作，但通常可以被内存访问隐藏

---

## Variant B: BA/AB广播模式识别与优化
Source: masked_scatter_with_position

专家实现识别了两种典型广播模式：BA模式（mask在后几维广播）和AB模式（mask在前几维广播）。通过`CanBroadcastBAOrAB`函数在Host端识别广播模式，将模式信息通过tiling data传递给Kernel。Kernel根据`PATTERN_TYPE`模板参数选择不同计算分支，避免运行时条件判断。BA模式索引计算为`maskIdx = i % xInner`，AB模式为`maskIdx = rowIdx`。baseline仅处理2D情况，使用运行时条件判断，效率较低。

**Expert implementation:**
```cpp
// 专家实现 - BA/AB模式识别
if (BA) {
    xOuter_ = xEleNums_ / maskEleNums_;
    xInner_ = maskEleNums_;
    isBA_ = true;
} else if (AB) {
    xOuter_ = maskEleNums_;
    xInner_ = xEleNums_ / maskEleNums_;
    isBA_ = false;
}

// Kernel端编译期选择分支
if constexpr (PATTERN_TYPE == PATTERN_AB) {
    if (maskGm[rowidx] == true) { /* AB模式 */ }
} else {
    if (maskGm[i % xInner] == true) { /* BA模式 */ }
}
```

**vs. baseline:**
```cpp
// baseline - 简单2D广播处理
uint32_t maskRow = (this->maskDim0 == 1) ? 0 : row;
uint32_t maskCol = (this->maskDim1 == 1) ? 0 : col;
uint32_t maskIdx = maskRow * this->maskDim1 + maskCol;
```

Benefit: 支持更通用的广播模式，编译期确定分支，零运行时开销，针对特定模式深度优化
Trade-off: Host端需要额外的广播模式识别逻辑，增加代码复杂度

---

## Variant C: Mask处理的精度保证
Source: scaled_masked_softmax_grad_v2

专家实现使用SelectWithBytesMask实现mask的精确应用，避免简单的乘法mask带来的精度问题。使用bool类型mask而非float（0.0/1.0）节省内存带宽，mask=true的位置保留原值，mask=false的位置置为MASK_VALUE，SelectWithBytesMask内部使用向量化实现保证性能。

**Expert implementation:**
```cpp
LocalTensor<uint8_t> maskTmpBuf = this->sharedBuffer.template Get<uint8_t>();
SelectWithBytesMaskShapeInfo shapeInfo;
shapeInfo.firstAxis = this->lineNum;
shapeInfo.srcLastAxis = this->paddedHeadDim_;
shapeInfo.maskLastAxis = this->paddedHeadDim_;
SelectWithBytesMask(tmpOutLocal, tmpOutLocal, MASK_VALUE, maskLocal, maskTmpBuf, shapeInfo);
```

**vs. baseline:**
```cpp
// baseline不支持mask
```

Benefit: 精确mask应用，避免乘法mask的精度损失
Trade-off: 需要额外buffer，增加计算步骤

---

## Variant D: Mask广播优化与Offset管理
Source: scaled_masked_softmax_v2

专家实现支持mask在batch和channel维度的广播，这是Transformer模型中常见的使用模式（例如causal mask通常是[1, 1, seq_len, seq_len]）。通过maskMode字段标识广播模式：bit0表示batch广播，bit1表示channel广播。Kernel端使用MaskOffset结构体管理复杂的mask偏移计算，支持跨batch、跨channel的灵活寻址。CopyMaskIn函数处理了多种边界情况：当当前batch和结束batch相同时，只需要在一个channel内处理；当不同时，需要跨batch处理。

**Expert implementation:**
```cpp
// 专家实现广播支持
struct MaskOffset {
    uint64_t batchOffset = 0;
    uint64_t channelOffset = 0;
    uint64_t lineOffset = 0;
    __aicore__ inline void NextChannel(uint64_t channelNum) {
        channelOffset = (channelOffset + 1) % channelNum;
        if (channelOffset == 0) batchOffset++;
        lineOffset = 0;
    }
    __aicore__ inline uint64_t GetOffset(uint64_t realBatch, uint64_t realChannel, uint64_t realLine) {
        return batchOffset * realBatch + channelOffset * realChannel + lineOffset * realLine;
    }
};

// 广播模式设置
void SetMaskAdapt() {
    uint64_t maskMode = 0;
    if (batch != maskBatch) { maskMode |= BROADCAST_BATCH; tiling.set_nStep(0); }
    if (channel != maskChannel) { maskMode |= BROADCAST_CHANNEL; tiling.set_cStep(0); }
    tiling.set_maskMode(maskMode);
}
```

**vs. baseline:**
```cpp
// baseline 无广播支持
uint32_t batchIdx = rowIdx / (totalRows / batchSize);
uint32_t queryIdx = rowIdx % seqLen;
uint32_t maskRowIdx = batchIdx * seqLen + queryIdx;
uint32_t maskOffset = maskRowIdx * seqLen;
```

Benefit: 支持常见mask广播模式，避免Host端预先展开mask的内存开销，提升易用性
Trade-off: Kernel端offset计算复杂，增加代码复杂度

---

## Variant E: Softmax高阶API使用与Tiling复用
Source: scaled_masked_softmax_v2

专家实现使用昇腾CANN提供的SoftMax高阶API替代手动实现的softmax计算。高阶API内部经过充分优化，包括数值稳定性处理、向量化指令调度、流水线优化等。为了使用SoftMax API，需要在Host端调用AscendC::SoftMaxTilingFunc计算并填充SoftMaxTiling结构体。Kernel端将softmaxTilingData和softmaxShapeInfo传递给API完成计算。此外，专家实现还复用了softmax的临时缓冲区作为SelectWithBytesMask的共享缓冲区，通过精细的内存复用减少UB占用。

**Expert implementation:**
```cpp
// 专家实现使用高阶API
// Host端
void SetSoftmaxTiling() {
    auto shape = ge::Shape({static_cast<int64_t>(tiling.get_lineHeadIter()),
                            static_cast<int64_t>(tiling.get_padLineNum())});
    auto size = AscendC::GetSoftMaxMaxTmpSize(shape, FP32_SIZE, false);
    AscendC::SoftMaxTilingFunc(shape, FP32_SIZE, size, tiling.softmaxTilingData);
}

// Kernel端
__aicore__ inline void SoftmaxX(LocalTensor<float>& dstTensor, LocalTensor<float>& srcTensor,
                                LocalTensor<uint8_t> sharedBuffer, uint64_t lines) {
    SoftMaxTiling softmaxTilingData = tilingData.softmaxTilingData;
    SoftMaxShapeInfo softmaxShapeInfoData = {
        static_cast<uint32_t>(lines),
        static_cast<uint32_t>(tilingData.padLineNum),
        static_cast<uint32_t>(lines),
        static_cast<uint32_t>(tilingData.width),
    };
    SoftMax<float, false, false>(dstTensor, srcTensor, sharedBuffer, softmaxTilingData, softmaxShapeInfoData);
}
```

**vs. baseline:**
```cpp
// baseline 手动softmax
AscendC::ReduceMax(maxLocal, scaledLocal, sharedLocal, tileLength);
float rowMax = maxLocal.GetValue(0);
AscendC::Duplicate(maxLocal, rowMax, tileLength);
AscendC::Sub(expLocal, scaledLocal, maxLocal, tileLength);
AscendC::Exp(expLocal, expLocal, tileLength);
AscendC::ReduceSum(sumLocal, expLocal, sharedLocal, tileLength);
float rowSum = sumLocal.GetValue(0);
float invSum = 1.0f / rowSum;
AscendC::Muls(outLocal, expLocal, invSum, tileLength);
```

Benefit: 性能提升10-20%，数值稳定性更好，代码更简洁
Trade-off: 需要学习和理解高阶API的使用方式

---

## Variant F: SelectWithBytesMask高阶API使用
Source: scaled_masked_softmax_v2

专家实现使用SelectWithBytesMask高阶API实现mask应用，替代手动if-else或Add操作。该API的语义是：当mask对应位置为true时，dst取value；否则取src。这种设计正好符合scaled masked softmax的需求：mask为true的位置应该被替换为MASK_VAL（-10000.0），这是一个极小的值，在softmax中会变成接近0的概率。相比baseline中使用Add操作（要求mask是float类型且值为0/-inf），这种bool mask方式更加内存高效（1字节 vs 4字节）且语义清晰。

**Expert implementation:**
```cpp
// 专家实现使用SelectWithBytesMask
AscendC::SelectWithBytesMaskShapeInfo selectShapeInfo;
selectShapeInfo.firstAxis = linePerIter;
selectShapeInfo.srcLastAxis = tilingData.padLineNum;
selectShapeInfo.maskLastAxis = tilingData.alignedMaskWidth;

// mask为true的位置替换为MASK_VAL (-10000.0)
AscendC::SelectWithBytesMask(
    scaledMaskedX, scaledMaskedX, MASK_VAL, maskTensor, sharedBuffer, selectShapeInfo);

// Input定义
this->Input("mask")
    .ParamType(OPTIONAL)
    .DataType({ge::DT_BOOL, ge::DT_BOOL, ge::DT_BOOL})  // bool类型
```

**vs. baseline:**
```cpp
// baseline 使用Add
AscendC::Muls(scaledLocal, xLocal, scale, tileLength);
AscendC::Add(scaledLocal, scaledLocal, maskLocal, tileLength);  // mask是float类型
```

Benefit: 内存占用减少75%（bool vs float），语义更清晰，使用高阶API性能更优
Trade-off: API使用有一定学习成本
