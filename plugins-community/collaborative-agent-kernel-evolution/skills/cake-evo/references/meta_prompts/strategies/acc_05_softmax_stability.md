# A5: Numerical Safety & Special Value Handling (数值安全与特殊值处理)
## Overview
专家实现在计算输出值时，通过Compare+Select组合实现了对NaN（Not a Number）的特殊处理。具体逻辑是：Compare(maskTemp, xLocal, xLocal, CMPMODE::EQ, calCount) - 如果x是NaN，则x != x，比较结果为false；然后通过Select(curTemp, maskTemp, curTemp, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, ...)，当mask为false（即x为NaN）时，输出0.0f。这种处理确保了量化结果在遇到NaN输入时不会传播NaN，而是输出一个确定值（0.0f）。这对于神经网络的量化训练尤为重要，因为NaN的传播可能导致训练不稳定。

## When to Use
- Softmax, attention score ops, any op with NaN/Inf risk
- 保证极端输入下的数值稳定性和正确性
- 避免FP16累加的精度损失，保证大hidden size场景下的数值稳定性
- 避免低精度累加误差，提高数值稳定性

## Trade-off
- 增加tiling阶段的开销
- 需要更多的UB内存用于FP32缓冲区
- 增加UB内存占用(4字节 vs 2字节)

**Source operators**: adaptive_max_pool3d_grad, add_rms_norm_cast, add_rms_norm_dynamic_quant, apply_adam_w_v2, batch_norm_v3, clipped_swiglu, deep_norm, dynamic_block_quant, dynamic_mx_quant, fake_quant_affine_cachemask, flash_attention_score, foreach_addcdiv_list, fused_floyd_attention, gemma_rms_norm, incre_flash_attention, inplace_add_rms_norm, layer_norm_v3, max_pool_grad_with_argmax_common, max_pool_with_argmax_v3, multi_scale_deformable_attn_function, rms_norm_quant, scaled_masked_softmax_v2, trans_quant_param_v2

---

## Variant A: 边界检查与数值稳定性
Source: adaptive_max_pool3d_grad

专家实现在tiling阶段进行了严格的边界检查：验证输入维度是否为5D（NCDHW），验证索引范围diDim * hiDim * wiDim > MAX_INT32防止索引溢出，在overlap场景计算D维度的GCD优化切分策略。这些检查确保了在极端输入下的数值稳定性和正确性。

**Expert implementation:**
```cpp
OP_CHECK_IF((xDimNum != NCDHW_DIM_NUM) || (gradDimNum != NCDHW_DIM_NUM),
    OP_LOGE(context_->GetNodeName(), "Input dim num should equal = %lu", NCDHW_DIM_NUM),
    return false);

OP_CHECK_IF(maxPoolGradParams.diDim * maxPoolGradParams.hiDim * maxPoolGradParams.wiDim > MAX_INT32,
    OP_LOGE(context_->GetNodeName(), "Shape too big"),
    return ge::GRAPH_FAILED);

maxPoolGradParams.dGcd = Gcd(maxPoolGradParams.doDim, maxPoolGradParams.diDim);
```

**vs. baseline:**
```cpp
// 基础形状检查，无索引范围检查
```

Benefit: 保证极端输入下的数值稳定性和正确性
Trade-off: 增加tiling阶段的开销

---

## Variant B: FP32中间计算保证数值稳定性
Source: add_rms_norm_cast

专家实现在所有涉及累加和归约的计算中都使用FP32作为中间类型，即使输入输出是FP16/BF16。这是防止低精度数据类型在大量累加时产生数值溢出的关键策略。精度保证措施包括：输入数据从FP16/BF16转换为FP32后再进行Add和Mul操作；RMS计算（平方、累加、开方）全程使用FP32；仅在最后输出时转换为目标数据类型。

**Expert implementation:**
```cpp
// FP32中间计算
Cast(x_fp32, xLocal, RoundMode::CAST_NONE, numCol);
PipeBarrier<PIPE_V>();
Mul(sqx, x_fp32, x_fp32, numCol);  // FP32平方
PipeBarrier<PIPE_V>();
Muls(sqx, sqx, avgFactor, numCol);  // FP32缩放
PipeBarrier<PIPE_V>();
ReduceSumCustom(sqx, sqx, reduce_buf_local, numCol);  // FP32归约
```

**vs. baseline:**
```cpp
// 简单的FP32转换
AscendC::Cast(addedLocal, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
AscendC::Cast(tempLocal, residualLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
AscendC::Add(addedLocal, addedLocal, tempLocal, this->tileLength);
```

Benefit: 避免FP16累加的精度损失，保证大hidden size场景下的数值稳定性
Trade-off: 需要更多的UB内存用于FP32缓冲区

---

## Variant C: Float32中间计算保证数值稳定性
Source: add_rms_norm_dynamic_quant

专家实现强制在UB中使用Float32进行所有中间计算(RMS求和、归一化、量化前的缩放)，仅在最终结果输出时才转换回目标格式。这种设计避免了FP16/BF16在累加过程中的精度损失，特别是在RMS计算涉及大量元素求和时。baseline实现虽然也有Float32中间计算，但专家实现通过更严格的类型转换策略进一步保证了精度。

**Expert implementation:**
```cpp
Cast(xLocalFp32, x1Local, RoundMode::CAST_NONE, elementCount);
Cast(yLocalFp32, x2Local, RoundMode::CAST_NONE, elementCount);
PipeBarrier<PIPE_V>();
Add(xLocalFp32, xLocalFp32, yLocalFp32, elementCount);
float squareSumTemp = ReduceSumHalfInterval(yLocalFp32, this->numLastDim);
float rstdLocalTemp = 1 / sqrt(squareSumTemp * this->aveNum + this->eps);
```

**vs. baseline:**
```cpp
AscendC::Cast(addOutLocal, x1Local, AscendC::RoundMode::CAST_RINT, tileLength);
AscendC::Add(addOutLocal, addOutLocal, tempLocal, tileLength);
float rowMeanSquare = rowSquareSum / hiddenSize;
```

Benefit: 避免低精度累加误差，提高数值稳定性
Trade-off: 增加UB内存占用(4字节 vs 2字节)

---

## Variant D: 动态量化Scale计算的数值稳定性
Source: add_rms_norm_dynamic_quant

专家实现在计算动态量化scale时，使用DYNAMIC_QUANT_DIVIDEND / maxTemp计算scale，然后取倒数1 / scaleTemp作为最终输出scale。这种两步计算确保了量化时可以直接使用scaleTemp进行乘法(而非除法)，提高计算效率，同时输出的scale是max/127，符合动态量化的标准定义。

**Expert implementation:**
```cpp
constexpr float DYNAMIC_QUANT_DIVIDEND = 127.0;
float maxTemp = tmpTensor.GetValue(0);
float scaleTemp = DYNAMIC_QUANT_DIVIDEND / maxTemp;
scaleTensor.SetValue(idx, 1 / scaleTemp);  // 输出scale = max / 127
Muls(srcTensor, srcTensor, scaleTemp, this->numLastDim);  // 量化使用乘法
```

**vs. baseline:**
```cpp
float rowScale = rowAbsMax / quantMax;
// 量化
AscendC::Muls(tempLocal, normalizedLocal, scaleRecip, tileLength);
```

Benefit: 量化使用乘法而非除法，符合标准动态量化定义
Trade-off: 需要额外的倒数计算

---

## Variant E: 高精度中间计算与类型转换
Source: apply_adam_w_v2

在FP16/BF16场景中，专家实现采用FP32中间计算策略确保数值稳定性。`ApplyAdamWV2B16`类在UB中分配了`inCastBuf_`和`outCastBuf_`两个FP32缓冲区，用于存储类型转换后的数据。计算流程为：(1) 输入数据从低精度Cast到FP32；(2) 所有计算在FP32域执行；(3) 结果根据数据类型选择不同的舍入模式Cast回低精度。

**Expert implementation:**
```cpp
LocalTensor<float> inCastLocal = inCastBuf_.Get<float>();
LocalTensor<float> outCastLocal = outCastBuf_.Get<float>();

// Cast input to FP32
Cast(inCastLocal[gradOffset_], dataLocal[gradOffset_], RoundMode::CAST_NONE, dataCount);

// FP32 computation
Muls(outCastLocal[varOffset_], inCastLocal[varOffset_], realWeightDecay_, dataCount);

// Cast back with appropriate rounding
if (isBfloat16_) {
    Cast(dataOutLocal[varOffset_], outCastLocal[varOffset_], RoundMode::CAST_ROUND, dataCount);
} else {
    Cast(dataOutLocal[varOffset_], outCastLocal[varOffset_], RoundMode::CAST_RINT, dataCount);
}
```

**vs. baseline:**
```cpp
// Direct computation in input dtype (FP32 only)
Muls(outLocal[0], var, decayFactor, count);
```

Benefit: 避免低精度累积误差；保证数值稳定性；支持混合精度训练
Trade-off: UB占用增加；需要额外的Cast操作

---

## Variant F: Epsilon数值稳定性
Source: batch_norm_v3

BatchNorm在计算1/sqrt(var + epsilon)时，如果var很小或为0，可能导致数值不稳定。专家实现通过以下方式保证数值稳定性：1）Host端默认值：epsilon默认值为1e-5，通过attr传入；2）Kernel端应用：在计算标准差时始终加上epsilon；3）FP32精度计算：整个归一化过程使用FP32精度，避免低精度下的数值截断。

**Expert implementation:**
```cpp
// Host端默认值
static constexpr float DEFAULT_EPSILON = 1e-5;
commonParams.epsilon = (epsilonPtr == nullptr) ? DEFAULT_EPSILON : *epsilonPtr;

// Kernel端应用
float weightMulInvstd = static_cast<float>(weightValue) / sqrt(finalVar + static_cast<float>(this->epsilon));
```

**vs. baseline:**
```cpp
# baseline epsilon处理
inv_std = 1.0 / ((var_val + eps) ** 0.5)
```

Benefit: 避免除以0错误；与标准深度学习框架行为一致
Trade-off: 引入极小的计算开销

---

## Variant G: Clip前置防止数值溢出
Source: clipped_swiglu

在计算Exp之前先进行Clip操作，将输入限制在[-limit, limit]范围内，可以有效防止：1)exp(x)在x很大时的数值溢出；2)exp(-x)在x为很大的负数时下溢为0；3)数值不稳定导致的精度损失。默认limit值为7.0，确保exp(7) ≈ 1096在合理范围内。

**Expert implementation:**
```cpp
// Clip前置
Mins(tmpA, tmpA, tl_->gluLimit, calPairNum_);
PipeBarrier<PIPE_V>();
Maxs(tmpA, tmpA, -1 * tl_->gluLimit, calPairNum_);
// then compute exp
Muls(xFloatLocal, tmpA, -1 * tl_->gluAlpha, calPairNum_);
Exp(xFloatLocal, xFloatLocal, calPairNum_);
```

**vs. baseline:**
```cpp
// clip after compute
AscendC::Maxs(outLocal, prodLocal, minVal, tileSize);
AscendC::Mins(outLocal, outLocal, maxVal, tileSize);
```

Benefit: 防止数值溢出，提高数值稳定性
Trade-off: 增加了Clip指令开销，但值得

---

## Variant H: FP32 中间计算保证数值稳定性
Source: deep_norm

DeepNorm 算子涉及均值和方差计算，这些操作对数值精度敏感。专家实现对于 FP16 和 BF16 输入，在计算过程中转换为 FP32 进行中间计算，最后再将结果转换回原精度。这种策略在统计计算中尤为重要，因为 FP16 的有效位数只有 10 位，大量数据的累加容易导致精度丢失；方差计算涉及平方操作，数值范围扩大，低精度容易溢出。

**Expert implementation:**
```cpp
// 专家实现 - 类型提升后计算
// FP16 输入转为 FP32
Cast(local_y_fp32, x_local, RoundMode::CAST_NONE, stepSize);
Cast(local_x_fp32, gx_local, RoundMode::CAST_NONE, stepSize);
// FP32 精度进行累加和方差计算
Axpy(local_x_fp32, local_y_fp32, alphaVal, stepSize);
float mean_local_temp = ReduceSumCustom(local_y_fp32[offset], num_last_dim);
Mul(local_x_fp32, local_y_fp32, local_y_fp32, stepSize);  // 方差
float var_local_temp = ReduceSumCustom(local_x_fp32[offset], num_last_dim) * meanNum;
```

**vs. baseline:**
```cpp
// baseline - 全程 FP32
float rowVarSum = 0.0f;
for (uint32_t tileId = 0; tileId < nTiles; tileId++) {
    // (x - mean)^2
    AscendC::Adds(tempTensor, inputTensor, -rowMean, actualTileLen);
    AscendC::Mul(tempTensor, tempTensor, tempTensor, actualTileLen);
    float tileVar = sharedTensor.GetValue(0);
    rowVarSum += tileVar;
}
```

Benefit: 避免低精度累加误差和溢出，保证统计计算精度
Trade-off: 需要额外的 FP32 缓冲区和类型转换开销

---

## Variant I: Epsilon 和 Alpha 参数的精度处理
Source: deep_norm

专家实现对 epsilon（防止除零）和 alpha（DeepNorm 的加权系数）参数进行了特殊的精度处理。在 Host 端，这些 float 参数通过 memcpy_s 转换为 uint32_t 存储在 TilingData 中，在 Kernel 端再通过 reinterpret_cast 转回 float。这种设计确保了参数在 Host 和 Device 之间传输时的位级精确性，避免了任何潜在的精度损失。

**Expert implementation:**
```cpp
// 专家实现 - 位级精确传输
// Host 端
float tempAlpha = *context->GetAttrs()->GetFloat(0);
float eps = *context->GetAttrs()->GetFloat(1);
uint32_t temp_eps;
memcpy_s(&temp_eps, sizeof(float), &eps, sizeof(float));
tiling.set_eps_str(temp_eps);

// Kernel 端
eps = *reinterpret_cast<float*>(&eps_);
alphaVal = *reinterpret_cast<float*>(&alpha_);
```

**vs. baseline:**
```cpp
// baseline - 直接传递 float
tiling.set_eps(eps);
```

Benefit: 确保浮点参数的位级精确性，避免传输过程中的精度损失
Trade-off: 代码可读性略降，需要理解位操作

---

## Variant J: minScale精度保护
Source: dynamic_block_quant

专家实现支持minScale属性，用于防止scale过小导致的量化精度问题。当计算的maxAbs小于minScale时，使用minScale作为实际scale。这种设计在数据分布稀疏（大部分值接近0但有个别异常值）的场景下特别有用，可以避免因个别大值导致的大量小值被量化为0。实现上，在ComputeScaleVF中使用Max指令将scale与minScale比较取大值，同时计算inverse minScale用于限制scale的倒数。这种保护机制确保了量化结果的数值稳定性。

**Expert implementation:**
```cpp
if (hasMinScale) {
    Max(scaleLocalTmp, scaleLocalTmp, minScaleTensor, NUM_SIX_FOUR, repeatTimes, MultiOneBlockParams);
    Min(scaleLocal, scaleLocal, invMinScaleTensor, NUM_SIX_FOUR, repeatTimes, MultiOneBlockParams);
}
```

Benefit: 防止scale过小导致的精度损失，适应稀疏数据分布
Trade-off: 需要额外的比较操作，增加计算量

---

## Variant K: 除零保护
Source: dynamic_block_quant

专家实现使用MicroAPI的DivSpecificMode配置提供了细粒度的除零保护。通过设置MaskMergeMode::ZEROING，当除数为0时结果被置为0而非产生异常。在ComputeScaleVF中，先使用Compare比较scale是否为0，然后通过mask选择保留原始值或计算结果。这种设计避免了显式的if-else分支，使用向量化条件操作实现保护，既保证了数值稳定性又保持了高性能。baseline仅做了简单的maxAbs == 0检查，无法处理向量化场景。

**Expert implementation:**
```cpp
static constexpr AscendC::MicroAPI::DivSpecificMode mode = 
    {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
AscendC::MicroAPI::Div<float, &mode>(reciprocalScale, reciprocalScale, minScaleReg, preg0);
AscendC::MicroAPI::CompareScalar<float, CMPMODE::NE>(preg1, vreg2, (float)0.0, preg0);
AscendC::MicroAPI::Select(vreg7, vreg4, vreg5, preg1);
```

**vs. baseline:**
```cpp
if (maxAbs == 0.0f) {
    maxAbs = 1.0f;
}
```

Benefit: 向量化除零保护，无分支性能损失
Trade-off: 需要理解MicroAPI的mask机制

---

## Variant L: 特殊值处理 (NaN/Inf/Subnormal)
Source: dynamic_mx_quant

浮点数的特殊值（NaN、Inf、Subnormal）需要精心处理以避免量化错误。专家实现通过位操作检测和修正这些特殊值：Inf 检测（通过比较指数位 MAX_EXP 识别 Inf）、NaN 处理（使用 NAN_CUSTOMIZATION 值替换 NaN）、Zero 检测（确保零值在量化后仍为零）、FP4 特殊指数调整（FP4 的动态范围有限，需要裁剪指数到有效范围）。

**Expert implementation:**
```cpp
// 专家实现：完整的特殊值处理
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;
constexpr uint32_t NAN_CUSTOMIZATION_FP32 = 0x7f810000;
constexpr uint16_t MAX_EXP_FOR_BF16 = 0x7f80;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;

// Inf 检测
AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(infMask, expMaxRegTensor, maxEleRegTensor, p0);
// Zero 检测
AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::NE>(zeroMask, expMaxRegTensor, zeroRegTensor, p0);
// NaN 替换
AscendC::MicroAPI::Select<calcTypeInt>(mxScaleRegTensor, mxScaleRegTensor, fp8NanRegTensor, infMask);
// Zero 处理
AscendC::MicroAPI::Select<calcTypeInt>(mxScaleRegTensor, mxScaleRegTensor, zeroRegTensor, zeroMask);
// FP4 指数裁剪
if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
    AscendC::MicroAPI::Compare<calcTypeInt, CMPMODE::LE>(
        invalidDataMask, expMaxRegTensor, fp4MaxExpRegTensor, p0);
    AscendC::MicroAPI::Select<calcTypeInt>(
        expMaxRegTensor, fp4MaxExpRegTensor, expMaxRegTensor, invalidDataMask);
}
```

**vs. baseline:**
```cpp
// baseline：简单的数值裁剪
if (blockMax < 1e-8f) {
    blockMax = 1e-8f;
}
AscendC::ClampMax(quantLocal, quantLocal, static_cast<float>(maxVal), blockSize);
AscendC::ClampMin(quantLocal, quantLocal, static_cast<float>(minVal), blockSize);
```

Benefit: 量化结果符合 IEEE 754 标准，Inf/NaN 处理与 CPU/GPU 行为一致
Trade-off: 增加常量定义和条件判断，代码体积增加约 10-15%

---

## Variant M: 特殊值处理 (NaN处理)
Source: fake_quant_affine_cachemask

专家实现在计算输出值时，通过Compare+Select组合实现了对NaN（Not a Number）的特殊处理。具体逻辑是：Compare(maskTemp, xLocal, xLocal, CMPMODE::EQ, calCount) - 如果x是NaN，则x != x，比较结果为false；然后通过Select(curTemp, maskTemp, curTemp, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, ...)，当mask为false（即x为NaN）时，输出0.0f。这种处理确保了量化结果在遇到NaN输入时不会传播NaN，而是输出一个确定值（0.0f）。这对于神经网络的量化训练尤为重要，因为NaN的传播可能导致训练不稳定。

**Expert implementation:**
```cpp
// 专家实现 - NaN检测与处理
// NaN != NaN，所以如果x是NaN，Compare结果为false
Compare(maskTemp, xLocal, xLocal, CMPMODE::EQ, calCount);
// 如果mask为false（NaN情况），选择0.0f；否则保留原值
Select(curTemp, maskTemp, curTemp, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, 
       this->mask, repeatTimes, repeatParams);
Adds(curTemp, curTemp, static_cast<yType>(-1 * zeroPointValue), calCount);
Muls(yLocal, curTemp, static_cast<yType>(scaleValue), calCount);
```

**vs. baseline:**
```cpp
// baseline无NaN处理
AscendC::Muls(outputLocal, tempLocal, this->scale, this->tileSize);
```

Benefit: 确保量化结果稳定性，避免NaN传播导致训练失败；输出确定值便于调试
Trade-off: 增加两条指令（Compare+Select）的开销；对正常输入无影响

---

## Variant N: bfloat16_t高精度计算路径
Source: foreach_addcdiv_list

专家实现针对bfloat16_t类型提供了高精度计算路径。在InnerComputer<bfloat16_t, float, op>特化中，数据被转换为float32进行计算，使用RoundMode::CAST_NONE进行无精度损失的类型转换，计算完成后再使用RoundMode::CAST_RINT（四舍五入）转回bfloat16_t。这确保了在复杂计算（如除法、乘法累加）中的数值稳定性。baseline实现只支持float32，没有考虑低精度类型的数值稳定性问题。

**Expert implementation:**
```cpp
// 专家实现: bfloat16_t高精度计算
Cast(float32Tensor, inLocal_1[index * maxCastDataCount], RoundMode::CAST_NONE, maxCastDataCount);
Cast(float32Tensor[maxCastDataCount], inLocal_2[index * maxCastDataCount], RoundMode::CAST_NONE, maxCastDataCount);
// ... float32精度计算 ...
Cast(outLocal[index * maxCastDataCount], float32Tensor, RoundMode::CAST_RINT, maxCastDataCount);
```

**vs. baseline:**
```cpp
// baseline: 仅float32，无类型转换
AscendC::LocalTensor<float> tensor1Local = tensor1Queue.DeQue<float>();
AscendC::Div(divLocal, tensor1Local, tensor2Local, current_tile_size);
```

Benefit: bfloat16_t类型复杂计算时保持数值稳定性
Trade-off: 需要额外的类型转换开销和UB内存

---

## Variant O: Gemma 特殊语义处理（gamma+1）
Source: gemma_rms_norm

Gemma 模型的 RMSNorm 与传统 RMSNorm 不同，其计算公式为 y = (1 + gamma) * (x * rstd)，即权重需要加 1 后再进行缩放。专家实现通过 is_gemma 标志控制这一行为，在 Host 端根据算子类型设置该标志。baseline实现完全缺少这一特性，无法正确计算 Gemma 模型的 RMSNorm。

**Expert implementation:**
```cpp
// 专家实现 - Gemma 特殊处理
if (is_gemma == 1) {
    LocalTensor<float> gammaFp32 = x_fp32_buf.Get<float>();
    Adds(gammaFp32, gammaLocal, static_cast<float>(1.0), elementNum);  // gamma + 1
    PipeBarrier<PIPE_V>();
    Mul(yLocal, sqx, gammaFp32, elementNum);
} else {
    Mul(yLocal, sqx, gammaLocal, elementNum);
}

// Host 端设置 is_gemma
is_gemma = tiling->is_gemma;
```

**vs. baseline:**
```cpp
// baseline - 标准 RMSNorm
// y = x / rms * weight
// 缺少 Gemma 的 (1+gamma) 语义
AscendC::Muls(tempLocal, inputLocal, 1.0f / rms, actualTileLen);
AscendC::Mul(outputLocal, tempLocal, weightLocal, actualTileLen);
```

Benefit: 正确支持 Gemma 模型的 RMSNorm 变体，计算结果与标准一致
Trade-off: 需要额外的 Adds 操作，略有性能开销

---

## Variant P: FP16/BF16的FP32中间计算
Source: inplace_add_rms_norm

为了保证混合精度计算的数值稳定性，专家实现在FP16和BF16计算中使用了FP32作为中间计算类型。这种策略在卷积、归约等累加操作中尤为重要，可以有效避免半精度累加的精度损失。具体实现中，输入数据从GM加载到UB后，首先通过Cast指令转换为FP32，然后在FP32精度下进行累加、乘法等操作，最后再将结果转换回原始数据类型。

**Expert implementation:**
```cpp
// 专家实现 - 混合精度
if constexpr (is_same<T, half>::value) {
    LocalTensor<float> x1_fp32 = xFp32Buf.Get<float>();
    Add(xLocal, x1Local, x2Local, numCol);
    PipeBarrier<PIPE_V>();
    Cast(x1_fp32, xLocal, RoundMode::CAST_NONE, numCol);
    // ... 在FP32下计算
    Cast(yLocal, x1_fp32, RoundMode::CAST_NONE, numCol);
}
```

**vs. baseline:**
```cpp
// baseline - 单精度计算
AscendC::Add(sumLocal, xLocal, yLocal, this->cols);
AscendC::Mul(squareLocal, sumLocal, sumLocal, this->cols);
AscendC::ReduceSum(sharedLocal, squareLocal, sharedLocal, this->cols);
```

Benefit: 避免半精度累加的精度损失，保持数值稳定性
Trade-off: 需要更多的UB内存来存储FP32中间结果

---

## Variant Q: Epsilon的灵活配置
Source: layer_norm_v3

专家实现支持通过attribute传入epsilon，默认1e-5（LayerNormV3）或1e-7（LayerNormV1）。在CalculateRstdByHighPrecision中，Adds(var, var, epsilon, pregMerge)确保variance不会为0，避免除零错误。

**Expert implementation:**
```cpp
// 可配置epsilon
const float* epsilonPtr = attrs->GetFloat(INPUT_IDX_BETA);
commonParams.eps = (epsilonPtr == nullptr) 
                   ? (commonParams.isV1 ? DEFAULT_EPSILON_V1 : DEFAULT_EPSILON_V3)
                   : *epsilonPtr;
// CalculateRstdByHighPrecision中使用
Adds(var, var, epsilon, pregMerge);
```

**vs. baseline:**
```cpp
// 硬编码epsilon
float stdVal = sqrt(varVal + this->eps);
// eps为固定值1e-05
```

Benefit: 灵活适应不同模型需求，避免除零错误
Trade-off: 需要从attribute读取，增加少量Host端开销

---

## Variant R: 统一计算类型
Source: max_pool_grad_with_argmax_common

专家实现定义了统一的计算类型computeType = float，无论输入数据类型是FP16还是FP32，内部梯度累加都使用FP32进行。这种设计避免了低精度数据类型的累加误差，特别是在梯度反向传播场景中，多个梯度值可能累加到同一个输入位置。通过使用高精度进行中间计算，最后再将结果转换回原始数据类型，可以在保持性能的同时确保数值稳定性。

**Expert implementation:**
```cpp
// 专家实现：统一使用float进行计算
using computeType = float;

template <typename T>
__aicore__ inline void GradientAcc(__local_mem__ computeType* yAddr, 
                                   MicroAPI::RegTensor<computeType>& gradReg,
                                   MicroAPI::RegTensor<T>& argmaxReg, ...)
{
    AscendC::MicroAPI::RegTensor<computeType> scatterAccResReg;
    // 使用float进行累加
    AscendC::MicroAPI::DataCopyGather(scatterAccResReg, yAddr, ...);
    AscendC::MicroAPI::Add(scatterAccResReg, scatterAccResReg, gradReg, pregArgmax);
    AscendC::MicroAPI::DataCopyScatter(yAddr, scatterAccResReg, ...);
}
```

**vs. baseline:**
```cpp
// baseline实现：直接使用输入类型计算
float grad_val = gradOutLocal.GetValue(c_offset);
float current_val = gradInLocal.GetValue(0);
float new_val = current_val + grad_val;
tempLocal.SetValue(0, new_val);
```

Benefit: 避免低精度累加误差，确保梯度反向传播的数值稳定性
Trade-off: 增加UB内存占用（float比FP16占用多一倍）

---

## Variant S: 负无穷值的位模式初始化
Source: max_pool_with_argmax_v3

专家实现使用位模式直接设置负无穷值，避免浮点运算开销。对于FP32/FP16/BF16分别使用IEEE 754负无穷位模式（0xFF800000/0xFC00/0xFF80），通过constexpr在编译期确定，执行时使用MicroAPI::Duplicate向量化填充。

**Expert implementation:**
```cpp
// 专家实现 - 位模式负无穷
template <typename T>
__aicore__ inline void DuplicateNegInfReg(MicroAPI::RegTensor<T>& negInfReg) {
    constexpr uint32_t FLOAT32_NEG_INF = 0xFF800000;
    constexpr uint16_t FLOAT16_NEG_INF = 0xFC00;
    constexpr uint16_t BFLOAT16_NEG_INF = 0xFF80;
    
    if constexpr (std::is_same<T, float>::value) {
        AscendC::MicroAPI::Duplicate((RegTensor<uint32_t>&)negInfReg, FLOAT32_NEG_INF);
    } else if constexpr (std::is_same<T, half>::value) {
        AscendC::MicroAPI::Duplicate((RegTensor<uint16_t>&)negInfReg, FLOAT16_NEG_INF);
    }
}
```

**vs. baseline:**
```cpp
// baseline - 近似负无穷
AscendC::Duplicate(maxLocal, -3.402823e+38f, count);
```

Benefit: 精确的负无穷值，无浮点转换开销，向量化初始化
Trade-off: 需要对IEEE 754格式有深入理解

---

## Variant T: NaN 处理与数值稳定性
Source: max_pool_with_argmax_v3

专家实现在 CycleUpdate 函数中处理 NaN 值：如果输入是 NaN（通过 Simt::IsNan 检查），则将其视为最大值。这与 PyTorch 的 max pool 行为一致。baseline 完全没有处理 NaN 场景。

**Expert implementation:**
```cpp
// 专家实现 - NaN处理
__aicore__ inline static void CycleUpdate(float val, idx_accscalar_t idxOffset, 
                                           float* maxval, idx_accscalar_t* maxidx) {
    if ((static_cast<float>(val) > *maxval) || Simt::IsNan(val)) {
        *maxidx = idxOffset;
        *maxval = val;
    }
}
```

**vs. baseline:**
```cpp
// baseline - 无NaN处理
if (inputVal > maxVal) {
    maxLocal.SetValue(i, inputVal);
}
```

Benefit: 数值稳定性，与PyTorch语义一致
Trade-off: 每次比较增加一次IsNan检查开销（SIMT模式下可忽略）

---

## Variant U: Ceil Mode 的边界修正
Source: max_pool_with_argmax_v3

专家实现在 shape 推导中正确处理 ceil mode 的边界情况。UpdateMaxShape 函数计算输出尺寸后，在 ceil mode 下检查 (out_max_shape - 1) * strides >= dim_size + pad 条件，必要时调整输出尺寸。baseline 虽有 ceil mode 计算，但没有边界修正。

**Expert implementation:**
```cpp
// 专家实现 - ceil mode边界修正
static void UpdateMaxShape(...) {
    int64_t exact_size = dim_size + 2 * pad - dilation * (ksize - 1) - 1 + (ceil_mode ? (strides - 1) : 0);
    out_max_shape = DivRtn(exact_size, strides) + 1;
    if (ceil_mode) {
        if ((out_max_shape - 1) * strides >= dim_size + pad) {
            out_max_shape = out_max_shape - 1;  // 修正边界
        }
    }
}
```

**vs. baseline:**
```cpp
// baseline - 简单ceil mode
if (ceilMode) {
    outH = (height + 2 * padH - dilationH * (kernelH - 1) - 1 + strideH - 1) / strideH + 1;
} else {
    outH = (height + 2 * padH - dilationH * (kernelH - 1) - 1) / strideH + 1;
}
```

Benefit: 与PyTorch形状计算完全一致，避免极端场景下的形状不匹配
Trade-off: Host端计算略复杂，增加少量CPU开销

---

## Variant V: Padding 区域 Argmax 修正
Source: max_pool_with_argmax_v3

当存在 padding 时，专家实现会对计算出的 argmax 进行修正：减去 padH/padW 得到相对于原始输入的坐标，使用 Compare 和 Select 将负坐标裁剪为 0。这确保返回的索引始终在有效范围内。

**Expert implementation:**
```cpp
// 专家实现 - padding区域argmax修正
if constexpr (IS_PAD == 1) {
    MicroAPI::Adds(argmaxHRes, argmaxHRes, -padH, computeT2);
    MicroAPI::Adds(argmaxWRes, argmaxWRes, -padW, computeT2);
    
    AscendC::MicroAPI::MaskReg hMask, wMask;
    MicroAPI::RegTensor<T2> argmaxZero;
    AscendC::MicroAPI::Duplicate(argmaxZero, 0);
    
    AscendC::MicroAPI::Compare<T2, CMPMODE::GE>(hMask, argmaxHRes, argmaxZero, computeT2);
    AscendC::MicroAPI::Select(argmaxHRes, argmaxHRes, argmaxZero, hMask);
    AscendC::MicroAPI::Compare<T2, CMPMODE::GE>(wMask, argmaxWRes, argmaxZero, computeT2);
    AscendC::MicroAPI::Select(argmaxWRes, argmaxWRes, argmaxZero, wMask);
}
```

**vs. baseline:**
```cpp
// baseline - 无padding修正
if (ih >= 0 && ih < (int32_t)height && iw >= 0 && iw < (int32_t)width) {
    uint32_t perChannelIdx = ih * width + iw;
    idxLocal.SetValue(i, static_cast<int32_t>(perChannelIdx));
}
```

Benefit: 索引始终在有效范围内，避免越界访问
Trade-off: 增加额外的向量化比较和选择操作

---

## Variant W: 边界条件处理与数值稳定性
Source: multi_scale_deformable_attn_function

专家实现中对边界条件进行了精细处理，确保数值稳定性。在ComputeBilinearInterpolation中，通过条件判断（0 <= y0 && y0 < h等）处理超出图像边界的采样点，避免访问越界。对于部分在边界内的采样点，采用分支处理策略（ComputeGradSeparate vs ComputeGradTogether），确保只有在有效区域内的点才参与计算。这种策略避免了无效计算和潜在的数值异常。

**Expert implementation:**
```cpp
// 边界条件处理
if (0 <= y0 && y0 < h) {
    if (0 < x1 && x1 < w) {
        DataCopy(value[pingOffset + point * alignedEmbedDims_],
            valueGm_[srcOffset_ + (y0 * w + x0) * outDims_], cpDoubleValParams_);
    } else if (0 <= x0 && x0 < w) {
        DataCopy(value[pingOffset + point * alignedEmbedDims_],
            valueGm_[srcOffset_ + (y0 * w + x0) * outDims_], cpOneValParams_);
    }
}
```

**vs. baseline:**
```cpp
// baseline实现不存在，无法提供对比代码
```

Benefit: 确保数值稳定性，避免越界访问导致的数值异常
Trade-off: 增加了条件分支，可能影响性能

---

## Variant X: 除零保护机制
Source: rms_norm_quant

专家实现在RMS计算中加入了除零保护机制。虽然RMS值理论上不会为零（因为有epsilon保护），但在极端情况下（如输入全零且epsilon设置过小），RMS可能接近零导致除法溢出。实现中通过条件表达式factor = (rms != 0) ? 1 / rms : 1.0f确保即使RMS为零，归一化因子也为1.0。

**Expert implementation:**
```cpp
float avg = squareSum * avg_factor_;
float rms = sqrt(avg + epsilon_);
float factor = (rms != 0) ? 1 / rms : 1.0f; // 避免除0问题
```

**vs. baseline:**
```cpp
float variance = varianceSum / colsFloat;
float rms = sqrt(variance + eps);
float rmsInv = 1.0f / rms;
```

Benefit: 边缘情况下的数值稳定性，避免产生Inf或NaN
Trade-off: 每个元素增加一个条件判断（可忽略不计）

---

## Variant Y: Softmax数值稳定性保障
Source: scaled_masked_softmax_v2

专家实现通过使用高阶SoftMax API确保了数值稳定性。Softmax计算的数值稳定性主要涉及两个环节：一是减去最大值防止指数溢出，二是防止除零错误。手动实现虽然也进行了max减法和sum非零判断，但高阶API内部实现了更鲁棒的策略，包括使用更高精度的中间计算、边界条件的特殊处理等。此外，专家实现在不同数据类型间的转换也考虑了精度损失：从低精度到高精度使用CAST_NONE模式保持数值，从高精度到低精度使用CAST_RINT模式进行四舍五入，减少精度截断误差。

**Expert implementation:**
```cpp
// 专家实现使用高阶API
SoftMax<float, false, false>(dstTensor, srcTensor, sharedBuffer, softmaxTilingData, softmaxShapeInfoData);

// 类型转换的精度控制
if constexpr (std::is_same<T, bfloat16_t>::value) {
    Cast(scaledMaskedX, xTensor, RoundMode::CAST_NONE, this->elePerIter);  // 低精度转高精度
}
// 输出时高精度转低精度，四舍五入
Cast(yTensor, scaledMaskedX, RoundMode::CAST_RINT, this->elePerIter);
```

**vs. baseline:**
```cpp
// baseline 基本处理
AscendC::ReduceMax(maxLocal, scaledLocal, sharedLocal, tileLength);
float rowMax = maxLocal.GetValue(0);
AscendC::Duplicate(maxLocal, rowMax, tileLength);
AscendC::Sub(expLocal, scaledLocal, maxLocal, tileLength);
float rowSum = sumLocal.GetValue(0);
float invSum = 1.0f / rowSum;
```

Benefit: 数值稳定性更好，避免极端输入下的数值溢出或下溢
Trade-off: 高阶API可能引入少量额外开销

---

## Variant Z: 特殊值处理（0, Inf, NaN）
Source: trans_quant_param_v2

在FP32到FP19转换过程中，专家实现专门处理了三种特殊值：0（exponent=0）、无穷大（Inf）和非数字（NaN，exponent=255）。对于0，直接生成符号位；对于Inf/NaN，保留符号位、指数位和尾数位的高位。这种处理确保了量化参数转换的数学正确性，避免了特殊值在转换过程中产生未定义行为。

**Expert implementation:**
```cpp
if (exponent == 0) {
    fp19Scale = sign << FP19_SIGN_SHIFT;
} else if (exponent == MAX_INT9) {
    fp19Scale = (sign << FP19_SIGN_SHIFT) | (exponent << FP19_EXP_SHIFT) | fp19Mantissa;
} else {
    if ((roundBit != 0) && ((stickyBits != 0) || ((fp19Mantissa & 0x1) != 0))) {
        fp19Mantissa += 1;
    }
}
```

**vs. baseline:**
```cpp
# 无特殊值处理
```

Benefit: 确保特殊值的数学正确性，避免未定义行为
Trade-off: 增加了代码分支复杂度

---

## Variant 27: Flash Attention在线Softmax数值稳定性
Source: flash_attention_score, fused_floyd_attention

Flash Attention采用在线Softmax算法(Online Softmax)，在S2维度分块迭代中维护全局running max和running sum。每个S2块计算局部max后，与全局max比较并rescale之前的累加结果，避免了完整S1×S2矩阵的exp计算溢出。支持多种稀疏掩码模式(Causal/Band/Prefix)下的无效位置处理：通过预计算偏移量将无效位置设为极小值(-10000.0)，确保Softmax后这些位置概率接近0。

**Expert implementation:**
```cpp
// Online Softmax: 分块迭代维护全局max和sum
constexpr static AscendC::SoftmaxConfig SOFTMAX_DEFAULT_CFG = {false};
constexpr static int32_t SOFTMAX_M_ALIGNED_SIZE = 8;
constexpr static int32_t SOFTMAX_K_ALIGNED_SIZE = 64;

// Per S2-block iteration:
// 1. Compute BMM1: scores[S1, S2_block] = Q @ K_block^T
// 2. Apply attention mask (set invalid positions to -10000.0)
// 3. Online softmax update:
//    new_max = max(running_max, block_max)
//    rescale_factor = exp(running_max - new_max)
//    running_sum = running_sum * rescale_factor + block_sum
//    running_max = new_max
// 4. Rescale previous output: output *= rescale_factor
// 5. BMM2: output += softmax_scores @ V_block

// Causal mask: precomputed offset avoids per-element branching
__aicore__ inline int64_t ComputeOffsetForCausal(const int64_t &delta,
    const uint32_t &s1BaseSize, const uint32_t &s2BaseSize, const uint32_t &attenMaskS2Size) {
    if (delta <= 0) return Min(-1 * delta, s1BaseSize);
    else return Min(delta, s2BaseSize) * attenMaskS2Size;
}
```

**vs. baseline:**
```cpp
// 标准Softmax：需要完整S1×S2矩阵，大序列长度时溢出风险
AscendC::ReduceMax(maxLocal, scoresLocal, sharedLocal, seqLength);  // 全局max
float rowMax = maxLocal.GetValue(0);
AscendC::Adds(expLocal, scoresLocal, -rowMax, seqLength);
AscendC::Exp(expLocal, expLocal, seqLength);  // exp(x-max), 可能需要O(S1×S2)内存
AscendC::ReduceSum(sumLocal, expLocal, sharedLocal, seqLength);
```

Benefit: 内存从O(S1×S2)降至O(S1×block_size)；避免完整矩阵的exp溢出；支持稀疏掩码预计算偏移
Trade-off: 在线更新需要额外的rescaling计算；分块策略增加控制流复杂度

---

## Variant 28: 可变形注意力双线性插值边界安全
Source: multi_scale_deformable_attn_function

多尺度可变形注意力的采样点可能落在特征图边界外，专家实现通过三级条件判断确保边界安全：1）完全在界内(0<=y0<h && 0<x1<w)：使用双倍拷贝参数一次加载两个角点值；2）部分在界内(0<=x0<w)：使用单次拷贝参数仅加载有效角点；3）完全越界：跳过拷贝（UB中该位置保持零值初始化）。这种分级处理避免了越界访问导致的未定义行为，同时最大化有效数据的传输效率。

**Expert implementation:**
```cpp
// 三级边界条件处理
if (0 <= y0 && y0 < h) {
    if (0 < x1 && x1 < w) {
        // 两个角点都有效：double copy
        DataCopy(value[pingOffset + point * alignedEmbedDims_],
            valueGm_[srcOffset_ + (y0 * w + x0) * outDims_], cpDoubleValParams_);
    } else if (0 <= x0 && x0 < w) {
        // 仅左角点有效：single copy
        DataCopy(value[pingOffset + point * alignedEmbedDims_],
            valueGm_[srcOffset_ + (y0 * w + x0) * outDims_], cpOneValParams_);
    }
    // 完全越界：跳过（零值）
}
```

**vs. baseline:**
```cpp
// 简单clamp，不区分边界状态
int x0 = max(0, min((int)floor(x), w-1));
int y0 = max(0, min((int)floor(y), h-1));
float val = getValue(y0, x0);  // 边界处重复边缘值，精度不如零填充
```

Benefit: 避免越界访问；分级拷贝最大化有效数据传输效率；零填充语义与标准CV框架一致
Trade-off: 三级条件分支增加控制流复杂度
